import logging
import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.amp import autocast, GradScaler
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from stabddg.intact.dataset import IntactDataset, IntactContrastiveStream, intact_collate_fn, passthrough_collate_fn
from stabddg.model import StaBddG
from stabddg.training import _is_dist_initialized, _is_main_process, _unwrap_model

torch.set_float32_matmul_precision("high")
default_amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContrastiveLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        reduction: str = "mean",
        normalize: bool = False,
    ):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.normalize = normalize
        super(ContrastiveLoss, self).__init__(reduction=reduction)

    def _component_losses(self, z, z_sign, z_pos, z_neg, z_neu):
        """
        Compute per-sample component losses before weighting and final reduction.
        Returns:
          loss_pos_vec: (B,) tensor
          loss_neg_vec: (B,) tensor
        """
        # Normalising inputs
        if self.normalize:
            z = F.normalize(z, dim=0)
            z_pos = F.normalize(z_pos, dim=0)
            z_neg = F.normalize(z_neg, dim=0)
            z_neu = F.normalize(z_neu, dim=0)

        # Concatenating positive and negative samples
        z_posneg = torch.cat([z_pos, z_neg], dim=0)
        z_posneg_sign = torch.cat([torch.ones_like(z_pos), -torch.ones_like(z_neg)], dim=0)

        # Weights matrix indicating positive and negative samples
        w = (z_sign * z_posneg_sign.T)
        same_mask = (w > 0)
        opp_mask = (w < 0)

        # Calculating cosine similarities
        s = (z @ z_posneg.T)         # (B, N)
        s_pos = s / self.tau_pos     # (B, N)
        s_neg = s / self.tau_neg     # (B, N)

        # Neutral denominators (mirrored space for the opp term)
        s_neu = (z @ z_neu.T)  # (B, B_neu)
        lse_neu_pos = torch.logsumexp(s_neu / self.tau_pos, dim=1, keepdim=True)  # [B, 1]
        lse_neu_neg = torch.logsumexp(-s_neu / self.tau_neg, dim=1, keepdim=True)  # [B, 1]

        # Counts per anchor
        pos_count = same_mask.sum(dim=1, keepdim=True)  # [B, 1]
        neg_count = opp_mask.sum(dim=1, keepdim=True)  # [B, 1]

        # Sums over selected pairs
        same_mask_f = same_mask.to(s.dtype)
        opp_mask_f = opp_mask.to(s.dtype)
        pos_sum = (s_pos * same_mask_f).sum(dim=1, keepdim=True)  # [B, 1]
        neg_sum = (s_neg * opp_mask_f).sum(dim=1, keepdim=True)  # [B, 1]

        # Safe divisors
        pos_count_safe = torch.clamp(pos_count, min=1)
        neg_count_safe = torch.clamp(neg_count, min=1)

        # Averaging: subtract count * LSE, then divide by count
        # Note that the ddG value for +ve pairs should be negative
        loss_pos_vec = -(pos_sum - pos_count * lse_neu_pos) / pos_count_safe
        loss_neg_vec = -(-neg_sum - neg_count * lse_neu_neg) / neg_count_safe

        # Zero-out anchors with no samples of that type
        loss_pos_vec = loss_pos_vec * (pos_count > 0).to(s.dtype)
        loss_neg_vec = loss_neg_vec * (neg_count > 0).to(s.dtype)

        return loss_pos_vec.squeeze(1), loss_neg_vec.squeeze(1)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        if self.reduction == "sum":
            return x.sum()
        # 'none' or any other value: return as-is
        return x

    def forward(self, z, z_sign, z_pos, z_neg, z_neu):
        """
        Returns a differentiable scalar (or vector if reduction='none') loss.
        """
        loss_pos_vec, loss_neg_vec = self._component_losses(z, z_sign, z_pos, z_neg, z_neu)
        total_vec = self.lambda_pos * loss_pos_vec + self.lambda_neg * loss_neg_vec
        loss = self._reduce(total_vec)
        return loss

    def return_losses_and_metrics(
        self,
        z: torch.Tensor,
        z_sign: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        z_neu: torch.Tensor
    ):
        """
        Returns:
          loss: differentiable tensor (scalar unless reduction='none')
          metrics: dict of detached scalar components:
            - 'loss': reduced total loss
            - 'loss_pos': reduced positive component (unweighted)
            - 'loss_neg': reduced negative component (unweighted)
        """
        loss_pos_vec, loss_neg_vec = self._component_losses(z, z_sign, z_pos, z_neg, z_neu)
        total_vec = self.lambda_pos * loss_pos_vec + self.lambda_neg * loss_neg_vec
        loss = self._reduce(total_vec)

        metrics = {
            "loss": self._reduce(total_vec.detach()).item(),
            "loss_pos": self._reduce(loss_pos_vec.detach()).item(),
            "loss_neg": self._reduce(loss_neg_vec.detach()).item(),
        }
        return loss, metrics


def train_step(
    model: StaBddG,
    dataloader: DataLoader,
    dataloader_contrastive: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveLoss,
    epoch: int,
    output_logger = logger,
    grad_accum_steps: int = 1,
    amp_dtype: torch.dtype = default_amp_dtype
) -> tuple[dict[str, float], pd.DataFrame]:
    # Preparing the model and the optimiser for training
    _unwrap_model(model).train()
    scaler = getattr(train_step, "_scaler", None)
    if scaler is None:
        scaler = GradScaler(enabled=torch.cuda.is_available() and amp_dtype == torch.float16)
        train_step._scaler = scaler
    optimizer.zero_grad(set_to_none=True)
    print_prefix = f"Epoch {epoch + 1}"

    df_forecasts = pd.DataFrame()
    all_train_metrics = None

    i = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, disable=not _is_main_process())
    for batch in pbar:
        # Fetching a sample of contrastive datapoints
        batch_contrast = next(iter(dataloader_contrastive))

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            # Use fused model method to reduce graph fragmentation and allocations
            z_anchor = _unwrap_model(model).fused_forward_intact_datapoint(batch)
            z_pos = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["positive"])
            z_neg = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["negative"])
            z_neu = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["neutral"])

            z_sign = batch["sign"].to(z_anchor.device, non_blocking=True)

            loss, metrics = loss_fn.return_losses_and_metrics(
                z=z_anchor.unsqueeze(-1),
                z_sign=z_sign.unsqueeze(-1),
                z_pos=z_pos.unsqueeze(-1),
                z_neg=z_neg.unsqueeze(-1),
                z_neu=z_neu.unsqueeze(-1),
            )

        # AMP-aware backward
        if scaler.is_enabled():
            scaler.scale(loss / max(grad_accum_steps, 1)).backward()
        else:
            (loss / max(grad_accum_steps, 1)).backward()

        if ((i + 1) % max(grad_accum_steps, 1)) == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if _is_main_process():
            pbar.set_description(f"Train Batch {i + 1}")
            pbar.set_postfix(metrics)
            # pbar.update(1)
            output_logger.info(
                f'{print_prefix}: Batch {i + 1} Train metrics: '
                f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }'
            )

        if not all_train_metrics:
            all_train_metrics = metrics
        else:
            for k, v in metrics.items():
                all_train_metrics[k] = (all_train_metrics[k] * i + v) / (i + 1)

        i += 1

        df_forecasts_cur = pd.DataFrame(
            {
                "anchor_idx": batch["anchor_idx"].detach().cpu().numpy(),
                "forecast": z_anchor.detach().cpu().numpy(),
            }
        )
        df_forecasts = pd.concat([df_forecasts, df_forecasts_cur], ignore_index=True)

    # Final optimizer step if loop ended mid-accumulation
    if (i % max(grad_accum_steps, 1)) != 0:
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    df_forecasts["epoch"] = epoch + 1
    df_forecasts["split"] = "Train"

    return all_train_metrics, df_forecasts


def validation_step(
    model: StaBddG,
    dataloader: DataLoader,
    dataloader_contrastive: DataLoader,
    loss_fn: ContrastiveLoss,
    epoch: int,
    step_name: str = "Validation",
    output_logger = logger,
    amp_dtype: torch.dtype = default_amp_dtype,
) -> tuple[dict[str, float], pd.DataFrame]:
    _unwrap_model(model).eval()
    print_prefix = f"Epoch {epoch + 1}"

    df_forecasts = pd.DataFrame()
    all_metrics = None
    i = 0
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        pbar = tqdm(dataloader, desc=f"{step_name} Epoch {epoch}", leave=False, disable=not _is_main_process())
        for batch in pbar:
            # Fetching a sample of contrastive datapoints
            batch_contrast = next(iter(dataloader_contrastive))

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                # Use fused model method to reduce graph fragmentation and allocations
                z_anchor = _unwrap_model(model).fused_forward_intact_datapoint(batch)
                z_pos = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["positive"])
                z_neg = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["negative"])
                z_neu = _unwrap_model(model).fused_forward_intact_datapoint(batch_contrast["neutral"])

                z_sign = batch["sign"].to(z_anchor.device, non_blocking=True)

                loss, metrics = loss_fn.return_losses_and_metrics(
                    z=z_anchor.unsqueeze(-1),
                    z_sign=z_sign.unsqueeze(-1),
                    z_pos=z_pos.unsqueeze(-1),
                    z_neg=z_neg.unsqueeze(-1),
                    z_neu=z_neu.unsqueeze(-1),
                )

            if _is_main_process():
                if isinstance(pbar, tqdm):
                    pbar.set_description(f"{step_name} Batch {i + 1}")
                    pbar.set_postfix(metrics)
                    # pbar.update(1)
                output_logger.info(
                    f'{print_prefix}: Batch {i + 1} {step_name} metrics: '
                    f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }'
                )

            if not all_metrics:
                all_metrics = metrics
            else:
                for k, v in metrics.items():
                    all_metrics[k] = (all_metrics[k] * i + v) / (i + 1)
            i += 1

            df_forecasts_cur = pd.DataFrame(
                {
                    "anchor_idx": batch["anchor_idx"].detach().cpu().numpy(),
                    "forecast": z_anchor.detach().cpu().numpy(),
                }
            )
            df_forecasts = pd.concat([df_forecasts, df_forecasts_cur], ignore_index=True)

    if _is_dist_initialized() and all_metrics:
        device = next(_unwrap_model(model).parameters()).device
        for k, v in list(all_metrics.items()):
            t = torch.tensor([v], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            all_metrics[k] = (t / max(world_size, 1)).item()

    all_metrics = all_metrics if all_metrics is not None else {"loss": 0.0, "loss_pos": 0.0, "loss_neg": 0.0}
    df_forecasts["epoch"] = epoch + 1
    df_forecasts["split"] = step_name

    return all_metrics, df_forecasts


def pretrain(
    model: StaBddG,
    dataset_train: IntactDataset,
    dataset_valid: IntactDataset,
    dataset_test: IntactDataset,
    save_dir: str,
    batch_size: int = 4,
    num_dataloader_workers: int = 1,
    lr: float = 1e-4,
    normalize_loss: bool = True,
    n_epochs: int = 10,
    model_val_freq: int = 2,
    model_save_freq: int = 1,
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss(
        lambda_pos=1.0,
        lambda_neg=dataset_train.df_pos.shape[0] / dataset_train.df_neg.shape[0],
        normalize=normalize_loss
    )

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()
    # DataFrame with forecasts
    df_forecasts = pd.DataFrame()

    # Directory to save model checkpoints
    model_save_dir = os.path.join(save_dir, "model")
    metrics_save_dir = os.path.join(save_dir, "metrics")
    if _is_main_process():
        logger.info(f"Creating directory {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(metrics_save_dir, exist_ok=True)

    # Creating a logging file logs.txt (main process only)
    if _is_main_process():
        log_path = os.path.join(save_dir, "logs.txt")
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger_file = logging.getLogger("file_only")
        logger_file.setLevel(logging.INFO)
        logger_file.propagate = False
        logger_file.addHandler(file_handler)

        logger.info(f"Logging to {log_path}")

    # Preparing data loaders (batch_size fixed to 1, use a collate_fn that returns the single element)
    train_sampler = DistributedSampler(dataset_train, shuffle=True) if _is_dist_initialized() else None
    valid_sampler = DistributedSampler(dataset_valid, shuffle=False) if _is_dist_initialized() else None
    test_sampler = DistributedSampler(dataset_test, shuffle=False) if _is_dist_initialized() else None

    ds_contrastive = IntactContrastiveStream(dataset_train)

    dl_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=intact_collate_fn,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=intact_collate_fn,
        pin_memory=True,
    )
    dl_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=intact_collate_fn,
        pin_memory=True,
    )
    dl_contrastive = DataLoader(
        ds_contrastive,
        batch_size=1,
        shuffle=False,
        num_workers=num_dataloader_workers,
        collate_fn=passthrough_collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    # Saving the initial model checkpoint
    if _is_main_process():
        logger.info("Saving model checkpoint at epoch 0")
        torch.save(
            _unwrap_model(model).pmpnn.state_dict(),
            f"{model_save_dir}/initial.pt"
        )

    # Iterating over epochs
    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        # Running training step
        train_metrics, df_forecasts_train = train_step(
            model=model,
            dataloader=dl_train,
            dataloader_contrastive=dl_contrastive,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            grad_accum_steps=grad_accum_steps,
            output_logger=logger_file,
        )

        # Collecting metrics from the training step
        if _is_main_process():
            logger.info(
                f'Epoch {epoch + 1}: Train metrics: '
                f'{ {k: "{0:0.4f}".format(v) for k, v in train_metrics.items() if v is not None} }'
            )

            df_train_metrics = pd.DataFrame(
                {"epoch": epoch + 1, "split": "Train"} | train_metrics,
                index=[0]
            )
            df_metrics = pd.concat([df_metrics, df_train_metrics], ignore_index=True)
            df_forecasts = pd.concat([df_forecasts, df_forecasts_train], ignore_index=True)
            
            # Potentially saving model checkpoint
            if (epoch + 1) % model_save_freq == 0:
                logger.info(f"Saving model checkpoint at epoch {epoch + 1}")
                torch.save(
                    _unwrap_model(model).pmpnn.state_dict(),
                    f"{model_save_dir}/epoch_{epoch + 1}.pt",
                )

        # Running validation steps across all ranks
        if (epoch + 1) % model_val_freq == 0:
            valid_metrics, df_forecasts_valid = validation_step(
                model=model,
                dataloader=dl_valid,
                dataloader_contrastive=dl_contrastive,
                loss_fn=loss_fn,
                epoch=epoch,
                output_logger=logger_file,
                step_name="Validation",
            )

            if _is_main_process():
                logger.info(
                    f'Epoch {epoch + 1}: Validation metrics: '
                    f'{ {k: "{0:0.4f}".format(v) for k, v in valid_metrics.items() if v is not None} }'
                )

                df_valid_metrics = pd.DataFrame(
                    {"epoch": epoch + 1, "split": "Validation"} | valid_metrics,
                    index=[0]
                )
                df_metrics = pd.concat([df_metrics, df_valid_metrics], ignore_index=True)
                df_forecasts = pd.concat([df_forecasts, df_forecasts_valid], ignore_index=True)

        # Save metrics and forecasts as CSV
        if _is_main_process():
            df_metrics.to_csv(os.path.join(metrics_save_dir, "metrics.csv"), index=False)
            df_forecasts.to_csv(os.path.join(metrics_save_dir, "forecasts.csv"), index=False)

            if use_wandb:
                wandb.log(valid_metrics, step=epoch + 1)
                wandb.log({"lr": optimizer.param_groups[0]["lr"],}, step=epoch + 1)

    # Calculating the test metrics at the end of the training loop
    logger.info("Calculating test metrics at the end of the training loop")
    test_metrics, df_forecasts_test = validation_step(
        model=model,
        dataloader=dl_test,
        dataloader_contrastive=dl_contrastive,
        loss_fn=loss_fn,
        epoch=n_epochs,
        step_name="Test",
        output_logger=logger_file,
    )

    if _is_main_process():
        logger.info(
            f'Epoch {epoch + 1}: Test metrics: '
            f'{ {k: "{0:0.4f}".format(v) for k, v in test_metrics.items() if v is not None} }'
        )

        df_test_metrics = pd.DataFrame(
            {"epoch": n_epochs, "split": "Test"} | test_metrics,
            index=[0]
        )
        df_metrics = pd.concat([df_metrics, df_test_metrics], ignore_index=True)
        df_forecasts = pd.concat([df_forecasts, df_forecasts_test], ignore_index=True)

        df_metrics.to_csv(os.path.join(metrics_save_dir, "metrics.csv"), index=False)
        df_forecasts.to_csv(os.path.join(metrics_save_dir, "forecasts.csv"), index=False)

    # Saving the final model checkpoint
    if _is_main_process():
        torch.save(
            _unwrap_model(model).pmpnn.state_dict(), f"{model_save_dir}/final.pt"
        )
