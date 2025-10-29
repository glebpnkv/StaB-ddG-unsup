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

        # Calculating cosine similarities
        s_pos = (z * z_sign) @ z_pos.T / self.tau_pos     # (B, B_pos)
        s_neg = (z * z_sign) @ z_neg.T / self.tau_neg     # (B, B_neg)
        s_neu = z @ z_neu.T                               # (B, B_neu)

        # Per-sample vectors (B,)
        loss_pos_vec = -torch.mean(s_pos - torch.logsumexp(s_neu, dim=-1, keepdims=True), dim=-1)
        loss_neg_vec = -torch.mean(-s_neg - torch.logsumexp(-s_neu, dim=-1, keepdims=True), dim=-1)
        return loss_pos_vec, loss_neg_vec

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
    grad_accum_steps: int = 1,
    amp_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
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
            pbar.update(1)
            logger.info(f'{print_prefix}: Train metrics: '
                        f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }')

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
    amp_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
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
                    pbar.update(1)
                logger.info(f'{print_prefix}: {step_name} metrics: '
                            f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }')

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
    run_name: str,
    model_save_dir: str,
    num_dataloader_workers: int = 1,
    lr: float = 1e-4,
    n_epochs: int = 10,
    model_save_freq: int = 1,
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss()

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()
    # DataFrame with forecasts
    df_forecasts = pd.DataFrame()

    # Directory to save model checkpoints
    if _is_main_process() and not os.path.exists(model_save_dir):
        logger.info(f"Creating directory {model_save_dir}")
        os.makedirs(model_save_dir, exist_ok=True)

    # Creating a logging file logs.txt (main process only)
    if _is_main_process():
        log_path = os.path.join(model_save_dir, "logs.txt")
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")

    # DDP state
    world_size = dist.get_world_size() if _is_dist_initialized() else 1
    rank = dist.get_rank() if _is_dist_initialized() else 0

    # Preparing data loaders (batch_size fixed to 1, use a collate_fn that returns the single element)
    train_sampler = DistributedSampler(dataset_train, shuffle=True) if _is_dist_initialized() else None
    valid_sampler = DistributedSampler(dataset_valid, shuffle=False) if _is_dist_initialized() else None
    test_sampler = DistributedSampler(dataset_test, shuffle=False) if _is_dist_initialized() else None

    ds_contrastive = IntactContrastiveStream(dataset_train)

    dl_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=intact_collate_fn,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=intact_collate_fn,
        pin_memory=True,
    )
    dl_test = DataLoader(
        dataset_test,
        batch_size=1,
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
        )

        # Collecting metrics from the training step
        if _is_main_process():
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
                    f"{model_save_dir}/{run_name}_epoch{epoch}.pt",
                )

        # Running validation and test steps across all ranks
        valid_metrics, df_forecasts_valid = validation_step(
            model=model,
            dataloader=dl_valid,
            dataloader_contrastive=dl_contrastive,
            loss_fn=loss_fn,
            epoch=epoch,
            step_name="Validation"
        )
        test_metrics, df_forecasts_test = validation_step(
            model=model,
            dataloader=dl_test,
            dataloader_contrastive=dl_contrastive,
            loss_fn=loss_fn,
            epoch=epoch,
            step_name="Test"
        )

        if _is_main_process():
            df_valid_metrics = pd.DataFrame(
                {"epoch": epoch + 1, "split": "Validation"} | valid_metrics,
                index=[0]
            )
            df_test_metrics = pd.DataFrame(
                {"epoch": epoch + 1, "split": "Test"} | test_metrics,
                index=[0]
            )
            df_metrics = pd.concat([df_metrics, df_valid_metrics, df_test_metrics], ignore_index=True)
            df_forecasts = pd.concat([df_forecasts, df_forecasts_valid, df_forecasts_test], ignore_index=True)

            # Save metrics and forecasts as CSV
            df_metrics.to_csv(os.path.join(model_save_dir, "metrics.csv"), index=False)
            df_forecasts.to_csv(os.path.join(model_save_dir, "forecasts.csv"), index=False)

            if use_wandb:
                wandb.log(valid_metrics, step=epoch + 1)
                wandb.log(test_metrics, step=epoch + 1)
                wandb.log({"lr": optimizer.param_groups[0]["lr"],}, step=epoch + 1)
        
    # Saving the final model checkpoint
    if _is_main_process():
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(
            _unwrap_model(model).pmpnn.state_dict(), f"{model_save_dir}/{run_name}_final.pt"
        )
