import logging
import os

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from stabddg.intact.dataset import IntactDataset, IntactContrastiveStream, intact_collate_fn, passthrough_collate_fn
from stabddg.intact.losses import ContrastiveLoss
from stabddg.model import StaBddG
from stabddg.training import _is_dist_initialized, _is_main_process, _unwrap_model

torch.set_float32_matmul_precision("high")
default_amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_epoch_metrics(tb_writer, metrics, split_name, step):
    if not metrics:
        return

    payload = {}
    for k, v in metrics.items():
        v_float = _maybe_float(v)
        if v_float is None:
            continue
        payload[f"{split_name}/epoch/{k}"] = v_float

    if tb_writer is not None:
        for k, v in payload.items():
            tb_writer.add_scalar(k, v, step)


def train_step(
    model: StaBddG,
    dataloader: DataLoader,
    dataloader_contrastive: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveLoss,
    epoch: int,
    output_logger = logger,
    grad_accum_steps: int = 1,
    amp_dtype: torch.dtype = default_amp_dtype,
    tb_writer = None,
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
    
    # Iterating over batches
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
            if tb_writer is not None:
                step = epoch * len(dataloader) + i + 1
                for k, v in metrics.items():
                    v_float = _maybe_float(v)
                    if v_float is not None:
                        tb_writer.add_scalar(f"train/step/{k}", v_float, step)

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
    tb_writer = None,
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
                if tb_writer is not None:
                    step = epoch * len(dataloader) + i + 1
                    split_prefix = step_name.lower()
                    for k, v in metrics.items():
                        v_float = _maybe_float(v)
                        if v_float is not None:
                            tb_writer.add_scalar(f"{split_prefix}/step/{k}", v_float, step)

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
    lambda_supcon: float = 1.0,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    n_epochs: int = 10,
    model_val_freq: int = 2,
    model_save_freq: int = 1,
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss(
        lambda_supcon=lambda_supcon,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
    )

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()
    # DataFrame with forecasts
    df_forecasts = pd.DataFrame()

    # Directory to save model checkpoints
    model_save_dir = os.path.join(save_dir, "model")
    metrics_save_dir = os.path.join(save_dir, "metrics")
    default_tensorboard_log_dir = os.path.join(save_dir, "tb_logs")
    if _is_main_process():
        logger.info(f"Creating directory {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(metrics_save_dir, exist_ok=True)
        os.makedirs(default_tensorboard_log_dir, exist_ok=True)

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
    tb_writer = None
    tensorboard_log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", default_tensorboard_log_dir)

    logger.info("AIP_TENSORBOARD_LOG_DIR env: %s", os.environ.get("AIP_TENSORBOARD_LOG_DIR"))
    logger.info("Resolved tensorboard_log_dir: %s", tensorboard_log_dir)
    logger.info("SummaryWriter available: %s", SummaryWriter is not None)

    if _is_main_process() and tensorboard_log_dir and SummaryWriter is not None:
        try:
            tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
            logger.info("TensorBoard writer ENABLED (log_dir=%s)", tensorboard_log_dir)
            tb_writer.flush()
        except Exception as exc:
            logger.warning("Failed to create TensorBoard writer: %s", exc)
    else:
        logger.info("TensorBoard writer DISABLED (rank!=0 or no log dir)")

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
            tb_writer=tb_writer,
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

            # Adding epoch train metrics to TensorBoard
            _log_epoch_metrics(tb_writer, train_metrics, "train", epoch + 1)
            _log_epoch_metrics(
                tb_writer,
                {"lr": optimizer.param_groups[0]["lr"]},
                "train",
                epoch + 1,
            )
            
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
                tb_writer=tb_writer,
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

                # Adding epoch validation metrics to TensorBoard
                _log_epoch_metrics(tb_writer, valid_metrics, "validation", epoch + 1)

                if use_wandb:
                    wandb.log(valid_metrics, step=epoch + 1)
                    wandb.log({"lr": optimizer.param_groups[0]["lr"],}, step=epoch + 1)

        # Save metrics and forecasts as CSV
        if _is_main_process():
            df_metrics.to_csv(os.path.join(metrics_save_dir, "metrics.csv"), index=False)
            df_forecasts.to_csv(os.path.join(metrics_save_dir, "forecasts.csv"), index=False)

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

        _log_epoch_metrics(tb_writer, test_metrics, "test", n_epochs)

        df_metrics.to_csv(os.path.join(metrics_save_dir, "metrics.csv"), index=False)
        df_forecasts.to_csv(os.path.join(metrics_save_dir, "forecasts.csv"), index=False)

    # Saving the final model checkpoint
    if _is_main_process():
        torch.save(
            _unwrap_model(model).pmpnn.state_dict(), f"{model_save_dir}/final.pt"
        )

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
