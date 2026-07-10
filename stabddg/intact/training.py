import atexit
import contextlib
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


class _RunLogger:
    """Best-effort experiment tracking to SageMaker Experiments and/or MLflow.

    Both sinks are entirely optional and failure-tolerant: a run that has neither available logs to
    neither and trains exactly as before. Used on the main process only.

    * **SageMaker Experiments** — ``load_run()`` attaches to the run SageMaker auto-creates for the
      training job (and, under Pipelines, the execution), so metrics show under Studio -> Experiments
      with no standing infrastructure cost.
    * **MLflow** — enabled only when ``MLFLOW_TRACKING_URI`` is set, so a cloud job never tries to
      reach a server that isn't there. To track locally: ``mlflow ui`` (or a bare ``./mlruns`` file
      store) and ``export MLFLOW_TRACKING_URI=http://127.0.0.1:5000`` before a local training run.
    """

    def __init__(self):
        self._stack = contextlib.ExitStack()
        self._sm_run = None
        self._mlflow = None

    def start(self, params: dict | None = None) -> "_RunLogger":
        try:
            import boto3
            import sagemaker
            from sagemaker.experiments.run import load_run

            # Build a region-aware session explicitly: inside the training container the default
            # boto session often has no region, which makes load_run() raise "Must setup local AWS
            # configuration with a region". SageMaker sets AWS_REGION/AWS_DEFAULT_REGION on the job.
            region = (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or boto3.session.Session().region_name
            )
            sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=region))
            self._sm_run = self._stack.enter_context(load_run(sagemaker_session=sm_session))
            logger.info("SageMaker Experiments run attached (region=%s)", region)
        except Exception as exc:  # not in SageMaker, SDK missing, no active run, etc.
            logger.info("SageMaker Experiments unavailable (%s); skipping", exc)
            self._sm_run = None

        if os.environ.get("MLFLOW_TRACKING_URI"):
            try:
                import mlflow

                self._mlflow = mlflow
                self._stack.enter_context(mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME")))
                logger.info("MLflow run started (tracking_uri=%s)", os.environ["MLFLOW_TRACKING_URI"])
            except Exception as exc:
                logger.info("MLflow unavailable (%s); skipping", exc)
                self._mlflow = None

        atexit.register(self.close)
        self.log_params(params or {})
        return self

    def log_params(self, params: dict):
        if not params:
            return
        if self._sm_run is not None:
            try:
                self._sm_run.log_parameters({k: v for k, v in params.items()})
            except Exception:
                pass
        if self._mlflow is not None:
            try:
                self._mlflow.log_params(params)
            except Exception:
                pass

    def log_metrics(self, metrics: dict | None, step: int):
        if not metrics:
            return
        for k, v in metrics.items():
            fv = _maybe_float(v)
            if fv is None:
                continue
            if self._sm_run is not None:
                try:
                    self._sm_run.log_metric(name=k, value=fv, step=step)
                except Exception:
                    pass
            if self._mlflow is not None:
                try:
                    self._mlflow.log_metric(k, fv, step=step)
                except Exception:
                    pass

    def close(self):
        # ExitStack.close() is idempotent, so an explicit close + the atexit fallback are both safe.
        self._stack.close()


def _slice_pool(batch: dict, start: int, end: int) -> dict:
    """Slice a collated anchor/pool batch along dim 0 (the datapoint axis).

    A batch is ``{"complex"/"binder1"/"binder2": {tensor_key: [B, ...]}, ...}`` (plus a scalar
    ``max_length`` per branch, and ``anchor_idx``/``sign`` on the anchor batch). We slice every tensor
    ``[start:end]`` and pass non-tensors (e.g. ``max_length``) through untouched.
    """
    out: dict = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            out[key] = {k: (v[start:end] if torch.is_tensor(v) else v) for k, v in val.items()}
        elif torch.is_tensor(val):
            out[key] = val[start:end]
        else:
            out[key] = val
    return out


def _microbatched_forward(model, batch: dict, micro_batch_size: int) -> torch.Tensor:
    """``fused_forward_intact_datapoint`` over a collated batch, optionally in chunks.

    When ``micro_batch_size > 0`` (and smaller than the batch), the batch is split into chunks of that
    many datapoints, each forwarded separately, and the scalar outputs concatenated back to ``[B]``.
    Concatenation is the whole "un-glue": the loss consumes these as a flat dim-0 stack, and grads
    flow through ``torch.cat``. This caps *forward-time* peak activation memory — fully so for
    no-grad pools (neutrals, validation), where each chunk's activations free immediately. For
    grad-requiring pools the backward still needs every chunk's activations at once, so pair this with
    gradient checkpointing to also cap the backward peak.
    """
    fwd = _unwrap_model(model).fused_forward_intact_datapoint
    B = batch["complex"]["S"].shape[0]
    if not micro_batch_size or micro_batch_size <= 0 or micro_batch_size >= B:
        return fwd(batch)
    outs = [
        fwd(_slice_pool(batch, s, min(s + micro_batch_size, B)))
        for s in range(0, B, micro_batch_size)
    ]
    return torch.cat(outs, dim=0)


def _forward_neutrals(model, loss_fn, neutral_batch, ref: torch.Tensor, micro_batch_size: int = 0) -> torch.Tensor:
    """Forward the neutral contrast pool, but only with the autograd graph it actually needs.

    Neutrals feed two things in ``ContrastiveLoss``: the (always-detached) robust centre/scale
    statistics, and — only when ``lambda_neutral > 0`` — a neutral-regularisation term. So:
      * ``lambda_neutral > 0``  -> forward WITH grad (the reg term needs it);
      * normaliser on, λ == 0    -> forward under ``no_grad`` (stats are detached anyway);
      * both off                 -> skip entirely, returning an empty ``(0,)`` tensor.

    Skipping / no-grad-ing the neutral pool frees the activations of the *largest* contrast pool
    (``k_neutral`` ≫ ``k_pos``/``k_neg``), which is the dominant lever against the high-k OOM.
    ``ContrastiveLoss`` already handles an empty neutral tensor (numel == 0).
    """
    need_grad = getattr(loss_fn, "lambda_neutral", 0.0) > 0.0
    need_neu = need_grad or getattr(loss_fn, "use_neutral_normalizer", False)
    # neutral_batch is None when the sampler skips neutrals (k_neutral == 0); an empty dict is also
    # treated as "nothing to forward". Either way, and whenever neutrals aren't needed, return empty.
    if not need_neu or not neutral_batch:
        return ref.new_zeros(0)
    if need_grad:
        return _microbatched_forward(model, neutral_batch, micro_batch_size)
    with torch.no_grad():
        return _microbatched_forward(model, neutral_batch, micro_batch_size)


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
    micro_batch_size: int = 0,
    log_every_batches: int = 1,
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
    n_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, disable=not _is_main_process())

    # Persistent iterator over the (infinite) contrastive stream: calling iter() on every batch
    # would restart the DataLoader workers and always return the very first sampled pool.
    contrast_iter = iter(dataloader_contrastive)

    # Iterating over batches
    for batch in pbar:
        # Fetching a fresh sample of contrastive datapoints
        try:
            batch_contrast = next(contrast_iter)
        except StopIteration:
            contrast_iter = iter(dataloader_contrastive)
            batch_contrast = next(contrast_iter)

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            # Fused per-pool forward, optionally micro-batched to cap peak activation memory.
            z_anchor = _microbatched_forward(model, batch, micro_batch_size)
            z_pos = _microbatched_forward(model, batch_contrast["positive"], micro_batch_size)
            z_neg = _microbatched_forward(model, batch_contrast["negative"], micro_batch_size)
            z_neu = _forward_neutrals(model, loss_fn, batch_contrast["neutral"], z_anchor, micro_batch_size)

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
            batch_line = (
                f'{print_prefix}: Batch {i + 1} Train metrics: '
                f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }'
            )
            # Full per-batch record to logs.txt; throttled copy to stdout so CloudWatch (and
            # metric_definitions) get per-batch metrics without one line per step drowning the log.
            output_logger.info(batch_line)
            if (i % max(log_every_batches, 1) == 0) or (i + 1 == n_batches):
                logger.info(batch_line)
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
                "sign": batch["sign"].detach().cpu().numpy(),
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
    micro_batch_size: int = 0,
    log_every_batches: int = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    _unwrap_model(model).eval()
    print_prefix = f"Epoch {epoch + 1}"

    df_forecasts = pd.DataFrame()
    all_metrics = None
    i = 0
    n_batches = len(dataloader)
    with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        pbar = tqdm(dataloader, desc=f"{step_name} Epoch {epoch}", leave=False, disable=not _is_main_process())
        contrast_iter = iter(dataloader_contrastive)
        for batch in pbar:
            # Fetching a fresh sample of contrastive datapoints
            try:
                batch_contrast = next(contrast_iter)
            except StopIteration:
                contrast_iter = iter(dataloader_contrastive)
                batch_contrast = next(contrast_iter)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                # Fused per-pool forward, optionally micro-batched to cap peak activation memory.
                z_anchor = _microbatched_forward(model, batch, micro_batch_size)
                z_pos = _microbatched_forward(model, batch_contrast["positive"], micro_batch_size)
                z_neg = _microbatched_forward(model, batch_contrast["negative"], micro_batch_size)
                z_neu = _forward_neutrals(model, loss_fn, batch_contrast["neutral"], z_anchor, micro_batch_size)

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
                batch_line = (
                    f'{print_prefix}: Batch {i + 1} {step_name} metrics: '
                    f'{ {k: "{0:0.4f}".format(v) for k, v in metrics.items() if v is not None} }'
                )
                output_logger.info(batch_line)
                if (i % max(log_every_batches, 1) == 0) or (i + 1 == n_batches):
                    logger.info(batch_line)
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
    micro_batch_size: int = 0,
    num_dataloader_workers: int = 1,
    lr: float = 1e-4,
    lambda_supcon: float = 1.0,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    use_neutral_normalizer: bool = False,
    n_epochs: int = 10,
    model_val_freq: int = 2,
    model_save_freq: int = 1,
    log_every_batches: int = 1,
    use_wandb: bool = False,
    grad_accum_steps: int = 1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss(
        lambda_supcon=lambda_supcon,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
        use_neutral_normalizer=use_neutral_normalizer,
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

    # Default to the module logger on every rank; the main process additionally attaches a file handler
    # below. Without this, non-main ranks would hit an UnboundLocalError when passing logger_file on.
    logger_file = logger

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
        persistent_workers=(num_dataloader_workers > 0),
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

    # Experiment tracking (SageMaker Experiments and/or MLflow); main process only, best-effort.
    run_logger = (
        _RunLogger().start(
            {
                "lr": lr,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "lambda_supcon": lambda_supcon,
                "lambda_sign": lambda_sign,
                "lambda_neutral": lambda_neutral,
                "model_val_freq": model_val_freq,
            }
        )
        if _is_main_process()
        else None
    )

    for epoch in tqdm(range(n_epochs), desc="Epoch"):
        # Reshuffle distinctly each epoch under DistributedSampler (no-op otherwise)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            micro_batch_size=micro_batch_size,
            log_every_batches=log_every_batches,
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
            if run_logger is not None:
                run_logger.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, epoch + 1)
                run_logger.log_metrics({"train/lr": optimizer.param_groups[0]["lr"]}, epoch + 1)


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
                micro_batch_size=micro_batch_size,
                log_every_batches=log_every_batches,
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
                if run_logger is not None:
                    run_logger.log_metrics({f"validation/{k}": v for k, v in valid_metrics.items()}, epoch + 1)

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
        micro_batch_size=micro_batch_size,
        log_every_batches=log_every_batches,
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
        if run_logger is not None:
            run_logger.log_metrics({f"test/{k}": v for k, v in test_metrics.items()}, n_epochs)

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

    if run_logger is not None:
        run_logger.close()
