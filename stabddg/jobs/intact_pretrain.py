import logging
import os
from typing import Optional

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from sklearn.model_selection import train_test_split

from stabddg.intact.dataset import IntactDataset
from stabddg.intact.training import pretrain
from stabddg.model import StaBddG
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.training import _is_dist_initialized, _is_main_process
from stabddg.utils.torch import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_intact_splits(
    data_dir: str,
    model_save_dir: str,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    splits_subdir: str = "data_splits",
    intact_sample_size: Optional[int] = None,
    write: bool = True,
) -> tuple[str, str, str]:
    """
    Creates df_train, df_valid, df_test and saves them into a dedicated sub-folder under model_save_dir.

    The split is deterministic in ``random_state``, so every rank computes identical paths/contents.
    Pass ``write=False`` on non-main ranks to avoid concurrent writes to the same files (the same
    paths are still returned); the caller is responsible for a barrier before reading them back.

    Returns
    -------
    tuple[str, str, str]
        Paths to (train_path, valid_path, test_path)
    """
    # Checks on valid_size and test_size
    assert valid_size + test_size <= 1.0, "valid_size + test_size must be <= 1.0"
    assert valid_size >= 0.0, "valid_size must be >= 0.0"
    assert test_size >= 0.0, "test_size must be >= 0.0"

    # Ensure output directory exists
    out_dir = os.path.join(model_save_dir, splits_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Load input parquet files
    df_mutations = pd.read_parquet(os.path.join(data_dir, "df_intact_mutations_filtered.parquet"))
    df_assemblies = pd.read_parquet(os.path.join(data_dir, "df_assemblies_filtered.parquet"))

    # Keep only mutations with the same length as the originals and not deletions (".")
    df_mutations = df_mutations.loc[
        (df_mutations["resulting_sequence"] != ".")
        & (df_mutations["original_sequence"].str.len() == df_mutations["resulting_sequence"].str.len())
    ].reset_index(drop=True)

    # Merge mutations and assemblies
    df = df_mutations.merge(
        right=df_assemblies,
        on=["participant_protein", "affected_protein_ac"],
        how="inner",
        validate="m:m",
    )

    # max_len drives the length filter in IntactDataset. The previously-computed
    # affected_protein_ac_seq_mut / *_len helper columns were unused downstream and were derived
    # via a buggy `.str.replace(r".", "")` (an unanchored regex that blanked the whole column), so
    # they have been removed.
    df["max_len"] = df["biological_assembly_length"]

    # Convert range indices back to 1-based
    df["feature_ranges_start"] += 1
    df["feature_ranges_end"] += 1

    # feature_type_category for stratification
    feature_type_neu = IntactDataset.feature_type_neutral
    feature_type_pos = IntactDataset.feature_type_pos
    feature_type_neg = IntactDataset.feature_type_neg

    df["feature_type_category"] = "unknown"
    df.loc[df["feature_type"].isin(feature_type_neu), "feature_type_category"] = "neutral"
    df.loc[df["feature_type"].isin(feature_type_pos), "feature_type_category"] = "positive"
    df.loc[df["feature_type"].isin(feature_type_neg), "feature_type_category"] = "negative"

    # Removing complexes where the mutation impact is unknown
    df = df.loc[df["feature_type_category"] != "unknown"].reset_index(drop=True)

    # Removing complexes formed by the same proteins
    df = df.loc[df["participant_protein"] != df["affected_protein_ac"]].reset_index(drop=True)

    if intact_sample_size:
        df = (
            df
            .groupby("feature_type_category")
            .sample(
                n=intact_sample_size,
                replace=True,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    # Stratified split: train vs (valid+test)
    if valid_size + test_size > 0:
        df_train, df_valid_all = train_test_split(
            df,
            test_size=(valid_size + test_size),
            stratify=df["feature_type_category"],
            random_state=random_state,
        )
    else:
        df_train = df
        df_valid_all = None

    # Split (valid+test) into valid and test
    if df_valid_all is None:
        df_valid, df_test = None, None
    else:
        df_valid, df_test = train_test_split(
            df_valid_all,
            test_size=test_size / (valid_size + test_size),
            stratify=df_valid_all["feature_type_category"],
            random_state=random_state,
        )

    # Drop helper column and reset indices
    for df_cur in [df_train, df_valid, df_test]:
        if df_cur is None:
            continue
        df_cur.drop(columns=["feature_type_category"], inplace=True)
        df_cur.reset_index(drop=True, inplace=True)

    # Save to the dedicated subfolder under model_save_dir
    train_path = os.path.join(out_dir, "df_intact_mutations_filtered_train.parquet")
    valid_path = os.path.join(out_dir, "df_intact_mutations_filtered_valid.parquet")
    test_path = os.path.join(out_dir, "df_intact_mutations_filtered_test.parquet")

    if write:
        df_train.to_parquet(train_path)

    if df_valid is None:
        valid_path = None
    elif write:
        df_valid.to_parquet(valid_path)

    if df_test is None:
        test_path = None
    elif write:
        df_test.to_parquet(test_path)

    return train_path, valid_path, test_path


def intact_pretrain(
    run_name: str,
    data_dir: str,
    assemblies_dir: str,
    model_save_dir,
    max_length: int = 256,
    k_neutral: int = 32,
    k_pos: int = 16,
    k_neg: int = 16,
    batch_size: int = 4,
    micro_batch_size: int = 0,
    epochs: int = 5,
    lr: float = 1e-3,
    noise_level: float = 0.1,
    lambda_supcon: float = 1.0,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    use_neutral_normalizer: bool = False,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    num_dataloader_workers: int = 1,
    model_val_freq: int = 5,
    log_every_batches: int = 1,
    use_antithetic_variates: bool = True,
    use_grad_checkpoint: bool = False,
    model_existing_checkpoint: Optional[str] = None,
    use_wandb: bool = False,
    intact_sample_size: Optional[int] = None,
):
    """Contrastive weakly-supervised pretraining of a ProteinMPNN-based binding-ΔΔG predictor on IntAct.

    Launches (optionally multi-GPU via torchrun/DDP) supervised-contrastive pretraining where each
    anchor is an IntAct mutation with a weak sign label (+1 interaction-increasing, −1
    decreasing/disrupting, 0 neutral) and the model scores it as a binding ΔΔG. See
    ``stabddg/intact/training.py`` for the loop and ``ContrastiveLoss`` for the objective.

    Parameters
    ----------
    run_name : str
        Human-readable name for the run (used in logs / experiment tracking).
    data_dir : str
        Root of the training-ready IntAct bundle. Must contain
        ``df_intact_mutations_filtered.parquet`` and ``df_assemblies_filtered.parquet``.
    assemblies_dir : str
        Directory of per-assembly structure safetensors (``<biological_assembly>.safetensors``) —
        the only structures training loads. (There is intentionally no ``proteins_dir``: the AlphaFold
        monomers are extraction-only, consumed by the offline WT oracle, not by training.)
    model_save_dir : str
        Output directory for checkpoints, split parquet files, ``metrics.csv``/``forecasts.csv``,
        ``logs.txt`` and TensorBoard events.
    max_length : int
        Drop assemblies longer than this and truncate/pad to it. Caps per-datapoint memory.
    k_neutral, k_pos, k_neg : int
        Contrast-pool sizes sampled per step: neutrals (label 0), positives (+1), negatives (−1).
        Each pool item is a full ProteinMPNN forward *per rank*, so these dominate per-GPU memory.
        Neutrals are forwarded without grad (or skipped) unless ``lambda_neutral > 0``.
    batch_size : int
        Number of anchors per optimizer step (per rank).
    micro_batch_size : int
        If > 0, split each pool (anchor/pos/neg/neutral) into chunks of this many datapoints, forward
        each chunk separately, and concatenate the scalar outputs. 0 disables chunking (whole pool at
        once). Reduces peak activation memory for the no-grad neutral pool and validation; for the
        grad-requiring pools it lowers the forward-time peak but not the backward peak (pair with
        gradient checkpointing for that). Lets larger ``k_*`` fit a fixed GPU.
    epochs : int
        Number of training epochs.
    lr : float
        Adam learning rate.
    noise_level : float
        Backbone-noise magnitude (Å) for the StaB-ddG antithetic variates.
    lambda_supcon, lambda_sign, lambda_neutral : float
        Loss weights: supervised-contrastive term, sign-direction margin term, and neutral
        regularisation. ``lambda_neutral > 0`` makes neutrals require gradients.
    use_neutral_normalizer : bool
        Centre/scale all ΔΔG values by the neutral pool's robust median/MAD before the contrastive +
        sign terms. Removes an arbitrary global offset and sets a common scale (it does NOT equalize
        per-complex/length variance). At ``max_length=640`` neutrals are length-representative, so the
        estimate is well-calibrated; forces the neutral pool to be forwarded (see ``_forward_neutrals``).
    valid_size, test_size : float
        Fractions of the data held out for validation and test (stratified by feature-type category).
    random_state : int
        Seed for the deterministic train/valid/test split (every rank computes identical paths).
    num_dataloader_workers : int
        DataLoader worker processes per rank.
    model_val_freq : int
        Run validation every this many epochs.
    log_every_batches : int
        Emit a per-batch metrics line to stdout (→ CloudWatch) every this many batches (1 = every
        batch). Per-epoch summaries are always logged. Raise it on full data to reduce log volume.
    use_antithetic_variates : bool
        Use antithetic decoding-order / backbone-noise variates in StaB-ddG (variance reduction).
    use_grad_checkpoint : bool
        Recompute the ProteinMPNN forward during backward instead of storing its activations (~30%
        slower, large memory cut). Complements ``micro_batch_size`` by also capping the *backward*
        peak, so larger ``k_*`` pools fit a fixed GPU.
    model_existing_checkpoint : str or None
        Path to a ProteinMPNN/StaB-ddG checkpoint to warm-start from; None trains from scratch.
    use_wandb : bool
        Also log to Weights & Biases (in addition to TensorBoard / SageMaker Experiments / MLflow).
    intact_sample_size : int or None
        If set, subsample this many rows per feature-type category before splitting (debugging only).
    """
    # Resolve local_rank from env when launched via torchrun
    local_rank = None
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        try:
            local_rank = int(env_local_rank)
        except ValueError:
            local_rank = -1

    logger.info(
        f"local_rank (arg): {local_rank}, "
        f"env LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
        f"RANK={os.environ.get('RANK')}, "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}"
    )

    # Initialise the distributed process group when launched under torchrun (WORLD_SIZE > 1).
    # Without this, dist.is_initialized() stays False, so DDP / DistributedSampler / metric
    # all-reduce below never activate and every rank would train the full dataset uncoordinated.
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available() and local_rank is not None and local_rank >= 0:
            torch.cuda.set_device(local_rank)
        logger.info(f"Initialized process group: backend={backend}, world_size={world_size}")

    # Getting the device
    device = get_device(local_rank=local_rank)
    logger.info(f"Using device: {device}")

    # Prepare IntAct train/valid/test splits and save into a dedicated subfolder under model_save_dir.
    # Only the main process writes the (deterministic) split files; other ranks compute the same paths
    # in-memory and wait on a barrier so they never read a half-written parquet.
    train_p, valid_p, test_p = prepare_intact_splits(
        data_dir=data_dir,
        model_save_dir=model_save_dir,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
        intact_sample_size=intact_sample_size,
        write=_is_main_process(),
    )
    if _is_dist_initialized():
        torch.distributed.barrier()
    logger.info(f"Saved splits to:\n  train: {train_p}\n  valid: {valid_p}\n  test:  {test_p}")

    # Preparing Datasets
    ds_train = IntactDataset(
        df_intact_path=train_p,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg
    )

    ds_valid = IntactDataset(
        df_intact_path=valid_p,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg,
    )

    ds_test = IntactDataset(
        df_intact_path=test_p,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg,
    )

    # Setting up the model
    pmpnn = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.0,
        augment_eps=0.0,
    )
    if model_existing_checkpoint:
        try:
            mpnn_checkpoint = torch.load(model_existing_checkpoint, map_location=device)
            if "model_state_dict" in mpnn_checkpoint.keys():
                pmpnn.load_state_dict(mpnn_checkpoint["model_state_dict"])
            else:
                pmpnn.load_state_dict(mpnn_checkpoint)
            logger.info(f"Successfully loaded model at {model_existing_checkpoint}")
        except Exception as e:
            logger.info(f"Unable to load model from {model_existing_checkpoint}: {e}")

    model = StaBddG(
        pmpnn=pmpnn,
        use_antithetic_variates=use_antithetic_variates,
        noise_level=noise_level,
        device=device,
        use_grad_checkpoint=use_grad_checkpoint,
    )

    _ = model.to(device)

    # Wrap with DDP if applicable
    if _is_dist_initialized():
        # find_unused_parameters=True for safety if some parameters aren't used every step
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # Initialize wandb logging (main process only)
    if _is_main_process() and use_wandb:
        logger.info("Initializing weights and biases.")
        wandb.init(
            project="",
            entity="",
            name=run_name,
        )
        logger.info("Weights and biases initialized.")

    logger.info("Starting intact pretraining job")
    pretrain(
        model=model,
        dataset_train=ds_train,
        dataset_valid=ds_valid,
        dataset_test=ds_test,
        save_dir=model_save_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_dataloader_workers=num_dataloader_workers,
        lambda_supcon=lambda_supcon,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
        use_neutral_normalizer=use_neutral_normalizer,
        n_epochs=epochs,
        model_val_freq=model_val_freq,
        log_every_batches=log_every_batches,
        lr=lr,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
