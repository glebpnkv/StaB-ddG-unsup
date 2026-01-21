import logging
import os
from typing import Optional

import pandas as pd
import torch
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
) -> tuple[str, str, str]:
    """
    Creates df_train, df_valid, df_test and saves them into a dedicated sub-folder under model_save_dir.

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

    # Expressing 'resulting_sequence' as sequence codes (replicate notebook)
    # Note: the notebook used a regex pattern that removes all characters; we
    # reproduce it literally to preserve identical behavior.
    df["affected_protein_ac_seq_mut"] = df["affected_protein_ac_seq_mut"].str.replace(r".", "", regex=True)

    # Calculating lengths
    df["participant_protein_len"] = df["participant_protein_seq"].str.len()
    df["affected_protein_ac_len"] = df["affected_protein_ac_seq"].str.len()
    df["affected_protein_ac_mut_len"] = df["affected_protein_ac_seq_mut"].str.len()

    # Explicitly creating a max_len column
    df["max_len"] = df["biological_assembly_length"]

    # Convert range indices back to 1-based
    df["feature_ranges_start"] += 1
    df["feature_ranges_end"] += 1

    # feature_type_category for stratification
    feature_type_pos = IntactDataset.feature_type_pos
    feature_type_neg = IntactDataset.feature_type_neg

    df["feature_type_category"] = "neutral"
    df.loc[df["feature_type"].isin(feature_type_pos), "feature_type_category"] = "positive"
    df.loc[df["feature_type"].isin(feature_type_neg), "feature_type_category"] = "negative"

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

    df_train.to_parquet(train_path)

    if df_valid is None:
        valid_path = None
    else:
        df_valid.to_parquet(valid_path)

    if df_test is None:
        test_path = None
    else:
        df_test.to_parquet(test_path)

    return train_path, valid_path, test_path


def intact_pretrain(
    run_name: str,
    data_dir: str,
    proteins_dir: str,
    assemblies_dir: str,
    model_save_dir,
    max_length: int = 256,
    k_neutral: int = 32,
    k_pos: int = 16,
    k_neg: int = 16,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 1e-3,
    noise_level: float = 0.1,
    lambda_supcon: float = 1.0,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    num_dataloader_workers: int = 1,
    model_val_freq: int = 5,
    use_antithetic_variates: bool = True,
    model_existing_checkpoint: Optional[str] = None,
    use_wandb: bool = False,
    intact_sample_size: Optional[int] = None,
):
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

    # Getting the device
    device = get_device(local_rank=local_rank)
    logger.info(f"Using device: {device}")

    # Prepare IntAct train/valid/test splits and save into a dedicated subfolder under model_save_dir
    train_p, valid_p, test_p = prepare_intact_splits(
        data_dir=data_dir,
        model_save_dir=model_save_dir,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
        intact_sample_size=intact_sample_size,
    )
    logger.info(f"Saved splits to:\n  train: {train_p}\n  valid: {valid_p}\n  test:  {test_p}")

    # Preparing Datasets
    ds_train = IntactDataset(
        df_intact_path=train_p,
        proteins_dir=proteins_dir,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg
    )

    ds_valid = IntactDataset(
        df_intact_path=valid_p,
        proteins_dir=proteins_dir,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg,
    )

    ds_test = IntactDataset(
        df_intact_path=test_p,
        proteins_dir=proteins_dir,
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
        num_dataloader_workers=num_dataloader_workers,
        lambda_supcon=lambda_supcon,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
        n_epochs=epochs,
        model_val_freq=model_val_freq,
        lr=lr,
    )
