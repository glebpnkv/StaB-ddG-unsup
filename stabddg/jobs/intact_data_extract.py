#!/usr/bin/env python3
"""
Steps:
  1) Prepare IntAct mutations with IntactDataController
  2) Fetch AlphaFold protein atom data for all UniProts present in mutations
  3) Filter mutations to those with available structures for both partners
  4) Fetch assemblies metadata for each UniProt pair
  5) Select the best assembly per pair, normalize entry id, and save
  6) Fetch assemblies atom data and compute complex lengths
  7) Optionally upload/download data to/from GCP
"""

import logging
import os
from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from stabddg.intact.data import (
    IntactDataController,
    fetch_alphafold_atoms_parallel,
    fetch_assemblies_atoms_parallel,
    fetch_assemblies_for_uniprots_parallel,
    normalize_entry,
)
from stabddg.intact.dataset import IntactDataset
from stabddg.utils.gcp import download_dir_from_gcp, upload_dir_to_gcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_mutations(data_dir: str) -> pd.DataFrame:
    output_dir = os.path.join(data_dir, "intact")
    os.makedirs(output_dir, exist_ok=True)

    dc_intact = IntactDataController(output_dir=output_dir)
    dc_intact.prepare_data()

    df_mutations_path = os.path.join(output_dir, "df_intact_mutations.parquet")
    df_mutations = pd.read_parquet(df_mutations_path)
    return df_mutations


def fetch_alphafold_for_uniprots(df_mutations: pd.DataFrame, proteins_dir: str, max_workers: int) -> Dict[str, Any]:
    parquet_dir = os.path.join(proteins_dir, "parquet")
    safetensors_dir = os.path.join(proteins_dir, "safetensors")
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(safetensors_dir, exist_ok=True)

    # Collect all unique UniProt codes from both protein columns
    uniprot_codes = pd.concat(
        [
            df_mutations["participant_protein"],
            df_mutations["affected_protein_ac"],
        ],
        ignore_index=True,
    ).unique().tolist()

    logger.info(f"Fetching AlphaFold structures for {len(uniprot_codes)} UniProt proteins")
    step_outcome = fetch_alphafold_atoms_parallel(
        uniprot_codes=uniprot_codes,
        parquet_dir=parquet_dir,
        safetensors_dir=safetensors_dir,
        max_workers=max_workers,
    )
    return step_outcome


def filter_mutations_by_available_structures(df_mutations: pd.DataFrame, step_outcome: Dict[str, Any]) -> pd.DataFrame:
    step_outcome_success = {k: v for k, v in step_outcome.items() if v.get("status") == "success"}
    uniprot_codes_success = list(step_outcome_success.keys())

    df_filtered = df_mutations.loc[
        (df_mutations["participant_protein"].isin(uniprot_codes_success))
        & (df_mutations["affected_protein_ac"].isin(uniprot_codes_success))
    ].reset_index(drop=True)
    return df_filtered


def get_uniprot_pairs(df_mutations_filtered: pd.DataFrame) -> List[List[str]]:
    uniprots_pairs = [
        list(k)
        for k, _ in df_mutations_filtered.groupby(["participant_protein", "affected_protein_ac"]).groups.items()
    ]
    return uniprots_pairs


def fetch_assemblies_metadata(uniprots_pairs: List[List[str]], max_workers: int) -> pd.DataFrame:
    df_assemblies_raw = fetch_assemblies_for_uniprots_parallel(
        uniprots_pairs=uniprots_pairs,
        max_workers=max_workers,
    )
    return df_assemblies_raw


def select_and_normalize_assemblies(df_assemblies_raw: pd.DataFrame) -> pd.DataFrame:
    df_assemblies = df_assemblies_raw.copy()

    # Keep only entries with biological assemblies present
    df_assemblies = df_assemblies.loc[df_assemblies["biological_assembly"].notna()].reset_index(drop=True)

    # Choose the best (lowest score) per pair
    df_assemblies = df_assemblies.loc[
        df_assemblies["score"] == df_assemblies.groupby("pair_idx")["score"].transform("min")
    ].reset_index(drop=True)

    # Normalize entry id
    df_assemblies["entry_id"] = df_assemblies["biological_assembly"].apply(normalize_entry)
    return df_assemblies


def fetch_and_summarize_assemblies_atoms(
    df_assemblies: pd.DataFrame,
    assemblies_dir: str,
    max_workers: int,
) -> pd.DataFrame:
    parquet_dir = os.path.join(assemblies_dir, "parquet")
    safetensors_dir = os.path.join(assemblies_dir, "safetensors")
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(safetensors_dir, exist_ok=True)

    assemblies = list(
        df_assemblies[["biological_assembly", "participant_protein", "affected_protein_ac"]]
        .to_dict(orient="index")
        .values()
    )

    step_outcome = fetch_assemblies_atoms_parallel(
        assemblies=assemblies,
        parquet_dir=parquet_dir,
        safetensors_dir=safetensors_dir,
        max_workers=max_workers,
    )

    # Filter to successful assemblies
    step_outcome_success = {k: v for k, v in step_outcome.items() if v.get("status") == "success"}
    assembly_ids_success = list(step_outcome_success.keys())

    df_assemblies_filtered = df_assemblies.loc[
        df_assemblies["biological_assembly"].isin(assembly_ids_success)
    ].reset_index(drop=True)

    # Compute complex lengths from parquet outputs
    dict_lengths = {}
    for k, v in tqdm(step_outcome_success.items(), desc="Compute assembly lengths"):
        df_cur = pd.read_parquet(v["parquet_path"])  # expects columns [chain, resnum_label]
        dict_lengths[k] = df_cur.groupby(["chain", "resnum_label"]).ngroups

    df_dict_lengths = pd.DataFrame.from_dict(
        dict_lengths,
        orient="index",
        columns=["biological_assembly_length"],
    )

    df_assemblies_filtered = df_assemblies_filtered.merge(
        df_dict_lengths, left_on="biological_assembly", right_index=True
    )

    return df_assemblies_filtered


def maybe_upload_to_gcp(
    data_dir: str,
    bucket: str,
    prefix: str,
    region: str,
    do_upload: bool
):
    if not do_upload:
        return
    upload_dir_to_gcp(
        bucket_name=bucket,
        local_dir=os.path.join(data_dir, "intact") + "/",
        dst_prefix=prefix,
        region=region,
    )


def maybe_download_from_gcp(
    data_dir: str,
    bucket: str,
    prefix: str,
    do_download: bool
):
    if not do_download:
        return
    download_dir_from_gcp(
        bucket_name=bucket,
        src_prefix=prefix,
        local_dir=os.path.join(data_dir, "intact") + "/",
    )


def build_splits_and_save(
    data_dir: str,
    valid_size: float,
    test_size: float,
    random_state: int,
):
    feature_type_pos = IntactDataset.feature_type_pos
    feature_type_neg = IntactDataset.feature_type_neg

    df_mutations = pd.read_parquet(os.path.join(data_dir, "intact", "df_intact_mutations_filtered.parquet"))
    df_assemblies = pd.read_parquet(os.path.join(data_dir, "intact", "df_assemblies_filtered.parquet"))

    # Only keep same-length mutations (consistent with referenced paper)
    df_mutations = df_mutations.loc[
        (df_mutations["resulting_sequence"] != ".")
        & (df_mutations["original_sequence"].str.len() == df_mutations["resulting_sequence"].str.len())
    ].reset_index(drop=True)

    # Merge mutations with assemblies on pair columns
    df = df_mutations.merge(
        right=df_assemblies,
        on=["participant_protein", "affected_protein_ac"],
        how="inner",
        validate="m:m",
    )

    # Express 'resulting_sequence' as sequence codes
    # Note: This mirrors the notebook, which used regex "." (matches any char). Kept for parity.
    df["affected_protein_ac_seq_mut"] = df["affected_protein_ac_seq_mut"].str.replace(r".", "")

    # Lengths
    df["participant_protein_len"] = df["participant_protein_seq"].str.len()
    df["affected_protein_ac_len"] = df["affected_protein_ac_seq"].str.len()
    df["affected_protein_ac_mut_len"] = df["affected_protein_ac_seq_mut"].str.len()

    # Use assembly length as max_len
    df["max_len"] = df["biological_assembly_length"]

    # Convert range indices back to 1-based
    df["feature_ranges_start"] += 1
    df["feature_ranges_end"] += 1

    # For stratification
    df["feature_type_category"] = "neutral"
    df.loc[df["feature_type"].isin(feature_type_pos), "feature_type_category"] = "positive"
    df.loc[df["feature_type"].isin(feature_type_neg), "feature_type_category"] = "negative"

    # Train/valid/test split
    df_train, df_valid = train_test_split(
        df,
        test_size=(valid_size + test_size),
        stratify=df["feature_type_category"],
        random_state=random_state,
    )

    df_valid, df_test = train_test_split(
        df_valid,
        test_size=test_size / (valid_size + test_size),
        stratify=df_valid["feature_type_category"],
        random_state=random_state,
    )

    # Drop the helper column and save
    for df_cur, name in [
        (df_train, "train"),
        (df_valid, "valid"),
        (df_test, "test"),
    ]:
        df_cur.drop(columns=["feature_type_category"], inplace=True)
        df_cur.reset_index(drop=True, inplace=True)
        df_cur.to_parquet(os.path.join(data_dir, "intact", f"df_intact_mutations_filtered_{name}.parquet"))


def extract_intact_data(
    data_dir: str,
    alphafold_workers: int,
    metadata_workers: int,
    assemblies_workers: int,
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str,
    gcp_upload: bool,
    gcp_download: bool,
):
    data_dir = data_dir
    intact_dir = os.path.join(data_dir, "intact")
    proteins_dir = os.path.join(intact_dir, "proteins")
    assemblies_dir = os.path.join(intact_dir, "assemblies")

    os.makedirs(proteins_dir, exist_ok=True)
    os.makedirs(assemblies_dir, exist_ok=True)

    # 1) Prepare IntAct mutations
    logger.info("[1/7] Preparing IntAct mutations…")
    df_mutations = prepare_mutations(data_dir)

    # 2) Fetch AlphaFold atoms for UniProts
    logger.info("[2/7] Fetching AlphaFold atom data…")
    step_outcome_alpha = fetch_alphafold_for_uniprots(df_mutations, proteins_dir, alphafold_workers)

    # 3) Filter mutations by available structures
    logger.info("[3/7] Filtering mutations by available structures…")
    df_mutations_filtered = filter_mutations_by_available_structures(df_mutations, step_outcome_alpha)
    df_mutations_filtered.to_parquet(os.path.join(intact_dir, "df_intact_mutations_filtered.parquet"))

    # 4) Fetch assemblies metadata
    logger.info("[4/7] Fetching assemblies metadata…")
    uniprots_pairs = get_uniprot_pairs(df_mutations_filtered)
    df_assemblies_raw = fetch_assemblies_metadata(
        uniprots_pairs=uniprots_pairs,
        max_workers=metadata_workers
    )
    df_assemblies_raw = df_assemblies_raw.drop(columns=["mutations"], errors="ignore")
    df_assemblies_raw.to_parquet(
        os.path.join(intact_dir, "df_assemblies_raw.parquet")
    )

    # 5) Select and normalize assemblies
    logger.info("[5/7] Selecting best assemblies and normalizing entry ids…")
    df_assemblies = select_and_normalize_assemblies(df_assemblies_raw)
    df_assemblies.to_parquet(os.path.join(intact_dir, "df_assemblies.parquet"))

    # 6) Fetch assemblies atoms and compute lengths
    logger.info("[6/7] Fetching assemblies atom data and computing lengths…")
    df_assemblies_filtered = fetch_and_summarize_assemblies_atoms(
        df_assemblies, assemblies_dir, assemblies_workers
    )
    df_assemblies_filtered.to_parquet(os.path.join(intact_dir, "df_assemblies_filtered.parquet"))

    # 7) Optional GCP sync
    logger.info("[7/7] Optional GCP sync…")
    maybe_upload_to_gcp(data_dir, gcp_bucket, gcp_prefix, gcp_region, gcp_upload)
    maybe_download_from_gcp(data_dir, gcp_bucket, gcp_prefix, gcp_download)

