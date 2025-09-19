import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from stabddg.constants import AA3_TO_1, ALPHABET
from stabddg.model import StaBddG
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB
from stabddg.training import _is_dist_initialized, _is_main_process, _unwrap_model

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IntactDataset(Dataset):

    feature_type_pos = [
        "mutation causing(MI:2227)",
        "mutation increasing(MI:0382)",
        "mutation increasing rate(MI:1131)",
        "mutation increasing strength(MI:1132)"
    ]
    feature_type_neg = [
        "mutation decreasing(MI:0119)",
        "mutation decreasing rate(MI:1130)",
        "mutation decreasing strength(MI:1133)",
        "mutation disrupting(MI:0573)",
        "mutation disrupting rate(MI:1129)",
        "mutation disrupting strength(MI:1128)",
    ]
    feature_type_neutral = [
        "mutation with no effect(MI:2226)"
    ]

    sequence_unknown = AA3_TO_1["UNK"]

    def __init__(
        self,
        df_intact_path,
        structures_dir,
        max_length: int = 1024,
        k_neutral: int = 32,
        k_same: int = 4,
        k_opp: int = 4,
    ):
        self.structures_dir = structures_dir
        self.max_length = max_length
        self.k_neutral = k_neutral
        self.k_same = k_same
        self.k_opp = k_opp

        # Reading the metadata dataframe
        df = pd.read_parquet(df_intact_path)
        df = df.reset_index(drop=True)

        len_raw = len(df)

        # Removing rows with sequence length > max_length
        df = df.loc[
            (df["participant_protein_seq"].str.len() <= max_length) &
            (df["affected_protein_ac_seq"].str.len() <= max_length) &
            (df["affected_protein_ac_seq_mut"].str.len() <= max_length)
        ].reset_index(drop=True)

        len_filtered = len(df)

        logger.info(f"Removed {len_raw - len_filtered} rows out of {len_raw} with sequence length > {max_length}")

        # Separating intact data into +ve, -ve and neutrals
        self.df_pos = df.loc[
            df["feature_type"].isin(self.feature_type_pos)
        ].copy()
        self.df_neg = df.loc[
            df["feature_type"].isin(self.feature_type_neg)
        ].copy()
        self.df_neutral = df.loc[
            df["feature_type"].isin(self.feature_type_neutral)
        ].copy()
        self.df = pd.concat([self.df_pos, self.df_neg, self.df_neutral])

        # Making an index of datapoints for sampling
        self.df_idx = pd.concat([self.df_pos, self.df_neg])["feature_type"]

        # Creating a cache of structures
        self.structures_cache = {}

    def _combine_items(self, items: list[dict], add_mask: bool = True):
        """
        Combine a list of dictionaries into a single dictionary with padding.
        - Pads arrays along the first dimension to self.max_length with zeros.
        - Adds a 'mask' key (float32) of shape (len(items), self.max_length) with 1.0 for
          valid (unpadded) positions and 0.0 for padded positions.
        """
        out = {}
        keys = list(items[0].keys())

        # Determine lengths for mask using sequence if available, otherwise the first ndarray key
        lengths = None
        if "seq_chain_A" in keys and isinstance(items[0]["seq_chain_A"], np.ndarray):
            lengths = [min(len(x["seq_chain_A"]), self.max_length) for x in items]
        else:
            first_arr_key = next(
                k for k in keys if isinstance(items[0][k], np.ndarray)
            )
            lengths = [min(items[i][first_arr_key].shape[0], self.max_length) for i in range(len(items))]

        if add_mask:
            mask = np.zeros((len(items), self.max_length), dtype=np.float32)
            for i, l in enumerate(lengths):
                mask[i, :l] = 1.0
            out["mask"] = mask

        for key in keys:
            v0 = items[0][key]
            if isinstance(v0, np.ndarray):
                stacked = []
                for i, x in enumerate(items):
                    a = x[key]
                    l = min(a.shape[0], self.max_length)
                    a = a[:l]  # truncate if necessary (shouldn't normally happen after filtering)
                    pad_len = self.max_length - l
                    # Build pad width: pad the first dimension only
                    pad_width = [(0, pad_len)]
                    if a.ndim > 1:
                        pad_width += [(0, 0)] * (a.ndim - 1)
                    a_padded = np.pad(a, pad_width, mode="constant", constant_values=0)
                    stacked.append(a_padded)
                out[key] = np.stack(stacked, axis=0)
            elif isinstance(v0, dict):
                # Recurse into nested dicts but only add 'mask' at the current level (avoid duplicating masks)
                out[key] = self._combine_items([x[key] for x in items], add_mask=False)

        return out


    def __len__(self):
        return len(self.df_idx)

    def _fetch_sequence(self, seq_str: str) -> np.ndarray:
        """
        Convert sequences from strings to arrays of indices in ALPHABET (consistent with ProteinMPNN)
        Note: index 0 remains a valid amino acid index as well as the implicit pad used elsewhere 😭.
        """
        seq_list = [ch if ch in ALPHABET else self.sequence_unknown for ch in seq_str]
        return np.asarray([ALPHABET.index(ch) for ch in seq_list], dtype=np.int32)

    def _fetch_structure(self, name: str):
        """
        Load protein structure from cache or disk.
        """
        # Loading structures from cache if they exist
        if name in self.structures_cache.keys():
            return self.structures_cache[name]

        # Loading structures from the disk
        with safe_open(os.path.join(self.structures_dir, f"{name}.safetensors"), framework="pt", device="cpu") as f:
            coords = {}
            for key in f.keys():
                # TODO It's wasteful to convert tensors back to numpy arrays, but it makes _combine_items simpler
                coords[key] = f.get_tensor(key).numpy()

        # Adding to cache
        self.structures_cache[name] = coords
        return self.structures_cache[name]

    def _fetch_item(self, idx):
        """
        Fetch a single item from the dataset.
        """
        # Forming the output dictionary
        out = {}

        # A sample is a single row from the DataFrame
        sample = self.df.loc[idx]

        # Getting names of the proteins
        name_a = sample["participant_protein"]
        name_b = sample["affected_protein_ac"]

        # Getting amino acid sequences of the proteins
        out["seq_chain_A"] = self._fetch_sequence(sample["participant_protein_seq"])
        out["seq_chain_B"] = self._fetch_sequence(sample["affected_protein_ac_seq"])

        # Getting coordinates of the proteins
        out["coords_chain_A"] = self._fetch_structure(name_a)
        out["coords_chain_B"] = self._fetch_structure(name_b)

        return out

    def __getitem__(self, idx):
        # Getting the absolute index from the relative index
        cur_idx_row = self.df_idx.iloc[[idx]]
        idx = cur_idx_row.index[0]
        # Getting the "sign" of the anchor datapoint (idx)
        sign = cur_idx_row.values[0]

        sign_same = self.feature_type_pos if sign in self.feature_type_pos else self.feature_type_neg
        sign_diff = self.feature_type_neg if sign in self.feature_type_pos else self.feature_type_pos

        sample_same = self.df.loc[
            self.df["feature_type"].isin(sign_same)
        ].sample(self.k_same).index

        sample_opp = self.df.loc[
            self.df["feature_type"].isin(sign_diff)
        ].sample(self.k_opp).index

        sample_neutral = self.df.loc[
            self.df["feature_type"].isin(self.feature_type_neutral)
        ].sample(self.k_neutral).index

        # Getting the datapoint itself (anchor)
        out_anchor = self._combine_items(
            [self._fetch_item(idx)]
        )

        # Getting "same" datapoints
        out_same = self._combine_items(
            [self._fetch_item(x) for x in sample_same]
        )
        # Getting "opposite" datapoints
        out_opp = self._combine_items(
            [self._fetch_item(x) for x in sample_opp]
        )
        # Getting "neutral" datapoints
        out_neutral = self._combine_items(
            [self._fetch_item(x) for x in sample_neutral]
        )

        # Merging all datapoints into a single dictionary
        out = {
            "anchor": out_anchor,
            "same": out_same,
            "opp": out_opp,
            "neutral": out_neutral,
        }

        return out


class ContrastiveLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        reduction: str = "mean",
    ):
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        super(ContrastiveLoss, self).__init__(reduction=reduction)

    def forward(self, z, z_pos, z_neg, z_neu):
        # Normalising inputs
        z = F.normalize(z, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        z_neg = F.normalize(z_neg, dim=-1)
        z_neu = F.normalize(z_neu, dim=-1)

        # Calculating cosine similarities
        s_pos = z @ z_pos.T / self.tau_pos
        s_neg = z @ z_neg.T / self.tau_neg
        s_neu = z @ z_neu.T

        # Calculating contrastive losses
        loss_pos = -torch.mean(s_pos - torch.logsumexp(s_neu, dim=-1), dim=-1)
        loss_neg = -torch.mean(-s_neg - torch.logsumexp(-s_neu, dim=-1), dim=-1)
        loss = torch.sum(self.lambda_pos * loss_pos + self.lambda_neg * loss_neg)

        return loss


def pretrain(
    model,
    dataset_train,
    dataset_valid,
    dataset_test,
    ddG_data,
    args,
    batch_size=10000,
    device="cuda",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = ContrastiveLoss()

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()

    # Directory to save model checkpoints
    if _is_main_process() and not os.path.exists(args.model_save_dir):
        logger.info(f"Creating directory {args.model_save_dir}")
        os.makedirs(args.model_save_dir)

    # Creating a logging file logs.txt (main process only)
    if _is_main_process():
        log_path = os.path.join(args.model_save_dir, "logs.txt")
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")

    world_size = dist.get_world_size() if _is_dist_initialized() else 1
    rank = dist.get_rank() if _is_dist_initialized() else 0


