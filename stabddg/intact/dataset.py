import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from stabddg.constants import AA3_TO_1, ALPHABET, SEQUENCE_DELETION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DICT_KEYS = [
    "complex",
    "binder1",
    "binder2",
]

INTERNAL_TENSOR_KEYS = [
    "S",
    "X",
    "chain_encoding_all",
    "mask",
    "residue_idx",
    "resnums",
    "mut_seqs"
]


def pad_or_trim_dim1(t: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = t.size(1)
    if cur == target_len:
        return t
    if cur > target_len:
        return t.narrow(1, 0, target_len)
    pad_right = target_len - cur
    pads = []
    # Build (last->first) pairs; only dim=1 gets right pad
    for d in range(t.dim() - 1, -1, -1):
        if d == 1:
            pads.extend([0, pad_right])
        else:
            pads.extend([0, 0])
    return F.pad(t, tuple(pads))


def intact_collate_fn(x):
    out = {}

    # Combining "anchor_idx" and "sign" into a single tensor
    out["anchor_idx"] = torch.stack([it["anchor_idx"] for it in x])
    out["sign"] = torch.stack([it["sign"] for it in x])

    # Concatenating remaining sequences
    for k_dict in OUTPUT_DICT_KEYS:
        # Getting the max length
        max_length = max([d[k_dict]["max_length"] for d in x])

        out[k_dict] = {
            k: torch.cat([
                pad_or_trim_dim1(m, max_length)
                for m in (it[k_dict][k] for it in x)
            ])
            for k in INTERNAL_TENSOR_KEYS
        } | {
            "max_length": max_length
        }

    return out


def passthrough_collate_fn(x):
    return x[0]


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
        proteins_dir,
        assemblies_dir,
        max_length: int = 1024,
        k_neutral: int = 32,
        k_pos: int = 4,
        k_neg: int = 4,
    ):
        self.proteins_dir = proteins_dir
        self.assemblies_dir = assemblies_dir
        self.k_neutral = k_neutral
        self.k_pos = k_pos
        self.k_neg = k_neg

        # Reading the metadata dataframe
        df = pd.read_parquet(df_intact_path)
        df = df.reset_index(drop=True)

        len_raw = len(df)

        # Removing rows with sequence length > max_length
        df = df.loc[df["max_len"] <= max_length].reset_index(drop=True)

        len_filtered = len(df)

        logger.info(
            f"Kept {len_filtered} rows out of {len_raw} with sequence length > {max_length} ("
            f"{len_raw - len_filtered} rows were removed)"
        )

        # Separating intact data into +ve, -ve and neutrals
        self.df_pos = df.loc[
            df["feature_type"].isin(self.feature_type_pos)
        ].copy()
        self.df_pos["sign"] = 1
        self.df_neg = df.loc[
            df["feature_type"].isin(self.feature_type_neg)
        ].copy()
        self.df_neg["sign"] = -1
        self.df_neutral = df.loc[
            df["feature_type"].isin(self.feature_type_neutral)
        ].copy()
        self.df_neutral["sign"] = 0
        self.df = pd.concat([self.df_pos, self.df_neg, self.df_neutral])

        # Making an index of datapoints for sampling
        self.df_idx = pd.concat([self.df_pos, self.df_neg])["feature_type"]

        # Creating a cache of structures
        self.structures_cache = {}

    def __len__(self):
        return len(self.df_idx)

    def _fetch_sequence(self, seq_str: str) -> np.ndarray:
        """
        Convert sequences from strings to arrays of indices in ALPHABET (consistent with ProteinMPNN)
        Note: index 0 remains a valid amino acid index as well as the implicit pad used elsewhere 😭.
        """
        seq_list = [ch if ch in ALPHABET else self.sequence_unknown for ch in seq_str]
        return np.asarray([ALPHABET.index(ch) for ch in seq_list], dtype=np.int32)

    def _load_assembly(self, name: str):
        """
        Load complex structure from cache or disk.
        """
        # Loading structures from cache if they exist
        if name in self.structures_cache.keys():
            return self.structures_cache[name]

        # Loading proteins from the disk
        f = safe_open(os.path.join(self.assemblies_dir, f"{name}.safetensors"), framework="pt", device="cpu")
        metadata = f.metadata()
        structure = {}
        for key in f.keys():
            # TODO It's wasteful to convert tensors back to numpy arrays, but it makes _combine_items simpler
            structure[key] = f.get_tensor(key).numpy()

        # Adding to cache
        out = {
            "metadata": metadata,
            "data": structure,
        }
        self.structures_cache[name] = out
        return self.structures_cache[name]

    def _load_protein_full(self, name: str):
        """
        Load protein structure from cache or disk.
        """
        # Loading structures from cache if they exist
        if name in self.structures_cache.keys():
            return self.structures_cache[name]

        # Loading proteins from the disk
        f = safe_open(os.path.join(self.proteins_dir, f"{name}.safetensors"), framework="pt", device="cpu")
        metadata = f.metadata()
        structure = {}
        for key in f.keys():
            # TODO It's wasteful to convert tensors back to numpy arrays, but it makes _combine_items simpler
            structure[key] = f.get_tensor(key).numpy()

        # Adding to cache
        out = {
            "metadata": metadata,
            "data": structure,
        }
        self.structures_cache[name] = out
        return self.structures_cache[name]

    def _make_mutation_sequence(
        self,
        datapoint: dict,
        sample: pd.Series
    ) -> np.ndarray:
        # Creating a map from the chain encoding ID to the protein accession/name
        complex_metadata = datapoint["metadata"]
        chain_encoding_dict = {
            x.strip("chain_encoding:"): v
            for x, v in complex_metadata.items() if "chain_encoding" in x
        }
        chain_encoding_dict = {
            int(k): complex_metadata[v]
            for k, v in chain_encoding_dict.items()
        }

        data = datapoint["data"]

        S = data["S"].copy()  # (L,) int32
        chain_enc = data["chain_encoding_all"].astype(np.int32)  # (L,)
        resnums = data["resnums"].astype(np.int32)  # (L,)

        # Map encoding id for the affected protein accession
        affected_ac = sample["affected_protein_ac"]
        # chain_encoding_dict maps id -> protein accession/name
        # Find all ids that correspond to the affected protein
        ids_for_affected = [cid for cid, prot in chain_encoding_dict.items() if prot == affected_ac]
        mask_chain = np.isin(chain_enc, np.asarray(ids_for_affected, dtype=np.int32)) \
            if ids_for_affected \
            else np.zeros_like(chain_enc, dtype=bool)

        # Mutation range on residue numbers (1-based start, open end)
        start_unp = int(sample["feature_ranges_start"])  # inclusive
        end_unp = int(sample["feature_ranges_end"])  # exclusive

        # Indices (in S) belonging to the affected chain(s)
        aff_idx = np.flatnonzero(mask_chain)
        # Sort affected positions by resnums to keep the natural order
        order = np.argsort(resnums[aff_idx], kind="stable")
        aff_idx = aff_idx[order]
        aff_res = resnums[aff_idx]

        # Build idx_mut_start and idx_mut_end (end exclusive) in terms of indices within S
        # Select continuous block(s) of affected residues where resnums fall into [start_unp, end_unp)
        in_range = (aff_res >= start_unp) & (aff_res < end_unp)
        if not in_range.any():
            return S

        # Find contiguous runs in the boolean mask 'in_range' over aff_idx
        # We only expect one range per sample, but this works for multiples as well
        runs_start = np.flatnonzero(in_range & (~np.roll(in_range, 1)))
        runs_end = np.flatnonzero(in_range & (~np.roll(in_range, -1))) + 1  # exclusive
        if in_range[0]:
            runs_start[0] = 0
        if in_range[-1]:
            runs_end[-1] = in_range.size - 1

        # Map runs in aff array back to absolute S indices
        idx_mut_start = aff_idx[runs_start]
        idx_mut_end = aff_idx[runs_end]  # make exclusive in S coordinates

        # Replacement payload ('.' means deletion → remove completely)
        repl_str = str(sample["resulting_sequence"]).replace(SEQUENCE_DELETION, "")
        repl_vals = self._fetch_sequence(repl_str) if repl_str else np.asarray([], dtype=np.int32)

        # Splice iteratively over ranges: S[:s0] + repl + S[e0:s1] + repl + ... + S[e_last:]
        parts = []
        prev = 0
        for i in range(len(idx_mut_start)):
            s_i = int(idx_mut_start[i])
            e_i = int(idx_mut_end[i])  # exclusive
            if prev < s_i:
                parts.append(S[prev:s_i])
            parts.append(repl_vals)
            prev = e_i
        if prev < S.shape[0]:
            parts.append(S[prev:])

        mut_seqs = np.concatenate(parts, axis=0) if parts else np.asarray([], dtype=np.int32)
        return mut_seqs

    def _fetch_complex(self, idx):
        # A sample is a single row from the DataFrame
        sample = self.df.loc[idx]

        # Getting the name of the complex
        name_complex = sample["biological_assembly"]
        cmplex = self._load_assembly(name_complex)
        # Making a mutation sequence
        complex_mut_seqs = self._make_mutation_sequence(cmplex, sample)
        # Getting the length
        max_length = cmplex["data"]["S"].shape[0]

        # Preparing output
        out = cmplex["data"] | {"mut_seqs": complex_mut_seqs, "max_length": max_length}

        return out

    def _fetch_binders_full(self, idx):
        """
        Fetch a pair of complete binders for a single complex in the dataset.
        """
        # A sample is a single row from the DataFrame
        sample = self.df.loc[idx]

        # Getting names of the proteins
        name_binder_1 = sample["participant_protein"]
        name_binder_2 = sample["affected_protein_ac"]

        binder1 = self._load_protein_full(name_binder_1)
        binder2 = self._load_protein_full(name_binder_2)
        # Making the mutation sequence
        binder2_mut_seqs = self._make_mutation_sequence(binder2, sample)

        out = {
            "binder1": binder1.get("data"),
            "binder2": binder2.get("data"),
            "binder1_mut_seqs": binder1.get("data").get("S"),
            "binder2_mut_seqs": binder2_mut_seqs,
        }

        return out

    def _fetch_binders(self, idx):
        sample = self.df.loc[idx]

        # Getting the name of the complex
        name_complex = sample["biological_assembly"]
        # Getting names of the proteins
        name_binder_1 = sample["participant_protein"]
        name_binder_2 = sample["affected_protein_ac"]
        cmplex = self._load_assembly(name_complex)
        metadata = cmplex["metadata"]

        # Getting chains that correspond to binders 1 and 2
        chains_binder_1 = [
            k
            for k, v in metadata.items() if v == name_binder_1
        ]
        chain_codes_binder_1 = [
            int(
                next(k for k, v in metadata.items() if v == x).strip("chain_encoding:")
            )
            for x in chains_binder_1 if x in metadata.values()
        ]
        chains_binder_2 = [
            k
            for k, v in metadata.items() if v == name_binder_2
        ]
        chain_codes_binder_2 = [
            int(
                next(k for k, v in metadata.items() if v == x).strip("chain_encoding:")
            )
            for x in chains_binder_2 if x in metadata.values()
        ]

        bool_mask_binder_1 = np.isin(
            cmplex["data"]["chain_encoding_all"],
            chain_codes_binder_1
        )
        bool_mask_binder_2 = np.isin(
            cmplex["data"]["chain_encoding_all"],
            chain_codes_binder_2
        )
        try:
            binder1 = {k: v[bool_mask_binder_1, ...] for k, v in cmplex["data"].items()}
            binder2 = {k: v[bool_mask_binder_2, ...] for k, v in cmplex["data"].items()}
        except IndexError:
            print(f"Error: binder1/binder2 mask is empty for sample {idx}")
        # Making the mutation sequence
        binder2_mut_seqs = self._make_mutation_sequence(
            datapoint={"data": binder2, "metadata": metadata},
            sample=sample
        )

        # Getting the length
        binder1_max_length = binder1.get("S").shape[0]
        binder2_max_length = binder2_mut_seqs.shape[0]

        out_binder1 = binder1 | {"mut_seqs": binder1.get("S"), "max_length": binder1_max_length}
        out_binder2 = binder2 | {"mut_seqs": binder2_mut_seqs, "max_length": binder2_max_length}

        out = {
            "binder1": out_binder1,
            "binder2": out_binder2
        }

        return out

    def _fetch(self, idx):
        out_complex = self._fetch_complex(idx)
        # out_binders = self._fetch_binders_full(idx)
        out_binders = self._fetch_binders(idx)

        out = {"complex": out_complex} | out_binders
        return out

    @staticmethod
    def _combine_items(
        items: list[dict]
    ):
        """
        Batch and pad a list of fetched items.
        Each item is a dict with 3 keys total:
          - 3 keys with dict values; each dict maps str -> array with first dim L.
        Output preserves the same top-level keys as _fetch returns.
        Output dicts with tensors of shape [B, max_length, ...]
        All outputs are torch.Tensors. Padding/truncation follows the current padding semantics.
        """
        def pad_stack_first_dim(arr_list: list[np.ndarray], target_len: int) -> np.ndarray:
            # Pads/truncates only along the first dimension to target_len, constant=0
            stacked = []
            for a in arr_list:
                l = min(int(a.shape[0]), target_len)
                a_cut = a[:l]
                pad_len = target_len - l
                pad_width = [(0, pad_len)]
                if a_cut.ndim > 1:
                    pad_width += [(0, 0)] * (a_cut.ndim - 1)
                a_padded = np.pad(a_cut, pad_width, mode="constant", constant_values=0)
                stacked.append(a_padded)
            return np.stack(stacked, axis=0)

        out: dict[str, dict | int] = {}

        # Process dict-like keys: pad/stack along first dim
        for k_dict in OUTPUT_DICT_KEYS:
            # Getting the max length
            max_length = max([d[k_dict]["max_length"] for d in items])
            out[k_dict] = {
                k: torch.from_numpy(
                    pad_stack_first_dim(
                        [m for m in (it[k_dict][k] for it in items)],
                        max_length
                    )
                )
                for k in INTERNAL_TENSOR_KEYS
            } | {
                "max_length": max_length
            }

        return out

    def __getitem__(self, idx):
        # Getting the absolute index from the relative index
        cur_idx_row = self.df_idx.iloc[[idx]]
        idx = cur_idx_row.index[0]

        sign = self.df.loc[idx, "sign"]

        # Getting the datapoint itself (anchor)
        out_data = self._combine_items(
            items=[self._fetch(idx)]
        )

        # Merging all datapoints into a single dictionary
        out = {
            "anchor_idx": torch.from_numpy(np.array(idx)),
            "sign": torch.from_numpy(np.array(sign)),
        } | out_data

        return out

    def _sample(self):
        # Sampling positive values
        sample_pos = self.df.loc[
            self.df["feature_type"].isin(self.feature_type_pos)
        ].sample(
            self.k_pos,
            replace=True  # Safety
        ).index

        sample_neg = self.df.loc[
            self.df["feature_type"].isin(self.feature_type_neg)
        ].sample(
            self.k_neg,
            replace=True  # Safety
        ).index

        sample_neutral = self.df.loc[
            self.df["feature_type"].isin(self.feature_type_neutral)
        ].sample(
            self.k_neutral,
            replace=True  # Safety
        ).index

        out_pos = self._combine_items(
            items=[self._fetch(x) for x in sample_pos]
        )
        # Getting "opposite" datapoints
        out_neg = self._combine_items(
            items=[self._fetch(x) for x in sample_neg]
        )
        # Getting "neutral" datapoints
        out_neutral = self._combine_items(
            items=[self._fetch(x) for x in sample_neutral]
        )

        return out_pos, out_neg, out_neutral


class IntactContrastiveStream(IterableDataset):
    def __init__(
        self,
        intact_ds: IntactDataset,
        steps_per_epoch: int | None = None,
        seed: int | None = None
    ):
        """
        intact_ds: an initialized IntactDataset (used as a helper/provider)
        steps_per_epoch: cap number of yielded batches per epoch (optional)
        seed: base RNG seed for reproducibility (worker-specific seeding applied)
        """
        self.ds = intact_ds
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

    def _seed_worker(self, worker_id: int):
        # Create a distinct seed for each worker to keep streams disjoint but reproducible
        base = self.seed if self.seed is not None else int(time.time())
        s = base + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0

        # Seed per-worker RNGs
        self._seed_worker(worker_id)

        step = 0
        while True:
            # Produce one contrastive batch by calling the existing sampler
            pos, neg, neutral = self.ds._sample()
            yield {"positive": pos, "negative": neg, "neutral": neutral}

            step += 1
            if self.steps_per_epoch is not None and step >= self.steps_per_epoch:
                break
