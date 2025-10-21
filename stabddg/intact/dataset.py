import logging
import os

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from stabddg.constants import AA3_TO_1, ALPHABET, SEQUENCE_DELETION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def intact_collate_fn(x):
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
        k_same: int = 4,
        k_opp: int = 4,
    ):
        self.proteins_dir = proteins_dir
        self.assemblies_dir = assemblies_dir
        self.max_length = max_length
        self.k_neutral = k_neutral
        self.k_same = k_same
        self.k_opp = k_opp

        # Reading the metadata dataframe
        df = pd.read_parquet(df_intact_path)
        df = df.reset_index(drop=True)

        len_raw = len(df)

        # Removing rows with sequence length > max_length
        df = df.loc[df["max_len"] <= max_length].reset_index(drop=True)

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

        # Preparing output
        out = {
            "complex": cmplex["data"],
            "complex_mut_seqs": complex_mut_seqs,
        }

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

        out = {
            "binder1": binder1,
            "binder2": binder2,
            "binder1_mut_seqs": binder1.get("S"),
            "binder2_mut_seqs": binder2_mut_seqs,
        }

        return out

    def _fetch(self, idx):
        out_complex = self._fetch_complex(idx)
        # out_binders = self._fetch_binders_full(idx)
        out_binders = self._fetch_binders(idx)

        out = out_complex | out_binders
        return out

    def _combine_items(self, items: list[dict], add_mask: bool = True):
        """
        Batch and pad a list of fetched items.
        Each item is a dict with 6 keys total:
          - 3 keys with 1D arrays (L,) representing sequences/labels
          - 3 keys with dict values; each dict maps str -> array with first dim L (or L')
        Output preserves the same top-level keys as _fetch returns.
        For the 1D-array keys: output tensors of shape [B, self.max_length]
        For the dict keys: output dicts with tensors of shape [B, self.max_length, ...]
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

        def make_mask_from_first_dim(arr_list: list[np.ndarray], target_len: int) -> np.ndarray:
            # Builds [B, target_len] mask with 1.0 for valid positions based on original lengths
            B = len(arr_list)
            mask = np.zeros((B, target_len), dtype=np.float32)
            for i, a in enumerate(arr_list):
                l = min(int(a.shape[0]), target_len)
                mask[i, :l] = 1.0
            return mask

        # Determine which top-level keys are arrays and which are dicts using the first item
        example = items[0]
        top_level_keys = list(example.keys())

        # Identify array-like (1D L) keys and dict-like keys
        array_keys = [k for k in top_level_keys if isinstance(example[k], np.ndarray)]
        dict_keys = [k for k in top_level_keys if isinstance(example[k], dict)]

        out: dict[str, torch.Tensor | dict] = {}

        # Process array-like keys (expect (L,) arrays); output [B, max_length] long tensors
        for k in array_keys:
            arr_list = [it[k] for it in items]
            stacked_np = pad_stack_first_dim(arr_list, self.max_length)
            # Default to int64 for sequences; if not integer dtype, cast to float32
            if np.issubdtype(stacked_np.dtype, np.integer):
                out[k] = torch.from_numpy(stacked_np.astype(np.int64))
            else:
                out[k] = torch.from_numpy(stacked_np.astype(np.float32))

        # Optional global mask: choose one representative array source for lengths if available,
        # otherwise try from first dict key's first inner array.
        if add_mask:
            mask_source_list: list[np.ndarray] | None = None
            if array_keys:
                mask_source_list = [it[array_keys[0]] for it in items]
            elif dict_keys:
                first_dict_key = dict_keys[0]
                inner_keys = list(example[first_dict_key].keys())
                if inner_keys:
                    mask_source_list = [it[first_dict_key][inner_keys[0]] for it in items]
            if mask_source_list is not None:
                out["mask"] = torch.from_numpy(make_mask_from_first_dim(mask_source_list, self.max_length))

        # Process dict-like keys: preserve subkeys; pad/stack along first dim
        for k in dict_keys:
            inner_example = example[k]
            inner_keys = list(inner_example.keys())
            out_inner: dict[str, torch.Tensor] = {}
            for subk in inner_keys:
                arr_list = [it[k][subk] for it in items]
                stacked_np = pad_stack_first_dim(arr_list, self.max_length)
                # Numeric dtype handling
                if np.issubdtype(stacked_np.dtype, np.floating):
                    out_inner[subk] = torch.from_numpy(stacked_np.astype(np.float32))
                elif np.issubdtype(stacked_np.dtype, np.integer):
                    out_inner[subk] = torch.from_numpy(stacked_np.astype(np.int64))
                else:
                    # Fallback: try float32
                    out_inner[subk] = torch.from_numpy(stacked_np.astype(np.float32))
            out[k] = out_inner

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
            [self._fetch(idx)]
        )

        # Getting "same" datapoints
        out_same = self._combine_items(
            [self._fetch(x) for x in sample_same]
        )
        # Getting "opposite" datapoints
        out_opp = self._combine_items(
            [self._fetch(x) for x in sample_opp]
        )
        # Getting "neutral" datapoints
        out_neutral = self._combine_items(
            [self._fetch(x) for x in sample_neutral]
        )

        # Merging all datapoints into a single dictionary
        out = {
            "anchor": out_anchor,
            "same": out_same,
            "opp": out_opp,
            "neutral": out_neutral,
        }

        return out
