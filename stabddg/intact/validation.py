"""Ground-truth validation for the IntAct mutation/structure alignment.

IntAct independently tells us the wild-type residue(s) of every annotated mutation
(``original_sequence``). That gives a self-contained oracle for the data-extraction
pipeline: load the assembly structure, look at the residue(s) the stored numbering
points to, and check they equal the wild type IntAct expects. No network calls.

This is used both to *measure* the data quality and as the regression gate for the
fix to the assembly residue-numbering (see ``stabddg/intact/data.py``).
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from safetensors import safe_open

from stabddg.constants import ALPHABET


@dataclass
class WTConsistencyStats:
    n: int                 # rows evaluated (affected chain found in assembly)
    match: int             # WT at the mapped position equals original_sequence
    mismatch_present: int  # WT differs, but the WT segment exists elsewhere in the chain
    mismatch_absent: int   # WT not found anywhere in the chain (missing/isoform)
    skipped: int           # assembly file missing or affected chain not in metadata

    @property
    def match_rate(self) -> float:
        return self.match / self.n if self.n else 0.0

    def __str__(self) -> str:
        return (
            f"WT-consistency: match={self.match}/{self.n} ({self.match_rate:.1%}), "
            f"mismatch_present={self.mismatch_present}, mismatch_absent={self.mismatch_absent}, "
            f"skipped={self.skipped}"
        )


def _decode(S: np.ndarray) -> str:
    return "".join(ALPHABET[int(i)] for i in S)


def _affected_chain_ids(metadata: dict, affected_ac: str) -> list[int]:
    """Resolve the chain-encoding ids that belong to ``affected_ac`` (mirrors
    IntactDataset._make_mutation_sequence's metadata interpretation)."""
    enc = {k.strip("chain_encoding:"): v for k, v in metadata.items() if "chain_encoding" in k}
    enc = {int(k): metadata.get(v) for k, v in enc.items()}
    return [cid for cid, prot in enc.items() if prot == affected_ac]


def load_eval_frame(mutations_path: str, assemblies_filtered_path: str) -> pd.DataFrame:
    """Reproduce the train-time view: filter to length-preserving substitutions, merge in the
    chosen ``biological_assembly``, and convert feature ranges to 1-based (as prepare_intact_splits does)."""
    df = pd.read_parquet(mutations_path)
    df = df.loc[
        (df["resulting_sequence"] != ".")
        & (df["original_sequence"].str.len() == df["resulting_sequence"].str.len())
    ].reset_index(drop=True)

    asm = pd.read_parquet(assemblies_filtered_path)
    keys = [c for c in ["participant_protein", "affected_protein_ac", "biological_assembly"] if c in asm.columns]
    df = df.merge(asm[keys].drop_duplicates(), on=["participant_protein", "affected_protein_ac"], how="inner")

    # prepare_intact_splits converts the stored 0-based ranges to 1-based.
    df["feature_ranges_start"] = df["feature_ranges_start"].astype(int) + 1
    df["feature_ranges_end"] = df["feature_ranges_end"].astype(int) + 1
    return df


def _load_canonical(proteins_dir: str, acc: str, cache: dict) -> str | None:
    if acc in cache:
        return cache[acc]
    seq = None
    path = os.path.join(proteins_dir, f"{acc}.safetensors")
    if os.path.exists(path):
        with safe_open(path, framework="np") as f:
            seq = _decode(f.get_tensor("S"))
    cache[acc] = seq
    return seq


def make_alignment_resnum_fn(proteins_dir: str):
    """Build a ``resnum_fn`` for ``evaluate_wt_consistency`` that re-derives each assembly's residue
    numbering by aligning every chain's stored sequence to its canonical monomer sequence — i.e. the
    proposed extraction fix, evaluated on the existing files without any re-download."""
    from stabddg.intact.data import _build_uniprot_aligner, assign_uniprot_positions_by_alignment

    aligner = _build_uniprot_aligner()
    canon_cache: dict = {}

    def fn(data: dict, meta: dict) -> np.ndarray:
        S = data["S"]
        chain = data["chain_encoding_all"].astype(int)
        label_by_id = {int(k.split(":", 1)[1]): v for k, v in meta.items() if k.startswith("chain_encoding:")}
        acc_by_id = {cid: meta.get(lbl) for cid, lbl in label_by_id.items()}

        resn = np.full(len(S), -1, dtype=np.int64)
        for cid in np.unique(chain):
            acc = acc_by_id.get(int(cid))
            if not acc:
                continue
            canon = _load_canonical(proteins_dir, acc, canon_cache)
            if not canon:
                continue
            idx = np.flatnonzero(chain == cid)
            resn[idx] = assign_uniprot_positions_by_alignment(_decode(S[idx]), canon, aligner=aligner)
        return resn

    return fn


def evaluate_wt_consistency(
    eval_df: pd.DataFrame,
    assemblies_dir: str,
    resnum_fn=None,
    sample: int | None = None,
    random_state: int = 1,
) -> WTConsistencyStats:
    """Measure how often the residue(s) the numbering points to match IntAct's wild type.

    resnum_fn: optional ``(safetensors_data: dict, metadata: dict) -> np.ndarray`` that returns a
    replacement ``resnums`` vector for the assembly (used to evaluate a *candidate* numbering, e.g.
    the alignment-based fix, against the same oracle). When None, the stored ``resnums`` are used.
    """
    rows = eval_df.sample(min(sample, len(eval_df)), random_state=random_state) if sample else eval_df

    match = mm_present = mm_absent = skipped = 0
    for _, r in rows.iterrows():
        path = os.path.join(assemblies_dir, f"{r['biological_assembly']}.safetensors")
        if not os.path.exists(path):
            skipped += 1
            continue
        with safe_open(path, framework="np") as f:
            data = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata() or {}

        ids = _affected_chain_ids(meta, r["affected_protein_ac"])
        if not ids:
            skipped += 1
            continue

        chain = data["chain_encoding_all"].astype(int)
        S = data["S"]
        resn = (resnum_fn(data, meta) if resnum_fn is not None else data["resnums"]).astype(int)

        s = int(r["feature_ranges_start"])
        e = int(r["feature_ranges_end"])
        orig = str(r["original_sequence"])

        # The affected protein may occupy several chain copies (e.g. a homomer); the mutation applies
        # to each. Evaluate per copy: numbering is correct if any copy places the WT where IntAct says.
        copy_match = False
        present_any = False
        for cid in ids:
            cidx = np.flatnonzero(chain == cid)
            sel = cidx[(resn[cidx] >= s) & (resn[cidx] < e)]
            if _decode(S[sel]) == orig:
                copy_match = True
            if len(orig) > 0 and orig in _decode(S[cidx]):
                present_any = True

        if copy_match:
            match += 1
        elif present_any:
            mm_present += 1
        else:
            mm_absent += 1

    n = match + mm_present + mm_absent
    return WTConsistencyStats(n=n, match=match, mismatch_present=mm_present,
                              mismatch_absent=mm_absent, skipped=skipped)
