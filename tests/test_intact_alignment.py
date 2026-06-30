"""Tests for the sequence-alignment-based UniProt residue numbering (the fix for the assembly
mis-numbering bug in stabddg/intact/data.py).
"""
import numpy as np
import pandas as pd

from stabddg.intact.data import (
    add_uniprot_by_alignment,
    assign_uniprot_positions_by_alignment,
)

# A canonical reference with no repeated substrings (so fragment placements are unambiguous).
CANON = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDCNWHYTPEM"


def test_exact_match_is_identity():
    pos = assign_uniprot_positions_by_alignment(CANON, CANON)
    assert pos.tolist() == list(range(1, len(CANON) + 1))


def test_internal_fragment_gets_offset_positions():
    frag = CANON[9:19]  # canonical positions 10..19
    pos = assign_uniprot_positions_by_alignment(frag, CANON)
    assert pos.tolist() == list(range(10, 20))


def test_leading_tag_does_not_shift_real_residues():
    # The point of aligning (vs offset arithmetic) is that an N-terminal tag absent from the canonical
    # sequence must not shift the numbering of the residues that ARE in the canonical sequence.
    frag = CANON[5:15]            # canonical positions 6..15
    observed = "WWWW" + frag      # 4-residue tag prefix
    pos = assign_uniprot_positions_by_alignment(observed, CANON)
    assert pos[4:].tolist() == list(range(6, 16))  # the real residues keep their correct positions


def test_internal_deletion_keeps_blocks_registered():
    # An unresolved internal loop must be skipped (gapped), not absorbed by shifting the sequence.
    observed = CANON[0:6] + CANON[12:18]  # canonical 7..12 missing
    pos = assign_uniprot_positions_by_alignment(observed, CANON)
    # Residues away from the gap boundary must land on their true canonical positions.
    assert pos[:5].tolist() == list(range(1, 6))      # block 1 head -> 1..5
    assert pos[-5:].tolist() == list(range(14, 19))   # block 2 tail -> 14..18


def test_empty_inputs():
    assert assign_uniprot_positions_by_alignment("", CANON).tolist() == []
    assert assign_uniprot_positions_by_alignment("ACD", "").tolist() == [-1, -1, -1]


def _atom_df(chain, one_letter_seq, start_label=1):
    """One CA atom per residue for a single chain."""
    rows = []
    for i, aa in enumerate(one_letter_seq):
        rows.append({
            "chain": chain,
            "res_name_1": aa,
            "resnum_label": start_label + i,
            "resnum_auth": start_label + i,
            "atom_name": "CA",
        })
    return pd.DataFrame(rows)


def test_add_uniprot_by_alignment_maps_each_chain():
    # Chain A is canonical 10..19; chain B carries a 2-residue tag then canonical 3..8.
    df = pd.concat([
        _atom_df("A", CANON[9:19], start_label=1),
        _atom_df("B", "WW" + CANON[2:8], start_label=1),
    ], ignore_index=True)
    out = add_uniprot_by_alignment(df, {"A": "P1", "B": "P2"}, {"P1": CANON, "P2": CANON})

    a = out.loc[out["chain"] == "A", "resnum_uniprot"].tolist()
    assert a == list(range(10, 20))

    # Chain B's real residues (after the 2-residue tag) keep their correct canonical positions.
    b = out.loc[out["chain"] == "B", "resnum_uniprot"].tolist()
    assert b[2:] == list(range(3, 9))


def test_add_uniprot_by_alignment_skips_unknown_chain():
    df = _atom_df("A", CANON[0:5])
    out = add_uniprot_by_alignment(df, {"A": "P1"}, {"P1": None})  # no canonical available
    assert out["resnum_uniprot"].isna().all()
