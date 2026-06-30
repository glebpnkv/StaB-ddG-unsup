"""Unit tests for IntactDataset._make_mutation_sequence.

Regression coverage for the C-terminal off-by-one that silently produced
mutated sequences longer than the structure (any mutation range touching the
last residue of the affected chain).
"""
import numpy as np
import pandas as pd
import pytest

from stabddg.constants import ALPHABET
from stabddg.intact.dataset import IntactDataset


@pytest.fixture
def ds():
    # Bypass the heavy __init__ (parquet/safetensors IO); only _fetch_sequence
    # and sequence_unknown are needed by _make_mutation_sequence.
    d = IntactDataset.__new__(IntactDataset)
    d.sequence_unknown = "X"
    return d


def _decode(arr):
    return "".join(ALPHABET[i] for i in arr)


def _datapoint(seq, chain, resnums, metadata):
    return {
        "metadata": metadata,
        "data": {
            "S": np.array([ALPHABET.index(c) for c in seq], dtype=np.int32),
            "chain_encoding_all": np.array(chain, dtype=np.int32),
            "resnums": np.array(resnums, dtype=np.int32),
        },
    }


def _apply(ds, dp, ac, start, end, repl):
    sample = pd.Series({
        "affected_protein_ac": ac,
        "feature_ranges_start": start,
        "feature_ranges_end": end,
        "resulting_sequence": repl,
    })
    return _decode(ds._make_mutation_sequence(dp, sample))


@pytest.fixture
def single_chain():
    # residues 1..6, sequence ACDEFG, one chain encoded as id 1 -> protein P1
    return _datapoint("ACDEFG", [1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6],
                      {"chain_encoding:1": "cA", "cA": "P1"})


@pytest.mark.parametrize("start,end,repl,expected", [
    (3, 5, "KL", "ACKLFG"),      # middle substitution
    (5, 7, "KL", "ACDEKL"),      # touches C-terminus (was the bug: ACDEKLG)
    (6, 7, "K", "ACDEFK"),       # single residue at the very end (was: ACDEFKG)
    (1, 2, "K", "KCDEFG"),       # first residue
    (1, 7, "MNPQRS", "MNPQRS"),  # whole chain
])
def test_substitutions_preserve_length(single_chain, ds, start, end, repl, expected):
    out = _apply(ds, single_chain, "P1", start, end, repl)
    assert out == expected
    assert len(out) == single_chain["data"]["S"].shape[0]


def test_homomer_mutates_all_copies(ds):
    # P1 occupies two chains; resnums restart per chain. Mutating resnum 2 hits
    # both copies (S indices 1 and 4), which are non-contiguous in S.
    dp = _datapoint("ACDEFG", [1, 1, 1, 2, 2, 2], [1, 2, 3, 1, 2, 3],
                    {"chain_encoding:1": "cA", "cA": "P1",
                     "chain_encoding:2": "cB", "cB": "P1"})
    assert _apply(ds, dp, "P1", 2, 3, "K") == "AKDEKG"


def test_out_of_range_returns_original(single_chain, ds):
    # A range with no matching residues leaves the sequence untouched.
    assert _apply(ds, single_chain, "P1", 100, 105, "K") == "ACDEFG"
