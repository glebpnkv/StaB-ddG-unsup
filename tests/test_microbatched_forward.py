"""Unit tests for the micro-batching helpers (memory lever for large contrast pools).

Micro-batching must be a pure memory optimisation: splitting a pool into chunks, forwarding each, and
concatenating the outputs has to reproduce the whole-batch result exactly (the "un-glue"). These are
data-free unit tests with a fake model.
"""
import torch

from stabddg.intact.training import _microbatched_forward, _slice_pool


class _CountingModel:
    def __init__(self):
        self.calls = 0

    def fused_forward_intact_datapoint(self, batch):
        self.calls += 1
        # A per-datapoint scalar derived from the input, so chunk-then-concat must equal whole-batch.
        return batch["complex"]["S"][:, 0].float()


def _make_batch(B: int, L: int = 4) -> dict:
    def blk():
        return {"S": torch.arange(B * L).reshape(B, L), "max_length": L}

    return {
        "complex": blk(), "binder1": blk(), "binder2": blk(),
        "anchor_idx": torch.arange(B), "sign": torch.ones(B),
    }


def test_microbatch_matches_whole_batch_exactly():
    batch = _make_batch(5)
    whole = _microbatched_forward(_CountingModel(), batch, micro_batch_size=0)
    chunked_model = _CountingModel()
    chunked = _microbatched_forward(chunked_model, batch, micro_batch_size=2)
    assert torch.equal(whole, chunked)   # un-glue is exact
    assert chunked_model.calls == 3      # ceil(5 / 2) chunks


def test_microbatch_disabled_and_oversized_are_single_forward():
    batch = _make_batch(5)
    for mbs in (0, 5, 99):
        m = _CountingModel()
        _microbatched_forward(m, batch, micro_batch_size=mbs)
        assert m.calls == 1              # no chunking when disabled or >= batch


def test_slice_pool_slices_tensors_and_passes_scalars():
    s = _slice_pool(_make_batch(4), 1, 3)
    assert s["complex"]["S"].shape[0] == 2
    assert s["complex"]["max_length"] == 4       # non-tensor passthrough, not sliced
    assert s["anchor_idx"].tolist() == [1, 2]
