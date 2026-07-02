"""Gradient checkpointing must be a pure memory optimisation: identical loss and gradients to the
plain forward. The ProteinMPNN forward draws a random decode order, so this only holds because
``checkpoint(use_reentrant=False)`` preserves+restores the RNG state for the recompute. We assert
that equivalence on the real (small) model.

Skips when the IntAct data is not present locally.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

from stabddg.intact.dataset import intact_collate_fn

sys.path.insert(0, os.path.dirname(__file__))
from test_train_step_intact import _build_dataset, _build_small_model  # noqa: E402


def _loss_and_grads(model, batch, use_ckpt):
    model.use_grad_checkpoint = use_ckpt
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)  # same RNG start -> same decode order for both runs
    z = model.fused_forward_intact_datapoint(batch)
    loss = z.sum()
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return loss.detach().clone(), grads


def test_grad_checkpoint_matches_plain_forward():
    ds = _build_dataset(max_length=192)
    model = _build_small_model(torch.device("cpu"))
    n = min(3, len(ds))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=n, shuffle=False, collate_fn=intact_collate_fn)
    batch = next(iter(dl))

    loss_plain, grads_plain = _loss_and_grads(model, batch, use_ckpt=False)
    loss_ckpt, grads_ckpt = _loss_and_grads(model, batch, use_ckpt=True)

    assert torch.allclose(loss_plain, loss_ckpt, atol=1e-5), (loss_plain, loss_ckpt)
    assert len(grads_plain) == len(grads_ckpt) and grads_plain
    for gp, gc in zip(grads_plain, grads_ckpt):
        assert torch.allclose(gp, gc, atol=1e-5), (gp - gc).abs().max()
