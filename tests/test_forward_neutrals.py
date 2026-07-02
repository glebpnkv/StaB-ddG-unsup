"""Unit tests for the conditional neutral forward (memory lever against high-k OOM).

Neutrals must only build the autograd graph they actually need:
  * both off (default)          -> skipped entirely (empty tensor, no forward);
  * normaliser on, lambda == 0  -> forwarded under no_grad (stats are detached);
  * lambda_neutral > 0          -> forwarded with grad (neutral reg term needs it).

These are pure unit tests with fakes, so they run without the IntAct data.
"""
import torch

from stabddg.intact.training import _forward_neutrals


_B = 3  # neutral pool size in these fakes


class _FakeModel:
    def __init__(self):
        self.calls = 0
        self.p = torch.nn.Parameter(torch.ones(1))

    def fused_forward_intact_datapoint(self, batch):
        self.calls += 1
        return self.p * torch.ones(_B)  # depends on p so grad can flow when enabled


class _FakeLoss:
    def __init__(self, lambda_neutral=0.0, use_neutral_normalizer=False):
        self.lambda_neutral = lambda_neutral
        self.use_neutral_normalizer = use_neutral_normalizer


_REF = torch.zeros(2)
# _forward_neutrals delegates to _microbatched_forward, which reads batch["complex"]["S"].shape[0].
_BATCH = {"complex": {"S": torch.zeros(_B, 2)}, "binder1": {}, "binder2": {}}


def test_neutrals_skipped_when_unused():
    model, loss_fn = _FakeModel(), _FakeLoss()
    out = _forward_neutrals(model, loss_fn, _BATCH, ref=_REF)
    assert out.numel() == 0
    assert model.calls == 0  # no forward at all -> frees the largest pool


def test_neutrals_no_grad_when_normalizer_only():
    model, loss_fn = _FakeModel(), _FakeLoss(use_neutral_normalizer=True)
    out = _forward_neutrals(model, loss_fn, _BATCH, ref=_REF)
    assert model.calls == 1
    assert out.numel() == _B
    assert out.requires_grad is False  # detached stats only -> no activations retained


def test_neutrals_with_grad_when_lambda_positive():
    model, loss_fn = _FakeModel(), _FakeLoss(lambda_neutral=0.5)
    out = _forward_neutrals(model, loss_fn, _BATCH, ref=_REF)
    assert model.calls == 1
    assert out.requires_grad is True  # neutral reg term differentiates through these
