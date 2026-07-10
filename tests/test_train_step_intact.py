"""Smoke test for the IntAct contrastive train_step on real local data.

Skips automatically when the IntAct safetensors/parquet are not present.
"""
import math
import os

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from stabddg.intact.dataset import (
    IntactContrastiveStream,
    IntactDataset,
    intact_collate_fn,
    passthrough_collate_fn,
)
from stabddg.intact.losses import ContrastiveLoss
from stabddg.intact.training import train_step
from stabddg.model import StaBddG
from stabddg.mpnn_utils import ProteinMPNN


def _build_small_model(device: torch.device) -> StaBddG:
    pmpnn = ProteinMPNN(
        node_features=64,
        edge_features=64,
        hidden_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=16,
        dropout=0.0,
        augment_eps=0.0,
    )
    model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=False, noise_level=0.1, device=device)
    model.to(device)
    return model


def _build_dataset(max_length: int = 192) -> IntactDataset:
    data_dir = os.path.join("data", "intact")
    assemblies_dir = os.path.join(data_dir, "assemblies", "safetensors")

    parquet_candidates = [
        os.path.join(data_dir, "df_intact_mutations_filtered_train.parquet"),
        os.path.join(data_dir, "df_intact_mutations_filtered.parquet"),
    ]
    df_intact_path = next((p for p in parquet_candidates if os.path.exists(p)), None)
    if df_intact_path is None or not os.path.isdir(assemblies_dir):
        pytest.skip("IntAct data not found under data/intact. Skipping.")

    ds = IntactDataset(
        df_intact_path=df_intact_path,
        assemblies_dir=assemblies_dir,
        max_length=max_length,
        k_neutral=8,
        k_pos=2,
        k_neg=2,
    )
    if len(ds) == 0:
        pytest.skip("IntAct dataset is empty. Skipping.")
    return ds


def _build_loaders(ds: IntactDataset, n_anchor: int = 2):
    anchor = DataLoader(
        Subset(ds, list(range(min(n_anchor, len(ds))))),
        batch_size=min(n_anchor, len(ds)),
        shuffle=False,
        collate_fn=intact_collate_fn,
    )
    contrastive = DataLoader(
        IntactContrastiveStream(ds, steps_per_epoch=4, seed=0),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=passthrough_collate_fn,
    )
    return anchor, contrastive


# Metric keys the current ContrastiveLoss reports.
_EXPECTED_METRIC_KEYS = ["loss_total", "loss_supcon", "loss_sign", "loss_neutral_reg"]


def test_train_step_single_batch_with_detect_anomaly():
    device = torch.device("cpu")
    model = _build_small_model(device)
    ds = _build_dataset()
    dl, dl_contrastive = _build_loaders(ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = ContrastiveLoss()

    with torch.autograd.detect_anomaly():
        metrics, df_forecasts = train_step(
            model=model,
            dataloader=dl,
            dataloader_contrastive=dl_contrastive,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=0,
            grad_accum_steps=1,
        )

    assert isinstance(metrics, dict)
    for key in _EXPECTED_METRIC_KEYS:
        assert key in metrics, f"missing metric {key}"
        assert math.isfinite(float(metrics[key])), f"metric {key} not finite: {metrics[key]}"
    assert len(df_forecasts) > 0


def test_train_step_no_neutrals_pure_sign():
    """The next-run config: k_neutral=0 (neutrals dropped) + pure sign loss + normaliser off.

    Exercises the empty-neutral path end-to-end (sampler yields neutral=None, _forward_neutrals
    returns empty, loss consumes an empty z_neu) and confirms a finite, SupCon-free loss.
    """
    device = torch.device("cpu")
    model = _build_small_model(device)
    ds = _build_dataset()
    ds.k_neutral = 0  # drop the neutral pool entirely
    dl, dl_contrastive = _build_loaders(ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = ContrastiveLoss(lambda_supcon=0.0, lambda_sign=1.0, lambda_neutral=0.0,
                              use_neutral_normalizer=False)

    metrics, df_forecasts = train_step(
        model=model, dataloader=dl, dataloader_contrastive=dl_contrastive,
        optimizer=optimizer, loss_fn=loss_fn, epoch=0, micro_batch_size=2,
    )
    assert math.isfinite(float(metrics["loss_total"]))
    # lambda_supcon=0 -> total is exactly the (weighted) sign term
    assert abs(float(metrics["loss_total"]) - float(metrics["loss_sign"])) < 1e-5
    assert len(df_forecasts) > 0
