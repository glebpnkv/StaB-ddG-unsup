import random
import time

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, IterableDataset, get_worker_info, DataLoader
from tqdm.auto import tqdm

from stabddg.intact.losses import ContrastiveLoss, ContrastiveLossOld


class RegressionModel(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.linear = torch.nn.Linear(k, 1)

    def forward(self, x):
        return self.linear(x)


class SynthDataset(Dataset):
    def __init__(
        self,
        x,
        df_synth,
        k_neutral: int = 16,
        k_pos: int = 8,
        k_neg: int = 8,
    ):
        self.k_neutral = k_neutral
        self.k_pos = k_pos
        self.k_neg = k_neg

        # Features as numpy array
        self.x = x
        df = df_synth.copy()

        # Split by sign
        self.df_pos = df.loc[df["sign"] == 1].copy()
        self.df_neg = df.loc[df["sign"] == -1].copy()
        self.df_neutral = df.loc[df["sign"] == 0].copy()
        self.df = pd.concat([self.df_pos, self.df_neg, self.df_neutral])

        # Index for sampling (pos + neg only)
        self.df_idx = pd.concat([self.df_pos, self.df_neg]).index

    def __len__(self):
        return len(self.df_idx)

    def __getitem__(self, idx):
        # Absolute index in the original dataframe
        cur_idx_row = self.df_idx[idx]
        idx = cur_idx_row

        sign = self.df.loc[idx, "sign"]

        # Base datapoint
        out_data = {
            "x": torch.as_tensor(self.x[idx, ...], dtype=torch.float32),
            "y": torch.as_tensor(self.df.loc[idx, "y"], dtype=torch.float32)
        }

        # Merge into a single dict
        out = {
            "anchor_idx": torch.as_tensor(idx, dtype=torch.long),
            "sign": torch.as_tensor(sign, dtype=torch.long),
        } | out_data

        return out

    @staticmethod
    def _combine_items(items: list[dict]) -> dict:
        """
        Combine a list of __getitem__ outputs into a mini-batch.
        Returns a dict with stacked tensors.
        """
        anchor_idx = torch.stack([torch.as_tensor(it["anchor_idx"], dtype=torch.long) for it in items])
        sign = torch.stack([torch.as_tensor(it["sign"], dtype=torch.long) for it in items])
        x = torch.stack([torch.as_tensor(it["x"], dtype=torch.float32) for it in items])
        y = torch.stack([torch.as_tensor(it["y"], dtype=torch.float32) for it in items])
        return {"anchor_idx": anchor_idx, "sign": sign, "x": x, "y": y}

    def _fetch(self, idx):
        sign = self.df.loc[idx, "sign"]

        # Base datapoint
        out_data = {
            "x": torch.as_tensor(self.x[idx, ...], dtype=torch.float32),
            "y": torch.as_tensor(self.df.loc[idx, "y"], dtype=torch.float32)
        }

        # Merge into a single dict
        out = {
            "anchor_idx": torch.as_tensor(idx, dtype=torch.long),
            "sign": torch.as_tensor(sign, dtype=torch.long),
        } | out_data

        return out

    def _sample(self):
        # Sampling positive values
        sample_pos = self.df.loc[
            self.df["sign"] == 1
        ].sample(
            self.k_pos,
            replace=True  # Safety
        ).index

        sample_neg = self.df.loc[
            self.df["sign"] == -1
        ].sample(
            self.k_neg,
            replace=True  # Safety
        ).index

        sample_neutral = self.df.loc[
            self.df["sign"] == 0
        ].sample(
            self.k_neutral,
            replace=True  # Safety
        ).index

        out_pos = self._combine_items([self._fetch(x) for x in sample_pos])
        out_neg = self._combine_items([self._fetch(x) for x in sample_neg])
        out_neutral = self._combine_items([self._fetch(x) for x in sample_neutral])

        return out_pos, out_neg, out_neutral


class SynthContrastiveStream(IterableDataset):
    def __init__(
        self,
        ds: SynthDataset,
        steps_per_epoch: int | None = None,
        seed: int | None = None,
    ):
        """
        synth_ds: an initialized SynthDataset (used as a helper/provider)
        steps_per_epoch: cap number of yielded batches per epoch (optional)
        seed: base RNG seed for reproducibility (worker-specific seeding applied)
        """
        self.ds = ds
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

    def _seed_worker(self, worker_id: int):
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
            pos, neg, neutral = self.ds._sample()
            yield {"positive": pos, "negative": neg, "neutral": neutral}

            step += 1
            if self.steps_per_epoch is not None and step >= self.steps_per_epoch:
                break


def generate_synth_data(
    n: int = 5000,
    k: int = 3,
    seed: int | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generates synthetic data frame and features"""
    rng = np.random.default_rng(seed=seed)

    beta = rng.normal(size=k + 1)
    x = rng.uniform(low=-10, high=10, size=(n, k))
    eps = rng.normal(scale=4.0, size=n)

    y = (x @ beta[1:]) + beta[0] + eps

    df_synth = pd.DataFrame(
        y.reshape(-1, 1),
        columns=["y"]
    )

    df_synth["sign"] = (~df_synth["y"].between(-1, 1)).astype("int")
    df_synth.loc[
        df_synth["y"] < -1,
        "sign"
    ] = -1

    print(beta)

    return df_synth, x


def synth_collate_fn(x):
    out = {}

    # Combining "anchor_idx" and "sign" into a single tensor
    out["anchor_idx"] = torch.stack([it["anchor_idx"] for it in x])
    out["sign"] = torch.stack([it["sign"] for it in x])
    out["x"] = torch.stack([it["x"] for it in x])
    out["y"] = torch.stack([it["y"] for it in x])

    return out


def passthrough_collate_fn(x):
    return x[0]


def _train(
    df_synth: pd.DataFrame,
    x: np.ndarray,
    model: RegressionModel,
    k_neutral: int = 16,
    k_pos: int = 8,
    k_neg: int = 8,
    batch_size: int = 64,
    n_epochs: int = 10,
    lr: float = 1e-3,
    lambda_supcon: float = 1.0,
    lambda_sign: float = 0.0,
    lambda_neutral: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Device
    accelerator = Accelerator()
    device = accelerator.device

    model = model.to(device)
    # Initialize dataset and model
    ds = SynthDataset(x, df_synth, k_neutral=k_neutral, k_pos=k_pos, k_neg=k_neg)
    ds_contrastive = SynthContrastiveStream(ds)

    pin_mem = (accelerator.device.type == "cuda")  # False on mps/cpu

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=synth_collate_fn,
        pin_memory=pin_mem,
    )
    dl_contrastive = DataLoader(
        ds_contrastive,
        batch_size=1,
        shuffle=False,
        pin_memory=pin_mem,
        collate_fn=passthrough_collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = ContrastiveLoss(
        lambda_supcon=lambda_supcon,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
    )
    # loss_fn = ContrastiveLossOld(
    #     lambda_supcon=lambda_supcon,
    #     lambda_sign=lambda_sign,
    #     lambda_neutral=lambda_neutral
    # )

    df_metrics = None
    df_forecasts = pd.DataFrame()
    model, optimizer, dl, dl_contrastive = accelerator.prepare(model, optimizer, dl, dl_contrastive)

    pbar_epoch = tqdm(range(n_epochs), desc="Epoch", leave=False)

    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"Epoch {epoch + 1}")

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Reset per-epoch aggregates
        all_train_metrics = None
        df_forecasts_epoch = pd.DataFrame()

        contrast_iter = iter(dl_contrastive)
        # pbar = tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)

        # for i, batch in enumerate(pbar):
        for i, batch in enumerate(dl):
            # Refresh contrastive iterator if exhausted
            try:
                batch_contrast = next(contrast_iter)
            except StopIteration:
                contrast_iter = iter(dl_contrastive)
                batch_contrast = next(contrast_iter)

            z_anchor = model(batch["x"])
            z_pos = model(batch_contrast["positive"]["x"])
            z_neg = model(batch_contrast["negative"]["x"])
            z_neu = model(batch_contrast["neutral"]["x"])

            z_sign = batch["sign"]

            loss, metrics = loss_fn.return_losses_and_metrics(
                z=z_anchor,
                z_sign=z_sign,
                z_pos=z_pos,
                z_neg=z_neg,
                z_neu=z_neu,
            )

            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Progress bar
            # pbar.set_postfix(metrics)

            # Running mean of metrics
            if all_train_metrics is None:
                all_train_metrics = metrics
            else:
                for k, v in metrics.items():
                    all_train_metrics[k] = (all_train_metrics[k] * i + v) / (i + 1)

            # Collect forecasts with ground truth
            df_forecasts_cur = pd.DataFrame(
                {
                    "anchor_idx": batch["anchor_idx"].detach().cpu().numpy(),
                    "forecast": z_anchor.detach().cpu().flatten().numpy(),
                    "y": batch["y"].detach().cpu().numpy(),
                }
            )
            df_forecasts_epoch = pd.concat([df_forecasts_epoch, df_forecasts_cur], ignore_index=True)

        # Annotate epoch/split and append
        df_forecasts_epoch["epoch"] = epoch + 1
        df_forecasts_epoch["split"] = "Train"
        df_forecasts = pd.concat([df_forecasts, df_forecasts_epoch], ignore_index=True)

        df_metrics_cur = pd.DataFrame({"epoch": epoch + 1} | all_train_metrics, index=[0])
        df_metrics = pd.concat([df_metrics, df_metrics_cur], ignore_index=True)

        # Print epoch-level aggregates
        if all_train_metrics:
            pbar_epoch.set_postfix(all_train_metrics)
            tqdm.write(
                f'Epoch {epoch + 1}: '
                f'{ {k: f"{v:.4f}" for k, v in all_train_metrics.items() if v is not None} }'
            )

    # Adding run's key hyperparameters to df_metrics
    df_metrics["lambda_supcon"] = lambda_supcon
    df_metrics["lambda_sign"] = lambda_sign
    df_metrics["lr"] = lr

    return df_metrics, df_forecasts


def train(
    n: int = 100,
    k: int = 3,
    seed: int | None = None,
    k_neutral: int = 16,
    k_pos: int = 8,
    k_neg: int = 8,
    batch_size: int = 64,
    n_epochs: int = 500,
    lr: float = 1e-2,
    lambda_sign: float = 5.0,
    lambda_neutral: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Generate synthetic data
    df_synth, x = generate_synth_data(n=n, k=k, seed=seed)
    model = RegressionModel(k)

    df_metrics, df_forecasts = _train(
        df_synth,
        x,
        model,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg,
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
    )

    return df_metrics, df_forecasts


if __name__ == "__main__":
    train()
