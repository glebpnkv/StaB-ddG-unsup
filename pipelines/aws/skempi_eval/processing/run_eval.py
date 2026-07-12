"""SageMaker Processing entry point for standalone SKEMPI v2 evaluation.

Loads a ProteinMPNN checkpoint produced by the IntAct pretrain job (from its ``model.tar.gz``), runs
the paper's ensemble ΔΔG prediction over the SKEMPI test split, and writes TWO outputs — the
per-mutation forecasts (the asset kept for any later diagnostics/plots) and a summary of the paper's
metrics computed *from* them:

  * predictions.csv       every per-mutation forecast: ``#Pdb, Mutation, ddG, ddG_pred``.
  * summary_metrics.json  baselines/eval_utils.py::compute_metrics — the pooled (rank-based, so
                          scale-agnostic) **Spearman** + bootstrap SE, plus Pearson / RMSE / MAE /
                          ROC-AUC / PR-AUC and the per-structure variants.

Container I/O (SageMaker Processing mounts):
  inputs   /opt/ml/processing/input/model/    the pretrain artifact (a single ``model.tar.gz``)
           /opt/ml/processing/input/skempi/   filtered_skempi.csv, test_pdb.pkl, *_pdb_dict.pkl
  outputs  /opt/ml/processing/output/predictions/predictions.csv
           /opt/ml/processing/output/summary/summary_metrics.json

``stabddg`` and ``baselines`` both come from the baked image (``COPY . .`` at /app), so only this file
is uploaded as the ScriptProcessor code.
"""
import argparse
import glob
import json
import math
import os
import sys
import tarfile

import pandas as pd
import torch

# baselines/ is copied to /app but is not part of the installed `stabddg` package; put /app on the
# path so `from baselines.eval_utils import compute_metrics` resolves (namespace package).
sys.path.insert(0, "/app")

INPUT_ROOT = "/opt/ml/processing/input"
OUTPUT_ROOT = "/opt/ml/processing/output"
WORK = "/opt/ml/processing/work"


def _extract_model(model_input_dir: str, dst: str) -> str:
    """Return a directory holding the pretrain artifact tree (model/, metrics/, ...).

    ProcessingInput mounts the ``model.tar.gz`` object as a file; extract it. If the caller instead
    mounted an already-unpacked tree, use it as-is.
    """
    tars = glob.glob(os.path.join(model_input_dir, "**", "*.tar.gz"), recursive=True)
    if not tars:
        return model_input_dir
    os.makedirs(dst, exist_ok=True)
    with tarfile.open(tars[0]) as t:
        t.extractall(dst)
    return dst


def _epoch_ckpts(model_dir: str) -> dict[int, str]:
    out = {}
    for p in glob.glob(os.path.join(model_dir, "epoch_*.pt")):
        try:
            out[int(os.path.basename(p)[len("epoch_"):-len(".pt")])] = p
        except ValueError:
            continue
    return out


def select_checkpoint(model_root: str, how: str) -> tuple[str, str]:
    """Resolve the requested checkpoint to (path, human_label). ``how`` is one of
    best_val | last | initial | epoch_<N>."""
    model_dir = os.path.join(model_root, "model")
    if how == "initial":
        return os.path.join(model_dir, "initial.pt"), "initial"
    if how.startswith("epoch_"):
        return os.path.join(model_dir, f"{how}.pt"), how
    if how == "last":
        final = os.path.join(model_dir, "final.pt")
        if os.path.exists(final):
            return final, "final"
        epochs = _epoch_ckpts(model_dir)
        n = max(epochs)
        return epochs[n], f"epoch_{n}"
    if how == "best_val":
        dfm = pd.read_csv(os.path.join(model_root, "metrics", "metrics.csv"))
        val = dfm[dfm["split"] == "Validation"]
        if len(val) == 0:
            raise ValueError("best_val requested but metrics.csv has no Validation rows; "
                             "use --checkpoint-select last or epoch_<N>.")
        best_epoch = int(val.loc[val["sign_violation_rate"].idxmin(), "epoch"])
        return os.path.join(model_dir, f"epoch_{best_epoch}.pt"), f"epoch_{best_epoch} (best val)"
    raise ValueError(f"Unknown --checkpoint-select {how!r}")


def _jsonable(v):
    """NaN/inf -> None; numpy scalars -> python floats (valid, comparable JSON)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if (math.isnan(f) or math.isinf(f)) else f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--noise-level", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-size", type=int, default=0, help="0 = full test set (debug only)")
    p.add_argument("--checkpoint-select", default="best_val",
                   help="best_val | last | initial | epoch_<N>")
    p.add_argument("--run-tag", default="run")
    args = p.parse_args()

    # Imports that need the baked image (kept after argparse so --help works anywhere).
    from stabddg.jobs.skempi_eval import skempi_eval
    from stabddg.model import StaBddG
    from stabddg.mpnn_utils import ProteinMPNN
    from stabddg.ppi_dataset import SKEMPIDataset

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[skempi-eval] device={device} run_tag={args.run_tag} ensemble={args.ensemble}")

    model_root = _extract_model(os.path.join(INPUT_ROOT, "model"), os.path.join(WORK, "model"))
    ckpt_path, ckpt_label = select_checkpoint(model_root, args.checkpoint_select)
    print(f"[skempi-eval] checkpoint: {ckpt_label} -> {ckpt_path}")

    skempi_dir = os.path.join(INPUT_ROOT, "skempi")
    dataset = SKEMPIDataset(
        csv_path=os.path.join(skempi_dir, "filtered_skempi.csv"),
        split_path=os.path.join(skempi_dir, "test_pdb.pkl"),
        pdb_dir=os.path.join(skempi_dir, "PDBs"),  # absent by design; the cache is authoritative
        pdb_dict_cache_path=os.path.join(skempi_dir, "skempi_full_mask_pdb_dict.pkl"),
    )

    # The eval model is the full ProteinMPNN the paper uses (the pretrain fine-tunes exactly this).
    pmpnn = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
                        dropout=0.0, augment_eps=0.0)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    pmpnn.load_state_dict(state)
    model = StaBddG(pmpnn=pmpnn, noise_level=args.noise_level, device=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        df_pred = skempi_eval(
            model=model, dataset=dataset, device=device,
            ensemble=args.ensemble, batch_size=args.batch_size,
            sample_size=(args.sample_size or None),
        )

    # ---- Output 1: per-mutation predictions (the primary asset — never aggregated away) ----
    pred_dir = os.path.join(OUTPUT_ROOT, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    df_pred.to_csv(os.path.join(pred_dir, "predictions.csv"), index=False)
    print(f"[skempi-eval] wrote predictions.csv: {len(df_pred)} rows / "
          f"{df_pred['#Pdb'].nunique()} complexes")

    # ---- Output 2: summary metrics (faithful replica via the paper's compute_metrics) ----
    from baselines.eval_utils import compute_metrics
    raw = compute_metrics(df_pred, bootstrap=True)
    summary = {
        "run_tag": args.run_tag,
        "checkpoint": ckpt_label,
        "n_mutations": int(len(df_pred)),
        "n_complexes": int(df_pred["#Pdb"].nunique()),
        "ensemble": args.ensemble,
        "seed": args.seed,
        **{k: _jsonable(v) for k, v in raw.items()},
    }
    sum_dir = os.path.join(OUTPUT_ROOT, "summary")
    os.makedirs(sum_dir, exist_ok=True)
    with open(os.path.join(sum_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # A single, greppable line for CloudWatch (and metric scraping if wired later).
    print(f"[skempi-eval] SKEMPI Spearman: {summary.get('Spearman')}")
    print(f"[skempi-eval] summary_metrics: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
