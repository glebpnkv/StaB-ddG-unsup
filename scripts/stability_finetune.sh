#!/usr/bin/env bash
set -euo pipefail

# -------- Configuration (edit here) --------
# Training args
num_epochs=70
batch_size=15000
model_save_freq=2
val_freq=2
num_dataloader_workers=4

# Data paths
data_dir="data"
pdb_dir="$data_dir/AlphaFold_model_PDBs"
stability_data="$data_dir/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"

# Multi-GPU launcher behavior
#   auto   -> use torchrun if >1 GPU; otherwise plain python
#   always -> always use torchrun
#   never  -> never use torchrun
use_torchrun="auto"
# Optionally force number of processes (defaults to GPU count when using torchrun)
nproc=""
# -------------------------------------------

source .venv/bin/activate

# Allow overriding the two launcher toggles from CLI if desired:
#   ./stability_finetune.sh --use-torchrun always --nproc 4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-torchrun) use_torchrun="${2:-$use_torchrun}"; shift 2 ;;
    --nproc)        nproc="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

gpu_count() {
  local count=0

  # Respect CUDA_VISIBLE_DEVICES if set
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra DEV_ARR <<< "${CUDA_VISIBLE_DEVICES}"
    local filtered=()
    for d in "${DEV_ARR[@]}"; do
      [[ -n "$d" ]] && filtered+=("$d")
    done
    if [[ ${#filtered[@]} -gt 0 ]]; then
      echo "${#filtered[@]}"
      return 0
    fi
  fi

  # Try nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    count=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
    if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
      echo "$count"
      return 0
    fi
  fi

  # Fallback: Python (Torch)
  if command -v python >/dev/null 2>&1; then
    count=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
    if [[ "$count" =~ ^[0-9]+$ ]]; then
      echo "$count"
      return 0
    fi
  fi

  echo 0
}

NUM_GPUS="$(gpu_count)"

should_use_torchrun=false
case "$use_torchrun" in
  always) should_use_torchrun=true ;;
  never)  should_use_torchrun=false ;;
  auto)
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
      should_use_torchrun=true
    fi
    ;;
  *)
    echo "Invalid use_torchrun value: $use_torchrun (expected auto|always|never)" >&2
    exit 1
    ;;
esac

# Common args
BASE_ARGS=(
  --num_epochs "$num_epochs"
  --batch_size "$batch_size"
  --model_save_freq "$model_save_freq"
  --val_freq "$val_freq"
  --pdb_dir "$pdb_dir"
  --stability_data "$stability_data"
  --num_dataloader_workers "$num_dataloader_workers"
)

if $should_use_torchrun; then
  NPROC_PER_NODE="${nproc:-$NUM_GPUS}"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NPROC_PER_NODE" stability_finetune.py --distributed "${BASE_ARGS[@]}"
  else
    python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" stability_finetune.py --distributed "${BASE_ARGS[@]}"
  fi
else
  python stability_finetune.py "${BASE_ARGS[@]}"
fi
