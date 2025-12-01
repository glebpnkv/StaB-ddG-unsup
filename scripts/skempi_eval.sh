#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Starting SKEMPI Evaluation Script"

# -------- Defaults (overridable via CLI) --------
# Run metadata
run_name="skempi-eval"

# GCS locations (if empty, no GCS copy is attempted)
skempi_data_gcs_uri=""
model_ckpt_gcs_uri=""
output_gcs_uri=""

# Local root inside container / VM
local_root="/app"

# Evaluation hyperparameters
batch_size=10000
ensemble=20
noise_level=0.1
seed=0
sample_size=""   # if set, limit number of SKEMPI samples for debugging

# Multi-GPU launcher behavior
#   auto   -> use torchrun if >1 GPU; otherwise plain python
#   always -> always use torchrun
#   never  -> never use torchrun
use_torchrun="auto"

# ------------------------------------------------
# Optional: activate a venv if present (safe no-op otherwise)
if [[ -d "/app/.venv" ]]; then
  # shellcheck disable=SC1091
  source /app/.venv/bin/activate || true
fi

# Helper for data transfer (wraps the python script)
transfer_script="/app/jobs/transfer_data.py"
training_script="/app/jobs/skempi_eval.py"

# -------- Argument parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name)            run_name="${2:-$run_name}"; shift 2 ;;
    --skempi_data_gcs_uri) skempi_data_gcs_uri="${2:-}"; shift 2 ;;
    --model_ckpt_gcs_uri)  model_ckpt_gcs_uri="${2:-}"; shift 2 ;;
    --output_gcs_uri)      output_gcs_uri="${2:-}"; shift 2 ;;
    --local_root)          local_root="${2:-$local_root}"; shift 2 ;;
    --batch_size)          batch_size="${2:-$batch_size}"; shift 2 ;;
    --ensemble)            ensemble="${2:-$ensemble}"; shift 2 ;;
    --noise_level)         noise_level="${2:-$noise_level}"; shift 2 ;;
    --seed)                seed="${2:-$seed}"; shift 2 ;;
    --sample_size)         sample_size="${2:-$sample_size}"; shift 2 ;;
    --use_torchrun)        use_torchrun="${2:-$use_torchrun}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[INFO] run_name=${run_name}"
echo "[INFO] skempi_data_gcs_uri=${skempi_data_gcs_uri}"
echo "[INFO] model_ckpt_gcs_uri=${model_ckpt_gcs_uri}"
echo "[INFO] output_gcs_uri=${output_gcs_uri}"
echo "[INFO] local_root=${local_root}"

# -------- Local paths derived from local_root --------
local_data_dir="${local_root}/data"
local_skempi_dir="${local_data_dir}/SKEMPI"
local_cache_dir="${local_root}/cache"
local_ckpt_dir="${local_root}/model_ckpts"
local_run_dir="${local_root}/runs/skempi_eval"

# Paths expected by jobs/skempi_eval.py
local_skempi_csv="${local_skempi_dir}/filtered_skempi.csv"
local_skempi_split="${local_skempi_dir}/test_pdb.pkl"
local_skempi_pdb_dir="${local_data_dir}/PDBs"
local_skempi_pdb_cache="${local_cache_dir}/skempi_full_mask_pdb_dict.pkl"
local_checkpoint="${local_ckpt_dir}/stabddg.pt"
local_output_dir="${local_run_dir}"

mkdir -p "${local_skempi_dir}" \
         "${local_cache_dir}" \
         "${local_ckpt_dir}" \
         "${local_output_dir}"

# -------- GCS → local copies (if URIs provided) --------
# skempi_data_gcs_uri
if [[ -n "${skempi_data_gcs_uri}" ]]; then
  echo "[INFO] Copying SKEMPI data from GCS to ${local_data_dir}"
  # Expect the GCS prefix to contain data/SKEMPI/... and cache/...
  python "${transfer_script}" download "${skempi_data_gcs_uri}" "${local_data_dir}/"
else
  echo "[INFO] skempi_data_gcs_uri not set; assuming data already present under ${local_root}"
fi

# model_ckpt_gcs_uri
if [[ -n "${model_ckpt_gcs_uri}" ]]; then
  echo "[INFO] Copying model checkpoint from GCS..."
  # Download into a temporary subdir under local_ckpt_dir to avoid collisions
  tmp_ckpt_dir="${local_ckpt_dir}/_download"
  mkdir -p "${tmp_ckpt_dir}"
  python "${transfer_script}" download "${model_ckpt_gcs_uri}" "${tmp_ckpt_dir}/" || echo "[WARN] No ckpts downloaded"

  # Find a .pt file under tmp_ckpt_dir
  downloaded_ckpt=""
  if ls "${tmp_ckpt_dir}"/*.pt >/dev/null 2>&1; then
    # Prefer a file literally named stabddg.pt if present
    if [[ -f "${tmp_ckpt_dir}/stabddg.pt" ]]; then
      downloaded_ckpt="${tmp_ckpt_dir}/stabddg.pt"
    else
      # Otherwise, take the first .pt file (or you could sort by mtime)
      downloaded_ckpt="$(ls "${tmp_ckpt_dir}"/*.pt | head -n 1)"
    fi
  fi

  if [[ -n "${downloaded_ckpt}" ]]; then
    echo "[INFO] Using downloaded checkpoint: ${downloaded_ckpt}"
    # Overwrite the canonical local_checkpoint name so the rest of the script
    # always passes a stable path to jobs/skempi_eval.py
    cp "${downloaded_ckpt}" "${local_checkpoint}"
  else
    echo "[WARN] No .pt files found under ${tmp_ckpt_dir}; falling back to local ${local_checkpoint} if it exists."
  fi
fi

if [[ -f "${local_checkpoint}" ]]; then
  echo "[INFO] Using checkpoint: ${local_checkpoint}"
else
  echo "[WARN] Checkpoint ${local_checkpoint} not found; evaluation will fail unless checkpoint is provided elsewhere."
fi

# -------- GPU detection --------
echo "[INFO] Python / Torch CUDA sanity check:"
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

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
echo "[INFO] Detected GPUs: ${NUM_GPUS}"

# Decide whether to use torchrun and what nproc_per_node to use
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

nproc="${NUM_GPUS}"

# -------- Launch evaluation --------
BASE_ARGS=(
  --run_name "${run_name}"
  --checkpoint "${local_checkpoint}"
  --skempi_path "${local_skempi_csv}"
  --skempi_pdb_dir "${local_skempi_pdb_dir}"
  --skempi_pdb_cache_path "${local_skempi_pdb_cache}"
  --skempi_split_path "${local_skempi_split}"
  --ensemble "${ensemble}"
  --output_dir "${local_output_dir}"
  --seed "${seed}"
  --noise_level "${noise_level}"
  --batch_size "${batch_size}"
)

# Only pass sample_size if explicitly set;
# otherwise argparse default (None) is used in Python.
if [[ -n "${sample_size}" ]]; then
  BASE_ARGS+=( --sample_size "${sample_size}" )
fi

echo "[INFO] Starting evaluation in ${local_root} ..."
cd "${local_root}"

if $should_use_torchrun; then
  echo "[INFO] Launching torchrun with nproc_per_node=${nproc}"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="${nproc}" "${training_script}" "${BASE_ARGS[@]}"
  else
    python -m torch.distributed.run --nproc_per_node="${nproc}" "${training_script}" "${BASE_ARGS[@]}"
  fi
else
  echo "[INFO] Launching single-process evaluation with python"
  python "${training_script}" "${BASE_ARGS[@]}"
fi

echo "[INFO] Evaluation completed."

# -------- Local → GCS copy of artifacts --------
if [[ -n "${output_gcs_uri}" ]]; then
  echo "[INFO] Copying evaluation outputs from ${local_output_dir} to ${output_gcs_uri}"
  python "${transfer_script}" upload "${local_output_dir}" "${output_gcs_uri%/}"
else
  echo "[INFO] output_gcs_uri not set; skipping upload of evaluation outputs."
fi

echo "[INFO] skempi_eval.sh finished successfully."