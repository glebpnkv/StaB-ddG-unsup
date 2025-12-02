#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Starting IntAct Pretraining Script"

# -------- Defaults (overridable via CLI) --------
# Run metadata
run_name="intact-pretrain"

# GCS locations (if empty, no GCS copy is attempted)
intact_data_gcs_uri=""
model_ckpt_gcs_uri=""
model_save_gcs_uri=""

# Local root inside container / VM
local_root="/app"

# Core training hyperparameters
max_length=400
batch_size=2
epochs=5
lr=1e-3
k_neutral=20
k_pos=5
k_neg=5
noise_level=0.1
normalize_loss="true"   # "true" -> --normalize_loss, "false" -> --no-normalize_loss

# Data split settings
intact_sample_size="" # 1.0 -> use all data
valid_size=0.1
test_size=0.1
random_state=42

# Dataloader / validation cadence
num_dataloader_workers=1
model_val_freq=5

# Model saving / logging (local, under $local_root)
local_model_subdir="intact_pretrain"
use_wandb="false"

# Optional: initialize from existing checkpoint (local path under $local_root)
model_existing_checkpoint=""

# Multi-GPU launcher behavior
#   auto   -> use torchrun if >1 GPU; otherwise plain python
#   always -> always use torchrun
#   never  -> never use torchrun
use_torchrun="auto"

# Optional: paths for KFP artifact outputs (within the container FS)
vertex_model_dir_path=""
vertex_metrics_dir_path=""

# ------------------------------------------------
# Optional: activate a venv if present (safe no-op otherwise)
if [[ -d "/app/.venv" ]]; then
  # shellcheck disable=SC1091
  source /app/.venv/bin/activate || true
fi

# Helper for data transfer (wraps the python script)
transfer_script="/app/jobs/transfer_data.py"
training_script="/app/jobs/intact_pretrain.py"

# -------- Argument parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name)                 run_name="${2:-$run_name}"; shift 2 ;;
    --intact_data_gcs_uri)      intact_data_gcs_uri="${2:-}"; shift 2 ;;
    --model_ckpt_gcs_uri)       model_ckpt_gcs_uri="${2:-}"; shift 2 ;;
    --model_save_gcs_uri)       model_save_gcs_uri="${2:-}"; shift 2 ;;
    --local_root)               local_root="${2:-$local_root}"; shift 2 ;;
    --max_length)               max_length="${2:-$max_length}"; shift 2 ;;
    --batch_size)               batch_size="${2:-$batch_size}"; shift 2 ;;
    --epochs)                   epochs="${2:-$epochs}"; shift 2 ;;
    --lr)                       lr="${2:-$lr}"; shift 2 ;;
    --k_neutral)                k_neutral="${2:-$k_neutral}"; shift 2 ;;
    --k_pos)                    k_pos="${2:-$k_pos}"; shift 2 ;;
    --k_neg)                    k_neg="${2:-$k_neg}"; shift 2 ;;
    --noise_level)              noise_level="${2:-$noise_level}"; shift 2 ;;
    --normalize_loss)           normalize_loss="${2:-$normalize_loss}"; shift 2 ;;
    --intact_sample_size)       intact_sample_size="${2:-$intact_sample_size}"; shift 2 ;;
    --valid_size)               valid_size="${2:-$valid_size}"; shift 2 ;;
    --test_size)                test_size="${2:-$test_size}"; shift 2 ;;
    --random_state)             random_state="${2:-$random_state}"; shift 2 ;;
    --num_dataloader_workers)   num_dataloader_workers="${2:-$num_dataloader_workers}"; shift 2 ;;
    --model_val_freq)           model_val_freq="${2:-$model_val_freq}"; shift 2 ;;
    --use_wandb)                use_wandb="${2:-$use_wandb}"; shift 2 ;;
    --use_antithetic_variates)  use_antithetic_variates="${2:-true}"; shift 2 ;; # kept for backward compat
    --model_existing_checkpoint) model_existing_checkpoint="${2:-}"; shift 2 ;;
    --use_torchrun)             use_torchrun="${2:-$use_torchrun}"; shift 2 ;;
    --vertex_model_dir_path)    vertex_model_dir_path="${2:-}"; shift 2 ;;
    --vertex_metrics_dir_path)  vertex_metrics_dir_path="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[INFO] run_name=${run_name}"
echo "[INFO] intact_data_gcs_uri=${intact_data_gcs_uri}"
echo "[INFO] model_ckpt_gcs_uri=${model_ckpt_gcs_uri}"
echo "[INFO] model_save_gcs_uri=${model_save_gcs_uri}"
echo "[INFO] local_root=${local_root}"

# -------- Local paths derived from local_root --------
local_data_dir="${local_root}/data/intact"
local_proteins_dir="${local_root}/data/intact/proteins/safetensors"
local_assemblies_dir="${local_root}/data/intact/assemblies/safetensors"
local_ckpt_dir="${local_root}/model_ckpts"
local_model_save_dir="${local_root}/runs/${local_model_subdir}"

mkdir -p "${local_data_dir}" \
         "${local_proteins_dir}" \
         "${local_assemblies_dir}" \
         "${local_ckpt_dir}" \
         "${local_model_save_dir}"

# -------- GCS → local copies (if URIs provided) --------
if [[ -n "${intact_data_gcs_uri}" ]]; then
  echo "[INFO] Copying IntAct parquet + metadata from GCS..."
  python "${transfer_script}" download "${intact_data_gcs_uri}" "${local_data_dir}/"
else
  echo "[INFO] intact_data_gcs_uri not set; assuming data already present at ${local_data_dir}"
fi

if [[ -n "${model_ckpt_gcs_uri}" ]]; then
  echo "[INFO] Copying existing model checkpoints from GCS..."
  python "${transfer_script}" download "${model_ckpt_gcs_uri}" "${local_ckpt_dir}/" || echo "[WARN] No ckpts to copy"
fi

# Determine existing checkpoint if not explicitly provided
if [[ -z "${model_existing_checkpoint}" ]]; then
  if [[ -f "${local_ckpt_dir}/proteinmpnn.pt" ]]; then
    model_existing_checkpoint="${local_ckpt_dir}/proteinmpnn.pt"
  fi
fi

if [[ -n "${model_existing_checkpoint}" ]]; then
  echo "[INFO] Using existing checkpoint: ${model_existing_checkpoint}"
else
  echo "[INFO] No existing checkpoint; training from scratch."
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

# -------- Launch training --------
BASE_ARGS=(
  --run_name "${run_name}"
  --data_dir "${local_data_dir}"
  --proteins_dir "${local_proteins_dir}"
  --assemblies_dir "${local_assemblies_dir}"
  --model_save_dir "${local_model_save_dir}"
  --max_length "${max_length}"
  --batch_size "${batch_size}"
  --epochs "${epochs}"
  --lr "${lr}"
  --k_neutral "${k_neutral}"
  --k_pos "${k_pos}"
  --k_neg "${k_neg}"
  --noise_level "${noise_level}"
  --valid_size "${valid_size}"
  --test_size "${test_size}"
  --random_state "${random_state}"
  --num_dataloader_workers "${num_dataloader_workers}"
  --model_val_freq "${model_val_freq}"
)

# Only pass intact_sample_size if explicitly set;
# otherwise argparse default (None) is used in Python.
if [[ -n "${intact_sample_size}" ]]; then
  BASE_ARGS+=( --intact_sample_size "${intact_sample_size}" )
fi

if [[ "${normalize_loss}" == "true" ]]; then
  BASE_ARGS+=( --normalize_loss )
else
  BASE_ARGS+=( --no-normalize_loss )
fi

if [[ "${use_wandb}" == "true" ]]; then
  BASE_ARGS+=( --wandb )
fi

# Enable antithetic variates (default True in code)
BASE_ARGS+=( --use_antithetic_variates )

if [[ -n "${model_existing_checkpoint}" ]]; then
  BASE_ARGS+=( --model_existing_checkpoint "${model_existing_checkpoint}" )
fi

echo "[INFO] Starting training in ${local_root} ..."
cd "${local_root}"

if $should_use_torchrun; then
  echo "[INFO] Launching torchrun with nproc_per_node=${nproc}"
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="${nproc}" "${training_script}" "${BASE_ARGS[@]}"
  else
    python -m torch.distributed.run --nproc_per_node="${nproc}" "${training_script}" "${BASE_ARGS[@]}"
  fi
else
  echo "[INFO] Launching single-process training with python"
  python "${training_script}" "${BASE_ARGS[@]}"
fi

echo "[INFO] Training completed."

# -------- Local → GCS copy of artifacts --------
if [[ -n "${model_save_gcs_uri}" ]]; then
  echo "[INFO] Copying artifacts from ${local_model_save_dir} to ${model_save_gcs_uri}"
  python "${transfer_script}" upload "${local_model_save_dir}" "${model_save_gcs_uri%/}"
else
  echo "[INFO] model_save_gcs_uri not set; skipping upload of artifacts."
fi

echo "[INFO] intact_pretrain.sh finished successfully."