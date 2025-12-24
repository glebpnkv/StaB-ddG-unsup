#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Starting IntAct Pretraining Script"

# -------- Defaults (overridable via CLI) --------
# Run metadata
run_name="intact-pretrain"

# GCS locations (if empty, no GCS copy is attempted)
intact_data_gcs_uri=""
model_ckpt_gcs_uri=""

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
lambda_sign=10.0
lambda_neutral=0.0

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
vertex_data_splits_dir_path=""

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
    --local_root)               local_root="${2:-$local_root}"; shift 2 ;;
    --max_length)               max_length="${2:-$max_length}"; shift 2 ;;
    --batch_size)               batch_size="${2:-$batch_size}"; shift 2 ;;
    --epochs)                   epochs="${2:-$epochs}"; shift 2 ;;
    --lr)                       lr="${2:-$lr}"; shift 2 ;;
    --k_neutral)                k_neutral="${2:-$k_neutral}"; shift 2 ;;
    --k_pos)                    k_pos="${2:-$k_pos}"; shift 2 ;;
    --k_neg)                    k_neg="${2:-$k_neg}"; shift 2 ;;
    --noise_level)              noise_level="${2:-$noise_level}"; shift 2 ;;
    --lambda_sign)              lambda_sign="${2:-$lambda_sign}"; shift 2 ;;
    --lambda_neutral)           lambda_neutral="${2:-$lambda_neutral}"; shift 2 ;;
    --intact_sample_size)       intact_sample_size="${2:-$intact_sample_size}"; shift 2 ;;
    --valid_size)               valid_size="${2:-$valid_size}"; shift 2 ;;
    --test_size)                test_size="${2:-$test_size}"; shift 2 ;;
    --random_state)             random_state="${2:-$random_state}"; shift 2 ;;
    --num_dataloader_workers)   num_dataloader_workers="${2:-$num_dataloader_workers}"; shift 2 ;;
    --model_val_freq)           model_val_freq="${2:-$model_val_freq}"; shift 2 ;;
    --use_wandb)                use_wandb="${2:-$use_wandb}"; shift 2 ;;
    --use_antithetic_variates)  use_antithetic_variates="${2:-true}"; shift 2 ;; # kept for backward compat
    --use_torchrun)             use_torchrun="${2:-$use_torchrun}"; shift 2 ;;
    --vertex_model_dir_path)    vertex_model_dir_path="${2:-}"; shift 2 ;;
    --vertex_metrics_dir_path)  vertex_metrics_dir_path="${2:-}"; shift 2 ;;
    --vertex_data_splits_dir_path) vertex_data_splits_dir_path="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[INFO] run_name=${run_name}"
echo "[INFO] intact_data_gcs_uri=${intact_data_gcs_uri}"
echo "[INFO] model_ckpt_gcs_uri=${model_ckpt_gcs_uri}"
echo "[INFO] local_root=${local_root}"
# -------- Vertex env debug --------
echo "[INFO] AIP_TENSORBOARD_LOG_DIR=${AIP_TENSORBOARD_LOG_DIR:-<unset>}"
echo "[INFO] AIP_MODEL_DIR=${AIP_MODEL_DIR:-<unset>}"
echo "[INFO] AIP_CHECKPOINT_DIR=${AIP_CHECKPOINT_DIR:-<unset>}"

# -------- TensorBoard shim: local write + periodic sync to GCS --------
TB_GCS_DIR="${AIP_TENSORBOARD_LOG_DIR:-}"

if [[ -n "${TB_GCS_DIR}" && "${TB_GCS_DIR}" == gs://* ]]; then
  TB_GCS_OUTPUT_DIR="${TB_GCS_DIR}${run_name}"
  TB_LOCAL_DIR="/tmp/tensorboard/${run_name}"
  mkdir -p "${TB_LOCAL_DIR}"

  # IMPORTANT: PyTorch SummaryWriter needs a local path. We overwrite the env var for the training process.
  export AIP_TENSORBOARD_LOG_DIR="${TB_LOCAL_DIR}"

  TB_SYNC_INTERVAL_SEC="${TB_SYNC_INTERVAL_SEC:-60}"
  echo "[INFO] TensorBoard: writing locally to ${TB_LOCAL_DIR}"
  echo "[INFO] TensorBoard: syncing to ${TB_GCS_OUTPUT_DIR} every ${TB_SYNC_INTERVAL_SEC}s"

  tb_sync_once() {
    # Prefer gsutil if present (fast + incremental).
    if command -v gsutil >/dev/null 2>&1; then
      gsutil -m rsync -r "${TB_LOCAL_DIR}" "${TB_GCS_OUTPUT_DIR}"
      return $?
    fi

    echo "[WARN] gsutil not found; cannot sync TB logs to GCS. Add gsutil/gcloud to the image."
    return 0
  }

  # First sync (so you see errors immediately if auth/path is wrong)
  tb_sync_once || echo "[WARN] Initial TensorBoard sync failed (will keep trying)."

  tb_sync_loop() {
    while true; do
      tb_sync_once >/dev/null 2>&1 || true
      sleep "${TB_SYNC_INTERVAL_SEC}"
    done
  }

  tb_sync_loop & TB_SYNC_PID=$!

  # Ensure we do a final sync and kill the background loop
  trap 'set +e; echo "[INFO] Final TensorBoard sync..."; tb_sync_once; [[ -n "${TB_SYNC_PID:-}" ]] && kill "${TB_SYNC_PID}" 2>/dev/null || true' EXIT
else
  echo "[INFO] AIP_TENSORBOARD_LOG_DIR is not a gs:// URI (or unset). TensorBoard streaming likely disabled."
fi

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
  echo "[INFO] Copying existing model checkpoint from GCS..."
  # Download into a temporary subdir under local_ckpt_dir to avoid collisions
  tmp_ckpt_dir="${local_ckpt_dir}/_download"
  mkdir -p "${tmp_ckpt_dir}"
  python "${transfer_script}" download "${model_ckpt_gcs_uri}" "${tmp_ckpt_dir}/" || echo "[WARN] No ckpts downloaded"

  # Find a .pt file under tmp_ckpt_dir
  if ls "${tmp_ckpt_dir}"/*.pt >/dev/null 2>&1; then
    # Take the first .pt file
    model_existing_checkpoint="$(ls "${tmp_ckpt_dir}"/*.pt | head -n 1)"
    echo "[INFO] Using downloaded checkpoint: ${model_existing_checkpoint}"
  else
    echo "[WARN] No .pt files found under ${tmp_ckpt_dir}"
  fi
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
  --lambda_sign "${lambda_sign}"
  --lambda_neutral "${lambda_neutral}"
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

# -------- Launch training (capture status, don't exit early) --------
set +e
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
train_status=$?
set -e

if [[ $train_status -eq 0 ]]; then
  echo "[INFO] Training completed successfully (status=${train_status})."
else
  echo "[ERROR] Training failed with status ${train_status}."
fi

# -------- Local → KFP artifact outputs --------
copy_dir_to_output() {
  local src="$1"
  local dest="$2"
  local label="$3"

  if [[ -z "$dest" ]]; then
    echo "[INFO] ${label} output path not set; skipping."
    return 0
  fi

  if [[ ! -d "$src" ]]; then
    echo "[WARN] ${label} source dir not found at ${src}; skipping copy."
    return 0
  fi

  mkdir -p "$dest"
  cp -R "${src}/." "$dest/"
  echo "[INFO] ${label} copied to ${dest}"
}

copy_dir_to_output "${local_model_save_dir}/model" "${vertex_model_dir_path}" "Model artifacts"
copy_dir_to_output "${local_model_save_dir}/metrics" "${vertex_metrics_dir_path}" "Metrics"
copy_dir_to_output "${local_model_save_dir}/data_splits" "${vertex_data_splits_dir_path}" "Data splits"

if [[ $train_status -eq 0 ]]; then
  echo "[INFO] intact_pretrain.sh finished successfully."
else
  echo "[ERROR] intact_pretrain.sh exiting with non-zero status ${train_status}."
fi

exit "$train_status"
