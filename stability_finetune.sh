#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

data_dir="data"

python stability_finetune.py \
  --num_epochs 70 \
  --batch_size 10000 \
  --model_save_freq 2 \
  --val_freq 2 \
  --pdb_dir $data_dir/AlphaFold_model_PDBs \
  --stability_data $data_dir/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
