#!/usr/bin/env bash
set -euo pipefail

data_dir="data"
mkdir -p "$data_dir"
cd -- "$data_dir" || { echo "Failed to cd into '$data_dir'"; exit 1; }
wget https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip 
wget https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip 
unzip AlphaFold_model_PDBs.zip 
unzip Processed_K50_dG_datasets.zip