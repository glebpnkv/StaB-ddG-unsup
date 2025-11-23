#!/usr/bin/env bash
set -euo pipefail

data_dir="${1:-data}"  # use first arg if given, otherwise "data"
mkdir -p "$data_dir"
cd -- "$data_dir" || { echo "Failed to cd into '$data_dir'"; exit 1; }

# Megascale protein folding stability data
wget https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip
wget https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip 
unzip AlphaFold_model_PDBs.zip 
unzip Processed_K50_dG_datasets.zip

# SKEMPI binding energy data
wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz
tar -xvzf SKEMPI2_PDBs.tgz

# Cleanup
rm AlphaFold_model_PDBs.zip Processed_K50_dG_datasets.zip SKEMPI2_PDBs.tgz
rm -rf __MACOSX  # Removing redundant folder that found its way into the archives