#!/bin/bash
epoch=$1
config=$2
out_path=$3
max_job=$4
batch_size=$5

###########################################################################################################################################################################################################
# ./FCGNN/train_model.sh 10 FCGNN/base_model.yaml test_base_model 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_2.yaml test_SNDShower_2 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_3.yaml test_SNDShower_3 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_4.yaml test_SNDShower_4 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_full.yaml test_SNDShower_full 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_Full_FC.yaml test_SNDShower_full_FC 8 32

# ./FCGNN/train_flavour_SNDShowerClassifier.sh 50 FCGNN/SNDShowerClassifier_Full_FC_flavour.yaml SNDShower_Full_FC_flavour 8 32

# ./FCGNN/train_flavour_SNDShowerClassifier.sh 50 FCGNN/SNDShowerClassifier_Full_FC_flavour.yaml SNDShower_Full_FC_flavour 8 32

###########################################################################################################################################################################################################

pwds=/afs/cern.ch/user/s/schuetha/work/private/GNN
mkdir -p "$pwds/$out_path"

# Activate env first
source /afs/cern.ch/user/s/schuetha/work/private/GNN/ML_env/bin/activate

# -----------------------------
# Torch JIT extension settings
# -----------------------------

# Default: persistent cache (good for interactive / repeated runs)
export TORCH_EXTENSIONS_DIR=/afs/cern.ch/work/s/schuetha/private/GNN/Forward_Centrality/torch_extensions_cache

# If running under HTCondor, prefer job-local scratch (faster, avoids AFS issues)
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ]; then
  export TORCH_EXTENSIONS_DIR="${_CONDOR_SCRATCH_DIR}/torch_extensions"
fi
mkdir -p "$TORCH_EXTENSIONS_DIR"

# CUDA arch list
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"

# Build parallelism for C++/CUDA extensions
export MAX_JOBS=$max_job

# Flush python prints immediately (useful for batch logs)
export PYTHONUNBUFFERED=1

# Python path for Exphormer / your repo
# export PYTHONPATH="$pwds/Exphormer:$pwds:$PYTHONPATH"

export PYTHONPATH="$pwds:$PYTHONPATH"
# Run
python3 -u "$pwds/train_flavour.py" -e ${epoch} -m "$pwds/${config}" -o "$pwds/$out_path"
