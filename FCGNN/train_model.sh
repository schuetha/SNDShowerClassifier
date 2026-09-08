#!/bin/bash
epoch=$1
config=$2
out_path=$3
max_job=$4
batch_size=$5

# ./FCGNN/train_model.sh 10 FCGNN/Full_model_GraphGPS.yaml test_GPS_2 4 32
# ./FCGNN/train_model.sh 10 FCGNN/DirGravNet_model.yaml test_Dir_2_GravNet 8 128
# ./FCGNN/train_model.sh 10 FCGNN/DirGravNet_model.yaml test_Dir_2_GravNet_no_E_in_mknn 8 32
# ./FCGNN/train_model.sh 10 FCGNN/Full_model_GraphGPS.yaml test_GPS_1_no_e_in_mknn 4 32
# ./FCGNN/train_model.sh 10 FCGNN/Full_model_GraphGPS.yaml test_GPS_1_no_E_in_mknn_new_z 8 32

# ./FCGNN/train_model.sh 10 FCGNN/Dir_local_GravNet.yaml test_Dir_GravNet_local_1 8 32

# ./FCGNN/train_model.sh 10 FCGNN/Dir_local_GravNet.yaml test_Dir_GravNet_local_1_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/GraphGPS_local.yaml test_GPS_local_1 8 32

################################################################################################################
# ./FCGNN/train_model.sh 10 FCGNN/base_model.yaml test_base_model 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_2.yaml test_SNDShower_2 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_3.yaml test_SNDShower_3 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_4.yaml test_SNDShower_4 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_full.yaml test_SNDShower_full 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_Full_FC.yaml test_SNDShower_full_FC 8 32

################################################################################################################

# ./FCGNN/train_model.sh 10 FCGNN/edge_conv.yaml test_edge 8 32


################################################################################################################
# ./FCGNN/train_model.sh 10 FCGNN/base_model.yaml test_base_model_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_2.yaml test_SNDShower_2_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_3.yaml test_SNDShower_3_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_4.yaml test_SNDShower_4_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_full.yaml test_SNDShower_full 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_Full_FC.yaml test_SNDShower_full_FC_new 8 32

# ./FCGNN/train_model.sh 10 FCGNN/SNDShowerClassifier_old.yaml SNDShowerClassifier_Full_FC_RD_300cut_old 8 32

################################################################################################################

pwds=/afs/cern.ch/user/s/schuetha/work/private/GNN
mkdir -p "$pwds/$out_path"

# Activate env first
source /afs/cern.ch/user/s/schuetha/work/private/GNN/ML_env/bin/activate

# -----------------------------
# Torch JIT extension settings
# -----------------------------

# Python path for Exphormer / your repo
# export PYTHONPATH="$pwds/Exphormer:$pwds:$PYTHONPATH"

export PYTHONPATH="$pwds:$PYTHONPATH"
# Run
python3 -u "$pwds/train_model_new.py" -e ${epoch} -b ${batch_size} -m "$pwds/${config}" -o "$pwds/$out_path"
