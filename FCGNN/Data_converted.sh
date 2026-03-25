#!/bin/bash

# Activate env first
source /afs/cern.ch/user/s/schuetha/work/private/GNN/ML_env/bin/activate

echo ">>> ${SNDSW_ROOT} <<<"

if [ -z ${SNDSW_ROOT+x} ]
then
    echo "Setting up SNDSW"
    export ALIBUILD_WORK_DIR=/afs/cern.ch/user/s/schuetha/work/public/SND_8Aug24/sw
    source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Jan30/setUp.sh  # recommended latest version
    eval `alienv -a slc9_x86-64 load --no-refresh sndsw/master-local1`
    export EOSSHIP=root://eosuser.cern.ch/
fi

python3 /afs/cern.ch/user/s/schuetha/work/private/GNN/FCGNN/GNN_s_b.py -m train
