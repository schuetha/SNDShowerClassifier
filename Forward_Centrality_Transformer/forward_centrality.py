import os
import torch
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(__file__)

path = "/afs/cern.ch/user/s/schuetha/work/private/GNN/Forward_Centrality_Transformer"

_forward_centrality = load(
    name="forward_centrality_ext",
    sources=[
        os.path.join(_this_dir, f"{path}/forward_centrality.cpp"),
        os.path.join(_this_dir, f"{path}/forward_centrality_cuda.cu"),
    ],
    verbose=False,
)

def forward_centrality(edge_index,
                       ptr,
                       include_self=False):

    return _forward_centrality.forward_centrality(edge_index, ptr, include_self=include_self)