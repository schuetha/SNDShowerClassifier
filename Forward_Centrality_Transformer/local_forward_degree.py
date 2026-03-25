import os
import torch
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(__file__)

path = "/afs/cern.ch/user/s/schuetha/work/private/GNN/Forward_Centrality_Transformer"

_local_centrality = load(
    name="local_centrality",
    sources=[f"{path}/local_forward_degree.cpp", f"{path}/local_forward_degree_cuda.cu"],
    verbose=True,
)

def local_centrality(edge_index_dir,
                     ptr,
                     normalize=True,
                     k_norm=-1 
                    ):
    
    return _local_centrality.local_forward_degree(edge_index_dir, ptr, normalize=normalize, k_norm=-1)