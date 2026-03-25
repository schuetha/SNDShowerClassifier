from torch.utils.cpp_extension import load

path = "/afs/cern.ch/user/s/schuetha/work/private/GNN/Forward_Centrality_Transformer"

# IMPORTANT:
# - sources must include the binding .cpp and the CUDA .cu
# - extra_cuda_cflags is often needed on lxplus to set a proper arch (optional)
_monoknn = load(
    name="monotonic_knn",
    sources=[f"{path}/monotonic_knn.cpp", f"{path}/monotonic_knn_cuda.cu"],
    verbose=True,
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

def monoknn(
    x, y,
    pos_x, pos_y,
    ptr_x=None, ptr_y=None,
    k: int = 16,
    cosine: bool = False,
    use_z_mono: bool = True,
    z_index: int = 2,
    use_time: bool = False,
    time_index: int = 3,
):
    """
    Python wrapper for CUDA monotonic KNN.

    Args:
      x, y:       [N,F] and [M,F] (CUDA tensors)
      pos_x,pos_y:[N,P] and [M,P] (CUDA tensors, same dtype as x/y)
      ptr_x,ptr_y:optional Long CUDA tensors (CSR ptr) of shape [B+1]
      k:          neighbors (<=100)
      cosine:     cosine distance if True else squared L2
      use_z_mono: enforce y_z > x_z using pos_*[:, z_index]
      use_energy: enforce y_e < x_e using pos_*[:, e_index]

    Returns:
      edge_index: Long tensor [2, E]
    """
    return _monoknn.mknn_cuda(
        x, y,
        pos_x, pos_y,
        ptr_x=ptr_x, ptr_y=ptr_y,
        k=k,
        cosine=cosine,
        use_z_mono=use_z_mono,
        z_index=z_index,
        use_time=use_time,
        time_index=time_index,
    )
