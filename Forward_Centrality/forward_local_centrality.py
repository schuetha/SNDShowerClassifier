"""
Pure PyTorch implementation of local forward degree centrality.

Replaces the CUDA kernel `local_centrality_cuda.cu`.
Works on any device: CPU, CUDA, MPS (Apple Silicon).

Usage (unchanged from CUDA version):
    from Forward_Centrality_Transformer.forward_local_centrality import forward_local_centrality
    out = forward_local_centrality(edge_index, ptr, normalize=True, k_norm=16)
"""

import torch
from torch import Tensor


def forward_local_centrality(
    edge_index: Tensor,
    ptr: Tensor,
    normalize: bool = True,
    k_norm: int = 0,
) -> Tensor:
    """
    Compute per-node forward out-degree: for each node u, count edges
    where u is the source and u < v (forward direction).

    Args:
        edge_index: [2, E] int64 tensor of (src, dst) pairs
        ptr:        [G+1] int64 tensor of graph boundaries (PyG Batch.ptr)
        normalize:  if True, return float32 normalized by k_norm or graph size
                    if False, return int32 raw counts
        k_norm:     if > 0 and normalize=True, divide counts by k_norm
                    otherwise divide by max(1, n_g - 1) per graph

    Returns:
        if normalize: [N] float32 tensor
        else:         [N] int32 tensor
    """
    device = edge_index.device
    N = ptr[-1].item()

    src = edge_index[0]
    dst = edge_index[1]

    # Keep only forward edges: src < dst
    fwd_mask = src < dst
    src_fwd = src[fwd_mask]

    # Count forward out-degree per node via bincount
    if src_fwd.numel() > 0:
        out_counts = torch.bincount(src_fwd, minlength=N).to(torch.int32)
    else:
        out_counts = torch.zeros(N, dtype=torch.int32, device=device)

    if not normalize:
        return out_counts

    # Normalize per graph using ptr
    G = ptr.numel() - 1
    out_norm = torch.zeros(N, dtype=torch.float32, device=device)

    for g in range(G):
        start_g = ptr[g].item()
        end_g = ptr[g + 1].item()
        n_g = end_g - start_g

        if n_g == 0:
            continue

        if k_norm > 0:
            denom = float(k_norm)
        else:
            denom = float(max(n_g - 1, 1))

        out_norm[start_g:end_g] = out_counts[start_g:end_g].float() / denom

    return out_norm