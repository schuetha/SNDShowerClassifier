"""
Pure PyTorch implementation of forward reachability centrality.

Replaces the CUDA kernel `forward_centrality_cuda.cu`.
Works on any device: CPU, CUDA, MPS (Apple Silicon).

Usage (unchanged from CUDA version):
    from Forward_Centrality_Transformer.forward_reachability import forward_reachability
    out = forward_reachability(edge_index, ptr, include_self=False)
"""

import torch
from torch import Tensor

def _reachability_single_graph(
    rowptr: Tensor,
    col: Tensor,
    n: int,
    offset: int,
    include_self: bool,
) -> Tensor:
    """
    Compute forward reachability for a single graph using set-based DP.

    Iterates nodes in reverse index order (which is reverse topological
    order for a DAG where all edges satisfy src < dst).

    Args:
        rowptr: CSR row pointers [n+1], LOCAL indexing (0..n-1)
        col:    CSR column indices,     LOCAL indexing (0..n-1)
        n:      number of nodes in this graph
        offset: global index offset for this graph (for returning global counts)
        include_self: whether a node counts itself in its reachable set

    Returns:
        counts: (n,) int32 tensor of reachable node counts
    """
    # Use Python sets for the DP — fast enough for graphs up to ~2000 nodes
    reachable = [set() for _ in range(n)]

    rowptr_cpu = rowptr.cpu().tolist()
    col_cpu = col.cpu().tolist()

    # Reverse topological order (highest index first)
    for i in range(n - 1, -1, -1):
        r = set()

        # Union over all forward neighbors
        start = rowptr_cpu[i]
        end = rowptr_cpu[i + 1]
        for e in range(start, end):
            j = col_cpu[e]
            r.add(j)                # direct neighbor
            r.update(reachable[j])  # neighbor's reachable set

        if include_self:
            r.add(i)

        reachable[i] = r

    counts = torch.tensor(
        [len(reachable[i]) for i in range(n)],
        dtype=torch.int32,
    )
    return counts


def _build_forward_csr(
    edge_index: Tensor,
    n: int,
    offset: int,
):
    """
    Build CSR (rowptr, col) from edge_index, keeping only forward edges
    (src < dst) and converting to local indexing (0..n-1).

    Args:
        edge_index: [2, E] global indices
        n:          number of nodes in this graph
        offset:     global index of first node in this graph

    Returns:
        rowptr: [n+1] int64
        col:    [E_fwd] int64, local indices
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Keep only forward edges (src < dst)
    fwd_mask = src < dst
    src_fwd = src[fwd_mask] - offset  # to local indexing
    dst_fwd = dst[fwd_mask] - offset

    if src_fwd.numel() == 0:
        rowptr = torch.zeros(n + 1, dtype=torch.long)
        col = torch.empty(0, dtype=torch.long)
        return rowptr, col

    # Sort by src for CSR
    perm = src_fwd.argsort()
    src_fwd = src_fwd[perm]
    dst_fwd = dst_fwd[perm]

    # Build rowptr via bincount
    deg = torch.bincount(src_fwd, minlength=n).to(torch.long)
    rowptr = torch.cat([
        torch.zeros(1, dtype=torch.long),
        deg.cumsum(0),
    ])

    return rowptr, dst_fwd


def forward_reachability(
    edge_index: Tensor,
    ptr: Tensor,
    include_self: bool = False,
) -> Tensor:
    """
    Compute per-node forward reachability, normalized per graph.

    For each node u in a DAG, counts how many other nodes are reachable
    by following forward (src → dst) edges, then normalizes by graph size.

    Args:
        edge_index: [2, E] int64 tensor of (src, dst) pairs
        ptr:        [G+1] int64 tensor of graph boundaries (PyG Batch.ptr)
        include_self: whether to count the node itself in its reachable set

    Returns:
        out_norm: [N] float32 tensor, per-node normalized reachability
    """
    device = edge_index.device
    N = ptr[-1].item()
    G = ptr.numel() - 1
    ptr_cpu = ptr.cpu().tolist()

    # Move edge_index to CPU for the set-based DP
    edge_index_cpu = edge_index.cpu()

    all_counts = torch.zeros(N, dtype=torch.int32)

    for g in range(G):
        start_g = ptr_cpu[g]
        end_g = ptr_cpu[g + 1]
        n_g = end_g - start_g

        if n_g == 0:
            continue

        # Extract edges belonging to this graph
        src = edge_index_cpu[0]
        dst = edge_index_cpu[1]
        mask = (src >= start_g) & (src < end_g) & (dst >= start_g) & (dst < end_g)
        graph_edges = edge_index_cpu[:, mask]

        # Build local CSR and compute reachability
        rowptr, col = _build_forward_csr(graph_edges, n_g, start_g)
        counts = _reachability_single_graph(rowptr, col, n_g, start_g, include_self)

        all_counts[start_g:end_g] = counts

    # Normalize per graph
    out_norm = torch.zeros(N, dtype=torch.float32)
    for g in range(G):
        start_g = ptr_cpu[g]
        end_g = ptr_cpu[g + 1]
        n_g = end_g - start_g

        if n_g == 0:
            continue

        denom = float(n_g) if include_self else float(max(n_g - 1, 1))
        out_norm[start_g:end_g] = all_counts[start_g:end_g].float() / denom

    return out_norm.to(device=device)