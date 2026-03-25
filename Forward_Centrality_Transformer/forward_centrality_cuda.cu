#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) TORCH_CHECK((x), "Input condition failed: " #x)

static inline __device__ int popcount64(uint64_t x) { return __popcll(x); }

// ---------------------------------------------------------------------
// Batched reachability DP (per-graph segment using ptr)
// ---------------------------------------------------------------------
// rowptr/col are for the *batched* graph with N = total nodes in batch.
// ptr describes graph boundaries: [0, n0, n0+n1, ...] on the same indexing.
// This kernel assumes a DAG topological order by node index *within each graph*.
// ---------------------------------------------------------------------
__global__ void reachability_bitset_kernel_batched(
    const int64_t* __restrict__ rowptr,   // [N+1]
    const int64_t* __restrict__ col,      // [E]
    const int64_t* __restrict__ ptr,      // [G+1]
    int64_t G,
    int64_t N,
    int64_t words,                        // words for GLOBAL N
    bool include_self,
    uint64_t* __restrict__ bits)          // [N*words]
{
    int tid = threadIdx.x;

    // Loop per graph (order doesn't matter since graphs are disjoint)
    for (int64_t g = 0; g < G; ++g) {
        int64_t start_g = ptr[g];
        int64_t end_g   = ptr[g + 1];

        // DP inside the graph segment only
        for (int64_t i = end_g - 1; i >= start_g; --i) {

            // clear bits for node i
            for (int64_t w = tid; w < words; w += blockDim.x) {
                bits[i * words + w] = 0ULL;
            }
            __syncthreads();

            // include self
            if (include_self && tid == 0) {
                int64_t wi = i >> 6;
                int64_t bi = i & 63;
                bits[i * words + wi] |= (1ULL << bi);
            }
            __syncthreads();

            // union over outgoing neighbors of i
            int64_t e0 = rowptr[i];
            int64_t e1 = rowptr[i + 1];

            for (int64_t e = e0; e < e1; ++e) {
                int64_t j = col[e];  // dst (global index)

                // IMPORTANT: because graphs are disjoint, j should also be in [start_g, end_g)
                // If you want a safety guard (slower), uncomment:
                // if (j < start_g || j >= end_g) continue;

                // OR child's bitset into parent
                for (int64_t w = tid; w < words; w += blockDim.x) {
                    bits[i * words + w] |= bits[j * words + w];
                }
                __syncthreads();

                // ensure direct neighbor j included
                if (tid == 0) {
                    int64_t wj = j >> 6;
                    int64_t bj = j & 63;
                    bits[i * words + wj] |= (1ULL << bj);
                }
                __syncthreads();
            }
        }
    }
}

__global__ void popcount_kernel(
    const uint64_t* __restrict__ bits,
    int64_t N,
    int64_t words,
    int32_t* __restrict__ out_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const uint64_t* row = bits + (int64_t)i * words;
    int32_t cnt = 0;
    for (int64_t w = 0; w < words; ++w) {
        cnt += popcount64(row[w]);
    }
    out_counts[i] = cnt;
}

__global__ void normalize_by_ptr_kernel(
    const int32_t* __restrict__ out_counts,  // [N]
    const int64_t* __restrict__ ptr,         // [G+1]
    int64_t G,
    bool include_self,
    float* __restrict__ out_norm)            // [N]
{
    int64_t g = (int64_t)blockIdx.x;
    if (g >= G) return;

    int64_t start = ptr[g];
    int64_t end   = ptr[g + 1];
    int64_t n_g   = end - start;

    float denom = include_self ? (float)n_g : (float)((n_g > 1) ? (n_g - 1) : 1);

    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        out_norm[i] = (float)out_counts[i] / denom;
    }
}

// Main entry
torch::Tensor forward_centrality_cuda(torch::Tensor edge_index,
                                      torch::Tensor ptr,          // NEW: required for batching
                                      bool include_self)
{
    CHECK_CUDA(edge_index);
    CHECK_CONTIGUOUS(edge_index);
    CHECK_INPUT(edge_index.dim() == 2);
    CHECK_INPUT(edge_index.size(0) == 2);
    CHECK_INPUT(edge_index.scalar_type() == torch::kLong);

    CHECK_CUDA(ptr);
    CHECK_CONTIGUOUS(ptr);
    CHECK_INPUT(ptr.dim() == 1);
    CHECK_INPUT(ptr.scalar_type() == torch::kLong);
    CHECK_INPUT(ptr.numel() >= 2);

    c10::cuda::CUDAGuard device_guard(edge_index.device());

    // total nodes in the batch = ptr[-1]
    const int64_t N = ptr[-1].item<int64_t>();
    const int64_t G = ptr.numel() - 1;

    // src, dst
    auto src = edge_index.select(0, 0);
    auto dst = edge_index.select(0, 1);

    // Optional: keep only src<dst (topo by index). Keep if you rely on index order.
    auto mask = src.lt(dst);
    src = src.masked_select(mask);
    dst = dst.masked_select(mask);

    // sort by src for CSR
    auto perm = std::get<1>(src.sort(0));     // argsort
    src = src.index_select(0, perm);
    dst = dst.index_select(0, perm);

    // degree per src (bincount)
    auto deg = at::bincount(src, /*weights=*/{}, /*minlength=*/N).to(torch::kLong);

    // rowptr = [0, cumsum(deg)]
    auto zero = at::zeros({1}, deg.options());
    auto csum = deg.cumsum(0);
    auto rowptr = at::cat({zero, csum}, 0).contiguous();   // [N+1]
    auto col = dst.contiguous();                            // [E_fwd]

    // Bitset storage uses GLOBAL N (inefficient but correct)
    const int64_t words = (N + 63) / 64;
    auto bits = torch::empty({N, words},
        torch::TensorOptions().device(edge_index.device()).dtype(torch::kUInt64));

    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;

    // Build reachability sets (single block DP)
    reachability_bitset_kernel_batched<<<1, threads, 0, stream>>>(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        ptr.data_ptr<int64_t>(),
        G,
        N,
        words,
        include_self,
        (uint64_t*)bits.data_ptr<uint64_t>());

    // Popcount counts
    auto out_counts = torch::zeros({N},
        torch::TensorOptions().device(edge_index.device()).dtype(torch::kInt32));

    int blocks = (N + threads - 1) / threads;
    popcount_kernel<<<blocks, threads, 0, stream>>>(
        (uint64_t*)bits.data_ptr<uint64_t>(),
        N,
        words,
        out_counts.data_ptr<int32_t>());

    // Normalize per graph using ptr
    auto out_norm = torch::empty({N},
        torch::TensorOptions().device(edge_index.device()).dtype(torch::kFloat));

    normalize_by_ptr_kernel<<<(unsigned)G, threads, 0, stream>>>(
        out_counts.data_ptr<int32_t>(),
        ptr.data_ptr<int64_t>(),
        G,
        include_self,
        out_norm.data_ptr<float>());

    return out_norm; // [N] float, per-node reachability normalized per graph
}
