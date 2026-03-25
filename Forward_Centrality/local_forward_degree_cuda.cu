// local_centrality_cuda.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) TORCH_CHECK((x), "Input condition failed: " #x)

#define THREADS 256

// ------------------------------------------------------------
// Count forward out-degree: counts[u] += 1 if (u < v)
// edge_index is [2, E] (src, dst) in contiguous layout
// ------------------------------------------------------------
__global__ void local_forward_degree_kernel(
    const int64_t* __restrict__ src,
    const int64_t* __restrict__ dst,
    int64_t E,
    int64_t N,
    int32_t* __restrict__ out_counts)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int64_t u = src[e];
    int64_t v = dst[e];

    // forward-only condition
    if (u < v) {
        if (0 <= u && u < N) {
            atomicAdd(&out_counts[u], 1);
        }
    }
}

// ------------------------------------------------------------
// Normalize per graph using ptr:
// for nodes i in graph g: out_norm[i] = counts[i] / denom_g
// denom_g = k_norm if k_norm>0 else max(1, n_g-1)
// ------------------------------------------------------------
__global__ void normalize_local_by_ptr_kernel(
    const int32_t* __restrict__ counts, // [N]
    const int64_t* __restrict__ ptr,    // [G+1]
    int64_t G,
    int64_t k_norm,
    float* __restrict__ out_norm)       // [N]
{
    int64_t g = (int64_t)blockIdx.x;
    if (g >= G) return;

    int64_t start = ptr[g];
    int64_t end   = ptr[g + 1];
    int64_t n_g   = end - start;

    float denom = 1.0f;
    if (k_norm > 0) {
        denom = (float)k_norm;
    } else {
        denom = (float)((n_g > 1) ? (n_g - 1) : 1);
    }

    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        out_norm[i] = (float)counts[i] / denom;
    }
}

// ------------------------------------------------------------
// Public entry point
//
// edge_index: [2, E] long cuda
// ptr: [G+1] long cuda (PyG Batch.ptr)
// normalize: if false -> return int32 counts [N]
//            if true  -> return float32 normalized [N]
// k_norm: if normalize and k_norm>0 -> divide by k_norm
//         else divide by max(1, n_g-1) per graph
// ------------------------------------------------------------
torch::Tensor local_forward_degree_cuda(
    torch::Tensor edge_index,
    torch::Tensor ptr,
    bool normalize,
    int64_t k_norm)
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

    // Total nodes in this batch = ptr[-1]
    const int64_t N = ptr.index({ptr.numel() - 1}).item<int64_t>();
    const int64_t E = edge_index.size(1);
    const int64_t G = ptr.numel() - 1;

    // edge_index contiguous: first row then second row
    const int64_t* p = edge_index.data_ptr<int64_t>();
    const int64_t* src = p;        // [E]
    const int64_t* dst = p + E;    // [E]

    auto out_counts = torch::zeros(
        {N},
        torch::TensorOptions().device(edge_index.device()).dtype(torch::kInt32));

    auto stream = at::cuda::getCurrentCUDAStream();

    // count
    {
        int blocks = (int)((E + THREADS - 1) / THREADS);
        local_forward_degree_kernel<<<blocks, THREADS, 0, stream>>>(
            src, dst, E, N, out_counts.data_ptr<int32_t>());
    }

    if (!normalize) {
        return out_counts; // [N] int32
    }

    // normalize per-graph using ptr
    auto out_norm = torch::empty(
        {N},
        torch::TensorOptions().device(edge_index.device()).dtype(torch::kFloat));

    normalize_local_by_ptr_kernel<<<(unsigned)G, THREADS, 0, stream>>>(
        out_counts.data_ptr<int32_t>(),
        ptr.data_ptr<int64_t>(),
        G,
        k_norm,
        out_norm.data_ptr<float>());

    return out_norm; // [N] float32
}
