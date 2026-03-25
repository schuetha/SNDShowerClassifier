#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#define THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(cond) TORCH_CHECK((cond), "Input condition failed: " #cond)

__device__ __forceinline__ int64_t get_example_idx(
    int64_t n_y, const int64_t* ptr, int64_t num_examples)
{
    int64_t left = 0;
    int64_t right = num_examples;
    while (left < right) {
        int64_t mid = (left + right) >> 1;
        if (ptr[mid + 1] <= n_y) left = mid + 1;
        else right = mid;
    }
    return left;
}

template <typename scalar_t>
struct Cosine {
    static __device__ scalar_t dot(const scalar_t* a, const scalar_t* b,
                                   int64_t ia, int64_t ib, int64_t dim) {
        scalar_t res = 0;
        for (int64_t i = 0; i < dim; i++)
            res += a[ia * dim + i] * b[ib * dim + i];
        return res;
    }
    static __device__ scalar_t norm(const scalar_t* a, int64_t ia, int64_t dim) {
        scalar_t res = 0;
        for (int64_t i = 0; i < dim; i++) {
            scalar_t v = a[ia * dim + i];
            res += v * v;
        }
        return sqrt(res);
    }
};

template <typename scalar_t>
__global__ void knn_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ pos_x,
    const scalar_t* __restrict__ pos_y,
    const int64_t* __restrict__ ptr_x,
    const int64_t* __restrict__ ptr_y,
    int64_t* __restrict__ row,
    int64_t* __restrict__ col,
    int64_t k,
    int64_t /*n*/,
    int64_t m,
    int64_t dim_x,
    int64_t dim_pos,
    int64_t num_examples,
    bool cosine,
    bool use_z_mono, int64_t z_index_pos,
    bool use_time, int64_t time_index_pos
) {
    int64_t ny = blockIdx.x * blockDim.x + threadIdx.x;
    if (ny >= m) return;

    int64_t ex = get_example_idx(ny, ptr_y, num_examples);

    scalar_t y_z = scalar_t(0);
    scalar_t y_t = scalar_t(0);
    if (use_z_mono) y_z = pos_y[ny * dim_pos + z_index_pos];
    if (use_time)   y_t = pos_y[ny * dim_pos + time_index_pos];

    // k is checked <= 100 on host
    scalar_t best_dist[100];
    int64_t best_idx[100];
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        best_dist[i] = scalar_t(1e20);
        best_idx[i]  = -1;
    }

    for (int64_t nx = ptr_x[ex]; nx < ptr_x[ex + 1]; nx++) {

        if (use_z_mono) {
            scalar_t x_z = pos_x[nx * dim_pos + z_index_pos];
            if (y_z <= x_z) continue;
        }
        if (use_time) {
            scalar_t x_t = pos_x[nx * dim_pos + time_index_pos];
            if (y_t <= x_t) continue;
        }

        scalar_t dist = 0;
        if (cosine) {
            scalar_t denom = Cosine<scalar_t>::norm(x, nx, dim_x) *
                             Cosine<scalar_t>::norm(y, ny, dim_x);
            if (denom <= scalar_t(0)) continue;
            dist = Cosine<scalar_t>::dot(x, y, nx, ny, dim_x) / denom;
            dist = 1 - dist;
        } else {
            for (int d = 0; d < dim_x; d++) {
                scalar_t diff = x[nx * dim_x + d] - y[ny * dim_x + d];
                dist += diff * diff;
            }
        }

        // insert into top-k (best_dist sorted ascending)
        for (int i = 0; i < k; i++) {
            if (dist < best_dist[i]) {
                for (int j = (int)k - 1; j > i; j--) {
                    best_dist[j] = best_dist[j - 1];
                    best_idx[j]  = best_idx[j - 1];
                }
                best_dist[i] = dist;
                best_idx[i]  = nx;
                break;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        int64_t src = best_idx[i];
        row[ny * k + i] = src;
        col[ny * k + i] = (src >= 0) ? ny : -1;
    }
}

torch::Tensor mknn_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor pos_x,
    torch::Tensor pos_y,
    c10::optional<torch::Tensor> ptr_x,
    c10::optional<torch::Tensor> ptr_y,
    int64_t k,
    bool cosine,
    bool use_z_mono, int64_t z_index_pos,
    bool use_time, int64_t time_index_pos
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x);
    CHECK_CUDA(y); CHECK_CONTIGUOUS(y);
    CHECK_INPUT(x.dim() == 2);
    CHECK_INPUT(y.dim() == 2);
    CHECK_INPUT(x.size(1) == y.size(1));

    CHECK_CUDA(pos_x); CHECK_CONTIGUOUS(pos_x);
    CHECK_CUDA(pos_y); CHECK_CONTIGUOUS(pos_y);
    CHECK_INPUT(pos_x.dim() == 2);
    CHECK_INPUT(pos_y.dim() == 2);
    CHECK_INPUT(pos_x.size(0) == x.size(0));
    CHECK_INPUT(pos_y.size(0) == y.size(0));
    CHECK_INPUT(pos_x.scalar_type() == x.scalar_type());
    CHECK_INPUT(pos_y.scalar_type() == y.scalar_type());

    CHECK_INPUT(k > 0 && k <= 100);

    if (use_z_mono)
        CHECK_INPUT(z_index_pos >= 0 && z_index_pos < pos_x.size(1));
    if (use_time)
        CHECK_INPUT(time_index_pos >= 0 && time_index_pos < pos_x.size(1));

    // Default ptr = single example if not provided
    if (!ptr_x.has_value()) {
        auto t_cpu = torch::tensor(
            {int64_t(0), int64_t(x.size(0))},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)
        );
        ptr_x = t_cpu.to(x.device(), /*non_blocking=*/false).contiguous();
    }
    if (!ptr_y.has_value()) {
        auto t_cpu = torch::tensor(
            {int64_t(0), int64_t(y.size(0))},
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU)
        );
        ptr_y = t_cpu.to(y.device(), /*non_blocking=*/false).contiguous();
    }

    // ptr checks (fixed: use scalar_type, not dtype())
    CHECK_CUDA(ptr_x.value());
    CHECK_CUDA(ptr_y.value());
    CHECK_CONTIGUOUS(ptr_x.value());
    CHECK_CONTIGUOUS(ptr_y.value());
    CHECK_INPUT(ptr_x.value().scalar_type() == at::kLong);
    CHECK_INPUT(ptr_y.value().scalar_type() == at::kLong);

    // (Optional but recommended) ptr length sanity
    CHECK_INPUT(ptr_x.value().numel() >= 2);
    CHECK_INPUT(ptr_y.value().numel() >= 2);
    CHECK_INPUT(ptr_x.value()[0].item<int64_t>() == 0);
    CHECK_INPUT(ptr_y.value()[0].item<int64_t>() == 0);

    auto row = torch::empty({y.size(0) * k}, y.options().dtype(torch::kLong));
    auto col = torch::full({y.size(0) * k}, -1, y.options().dtype(torch::kLong));

    int blocks = (int)((y.size(0) + THREADS - 1) / THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "monotonic_knn", [&] {
        knn_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            pos_x.data_ptr<scalar_t>(),
            pos_y.data_ptr<scalar_t>(),
            ptr_x.value().data_ptr<int64_t>(),
            ptr_y.value().data_ptr<int64_t>(),
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            k,
            x.size(0),
            y.size(0),
            x.size(1),
            pos_x.size(1),
            ptr_x.value().numel() - 1,
            cosine,
            use_z_mono, z_index_pos,
            use_time, time_index_pos
        );
    });

    // Debug / safety: catch launch errors early
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto mask = (col != -1) & (row >= 0);
    return torch::stack({ row.masked_select(mask), col.masked_select(mask) }, 0);
}
