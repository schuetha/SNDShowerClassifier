#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Declaration of the CUDA function implemented in monotonic_knn_cuda.cu
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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "mknn_cuda",
        &mknn_cuda,
        py::arg("x"),
        py::arg("y"),
        py::arg("pos_x"),
        py::arg("pos_y"),
        py::arg("ptr_x") = c10::nullopt,   // <-- important: matches c10::optional
        py::arg("ptr_y") = c10::nullopt,   // <-- important: matches c10::optional
        py::arg("k") = 16,
        py::arg("cosine") = false,
        py::arg("use_z_mono") = true,
        py::arg("z_index") = 2,
        py::arg("use_time") = false,
        py::arg("time_index") = 3,
        R"doc(
Monotonic KNN (CUDA).

Args:
  x, y:     [N, F] and [M, F] feature tensors (CUDA, contiguous)
  pos_x/y:  [N, P] and [M, P] position/aux tensors (same dtype/device as x/y)
  ptr_x/y:  optional CSR pointers for batching (Long CUDA). If not given, treated as single example.
  k:        number of neighbors (<=100)
  cosine:   if true, use cosine distance; else squared L2
  use_z_mono, z_index: enforce y_z > x_z using pos tensors column z_index
  use_energy, e_index: enforce y_e < x_e using pos tensors column e_index

Returns:
  edge_index: Long tensor of shape [2, E] with (row=src, col=dst).
)doc"
    );
}
