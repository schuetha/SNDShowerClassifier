// local_centrality.cpp  (pybind wrapper)
#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor local_forward_degree_cuda(torch::Tensor edge_index,
                                       torch::Tensor ptr,
                                       bool normalize,
                                       int64_t k_norm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_forward_degree", &local_forward_degree_cuda,
          pybind11::arg("edge_index"),
          pybind11::arg("ptr"),
          pybind11::arg("normalize") = false,
          pybind11::arg("k_norm") = -1,
          "Local forward out-degree per node (u<v).\n"
          "If normalize=True: divides per-graph using ptr by k_norm if >0 else max(1,n_g-1).");
}
