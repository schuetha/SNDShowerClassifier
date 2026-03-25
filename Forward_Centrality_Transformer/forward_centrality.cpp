#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor forward_centrality_cuda(torch::Tensor edge_index,
                                     torch::Tensor ptr,
                                     bool include_self);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_centrality", &forward_centrality_cuda,
          pybind11::arg("edge_index"),
          pybind11::arg("ptr"),
          pybind11::arg("include_self") = false,
          "Forward reachability centrality on a batched forward DAG.\n"
          "ptr defines graph boundaries: ptr[g]..ptr[g+1].\n"
          "Returns FC[i] = fraction of nodes reachable from i inside its own graph.\n"
          "Default excludes self (include_self=False).");
}
