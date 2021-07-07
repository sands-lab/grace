#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> rdxtopk_cuda(torch::Tensor input,torch::Tensor indices, unsigned int k);


// C++ interface
std::vector<torch::Tensor> topk(torch::Tensor input,torch::Tensor indices, unsigned int k) {
  return rdxtopk_cuda(input,indices, k);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk", &topk, "Radix TopK Selection");
}
