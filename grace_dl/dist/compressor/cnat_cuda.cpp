#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cnat_compress_cuda(torch::Tensor input);
torch::Tensor cnat_compress_deterministic_cuda(torch::Tensor input);
torch::Tensor cnat_decompress_cuda(torch::Tensor input);

// C++ interface

torch::Tensor cnat_compress(torch::Tensor input) {
  return cnat_compress_cuda(input);
}

torch::Tensor cnat_compress_deterministic(torch::Tensor input) {
  return cnat_compress_deterministic_cuda(input);
}

torch::Tensor cnat_decompress(torch::Tensor input) {
  return cnat_decompress_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress", &cnat_compress, "CNat compress (CUDA)");
  m.def("compress_deterministic", &cnat_compress_deterministic, "CNat compress deterministic (CUDA)");
  m.def("decompress", &cnat_decompress, "CNat decompress (CUDA)");
}
