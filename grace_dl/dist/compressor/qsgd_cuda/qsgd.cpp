#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> qsgd_compress_cuda(torch::Tensor input, int level, int bucket_size);

torch::Tensor  qsgd_decompress_cuda(torch::Tensor input,torch::Tensor scaler, int level, int bucket_size);


// C++ interface
std::vector<torch::Tensor> qsgd_compress(torch::Tensor input, int level, int bucket_size) {
  return qsgd_compress_cuda(input, level, bucket_size);
}

torch::Tensor qsgd_decompress(torch::Tensor input,torch::Tensor scaler, int level, int bucket_size) {
  return qsgd_decompress_cuda(input, scaler, level, bucket_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress", &qsgd_compress, "QSGD compression");
  m.def("decompress", &qsgd_decompress, "QSGD decompression");
}
