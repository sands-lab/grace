#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {
__constant__ __device__ uint8_t sign_and_exp_to_encoding[512] = 
{ 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
 25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
 39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
 53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
 81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,
 95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
123, 124, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
128, 128, 128, 128, 128, 128, 128, 128, 129, 130, 131, 132, 133, 134,
135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
247, 248, 249, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
255, 255, 255, 255, 255, 255, 255, 255};

__constant__ __device__ uint32_t encoding_to_sign_and_exp[256] = 
{ 0,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
 31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
 45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
 59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
 73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,
 87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
143, 144, 256, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368,
369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382,
383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396,
397, 398, 399, 400};

__global__ void cnat_compress_cuda_kernel(
    float* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ rand,
    int64_t len) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < len) {
    if (input[index] == 0)
      output[index] = 0;
    else {
      int exp;
      float prob = abs(frexpf(input[index], &exp)) / 0.5 - 1.; // [0.5, 1) -> [0, 1)
      if (rand[index] >= prob) exp -= 1;

      if (input[index] < 0) exp += 383; // 256+127
      else exp += 127;
      output[index] = sign_and_exp_to_encoding[exp];

      //exp += 127;
      //uint8_t encode;
      //if (exp<=17) encode = 0;
      //else if (exp<=143) encode = uint8_t(exp-17);
      //else encode = 127;
      //if (input[index] < 0) encode += 256;
      //output[index] = encode;
    }
  }
}

__global__ void cnat_compress_deterministic_cuda_kernel(
    float* __restrict__ input,
    uint8_t* __restrict__ output,
    int64_t len) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < len) {
    if (input[index] == 0)
      output[index] = 0;
    else {
      int exp;
      float prob = abs(frexpf(input[index], &exp)) / 0.5 - 1.; // [0.5, 1) -> [0, 1)
      if (0.5 >= prob) exp -= 1;

      if (input[index] < 0) exp += 383; // 256+127
      else exp += 127;
      output[index] = sign_and_exp_to_encoding[exp];

      //exp += 127;
      //uint8_t encode;
      //if (exp<=17) encode = 0;
      //else if (exp<=143) encode = uint8_t(exp-17);
      //else encode = 127;
      //if (input[index] < 0) encode += 256;
      //output[index] = encode;
    }
  }
}

__global__ void cnat_decompress_cuda_kernel(
    uint8_t* __restrict__ input,
    float* __restrict__ output,
    int64_t len) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < len) {
    uint32_t sign_and_exp = encoding_to_sign_and_exp[input[index]] << 23;
    output[index] = reinterpret_cast<float &>(sign_and_exp);
  }
}

} // namespace

torch::Tensor cnat_compress_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
  auto rand = torch::rand_like(input, torch::TensorOptions().device(input.device())); // [0, 1)
  const int threads = 1024;
  int64_t numel = input.numel();
  auto blocks = numel/threads;
  if (numel%threads || !blocks) blocks++;

  cnat_compress_cuda_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(),
    output.data_ptr<uint8_t>(),
    rand.data_ptr<float>(),
    numel);

  return output;
}

torch::Tensor cnat_compress_deterministic_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
  const int threads = 1024;
  int64_t numel = input.numel();
  auto blocks = numel/threads;
  if (numel%threads || !blocks) blocks++;

  cnat_compress_deterministic_cuda_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(),
    output.data_ptr<uint8_t>(),
    numel);

  return output;
}

torch::Tensor cnat_decompress_cuda(torch::Tensor input) {
  auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  const int threads = 1024;
  int64_t numel = input.numel();
  auto blocks = numel/threads;
  if (numel%threads || !blocks) blocks++;

  cnat_decompress_cuda_kernel<<<blocks, threads>>>(
    input.data_ptr<uint8_t>(),
    output.data_ptr<float>(),
    numel);

  return output;
}

