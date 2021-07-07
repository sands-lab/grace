#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>


using namespace std;

#define EIGEN_USE_GPU
#define maxThreadsPerBlock 1024

__global__ void _qsgdreduceSumV2(float *g_odata, float *g_idata, unsigned int n)
{
    extern __shared__ float sdata[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    unsigned int blockSize = blockDim.x;

    sdata[tid] = 0;
  
    while (i < n) {
        sdata[tid] += g_idata[i];// + g_idata[i + blockDim.x];
        i += gridSize;
    } 
    __syncthreads();
    
    // in-place reduction and complete unroll
    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // unrolling warp
    if (tid < 32)
    {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }
    
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void _qsgdreduceClipThresholdV2(float *g_odata, float *g_idata, unsigned int n)
{
    extern __shared__ float sdata[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int blockSize = blockDim.x;
   
    sdata[tid] = 0;
   
    while (i < n) {
        if (isfinite(g_idata[i])) {
            sdata[tid] += g_idata[i] * g_idata[i];// + g_idata[i + blockDim.x] * g_idata[i + blockDim.x];
        }
        i += gridSize;
    } 
    __syncthreads();
    
    // in-place reduction and complete unroll
    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    } 
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }
    
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void _qsgdreduceAbsMaxV2(float *g_odata, float *g_idata, unsigned int n)
{
    extern __shared__ float sdata[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int blockSize = blockDim.x;
   
    sdata[tid] = 0;
   
    while (i < n) {
        if (isfinite(g_idata[i]) && isfinite(sdata[tid]))
            sdata[tid] = fmaxf(sdata[tid], fabsf(g_idata[i])); //fmaxf(fabsf(g_idata[i]), fabsf(g_idata[i + blockDim.x])));
        else
            sdata[tid] = nanf("123");
        i += gridSize;
    } 
    __syncthreads();
    
    // in-place reduction and complete unroll
    if (blockSize >= 1024) {
        if (tid < 512) {
            if (isfinite(sdata[tid]) && isfinite(sdata[tid + 512])) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 512]);
            else sdata[tid] = nanf("123");
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            if (isfinite(sdata[tid]) && isfinite(sdata[tid + 256])) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
            else sdata[tid] = nanf("123");
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            if (isfinite(sdata[tid]) && isfinite(sdata[tid + 128])) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
            else sdata[tid] = nanf("123");
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            if (isfinite(sdata[tid]) && isfinite(sdata[tid + 64])) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
            else sdata[tid] = nanf("123");
        }
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile float *vsmem = sdata;
        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 32]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 32]);
        else vsmem[tid] = nanf("123");

        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 16]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid + 16]);
        else vsmem[tid] = nanf("123");

        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 8]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid +  8]);
        else vsmem[tid] = nanf("123");

        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 4]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid +  4]);
        else vsmem[tid] = nanf("123");

        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 2]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid +  2]);
        else vsmem[tid] = nanf("123");

        if (isfinite(vsmem[tid]) && isfinite(vsmem[tid + 1]))
            vsmem[tid] = fmaxf(vsmem[tid], vsmem[tid +  1]);
        else vsmem[tid] = nanf("123");

    }
    
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void _qsgdcomputeSqrt(float *scaler)
{
    *scaler = sqrt(*scaler);
    //printf("l2 norm result: %f\n", *scaler);
    //__syncthreads();
}

__global__ void _qsgdinitCURand(unsigned int len, unsigned int seed, curandState* states)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
  if (index < len)
    curand_init(seed + index, /* the seed can be the same for each core, here we pass the time in from the CPU */
                0, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[index]);
}

__global__ void _qsgdcompensateMemory(float *dst, const float *src, const float *local_mem, int len)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = index; i < len; i += stride){
        if (isfinite(src[i])) {
            //dst[i] = src[i]; // + local_mem[i]; //remove memory compensation for comparison purposes.
            dst[i] = src[i] + local_mem[i]; 
        }
        else {
            dst[i] = nanf("123"); 
        }
        //printf("CompensateMemory result: idx=%d, src=%f, mem=%f, dst=%f\n", i, src[i], local_mem[i], dst[i]);
        //__syncthreads();
    }
}

__global__ void _qsgdTernarizeValue(int8_t *dst, const float *src, float *scaler, float *local_mem, const int len, int level, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    curandState local_state = states[index];
    float norm_scaler = *scaler;
    
    // The input tensor here has been clipped.
    // Hence we have the ternarize formula: dst[i] = new_level[i] * sign(src[i])
    for (int i = index; i < len; i += stride) {
        if (isfinite(norm_scaler) && isfinite(src[i])) {
            float rand_sample = curand_uniform(&local_state);
            float level_float = (float)level / norm_scaler * fabsf(src[i]); 
            int8_t previous_level = floor(level_float);
            if (rand_sample < level_float - previous_level) {
                dst[i] = previous_level + 1;  // 1 is required by qsgd
            }
            else {
                dst[i] = previous_level; 
            } 
            if (src[i] < 0){
                dst[i] = -dst[i];
            }
                
            // update local memory
            local_mem[i] = src[i] - norm_scaler / (float)level * (float)dst[i]; // remove vanilla local memory update for comparison purposes.
        }
        else {
            // encode value to the minimum for Inf or NaN
            dst[i] = -128;
        }
        //printf("compressed result: idx=%d, scaler=%f, src=%f, dst=%d, update_mem=%f\n", i, *scaler, src[i], dst[i], local_mem[i]);
        //__syncthreads();
    }
}


// For qsgd allreduce
// __global__ void _qsgdDeternarizeValue(int len, float *dst, int8_t *src, float *scaler, int level)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     float norm_scaler = *scaler;
   
//     for (int i = index; i < len; i += stride)
//     {
//         dst[i] = norm_scaler / (float)level * (float)src[i];  
//     }
// }


// For qsgd allgather
__global__ void _qsgdDeternarizeAndAdd(int len, float *dst, int8_t *src, float *scaler, int level)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float norm_scaler = *scaler;

    for (int i = index; i < len; i += stride) {
        if (src[i] == -128) {
            dst[i] = nanf("123");
        }
        else {
            dst[i] += norm_scaler / (float)level * (float)src[i];    
        }
        //printf("decompressed result: idx=%d, scaler=%f, src=%d, dst=%f\n", i, *scaler, src[i], dst[i]);
        //__syncthreads();      
    }
}

__global__ void _bucket_l2norm(const int len, double *dst, float *src, const int bucket_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const int loop_times = len / bucket_size;
    const int remain_nums = len % bucket_size;
   
    for (int i = index; i < loop_times; i += stride)
    {
#pragma unroll
        for (int j = 0; j < bucket_size; j ++){
            if (isfinite(src[bucket_size*i+j])) {
              dst[i] += (double)(src[bucket_size*i+j]) * (double)(src[bucket_size*i+j]);
            }
        }
        dst[i] = sqrt(dst[i]);
    }
    if (remain_nums && index == loop_times){
#pragma unroll
        for (int i = 0; i < remain_nums; i++){
           if (isfinite(src[bucket_size*loop_times+i])) {
              dst[loop_times] += (double)(src[bucket_size*loop_times+i]) * (double)(src[bucket_size*loop_times+i]); 
           }
        }
        dst[loop_times] = sqrt(dst[loop_times]);
    }
        
}



__global__ void _bucket_qsgdTernarizeValue(int8_t *dst, const float *src, double *scaler, const int len, int level, const int bucket_size, unsigned int seed)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    // curandState local_state = states[index]; 
    curandState local_state;
    
    
    // The input tensor here has been clipped.
    // Hence we have the ternarize formula: dst[i] = new_level[i] * sign(src[i])
    for (int i = index; i < len; i += stride) {
        float norm_scaler = (float)(scaler[i/bucket_size]);
        curand_init(seed + index, 0, 0, &local_state);
        if (isfinite(norm_scaler) && isfinite(src[i])) {
            float rand_sample = curand_uniform(&local_state);
            float level_float = (float)level / norm_scaler * fabsf(src[i]); 
            int8_t previous_level = floor(level_float);
            if (rand_sample < level_float - previous_level) {
                dst[i] = previous_level + 1;  // 1 is required by qsgd
            }
            else {
                dst[i] = previous_level; 
            } 
            if (src[i] < 0){
                dst[i] = -dst[i];
            }
                
            // update local memory
            //local_mem[i] = src[i] - norm_scaler / (float)level * (float)dst[i]; // remove vanilla local memory update for comparison purposes.
        }
        else {
            // encode value to the minimum for Inf or NaN
            dst[i] = -128;
        }
        //printf("compressed result: idx=%d, scaler=%f, src=%f, dst=%d, update_mem=%f\n", i, *scaler, src[i], dst[i], local_mem[i]);
        //__syncthreads();
    }
}

// For qsgd allgather
__global__ void _bucket_qsgdDeternarizeAndAdd(int len, float *dst, int8_t *src, double *scaler, int level, const int bucket_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < len; i += stride) {
        float norm_scaler = (float)(scaler[i/bucket_size]);
        if (src[i] == -128) {
            dst[i] = nanf("123");
        }
        else {
            dst[i] = norm_scaler / (float)level * (float)src[i];
            //atomicAdd(dst+i, norm_scaler / (float)level * (float)src[i]);    
        }
        //printf("decompressed result: idx=%d, scaler=%f, src=%d, dst=%f\n", i, *scaler, src[i], dst[i]);
        //__syncthreads();      
    }
}


/*----------------------------------- Reduce Wrapper --------------------------------------------*/
void qsgdGPUReduce(int len, float *d_out, float *d_intermediate_res, float *result, int whichKernel, cudaStream_t stream) {
    // d_intermediate_res holds the input
    // setting up blocks
    int numBlocks = (int) ceil(1.0 * len / maxThreadsPerBlock); //(len / maxThreadsPerBlock) + 1;
    
    int prevNumBlocks = len;
    // recursively reduce to get the result
    while (numBlocks > maxThreadsPerBlock) {
        // clear d_out
        cudaMemset(d_out, 0, numBlocks * sizeof(float));
    
        switch (whichKernel) {
            // reduce sum
            case 0:
                _qsgdreduceSumV2<<<numBlocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, len);
                break;
            // reduce absmax
            case 1:
                _qsgdreduceAbsMaxV2<<<numBlocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, len);
                break;
            // reduce clip threshold
            case 2:
                _qsgdreduceClipThresholdV2<<<numBlocks, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, len);
                // we don't need to square the intermediate results.
                whichKernel = 0;
                break;
            default:
                break;
        }
        
        // by now, d_out holds the intermediate result, copy it to intermedaite_res for the next run
        cudaMemcpy(d_intermediate_res, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice); 
        // compute reduced problem size
        prevNumBlocks = numBlocks;
        len = numBlocks;
        numBlocks = (int) ceil(1.0 * numBlocks / maxThreadsPerBlock); //numBlocks / maxThreadsPerBlock + 1;
    } 

    // use one block to compute the rest.
    // clear d_out
    cudaMemset(d_out, 0, prevNumBlocks* sizeof(float));
    switch (whichKernel) {
        // reduce sum
        case 0:
            _qsgdreduceSumV2<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, prevNumBlocks);
            break;
        // reduce absmax
        case 1:
            _qsgdreduceAbsMaxV2<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, prevNumBlocks);
            break;
        // reduce clip threshold
        case 2:
            _qsgdreduceClipThresholdV2<<<1, maxThreadsPerBlock, maxThreadsPerBlock * sizeof(float)>>>(d_out, d_intermediate_res, prevNumBlocks);
            break;
        default:
            break;
    }
    // as we just use one block, just move the first element of d_out to result
    cudaMemcpy(result, d_out, sizeof(float), cudaMemcpyDeviceToDevice);
}

/*----------------------------------- Kernel Launch Wrappers ------------------------------------*/


void GPUReduceL2Norm(float *array, int len, double *l2norm_scaler, const int bucket_size)
{
    int blocksPerGrid = (int) ceil(1.0 * len / maxThreadsPerBlock);
    _bucket_l2norm<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(len, l2norm_scaler, array, bucket_size);
}

// void qsgdGPUInit_curand(int n, unsigned int seed, curandState* cuda_states)
// {
//     int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
//     _qsgdinitCURand<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(n, seed, cuda_states);
// }

// void qsgdGPUCompensateMemory(float *dst, const float *src, const float* local_mem, int len)
// {
//     int blocksPerGrid = (int) ceil(1.0 * len / maxThreadsPerBlock);
//     _qsgdcompensateMemory<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(dst, src, local_mem, len);
// }



void GPUTernarizeMultiLevelValue(int8_t *dst, const float *src, double *scaler, int len, int level, const int bucket_size)
{
    int blocksPerGrid = (int) ceil(1.0 * std::min(len, 1024 * 1024 * 25) / maxThreadsPerBlock);
    unsigned int seed = time(NULL);
    _bucket_qsgdTernarizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(dst, src, scaler, len, level, bucket_size, seed);
}

void GPUDeternarizeMultiLevelValue(int len, float *dst, int8_t *src, double *scaler, int level, const int bucket_size)
{
    int blocksPerGrid = (int) ceil(1.0 * len / maxThreadsPerBlock);
    _bucket_qsgdDeternarizeAndAdd<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(len, dst, src, scaler, level, bucket_size);
}




std::vector<torch::Tensor> qsgd_compress_cuda(torch::Tensor input, int level, int bucket_size) {
  
  int element_nums = input.numel();
  int num_buckets = ceil((float)element_nums / bucket_size);
  auto d_l2norm_scaler = torch::zeros(num_buckets, torch::TensorOptions().dtype(torch::kFloat64).device(input.device()));
  auto buffer_data = torch::empty(element_nums, torch::TensorOptions().dtype(torch::kInt8).device(input.device()));

//   curandState* cuda_states;
//   cuda_states = (curandState*)torch::empty(element_nums, torch::TensorOptions().dtype(torch::kInt).device(input.device())).data_ptr();
//   qsgdGPUInit_curand(element_nums, time(NULL), cuda_states);

  GPUReduceL2Norm((float*)input.data_ptr(), element_nums, (double*)d_l2norm_scaler.data_ptr(), bucket_size);
  GPUTernarizeMultiLevelValue((int8_t*)buffer_data.data_ptr(), (float*)input.data_ptr(), (double*)d_l2norm_scaler.data_ptr(), 
                              element_nums, level, bucket_size);

  return {buffer_data, d_l2norm_scaler};
}


torch::Tensor qsgd_decompress_cuda(torch::Tensor input, torch::Tensor d_l2norm_scaler, int level, int bucket_size) {
  int element_nums = input.numel();
  int num_buckets = ceil((float)element_nums / bucket_size);
  auto buffer_data = torch::empty(element_nums, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  GPUDeternarizeMultiLevelValue(element_nums, (float*)buffer_data.data_ptr(), (int8_t*)input.data_ptr(), 
  (double*)d_l2norm_scaler.data_ptr(), level, bucket_size);
  return buffer_data;
}