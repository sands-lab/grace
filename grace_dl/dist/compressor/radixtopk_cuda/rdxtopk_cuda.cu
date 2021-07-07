#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <curand.h>
#include <cuda_runtime_api.h>

#include "cub/device/device_radix_sort.cuh"
#include "cub/util_allocator.cuh"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>

// #include "printFunctions.cuh"
// #include "generateProblems.cuh"
// #include "topk.h"

using namespace std;
using namespace cub;
#define maxThreadsPerBlock 1024


/**
 * Computes the histogram over the digit values of an array of keys that MUST have a length of an integer multiple of (KPT * blockDim.x).
 * The padding to the integer multiple can be done by adding 0's at the end and subtracting the number of padded 0's from the final result's 0 bin.
 * The 2^NUM_BITS possible counts (0..2^NUM_BITSNUM_BITS-1) will be placed in global_histo.
 * @param keys            [IN]  The keys for which to compute the histogram
 * @param digit           [IN]
 * @param global_histo        [OUT] The array of element counts, MUST be 256 in size.
 * @param per_block_histo     [OUT]
 */
 template<
 typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
 typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
 int NUM_BITS,     // Number of bits being sorted at a time
 int KPT,          // Number of keys per thread
 int TPB,          // Number of threads per block
 int PRE_SORT_RUNS_LENGTH // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
>
__global__ void rdxsrt_histogram(KeyT *__restrict__ keys, const uint digit, IndexT *global_histo)
{
 /*** TYPEDEFs***/
 typedef Traits<KeyT>                        KeyTraits;
 typedef typename KeyTraits::UnsignedBits    UnsignedBits;
 /*typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;*/

 /*** DECLARATIONS ***/
 UnsignedBits tloc_keys[KPT]; // local keys in a thread
 uint tloc_masked[KPT]; 
 __shared__ uint shared_bins[0x01<<NUM_BITS]; // allocate a shared histogram in shared memory

 /*** INIT SHARED HISTO ***/
 if(threadIdx.x < 32){
   #pragma unroll
   for(int i=0;i<(0x01<<NUM_BITS);i+=32){
     shared_bins[i+threadIdx.x] = 0;
   }
 }
 __syncthreads();

 /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
 // Bucket index used to determine the memory offset of the bucket's global histogram
 const uint bucket_idx = 0;
 // This thread block's keys memory offset, pointing to the index of its first key
 const IndexT block_offset = (blockDim.x * blockIdx.x * KPT);

 // Load keys
 // KeyLoader(block_offset, threadIdx.x).template LoadStrided<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys);
 #pragma unroll
 for (int i=0; i<KPT; i++) {
   tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
 }

#if true || USE_RLE_HISTO
 // Mask
 #pragma unroll
 for (int i=0; i<KPT; i++) {
   tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
   tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);  // get the bin index
 }

 /*** COMPUTE HISTO ***/
 uint rle = 1;
 #pragma unroll
 for(int i=1; i<KPT; i++){
   if(tloc_masked[i] == tloc_masked[i-1]) // decrease the number of atomicAdd as much as possible
     rle++;
   else{
     atomicAdd(&shared_bins[tloc_masked[i-1]], rle);
     rle=1;
   }
 }
 atomicAdd(&shared_bins[tloc_masked[KPT-1]], rle);
#else
 #pragma unroll
 for(int i=0; i<KPT; i++){
   tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
   atomicAdd(&shared_bins[tloc_masked[i]], 1);
 }
#endif

 // Make sure we've got the counts from all threads
 __syncthreads();

 /*** Write shared histo to global histo ***/
 if(threadIdx.x < 32){
   for(int i=0;i<(0x01<<NUM_BITS);i+=32){
     atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]); // actually bucket_idx is 0 all the time (according to the code), thus we have global_histo index equal to shared_bins index
     // per_block_histo[blockIdx.x*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
   }
 }
}

template<
 typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
 typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
 int NUM_BITS,     // Number of bits being sorted at a time
 int KPT,          // Number of keys per thread
 int TPB,          // Number of threads per block
 int PRE_SORT_RUNS_LENGTH // For values greater than 1, this causes to sort a thread's keys by runs of a given length to improve run-length encoded updates to shared memory.
>
__global__ void rdxsrt_histogram_with_guards(KeyT *__restrict__ keys, const uint digit, IndexT *global_histo, const IndexT total_keys, const int block_index_offset)
{
 /*** TYPEDEFs***/
 typedef Traits<KeyT>                        KeyTraits;
 typedef typename KeyTraits::UnsignedBits    UnsignedBits;
 /*typedef LoadUnit<IndexT, RDXSRT_LOAD_STRIDE_WARP, KPT, TPB> KeyLoader;*/

 /*** DECLARATIONS ***/
 UnsignedBits tloc_keys[KPT];
 uint tloc_masked[KPT];
 __shared__ uint shared_bins[(0x01<<NUM_BITS) + 1];

 /*** INIT SHARED HISTO ***/
 if (threadIdx.x < 32) {
   #pragma unroll
   for(int i=0;i<(0x01<<NUM_BITS);i+=32){
     shared_bins[i+threadIdx.x] = 0;
   }
 }
 __syncthreads();

 /*** GET KEYS & PREPARE KEYS FOR HISTO ***/
 // Bucket index used to determine the memory offset of the bucket's global histogram
 const uint bucket_idx = 0;
 // This thread block's keys memory offset, pointing to the index of its first key
 const IndexT block_offset = (blockDim.x * (block_index_offset + blockIdx.x) * KPT);

 // Maximum number of keys the block may fetch
 const IndexT block_max_num_keys = total_keys - block_offset;
 // KeyLoader(block_offset, threadIdx.x).template LoadStridedWithGuards<UnsignedBits, KeyT, 0, KPT>(keys, tloc_keys, block_max_num_keys);
 #pragma unroll
 for (int i=0; i<KPT; i++) {
   if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
     tloc_keys[i] = reinterpret_cast<UnsignedBits*>(keys)[block_offset + threadIdx.x + blockDim.x * i];
   }
 }

 #pragma unroll
 for(int i=0; i<KPT; i++){
   // if(KeyLoader(block_offset, threadIdx.x).ThreadIndexInBounds(block_max_num_keys, i)){
   if ((threadIdx.x + blockDim.x * i) < block_max_num_keys) {
     tloc_keys[i] = KeyTraits::TwiddleIn(tloc_keys[i]);
     tloc_masked[i] = (tloc_keys[i]>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
     atomicAdd(&shared_bins[tloc_masked[i]], 1);
   }
 }

 // Make sure we've got the counts from all threads
 __syncthreads();

 /*** Write shared histo to global histo ***/
 if(threadIdx.x < 32){
   for(int i=0;i<(0x01<<NUM_BITS);i+=32){
     atomicAdd(&global_histo[(0x01<<NUM_BITS)*bucket_idx+i+threadIdx.x], shared_bins[i+threadIdx.x]);
     // per_block_histo[(block_index_offset + blockIdx.x)*(0x01<<NUM_BITS)+i+threadIdx.x] = shared_bins[i+threadIdx.x];
   }
 }
}


/**
* Makes a single pass over the input array to find entries whose digit is equal to selected digit value and greater than
* digit value. Entries equal to digit value are written to keys_buffer for future processing, entries greater
* are written to output array.
* @param d_keys_in        [IN] The keys for which to compute the histogram
* @param d_values_in      [IN] The values corresponding to the keys
* @param digit            [IN] Digit index (0 => highest digit, 3 => lowest digit for 32-bit)
* @param digit_val        [IN] Digit value.
* @param num_items        [IN] Number of entries.
* @param d_keys_buffer    [OUT] Entries with x[digit] = digit_val.
* @param d_keys_out       [OUT] Entries with x[digit] > digit_val.
* @param d_values_buffer  [OUT] Entry values with x[digit] = digit_val.
* @param d_values_out     [OUT] Entry values with x[digit] > digit_val.
* @param d_index_buffer   [OUT] Index into d_keys_buffer.
* @param d_index_out      [OUT] Index into d_keys_out.
*/
template<
 typename KeyT,    // Data type of the keys within device memory. Data will be twiddled (if necessary) to unsigned type
 typename IndexT,  // Data type used for key's offsets and counters (limits number of supported keys, uint = 2^32)
 int NUM_BITS,     // Number of bits being sorted at a time
 int KPT,          // Number of keys per thread
 int TPB           // Number of threads per block
>
__global__ void select_kth_bucket(KeyT* d_keys_in, unsigned int* d_values_in, const uint digit, const uint digit_val, uint num_items,
   KeyT* d_keys_buffer, KeyT* d_keys_out, unsigned int* d_values_buffer, unsigned int* d_values_out, uint* d_index_buffer, uint* d_index_out)
{
 typedef Traits<KeyT>                        KeyTraits;
 typedef typename KeyTraits::UnsignedBits    UnsignedBits;

 // Specialize BlockLoad for a 1D block of TPB threads owning KPT integer items each
 typedef cub::BlockLoad<UnsignedBits, TPB, KPT, BLOCK_LOAD_TRANSPOSE> BlockLoadT;

 // Specialize BlockScan type for our thread block
 typedef BlockScan<int, TPB, BLOCK_SCAN_RAKING> BlockScanT;

 // in some sense, tile means block
 const int tile_size = TPB * KPT;
 int tile_idx = blockIdx.x;    // Current tile index
 int tile_offset = tile_idx * tile_size;

 // Allocate shared memory for BlockLoad
 __shared__ union TempStorage
 {
   typename BlockLoadT::TempStorage    load_items;
   typename BlockScanT::TempStorage    scan;
   int offset[1];
   UnsignedBits raw_exchange[2 * TPB * KPT];
 } temp_storage;
 
 // Load a segment of consecutive items that are blocked across threads
 UnsignedBits key_entries[KPT];
 unsigned int value_entries[KPT];

 /*float payload_entries[KPT];*/
 int selection_flags[KPT];
 int selection_indices[KPT];

 int num_tiles = (num_items + tile_size - 1) / tile_size;
 int num_tile_items = tile_size;
 bool is_last_tile = false;
 if (tile_idx == num_tiles - 1) {
   num_tile_items = num_items - tile_offset;
   is_last_tile = true;
 }

 // Load keys and values
 if (is_last_tile) {
   BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys_in) + tile_offset, key_entries, num_tile_items);
   __syncthreads();
   BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<unsigned int*>(d_values_in) + tile_offset, value_entries, num_tile_items); 
 }
 else {
   BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<UnsignedBits*>(d_keys_in) + tile_offset, key_entries);
   __syncthreads();
   BlockLoadT(temp_storage.load_items).Load(reinterpret_cast<unsigned int*>(d_values_in) + tile_offset, value_entries);
 }
 
 __syncthreads();

 /*** Step 1: Find keys with digit value to selected digit value ***/
 #pragma unroll
 for (int ITEM = 0; ITEM < KPT; ++ITEM)
 {
   // Out-of-bounds items are selection_flags
   selection_flags[ITEM] = 0;
   
   if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
     UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
     uint masked_key = (key>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
     selection_flags[ITEM] = (masked_key > digit_val);
   }
 }

 __syncthreads();

 // Compute exclusive prefix sum
 int num_selected;
 BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

 __syncthreads();

 if (num_selected > 0) {
   int index_out;
   if (threadIdx.x == 0) {
     // Find index into keys_out array
     index_out = atomicAdd(d_index_out, num_selected);
     temp_storage.offset[0] = index_out;
   }

   __syncthreads();

   index_out = temp_storage.offset[0];

   __syncthreads();

   // Compact and scatter items
   #pragma unroll
   for (int ITEM = 0; ITEM < KPT; ++ITEM)
   {
     int local_scatter_offset = selection_indices[ITEM];
     if (selection_flags[ITEM])
     {
       temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
       temp_storage.raw_exchange[tile_size + local_scatter_offset] = value_entries[ITEM];
       /*temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM];*/
     }
   }

   __syncthreads();

   // Write out matched entries to output array
   for (int item = threadIdx.x; item < num_selected; item += TPB)
   {
     reinterpret_cast<UnsignedBits*>(d_keys_out)[index_out + item] = temp_storage.raw_exchange[item];
     d_values_out[index_out + item] = temp_storage.raw_exchange[tile_size + item];
   }

   __syncthreads();

#if 0
   for (int item = threadIdx.x; item < num_selected; item += TPB)
   {
     payload_out[num_selections_prefix + item] = temp_storage.raw_exchange[tile_size + item];
   }
#endif
 }

 /*** Step 2: Find entries that have digit equal to digit value ***/
 #pragma unroll
 for (int ITEM = 0; ITEM < KPT; ++ITEM)
 {
   // Out-of-bounds items are selection_flags
   selection_flags[ITEM] = 0;

   if (!is_last_tile || (int(threadIdx.x * KPT) + ITEM < num_tile_items)) {
     UnsignedBits key = KeyTraits::TwiddleIn(key_entries[ITEM]);
     uint masked_key = (key>>((sizeof(KeyT)*8)-(NUM_BITS*(digit+1))))&((0x01<<NUM_BITS)-1);
     selection_flags[ITEM] = (masked_key == digit_val);
   }
 }

 __syncthreads();

 // Compute exclusive prefix sum
 BlockScanT(temp_storage.scan).ExclusiveSum(selection_flags, selection_indices, num_selected);

 __syncthreads();

 if (num_selected > 0) {
   int index_buffer;
   if (threadIdx.x == 0) {
     index_buffer = atomicAdd(d_index_buffer, num_selected);
     temp_storage.offset[0] = index_buffer;
   }

   __syncthreads();

   index_buffer = temp_storage.offset[0];

   __syncthreads();

   // Compact and scatter items
   #pragma unroll
   for (int ITEM = 0; ITEM < KPT; ++ITEM)
   {
     int local_scatter_offset = selection_indices[ITEM];
     if (selection_flags[ITEM])
     {
       temp_storage.raw_exchange[local_scatter_offset] = key_entries[ITEM];
       temp_storage.raw_exchange[tile_size + local_scatter_offset] = value_entries[ITEM];
       /*temp_storage.raw_exchange[tile_size + local_scatter_offset] = payload_entries[ITEM];*/
     }
   }

   __syncthreads();

   // Write out output entries
   for (int item = threadIdx.x; item < num_selected; item += TPB)
   {
     reinterpret_cast<UnsignedBits*>(d_keys_buffer)[index_buffer + item] = temp_storage.raw_exchange[item];
     d_values_buffer[index_buffer + item] = temp_storage.raw_exchange[tile_size + item];
   }

   __syncthreads();
 }
}

__global__ void set_index_array(unsigned int* array, unsigned int len) {
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int gridSize = blockDim.x * gridDim.x;

   while (i < len) {
       array[i] = i;
       i += gridSize; 
   }
} 


#define KPT 16
#define TPB 384
#define DIGIT_BITS 8
cudaError_t CUDARadixSelectTopK(torch::Tensor d_keys_in,
                               torch::Tensor d_indices_in,
                               unsigned int num_items,
                               unsigned int k,
                               float *d_keys_out,
                               unsigned int *d_values_out) {
                               
   cudaError error = cudaSuccess;
   
   // get helper buffers
  //  unsigned int *d_histogram = buf->histogram;
  //  unsigned int *d_index_out = buf->index_out;
  //  unsigned int *d_index_buffer = buf->index_buffer;

  //  float* keys_double_buffer[2] = {buf->keys_buffer0, buf->keys_buffer1}; 
  //  unsigned int* values_double_buffer[2] = {buf->value_buffer0, buf->value_buffer1};
   unsigned char current_keys_buffer = 0;

   //initialize buffer with empty tensor
   //unsigned int *d_histogram = (uint*)torch::zeros(256*128, torch::TensorOptions().dtype(torch::kInt).device(d_keys_in.device())).data_ptr();
   //unsigned int *d_index_out = (uint*)torch::zeros(128, torch::TensorOptions().dtype(torch::kInt).device(d_keys_in.device())).data_ptr();
   //unsigned int *d_index_buffer = (uint*)torch::zeros(128, torch::TensorOptions().dtype(torch::kInt).device(d_keys_in.device())).data_ptr();
   unsigned int *d_histogram, *d_index_out, *d_index_buffer; 
   cudaMalloc(&d_histogram, 256*128);
   cudaMalloc(&d_index_out, 128);
   cudaMalloc(&d_index_buffer, 128);

   torch::Tensor keys_double_tensor[2] = {d_keys_in.clone(), d_keys_in.clone()}; 
   torch::Tensor indices_double_tensor[2] = {d_indices_in.clone(), d_indices_in.clone()}; 
   float* keys_double_buffer[2] = {(float*)keys_double_tensor[0].data_ptr(), (float*)keys_double_tensor[1].data_ptr()};
   unsigned int* values_double_buffer[2] = {(unsigned int*)indices_double_tensor[0].data_ptr(), (unsigned int*)indices_double_tensor[1].data_ptr()};
   //float* keys_double_buffer[2] = {(float*)d_keys_in.clone().data_ptr(), 
   // (float*)d_keys_in.clone().data_ptr()}; 
   //unsigned int* values_double_buffer[2] = {(uint*)d_indices_in.clone().data_ptr(), 
   // (uint*)d_indices_in.clone().data_ptr()}; 

   // Set the index into output array to 0.
   cudaMemset(d_index_out, 0, 4);
   
   unsigned int KPB = KPT * TPB;
   
   unsigned int *h_histogram = new unsigned int[256];

  //  set value array (index)
  //  int blocksPerGrid = (int) ceil(1.0 * num_items / TPB);
  //  set_index_array<<<blocksPerGrid, TPB, 0>>>(values_double_buffer[current_keys_buffer], num_items);
   
   
   // enumerate each digit (32-bit data (float32) / 8-bit/pass, so that's 4 digit in total)
   for (unsigned int digit = 0; digit < 4; digit++) {
       unsigned int num_blocks = num_items / KPB;// Pass-0 rough processing blocks (floor on purpose)
       unsigned int processed_elements = num_blocks * KPB;// Pass-0 number of rough processed elements
       unsigned int remaining_elements = num_items - processed_elements;// Do the remaining elements with a check in the inner loop
       unsigned int remainder_blocks = (KPB - 1 + remaining_elements) / KPB;// Number of blocks required for remaining elements (typically 0 or 1)

       /******************************************************************************************/
       /*  Caluclate Histogram                                                                   */
       /******************************************************************************************/
       // Zero out the histogram
       cudaMemset(d_histogram, 0, 256 * sizeof(int));

       float* d_current_keys_in = keys_double_buffer[current_keys_buffer];
       unsigned int* d_current_value_in = values_double_buffer[current_keys_buffer];
       if (num_blocks > 0)
           rdxsrt_histogram<float, uint, DIGIT_BITS, KPT, TPB, 9><<<num_blocks, TPB, 0>>>(d_current_keys_in, digit, d_histogram);
       if (remaining_elements > 0)
           rdxsrt_histogram_with_guards<float, uint, DIGIT_BITS, KPT, TPB, 9><<<remainder_blocks, TPB, 0>>>(d_current_keys_in, digit, d_histogram, num_items, num_blocks);

       /******************************************************************************************/
       /*  Find the bin which contains the Kth largest element                                   */
       /******************************************************************************************/
       cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(uint), cudaMemcpyDeviceToHost);
       
       // currently we find the bin on host, hence we need to synchronize the stream
      //  cudaStreamSynchronize(stream);

       unsigned int rolling_sum = 0;
       unsigned int digit_val;
       for (int i = 255; i >= 0; i--) {
           if ((rolling_sum + h_histogram[i]) > k) {
               digit_val = i;
               k -= rolling_sum;
               break;
           }
           rolling_sum += h_histogram[i];
       }

       cudaMemset(d_index_buffer, 0, 4);
       select_kth_bucket<float, unsigned int, DIGIT_BITS, KPT, TPB><<<num_blocks + remainder_blocks, TPB, 0>>>(d_current_keys_in,
                                                                                                                      d_current_value_in,
                                                                                                                      digit, 
                                                                                                                      digit_val, 
                                                                                                                      num_items, 
                                                                                                                      keys_double_buffer[1-current_keys_buffer],
                                                                                                                      d_keys_out,
                                                                                                                      values_double_buffer[1-current_keys_buffer],
                                                                                                                      d_values_out,
                                                                                                                      d_index_buffer,
                                                                                                                      d_index_out);
       uint h_index_out;
       uint h_index_buffer;

       cudaMemcpy(&h_index_out, d_index_out, sizeof(uint), cudaMemcpyDeviceToHost);
       cudaMemcpy(&h_index_buffer, d_index_buffer, sizeof(uint), cudaMemcpyDeviceToHost);

      //  cudaStreamSynchronize(stream);
       // Update number of items to reflect reduced number of elements.
       num_items = h_index_buffer;
       
       if (k == 0) break;
       else if (k != 0 && digit == 3) {
           // We are at last digit and k != 4 implies that kth value has repetition.
           // Copy any of the repeated values(and keys!) to out array to complete the array.
           cudaMemcpy(d_keys_out + h_index_out, keys_double_buffer[1-current_keys_buffer] ,k * sizeof(float), cudaMemcpyDeviceToDevice);
           cudaMemcpy(d_values_out + h_index_out, values_double_buffer[1-current_keys_buffer], k * sizeof(float), cudaMemcpyDeviceToDevice);
           k -= k;
       }

       current_keys_buffer = 1 - current_keys_buffer;
   }
   delete[] h_histogram;
   cudaFree(d_histogram);
   cudaFree(d_index_out);
   cudaFree(d_index_buffer);
}


// __global__ void _Uint32ToInt32(int *dst_data,
//   unsigned int *src_data,
//   unsigned int n)
// {
// // set thread ID
// unsigned int tid = threadIdx.x;
// unsigned int gridSize = blockDim.x * gridDim.x;
// unsigned int i = blockIdx.x * blockDim.x + tid;
// unsigned int blockSize = blockDim.x;

// while (i < n) {
//   dst_data[i] = (int)src_data[i];
// i += gridSize;
// }
// }

// void Uint32ToInt32(int *dst_data, 
// unsigned int *src_data,
// unsigned int num_elements)
// {
// int blocksPerGrid = (int) ceil(1.0 * num_elements / maxThreadsPerBlock);
// _Uint32ToInt32<<<blocksPerGrid, maxThreadsPerBlock, 0>>>(dst_data, src_data, num_elements);
// }



std::vector<torch::Tensor> rdxtopk_cuda(
  torch::Tensor input,torch::Tensor indices, unsigned int k) {

  unsigned int num_items = input.numel();
  
  auto d_keys_out = torch::zeros(k, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  auto d_values_out = torch::zeros(k, torch::TensorOptions().dtype(torch::kInt).device(input.device()));

  CUDARadixSelectTopK(input,indices,
    num_items,
    k,
    (float*)d_keys_out.data_ptr(),
    (uint*)d_values_out.data_ptr());

  // Uint32ToInt32((int*)d_values_out.data_ptr(), (uint*)d_values_out.data_ptr(), k);

return {d_keys_out, d_values_out};
}


