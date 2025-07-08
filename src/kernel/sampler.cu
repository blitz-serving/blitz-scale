#include "sampler.h"

#include <flashinfer/vec_dtypes.cuh>

namespace blitz::kernel {

template<typename DType, typename IdType>
__global__ void GatherLastTokenKernel(
	DType* __restrict__ output,			    // [batch_size, hidden_dim]
	const DType* __restrict__ input_tokens,	// [num_tokens, hidden_dim]
    const IdType* __restrict__ ragged_indptr,
	const uint32_t num_tokens,
	const uint32_t batch_size,
	const uint32_t hidden_size
) {
    constexpr uint32_t vec_size = 8;
    uint32_t batch_idx = blockIdx.x;
	uint32_t last_token_idx = ragged_indptr[batch_idx + 1] - 1;
    const DType* token_ptr = input_tokens + last_token_idx * hidden_size;

    // 1 warp 1 hidden size of token
    flashinfer::vec_t<DType, vec_size>::memcpy(
        output + batch_idx * hidden_size,
        token_ptr + threadIdx.x * vec_size
    );
}

template<typename DType, typename IdType>
void gather_last_token(
    DType* output,
    const DType* tokens,
    const IdType* ragged_indptr, 
    const uint32_t num_tokens,
    const uint32_t batch_size,
    const uint32_t hidden_size
) {
    cudaError_t status;
    {
        constexpr uint32_t vec_size = 8;
        dim3 nblks(batch_size);
        dim3 nthrs(hidden_size / vec_size);
        auto kernel = GatherLastTokenKernel<DType, IdType>;    
        void* args[] = {
            (void*)&output,
            (void*)&tokens,
            (void*)&ragged_indptr,
            (void*)&num_tokens,
            (void*)&batch_size,
            (void*)&hidden_size
        };
        status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args);    
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("GatherLastTokenKernel error: " + std::string(cudaGetErrorString(status)));
    }
}

#define FINAL_MASK 0xffffffffu

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    uint32_t lane = threadIdx.x & 0x1fu;  // in-warp idx
    uint32_t wid  = threadIdx.x >> 5;    // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

template<typename DType>
__global__ void BatchFindMaxKernel(
	uint32_t* output_indices,
	const DType* input,
	const uint32_t batch_size,
	const uint32_t length
) {
	__shared__ DType s_max;
	DType local_max = -65400.;
	uint32_t local_max_index;
	for (uint32_t i = threadIdx.x; i < length; i += blockDim.x) {
		if (input[i + length * blockIdx.x] > local_max) {
			local_max = input[i + length * blockIdx.x];
			local_max_index = i;
		}
	}
	DType max_val = (DType)(blockDim.x <= 32 ? warpReduceMax((float)local_max) : blockReduceMax((float)local_max));
	if (threadIdx.x == 0) {
		s_max = max_val;
	}
	__syncthreads();
	if (local_max == s_max) {
		output_indices[blockIdx.x] = local_max_index;
	}
}

template<typename DType>
void batch_find_max(
    uint32_t* output_indices,
    const DType* input,
    const uint32_t batch_size,
    const uint32_t length
) {
    cudaError_t status;
    {
        dim3 nblks(batch_size);
        dim3 nthrs(1024);
        auto kernel = BatchFindMaxKernel<DType>;    
        void* args[] = {
            (void*)&output_indices,
            (void*)&input,
            (void*)&batch_size,
            (void*)&length
        };
        status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args);    
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("BatchFindMaxKernel error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void gather_last_token(
    half* output,
    const half* tokens,
    const int32_t* ragged_indptr, 
    const uint32_t num_tokens,
    const uint32_t batch_size,
    const uint32_t hidden_size
);

template void batch_find_max(
    uint32_t* output_indices,
    const half* input,
    const uint32_t batch_size,
    const uint32_t length
);

}