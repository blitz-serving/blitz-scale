#include "embedding.h"

#include <flashinfer/vec_dtypes.cuh>

namespace blitz::kernel {

template<typename DType>
__global__ void EmbeddingKernel(
    DType* __restrict__ output,
    const uint32_t* __restrict__ token_ids,
    const DType* __restrict__ embed_tokens_weight,
    const uint32_t hidden_size
) {
    constexpr uint32_t vec_size = 8;
    uint32_t tx = threadIdx.x;
    uint32_t token_idx = blockIdx.x;
    const DType* embed_ptr = embed_tokens_weight + token_ids[token_idx] * hidden_size;

    // a warp a vector
    flashinfer::vec_t<DType, vec_size>::memcpy(
        output + token_idx * hidden_size + tx * vec_size,
        embed_ptr + tx * vec_size
    );
}

template<typename DType, typename IdType>
__global__ void EmbeddingKernel(
    DType* __restrict__ output,
    const uint32_t* __restrict__ token_ids,
    const IdType* __restrict__ out_loc,
    const DType* __restrict__ embed_tokens_weight,
    const uint32_t hidden_size
) {
    constexpr uint32_t vec_size = 8;
    uint32_t tx = threadIdx.x;
    uint32_t token_idx = blockIdx.x;
    const DType* embed_ptr = embed_tokens_weight + token_ids[token_idx] * hidden_size;
    DType* out_ptr = output + out_loc[token_idx] * hidden_size;

    // a warp a vector
    flashinfer::vec_t<DType, vec_size>::memcpy(
        out_ptr + tx * vec_size,
        embed_ptr + tx * vec_size
    );
}

template<typename DType>
void embedding(
    DType* output,
    const uint32_t* token_ids,
    const DType* embed_tokens_weight,
    const uint32_t num_tokens,
    const uint32_t hidden_size
) {
    cudaError_t status;
    {
        constexpr uint32_t vec_size = 8;
        dim3 nblks(num_tokens);
        dim3 nthrs(hidden_size / vec_size);
        auto kernel = EmbeddingKernel<DType>;    
        void* args[] = {(void*)&output, (void*)&token_ids, (void*)&embed_tokens_weight, (void*)&hidden_size};
        status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args);    
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("EmbeddingKernel error: " + std::string(cudaGetErrorString(status)));
    }
}

template<typename DType, typename IdType>
void embedding(
    DType* output,
    const uint32_t* token_ids,
    const IdType* out_loc,
    const DType* embed_tokens_weight,
    const uint32_t num_tokens,
    const uint32_t hidden_size
) {
    cudaError_t status;
    {
        constexpr uint32_t vec_size = 8;
        dim3 nblks(num_tokens);
        dim3 nthrs(hidden_size / vec_size);
        auto kernel = EmbeddingKernel<DType, IdType>;    
        void* args[] = {(void*)&output, (void*)&token_ids, (void*)&out_loc, (void*)&embed_tokens_weight, (void*)&hidden_size};
        status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args);    
    }
    if (status != cudaSuccess) {
        throw std::runtime_error("EmbeddingKernel error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void embedding<half>(half*, const uint32_t*, const half*, const uint32_t, const uint32_t);
template void embedding<half, int32_t>(half*, const uint32_t*, const int32_t*, const half*, const uint32_t, const uint32_t);

}	// namespace blitz::kernel