#include "add.h"

#include "cuda_helper.h"

#include <cstdint>
#include <numeric>

#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>

namespace blitz::kernel {

template<uint32_t VEC_SIZE, typename DType>
__global__ void AddKernel(DType* __restrict__ input,
                          DType* __restrict__ residual,
                          const uint32_t d) {
    const uint32_t bx = blockIdx.x;
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    constexpr uint32_t warp_size = 32;
    const uint32_t num_warps = blockDim.y;
    const uint32_t thread_id = tx + ty * warp_size;
    const uint32_t num_threads = num_warps * warp_size;
    const uint32_t rounds = flashinfer::ceil_div(d, VEC_SIZE * num_threads);

    for (uint32_t i = 0; i < rounds; i++) {
        if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            uint4 input_vec = *((uint4*)(input + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE));
            uint4 residual_vec = *((uint4*)(residual + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE));
            {
                *((half2*)(&residual_vec.x)) += *((half2*)(&input_vec.x));
                *((half2*)(&residual_vec.y)) += *((half2*)(&input_vec.y));
                *((half2*)(&residual_vec.z)) += *((half2*)(&input_vec.z));
                *((half2*)(&residual_vec.w)) += *((half2*)(&input_vec.w));
            }
            *((uint4*)(residual + bx * d + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE)) = residual_vec;
        }
    }
}

template<typename DType>
void add_residual(DType* input, DType* residual,
                  uint32_t batch_size, uint32_t d) {
    constexpr uint32_t vec_size = 8;
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = (block_size + 32 - 1 ) / 32;
    dim3 nblks(batch_size);
    dim3 nthrs(32, num_warps);

    void* args[] = {&input, &residual, &d};
    auto kernel = AddKernel<vec_size, DType>;
    cudaError_t status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0);
    if (status != cudaSuccess) {
        throw std::runtime_error("AddKernel error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void add_residual<half>(
    half* input, half* residual,
    uint32_t num_tokens, uint32_t dim
);

}

