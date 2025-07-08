#include "activation.h"


#include <flashinfer/activation.cuh>
#include <algorithm>

namespace blitz::kernel {

__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }

template<typename DType>
void silu_and_mul(
    DType *input,
    DType *output,
    uint32_t num_tokens,
    uint32_t inter_size
) {
    dim3 grid(num_tokens);
    uint32_t vec_size = 16 / sizeof(DType);
    dim3 block(std::min(inter_size / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<DType, silu>
        <<<grid, block, 0>>>(output, input, inter_size);
}

// instantiation
template void silu_and_mul<half>(half*, half*, uint32_t, uint32_t);

} // namespcae blitz::kernel