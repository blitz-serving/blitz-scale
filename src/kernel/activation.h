#pragma once

#include "cuda_helper.h"

namespace blitz::kernel {

/// \brief (Silu(input[:,:inter_size]) * input[:,inter_size:]) => output
/// \param input [num_tokens, 2 * inter_size]
/// \param output [num_tokens, inter_size]
template<typename DType>
void silu_and_mul(
    DType* input,
    DType* output,
    uint32_t num_tokens,
    uint32_t inter_size
);

// instantiation
// silu_and_mul<half>

}