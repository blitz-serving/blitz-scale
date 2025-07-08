#pragma once

#include "cuda_helper.h"

namespace blitz::kernel {

/// \brief add(input, residual) => residual
template<typename DType>
void add_residual(
    DType* input,
    DType* residual,
    uint32_t num_tokens,
    uint32_t dim
);

// instantiation
// add_residual<half>

}