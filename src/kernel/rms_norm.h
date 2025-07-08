#pragma once

#include "cuda_helper.h"

namespace blitz::kernel {

/// \brief RMSNorm<weight>(input) => output
/// \param input [num_tokens, hidden_size]
/// \param weight [hidden_size]
/// \param output [num_tokens, hidden_size]
template<typename DType>
void rms_norm(
    DType* input,
    DType* weight,
    DType* output,
    uint32_t num_tokens,
    uint32_t hidden_size,
    float eps = 1e-5
);

/// \brief RMSNorm<weight>(input + residual) => input
///        (input + residual) => residual
/// \param input [num_tokens, hidden_size]
/// \param residual [num_tokens, hidden_size]
/// \param weight [hidden_size]
template<typename DType>
void add_rms_norm(
    DType* input,
    DType* residual,
    DType* weight,
    uint32_t num_tokens,
    uint32_t hidden_size,
    float eps = 1e-5
);

}