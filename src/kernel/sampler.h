#pragma once

#include <cstdint>
#include "cuda_helper.h"

namespace blitz::kernel {

template<typename DType, typename IdType>
void gather_last_token(
    DType* output,
    const DType* tokens,
    const IdType* ragged_indptr, 
    const uint32_t num_tokens,
    const uint32_t batch_size,
    const uint32_t hidden_size
);

/// \brief argmax(input) => output
/// \param input [batch_size, length]
template<typename DType>
void batch_find_max(
    uint32_t* output_indices,
    const DType* input,
    const uint32_t batch_size,
    const uint32_t length
);

}  // namespace blitz::kernel