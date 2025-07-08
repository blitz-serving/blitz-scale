#pragma once

#include "cuda_helper.h"

namespace blitz::kernel {

template<typename DType>
void embedding(
    DType* output,
    const uint32_t* token_ids,
    const DType* embed_tokens_weight,
    const uint32_t num_tokens,
    const uint32_t hidden_size
);

/**
 * \brief retrieve shard embed locally
 *
 * only used for TP
 * \pre output is memset to 0
 * \post AllReduce
 *
 * \param output base address of full output
 * \param token_ids token id, used to identify embed
 * \param out_loc location within output buffer 
 */
template<typename DType, typename IdType>
void embedding(
    DType* output,
    const uint32_t* token_ids,
    const IdType* out_loc,
    const DType* embed_tokens_weight,
    const uint32_t num_tokens,
    const uint32_t hidden_size
);

}  // namespace blitz::kernel