#pragma once

#include "cuda_helper.h"
#include "flashinfer_types.h"
#include "page.h"

namespace blitz::kernel {

template<typename DType, typename IdType>
void batch_prefill_attention(
    BatchPrefillRaggedHandler<DType, IdType>& handler,
    DType* q_input,
    IdType* qo_indptr,
    DType* k_input,
    DType* v_input,
    IdType* kv_indptr,
    DType* output,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_size
);

template<typename DType, typename IdType>
void batch_decode_attention(
    BlitzBatchDecodeHandler<DType, IdType>& handler,
    DType* q_input,
    PageTable<DType, IdType>& paged_kv,
    DType* output,
    const uint32_t num_qo_heads
);

// instantiation
// batch_prefill_attention<half>
// batch_decode_attention<half>

}