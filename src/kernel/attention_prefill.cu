#include "flashinfer/layout.cuh"
#include "flashinfer/pos_enc.cuh"
// #include <flashinfer_ops.cuh>
// #include "flashinfer/attention/handler.cuh"
#include "flashinfer_ops.cuh"

#include "attention.h"

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
) {
    /// @note We assume that flashinfer has been well planned
    cudaError_t status = flashinfer::BatchPrefillWithRaggedKVCacheWrapper<DType, DType, DType, IdType>(
        static_cast<flashinfer::BatchPrefillHandler*>(handler.get()),
        q_input, qo_indptr,
        k_input, v_input, kv_indptr,
        nullptr, // q_offset
        nullptr, // k_rope_pos_offset
        output,
        nullptr, // lse
        batch_size,
        num_qo_heads, num_kv_heads, head_size,
        true, // Causal mask
        flashinfer::QKVLayout::kNHD,
        flashinfer::PosEncodingMode::kRoPELlama,
        true // allow fp16 kv reduction
    );
    /// @todo add debug
    if (status != cudaSuccess) {
        throw std::runtime_error("BatchPrefillWithRaggedKVCache error: " + std::string(cudaGetErrorString(status)));
    }
}

// instantiation
template void batch_prefill_attention<half, int32_t>(
    BatchPrefillRaggedHandler<half, int32_t>& handler,
    half* q_input,
    int32_t* qo_indptr,
    half* k_input,
    half* v_input,
    int32_t* kv_indptr,
    half* output,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_size
);

}