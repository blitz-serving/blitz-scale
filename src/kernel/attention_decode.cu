#include "flashinfer/layout.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer_ops.cuh"
#include "flashinfer/page.cuh"

#include "attention.h"
#include "flashinfer_types.h"
#include <cstdint>


namespace blitz::kernel {

template<typename DType, typename IdType>
void batch_decode_attention(
    BlitzBatchDecodeHandler<DType, IdType>& handler,
    DType* q_input,
    PageTable<DType, IdType>& page_table,
    DType* output,
    const uint32_t num_qo_heads
) {
    flashinfer::paged_kv_t<DType, IdType> paged_kv(
        page_table.num_heads, page_table.page_size, page_table.head_size,
        page_table.batch_size, flashinfer::QKVLayout::kNHD,
        page_table.k_data, page_table.v_data,
        page_table.indices_d, page_table.indptr_d, page_table.last_page_len_d
    );
    cudaError_t status = 
        flashinfer::BatchDecodeWithPagedKVCacheWrapper<DType, DType, DType, IdType>(
            static_cast<flashinfer::BatchDecodeHandler*>(handler.get()),
            q_input, nullptr,
            paged_kv,
            output,
            nullptr,
            num_qo_heads,
            flashinfer::PosEncodingMode::kRoPELlama
        );
    if (status != cudaSuccess) {
        throw std::runtime_error("batchDecodeAttention error: " + std::string(cudaGetErrorString(status)));
    } 
}

template void batch_decode_attention<half, int32_t>(
    BlitzBatchDecodeHandler<half, int32_t>& handler,
    half*,
    PageTable<half, int32_t>& paged_table,
    half*,
    const uint32_t
);

}