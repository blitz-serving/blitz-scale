// no cuh names header
#include "flashinfer_types.h"

// cuh names
#include <cstdint>
#include "flashinfer_ops.cuh"
#include <flashinfer/page.cuh>


namespace blitz::kernel {

void* create_flashinfer_prefill_handler() {
    flashinfer::BatchPrefillHandler* new_handler_ptr = 
        new flashinfer::BatchPrefillHandler;
    return (void*)new_handler_ptr;
}

void* create_flashinfer_decode_handler() {
    flashinfer::BatchDecodeHandler* new_handler_ptr = 
        new flashinfer::BatchDecodeHandler;
    return (void*)new_handler_ptr;
}

template<typename DType, typename IdType>
void BatchPrefillRaggedHandler<DType, IdType>::plan(
    void *float_buffer, size_t float_buffer_size,
    void *int_buffer, size_t int_buffer_size,
    IdType *qo_indptr_h, IdType *kv_indptr_h,
    uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size) {
    
    assert((flashinfer_handler != nullptr));
    flashinfer::BatchPrefillHandler* handler = reinterpret_cast
        <flashinfer::BatchPrefillHandler*>(flashinfer_handler);
    cudaError_t status;
    status = handler->Plan<DType, IdType>(
        float_buffer, float_buffer_size,
        int_buffer, int_buffer_size,
        qo_indptr_h, kv_indptr_h,
        qo_indptr_h[batch_size],
        batch_size, num_qo_heads, num_kv_heads,
        head_dim, page_size
    );
    if (status != cudaSuccess) {
        throw std::runtime_error("Handler plan: " + std::string(cudaGetErrorString(status)));
    }


}


template<typename DType, typename IdType>
void BlitzBatchDecodeHandler<DType, IdType>::plan(
   void *float_buffer, size_t float_buffer_size,
    void *int_buffer, size_t int_buffer_size,
    IdType *qo_indptr_h, IdType *last_page_len_h,
    uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim, uint32_t page_size
) {
    assert((flashinfer_handler != nullptr));
    cudaError_t status;
    flashinfer::BatchDecodeHandler* handler = reinterpret_cast<flashinfer::BatchDecodeHandler*>(flashinfer_handler);
    assert((handler != nullptr));
    assert((float_buffer != nullptr));
    assert((int_buffer != nullptr));
    assert((qo_indptr_h != nullptr));
    assert((last_page_len_h != nullptr));
    status = flashinfer::BatchDecodeHandlerPlan<DType, DType, DType, IdType>(
        handler,
        float_buffer, float_buffer_size,
        int_buffer, int_buffer_size,
        qo_indptr_h, last_page_len_h, 
        batch_size, num_qo_heads, num_kv_heads, 
        head_dim, page_size, 
        flashinfer::PosEncodingMode::kRoPELlama);

    if (status != cudaSuccess) {
        throw std::runtime_error("Handler plan: " + std::string(cudaGetErrorString(status)));
    }
}

}