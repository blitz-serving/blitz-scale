#pragma once

#include "cuda_helper.h"

namespace blitz::kernel {

void* create_flashinfer_prefill_handler();

void* create_flashinfer_decode_handler();

/// @note class must be complete
template<typename DType, typename IdType>
class BatchPrefillRaggedHandler {
  public:
    void plan(
        void* float_buffer,
        size_t float_buffer_size,
        void* int_buffer,
        size_t int_buffer_size,
        IdType* qo_indptr_h,
        IdType* kv_indptr_h,
        uint32_t batch_size,
        uint32_t num_qo_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t page_size
    );

    BatchPrefillRaggedHandler(void* handler_ptr = nullptr) : flashinfer_handler(handler_ptr) {}

    void* get() noexcept {
        return flashinfer_handler;
    }

    bool empty() noexcept {
        return flashinfer_handler == nullptr;
    }

  private:
    void* flashinfer_handler;
};

template<typename DType, typename IdType>
class BlitzBatchDecodeHandler {
  public:
    void plan(
        void* float_buffer,
        size_t float_buffer_size,
        void* int_buffer,
        size_t int_buffer_size,
        IdType* qo_indptr_h,
        IdType* last_page_len_h,
        uint32_t batch_size,
        uint32_t num_qo_heads,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t page_size
    );

    BlitzBatchDecodeHandler(void* handler_ptr = nullptr) : flashinfer_handler(handler_ptr) {}

    void* get() noexcept {
        return flashinfer_handler;
    }

    bool empty() noexcept {
        return flashinfer_handler == nullptr;
    }

  private:
    void* flashinfer_handler;
};

template class BatchPrefillRaggedHandler<half, int32_t>;
template class BlitzBatchDecodeHandler<half, int32_t>;

}  // namespace blitz::kernel