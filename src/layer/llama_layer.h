#pragma once

#include <cstdint>
#include <cstdio>

#include "attn.h"
#include "glu.h"

namespace blitz::layer {

template<typename DType, typename IdType>
class LlamaLayer {
  private:
    Attention<DType, IdType> self_attn;
    GatedLinearUnit<DType> ffn;

    const uint32_t head_size;
    const uint32_t hidden_size;
    const uint32_t num_qo_head;
    const uint32_t num_kv_head;
    const uint32_t tp_size;

  public:
    LlamaLayer(
        cublasHandle_t* cublas_handle_,
        util::CublasWrapper* cublas_wrapper_,
        blitz::kernel::BatchPrefillRaggedHandler<DType, IdType>* prefill_handle_ptr,
        blitz::kernel::BlitzBatchDecodeHandler<DType, IdType>* decode_handle_ptr,
        uint32_t head_dim_,
        uint32_t hidden_dim_,
        uint32_t num_qo_head_,
        uint32_t num_kv_head_,
        uint32_t inter_dim_,
        DType* workspace_ptr,
        DType** seg_iter,
        uint32_t tp_size_,
        ncclComm_t& nccl_tp_comm_ref
    ) :
        self_attn(
            cublas_handle_,
            cublas_wrapper_,
            prefill_handle_ptr,
            decode_handle_ptr,
            head_dim_,
            hidden_dim_,
            num_qo_head_,
            num_kv_head_,
            workspace_ptr,
            seg_iter,
            tp_size_,
            nccl_tp_comm_ref
        ),
        ffn(
            cublas_handle_,
            cublas_wrapper_,
            hidden_dim_,
            inter_dim_,
            workspace_ptr,
            seg_iter,
            tp_size_,
            nccl_tp_comm_ref
        ),
        head_size(head_dim_),
        hidden_size(hidden_dim_),
        num_qo_head(num_qo_head_),
        num_kv_head(num_kv_head_),
        tp_size(tp_size_) {}


    size_t size_in_bytes() const {
        return self_attn.size_in_bytes() + ffn.size_in_bytes();
    }

    void forward(const uint32_t num_tokens, IdType* qo_indptr_d, blitz::kernel::PageTable<DType, IdType>& page_table) {
        // BZ_DEBUG("[Unit test] forward self attention...");
        self_attn.forward(num_tokens, qo_indptr_d, page_table);
        // BZ_DEBUG("[Unit test] forward ffn...");
        ffn.forward(num_tokens);
    }

    Attention<DType, IdType>& get_self_attn() {
        return this->self_attn;
    }

    GatedLinearUnit<DType>& get_ffn() {
        return this->ffn;
    }
};

}  // namespace blitz::layer