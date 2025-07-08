#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <string>

#include "../kernel/attention.h"
#include "../kernel/flashinfer_types.h"
#include "../kernel/page.h"
#include "../kernel/rms_norm.h"
#include "util/cublas_wrapper.h"
#include "util/torch_utils.h"

namespace blitz::layer {

template<typename DType, typename IdType>
class Attention {
  private:
    cublasHandle_t* cublas_handle;
    blitz::util::CublasWrapper* cublas_wrapper;
    blitz::kernel::BatchPrefillRaggedHandler<DType, IdType>* prefill_handle_ptr;
    blitz::kernel::BlitzBatchDecodeHandler<DType, IdType>* decode_handle_ptr;
    ncclComm_t& nccl_tp_comm;

    const uint32_t num_qo_head;
    const uint32_t num_kv_head;
    const uint32_t head_size;
    const uint32_t hidden_size;
    const uint32_t tp_size;

    /// @brief cross-layer reusable device
    ///        buffer layout w/o ownership
    /// \param residual [N, dim]
    /// \param input [N, dim]
    /// \param qkv_proj [N, 3 * dim]
    /// \param attn_state [N, dim]
    /// \param o_proj [N, dim]
    DType* residual = nullptr;
    DType* input = nullptr;
    DType* qkv_proj = nullptr;
    DType* attn_state = nullptr;
    DType* o_proj = nullptr;
    /// @note parameters w/ ownership
    DType* input_norm_weight = nullptr;
    DType* q_weight = nullptr;
    DType* k_weight = nullptr;
    DType* v_weight = nullptr;
    DType* o_weight = nullptr;

  public:
    Attention(
        cublasHandle_t* cublas_handle_,
        util::CublasWrapper* cublas_wrapper_,
        blitz::kernel::BatchPrefillRaggedHandler<DType, IdType>*
            prefill_handle_,
        blitz::kernel::BlitzBatchDecodeHandler<DType, IdType>* decode_handle_,
        uint32_t head_dim_,
        uint32_t hidden_dim_,
        uint32_t num_qo_head_,
        uint32_t num_kv_head_,
        DType* workspace_ptr,
        DType** seg_iter,
        uint32_t tp_size_,
        ncclComm_t& nccl_tp_comm_
    ) :
        cublas_handle(cublas_handle_),
        cublas_wrapper(cublas_wrapper_),
        prefill_handle_ptr(prefill_handle_),
        decode_handle_ptr(decode_handle_),
        nccl_tp_comm(nccl_tp_comm_),
        num_qo_head(num_qo_head_),
        num_kv_head(num_kv_head_),
        head_size(head_dim_),
        hidden_size(hidden_dim_),
        tp_size(tp_size_),
        residual(workspace_ptr),
        input_norm_weight(*seg_iter),
        q_weight(input_norm_weight + hidden_size),
        k_weight(q_weight + (num_qo_head / tp_size) * head_size * hidden_size),
        v_weight(k_weight + (num_kv_head / tp_size) * head_size * hidden_size),
        o_weight(v_weight + (num_kv_head / tp_size) * head_size * hidden_size) {
        *seg_iter =
            o_weight + (num_qo_head / tp_size) * head_size * hidden_size;
    }

    ~Attention() {}

    size_t size_in_bytes() const noexcept {
        size_t res = (size_t)hidden_size
            + 2 * (size_t)num_qo_head * head_size * hidden_size / tp_size
            + 2 * (size_t)num_kv_head * head_size * hidden_size / tp_size;
        return res * sizeof(DType);
    }

    void forward(
        const uint32_t num_tokens,
        IdType* qo_indptr_d,  // maybe nullptr
        blitz::kernel::PageTable<DType, IdType>& page_table,
        DType* buf = nullptr
    ) {
        /// @brief Lay out buffer
        /// [N, dim] :: [N, dim] :: [N, (qo_h + 2 * kv_h) * h_size] :: [N, qo_h * h_size]
        /// residual    input       qkv_proj                           attn_state
        ///             o_proj
        /// @note o_output must be in the same offset as input,
        ///       this is used for GLU layer's post attention layer norm
        input = residual + num_tokens * hidden_size;
        qkv_proj = input + num_tokens * hidden_size;
        attn_state = qkv_proj
            + num_tokens * (num_qo_head + 2 * num_kv_head) / tp_size
                * head_size;
        o_proj = input;
        DType* swp = residual;
        residual = (buf == nullptr) ? swp : buf;

        // DType alpha = 1.0;
        // DType beta = 0.0;
        /// @todo change to a template
        // cudaDataType_t dtype = CUDA_R_16F;
        // cublasStatus_t status;

        /// 1. pre-layer norm
        blitz::kernel::rms_norm(
            residual,
            input_norm_weight,
            input,  // output
            num_tokens,
            hidden_size
        );
        /// 2. fused QKV projection
        DType* q_data = qkv_proj;
        DType* k_data = q_data + num_tokens * num_qo_head * head_size;
        DType* v_data = k_data + num_tokens * num_kv_head * head_size;
        if (num_qo_head == num_kv_head) {
            /// MHA, batch 3 matrices together
            cublas_wrapper->gemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                num_tokens,
                (num_qo_head + 2 * num_kv_head) / tp_size * head_size,
                hidden_size,
                input,
                q_weight,
                qkv_proj
            );
        } else {
            /// GQA, we can only batch KV matrices for now
            cublas_wrapper->gemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                num_tokens,
                hidden_size / tp_size,
                hidden_size,
                input,
                q_weight,
                q_data
            );
            cublas_wrapper->gemm(
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                num_tokens,
                (2 * num_kv_head) / tp_size * head_size,
                hidden_size,
                input,
                k_weight,
                k_data
            );
        }
        /// 3. Rope + Attention
        if (qo_indptr_d) {
            /// Prefill
            blitz::kernel::batch_prefill_attention(
                *prefill_handle_ptr,
                q_data,
                qo_indptr_d,
                k_data,
                v_data,
                qo_indptr_d,
                attn_state,  // output
                page_table.batch_size,
                num_qo_head / tp_size,
                num_kv_head / tp_size,
                head_size
            );
            blitz::kernel::store_prefill_kv_cache(
                page_table,
                k_data,
                v_data,
                qo_indptr_d
            );
        } else {
            /// Decode
            blitz::kernel::append_decode_kv_cache(page_table, k_data, v_data);
            blitz::kernel::batch_decode_attention(
                *decode_handle_ptr,
                q_data,
                page_table,
                attn_state,  // output
                num_qo_head / tp_size
            );
        }
        /// 4. O projection
        cublas_wrapper->gemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            num_tokens,
            hidden_size,
            num_qo_head / tp_size * head_size,
            attn_state,
            o_weight,
            o_proj
        );
        /// 5. TP All-Reduce
        if (tp_size > 1) {
            ncclResult_t result = ncclAllReduce(
                o_proj,
                o_proj,
                num_tokens * hidden_size,
                (std::is_same_v<DType, half> ? ncclFloat16 : ncclBfloat16),
                ncclSum,
                nccl_tp_comm,
                0
            );
            if (result != ncclSuccess) {
                BZ_ERROR(
                    "NCCL AllReduce Error: {}",
                    ncclGetErrorString(result)
                );
            }
        }
        /// \note avoid side effect
        residual = swp;
    }
};

}  // namespace blitz::layer