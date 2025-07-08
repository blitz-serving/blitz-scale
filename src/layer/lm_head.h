#pragma once

#include <cublas_v2.h>
#include <nccl.h>

#include <cstdint>

#include "../kernel/sampler.h"
#include "util/cublas_wrapper.h"

namespace blitz::layer {

template<typename DType, typename IdType>
class LmHead {
  private:
    cublasHandle_t* handle;
    util::CublasWrapper* cublas_wrapper;
    ncclComm_t& nccl_tp_comm;
    const uint32_t tp_size;
    const int tp_rank;

    const uint32_t vocab_size;
    const uint32_t hidden_size;

    DType* workspace;
    DType* input = nullptr;
    DType* output = nullptr;
    DType* last_tokens = nullptr;
    DType* weight = nullptr;

  public:
    LmHead(
        cublasHandle_t* handle_,
        util::CublasWrapper* cublas_wrapper_,
        uint32_t vocab_,
        uint32_t hidden_,
        DType* workspace_ptr,
        DType** seg_iter,
        int tp_rank_,
        uint32_t tp_size_,
        ncclComm_t& nccl_tp_comm_
    ) :
        handle(handle_),
        cublas_wrapper(cublas_wrapper_),
        nccl_tp_comm(nccl_tp_comm_),
        tp_size(tp_size_),
        tp_rank(tp_rank_),
        vocab_size(vocab_),
        hidden_size(hidden_),
        workspace(workspace_ptr),
        weight(*seg_iter) {
        *seg_iter = weight + vocab_size / tp_size * hidden_size;
        assert((tp_size == 1 || (vocab_size % tp_size == 0)));
    }

    ~LmHead() {}

    size_t size_in_bytes() const noexcept {
        return (size_t)vocab_size / tp_size * hidden_size * sizeof(DType);
    }

    void forward(
        DType** logits,
        const uint32_t num_tokens,
        const uint32_t batch_size,
        const IdType* ragged_indptr,
        DType* buf = nullptr
    ) {
        /// @brief Lay out buffer
        /// [N, dim] :: [Bs, dim] :: [Bs, vocab]
        /// input       last_tokens  output
        input = buf ? buf : workspace;
        last_tokens = input + num_tokens * hidden_size;
        output = last_tokens + batch_size * hidden_size;

        if (ragged_indptr) {
            blitz::kernel::gather_last_token(
                last_tokens,
                input,
                ragged_indptr,
                num_tokens,
                batch_size,
                hidden_size
            );
        } else {
            last_tokens = input;
        }
        size_t sendcount = vocab_size / tp_size * batch_size;
        cublas_wrapper->gemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            batch_size,
            vocab_size / tp_size,
            hidden_size,
            last_tokens,
            weight,
            output + tp_rank * sendcount
        );
        /// gather sharded vocab logits
        if (tp_size > 1) {
            ncclResult_t result = ncclAllGather(
                output + tp_rank * sendcount,
                output,
                sendcount * sizeof(DType),
                ncclChar,
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
        *logits = output;
    }
};

}  // namespace blitz::layer