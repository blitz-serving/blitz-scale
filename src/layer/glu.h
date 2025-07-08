#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "../kernel/activation.h"
#include "../kernel/add.h"
#include "../kernel/rms_norm.h"
#include "util/cublas_wrapper.h"
#include "util/torch_utils.h"

namespace blitz::layer {

template<typename DType>
class GatedLinearUnit {
  private:
    cublasHandle_t* handle;
    util::CublasWrapper* cublas_wrapper;
    ncclComm_t& nccl_tp_comm;

    const uint32_t hidden_size;
    const uint32_t inter_size;
    const uint32_t tp_size;

    /// @brief cross-layer reusable device
    ///        buffer layout w/o ownership
    /// \param residual [N, dim]
    /// \param input [N, dim]
    /// \param gate_up_proj [N, 2 * inter_dim]
    /// \param gate_up_state [N, inter_dim]
    /// \param down_proj [N, dim]
    DType* residual = nullptr;
    DType* input = nullptr;
    DType* gate_up_proj = nullptr;
    DType* gate_up_state = nullptr;
    DType* down_proj = nullptr;
    /// @brief parameters w/ ownership
    DType* post_attn_norm_weight = nullptr;
    DType* gate_up_weight = nullptr;
    DType* down_weight = nullptr;


  public:
    GatedLinearUnit(
        cublasHandle_t* handle_,
        util::CublasWrapper* cublas_wrapper_,
        uint32_t hidden_,
        uint32_t inter_,
        DType* workspace_ptr,
        DType** seg_iter,
        uint32_t tp_size_,
        ncclComm_t& nccl_tp_comm_
    ) :
        handle(handle_),
        cublas_wrapper(cublas_wrapper_),
        nccl_tp_comm(nccl_tp_comm_),
        hidden_size(hidden_),
        inter_size(inter_),
        tp_size(tp_size_),
        residual(workspace_ptr),
        post_attn_norm_weight(*seg_iter),
        gate_up_weight(post_attn_norm_weight + hidden_size),
        down_weight(gate_up_weight + 2 * hidden_size * (inter_size / tp_size)) {
        /**
         * \note `gate_up` is a row-major matrix w/ shape [hidden_size, 2 * inter_size]
         *       besides, each row is [gate:up], gate = [inter_size], up = [inter_size]
         *       where gate & up are both column-major in original safetensors 
         */
        *seg_iter = down_weight + hidden_size * (inter_size / tp_size);
    }

    ~GatedLinearUnit() {}

    size_t size_in_bytes() const {
        size_t res =
            hidden_size + 3 * (size_t)hidden_size * (inter_size / tp_size);
        return res * sizeof(DType);
    }

    void forward(const uint32_t num_tokens, DType* buf = nullptr) {
        /// @brief Lay out buffer
        /// [N, dim] [N, inter_dim // TP] [N, 2 * inter_dim // TP]
        /// residual gate_up_state        gate_up_proj
        ///          [N, dim]             [N, dim]
        ///          input                down_proj
        input = residual + num_tokens * hidden_size;
        gate_up_proj =
            input + num_tokens * std::max(hidden_size, inter_size / tp_size);
        gate_up_state = input;
        down_proj = gate_up_proj;
        DType* swp = residual;
        residual = (buf == nullptr) ? swp : buf;

        /// 1. post attention norm
        blitz::kernel::add_rms_norm(
            input,
            residual,
            post_attn_norm_weight,
            num_tokens,
            hidden_size
        );
        /// 2. fused Gate & Up projection
        cublas_wrapper->gemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            num_tokens,
            2 * inter_size / tp_size,
            hidden_size,
            input,
            gate_up_weight,
            gate_up_proj
        );
        /// 3. fused SiLU & Multiply
        blitz::kernel::silu_and_mul(
            gate_up_proj,
            gate_up_state,
            num_tokens,
            inter_size / tp_size
        );
        /// 4. Down projection
        cublas_wrapper->gemm(
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            num_tokens,
            hidden_size,
            inter_size / tp_size,
            gate_up_state,
            down_weight,
            down_proj
        );
        /// 5. TP All-Reduce
        if (tp_size > 1) {
            ncclResult_t result = ncclAllReduce(
                down_proj,
                down_proj,
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
        /// 6. Add residual
        blitz::kernel::add_residual(
            down_proj,
            residual,
            num_tokens,
            hidden_size
        );
        /// \note avoid side effect
        residual = swp;
    }
};

}  // namespace blitz::layer