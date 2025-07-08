#pragma once

#include <cublas_v2.h>
#include <nccl.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../kernel/flashinfer_types.h"
#include "../kernel/page.h"
#include "../kernel/rms_norm.h"
#include "../kernel/sampler.h"
#include "../layer/embed.h"
#include "../layer/llama_layer.h"
#include "../layer/lm_head.h"
#include "blitz/model_loader.h"
#include "fmt/format.h"
#include "include/logger.hpp"
#include "model.h"
#include "util/cublas_wrapper.h"

namespace blitz {

template<typename DType, typename IdType>
class Stub;

template<typename DType, typename IdType>
class ZigzagManager;
}  // namespace blitz

namespace blitz::model {

template<typename DType, typename IdType>
class Llama: public Model<DType, IdType> {
    friend class Stub<DType, IdType>;
    friend class ZigzagManager<DType, IdType>;
  private:
    // sublayers
    // use unique_ptr to bypass constructors
    std::unique_ptr<blitz::layer::EmbedLayer<DType, IdType>> embed;
    std::vector<blitz::layer::LlamaLayer<DType, IdType>> tfm_layers;
    DType* final_norm_weight;
    std::unique_ptr<blitz::layer::LmHead<DType, IdType>> lm_head;
    // memory layout
    DType* host_weight_segment;
    DType* weight_segment;
    DType* runtime_segment;
    size_t weight_segment_size_in_bytes;
    size_t runtime_segment_size_in_bytes;
    uint32_t* next_token_ids_d;
    uint32_t* next_token_ids_h;
    IdType* loc_h;
    IdType* loc_d;
    IdType* ragged_indptr_d;
    // model config
    const uint32_t num_layers;
    const uint32_t num_qo_head;
    const uint32_t num_kv_head;
    const uint32_t head_size;
    const uint32_t hidden_size;
    const uint32_t inter_size;
    const uint32_t vocab_size;
    const float eps = 1e-5;
    const uint32_t max_position_embeddings = 4096;
    const rank_t tp_size;
    ncclComm_t& nccl_tp_comm_ref;
    const rank_t tp_rank;
    // runtime config
    cublasHandle_t cublas_handle;
    util::CublasWrapper* cublas_wrapper;
    blitz::kernel::BatchPrefillRaggedHandler<DType, IdType> prefill_attn_handle;
    blitz::kernel::BlitzBatchDecodeHandler<DType, IdType> decode_attn_handle;
    void* attn_float_buffer;
    void* attn_int_buffer;
    const size_t float_buffer_size, int_buffer_size;

    const uint32_t max_batch_size;
    const uint32_t max_batch_tokens;

  public:
    // TODO: add max_position embedding
    Llama(
        uint32_t layers_,
        uint32_t qo_head_,
        uint32_t kv_head_,
        uint32_t head_size_,
        uint32_t hidden_size_,
        uint32_t inter_,
        uint32_t vocab_,
        uint32_t batch_size_,
        uint32_t batch_tokens_,
        const rank_t tp_size_,
        ncclComm_t& nccl_tp_comm_,
        const rank_t tp_rank_
    ) :
        num_layers(layers_),
        num_qo_head(qo_head_),
        num_kv_head(kv_head_),
        head_size(head_size_),
        hidden_size(hidden_size_),
        inter_size(inter_),
        vocab_size(vocab_),
        tp_size(tp_size_),
        nccl_tp_comm_ref(nccl_tp_comm_),
        tp_rank(tp_rank_),
        float_buffer_size(128 * 1024 * 1024),
        int_buffer_size(8 * 1024 * 1024),
        max_batch_size(batch_size_),
        max_batch_tokens(batch_tokens_)
    {
        // initialize handles for operator libraries
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cuBLAS error:" + std::string(cublasGetStatusString(status))
            );
        }
        cublas_wrapper = new util::CublasWrapper(
            std::shared_ptr<cublasHandle_t>(&cublas_handle)
        );
        BZ_INFO("After init cublas_wrapper");
        void* prefill_attn_handle_inner_ptr =
            blitz::kernel::create_flashinfer_prefill_handler();
        void* decode_attn_handle_inner_ptr =
            blitz::kernel::create_flashinfer_decode_handler();
        prefill_attn_handle =
            decltype(prefill_attn_handle) {prefill_attn_handle_inner_ptr};
        decode_attn_handle =
            decltype(decode_attn_handle) {decode_attn_handle_inner_ptr};
        cudaMalloc(&attn_float_buffer, float_buffer_size);
        cudaMalloc(&attn_int_buffer, int_buffer_size);

        // initialize memory segment
        weight_segment_size_in_bytes =
            (vocab_size * hidden_size / tp_size  // embed
             + num_layers
                 * (((size_t)2 * num_qo_head + 2 * num_kv_head) * head_size
                        * hidden_size / tp_size  // QKVO
                    + 2 * hidden_size  // RMSNorm
                    + 3 * hidden_size * inter_size / tp_size  // GLU
                 )
             + hidden_size  // final norm
             + vocab_size * hidden_size / tp_size  // final projection
            )
            * sizeof(DType);
        cudaMalloc(&weight_segment, weight_segment_size_in_bytes);
        BZ_INFO(
            "weight segment: [{}:{}]",
            ::fmt::ptr(weight_segment),
            ::fmt::ptr((char*)weight_segment + weight_segment_size_in_bytes)
        );

        runtime_segment_size_in_bytes =
            std::max({
                max_batch_tokens * hidden_size * 5 / tp_size,  // Attention
                max_batch_tokens * (hidden_size + 3 * std::max(hidden_size, inter_size / tp_size)),  // FFN
                max_batch_tokens * hidden_size + max_batch_size * (hidden_size + vocab_size)  // LM Head
            })
            * sizeof(DType);
        cudaMalloc(&runtime_segment, runtime_segment_size_in_bytes);
        BZ_INFO(
            "runtime segment: [{}:{}]",
            ::fmt::ptr(runtime_segment),
            ::fmt::ptr((char*)runtime_segment + runtime_segment_size_in_bytes)
        );

        // initialize small memory fragments
        cudaMalloc(&next_token_ids_d, max_batch_tokens * sizeof(uint32_t));
        cudaMallocHost(&next_token_ids_h, max_batch_tokens * sizeof(uint32_t));
        cudaMalloc(&ragged_indptr_d, (max_batch_size + 1) * sizeof(IdType));
        cudaMalloc(&loc_d, max_batch_tokens * sizeof(IdType));
        cudaMallocHost(&loc_h, max_batch_tokens * sizeof(IdType));

        // initialize sublayers
        DType* seg_iter = weight_segment;
        embed = std::make_unique<blitz::layer::EmbedLayer<DType, IdType>>(
            vocab_size,
            hidden_size,
            runtime_segment,
            &seg_iter,
            tp_size
        );
        for (size_t i = 0; i < num_layers; ++i) {
            tfm_layers.emplace_back(
                &cublas_handle,
                cublas_wrapper,
                &prefill_attn_handle,
                &decode_attn_handle,
                head_size,
                hidden_size,
                num_qo_head,
                num_kv_head,
                inter_size,
                runtime_segment,
                &seg_iter,
                tp_size,
                nccl_tp_comm_
            );
        }
        final_norm_weight = seg_iter;
        seg_iter += hidden_size;
        lm_head = std::make_unique<blitz::layer::LmHead<DType, IdType>>(
            &cublas_handle,
            cublas_wrapper,
            vocab_size,
            hidden_size,
            runtime_segment,
            &seg_iter,
            tp_rank,
            tp_size,
            nccl_tp_comm_
        );
        assert(((void*)seg_iter == (void*)((char*)weight_segment + weight_segment_size_in_bytes)));
    }

    ~Llama() {
        /// \note pass ownership to ModelLoader
        // cudaFree(weight_segment);
        cudaFree(runtime_segment);
        cudaFree(next_token_ids_d);
        cudaFreeHost(next_token_ids_h);
        cudaFree(ragged_indptr_d);
    }

    std::pair<void*, size_t> get_host_weight_segment() {
        return {
            (void*)this->host_weight_segment,
            this->weight_segment_size_in_bytes};
    }

    size_t get_embed_size_in_bytes() const noexcept {
        return embed->size_in_bytes();
    }

    size_t get_layer_size_in_bytes() const {
        return tfm_layers[0].size_in_bytes();
    }

    size_t get_lmhead_size_in_bytes() const noexcept {
        return hidden_size * sizeof(DType) + lm_head->size_in_bytes();
    }

    uint32_t get_layer_num() const noexcept {
        return num_layers;
    }

    std::tuple<void*, size_t> get_weight_segment() const noexcept {
        return {weight_segment, weight_segment_size_in_bytes};
    }

    void* get_runtime_segment() const noexcept {
        return runtime_segment;
    }

    uint32_t* get_io_token_ids_buf() noexcept override {
        return (uint32_t*)next_token_ids_h;
    }

    uint32_t get_max_position_embeddings() const noexcept override {
        return max_position_embeddings;
    }

    void forward(
        const uint32_t num_tokens,
        IdType* ragged_indptr_h,
        blitz::kernel::PageTable<DType, IdType>& page_table,
        const size_t max_num_pages
    ) override {
        assert(num_tokens);
        //* retrieve embeddings *//
        if (tp_size == 1) {
            CUDA_CHECK(cudaStreamSynchronize(0));
            CUDA_CHECK(cudaMemcpyAsync(
                next_token_ids_d,
                next_token_ids_h,
                num_tokens * sizeof(uint32_t),
                cudaMemcpyHostToDevice
            ));
            CUDA_CHECK(cudaStreamSynchronize(0));
            embed->forward(num_tokens, next_token_ids_d);
            CUDA_CHECK(cudaStreamSynchronize(0));
        } else {
            /// \pre memset for AllReduce
            CUDA_CHECK(cudaMemset(runtime_segment, 0, num_tokens * hidden_size * sizeof(DType)));
            /// \note change token id && mark output location 
            uint32_t lo = vocab_size / tp_size * tp_rank;
            uint32_t hi = vocab_size / tp_size * (tp_rank + 1);
            size_t loc = 0, t = 0;
            for (; t < num_tokens; ++t) {
                if (next_token_ids_h[t] >= lo && next_token_ids_h[t] < hi) {
                    next_token_ids_h[loc] = next_token_ids_h[t] - lo;
                    loc_h[loc++] = t;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(
                next_token_ids_d,
                next_token_ids_h,
                loc * sizeof(IdType),
                cudaMemcpyHostToDevice
            ));
            CUDA_CHECK(cudaMemcpyAsync(
                loc_d,
                loc_h,
                loc * sizeof(IdType),
                cudaMemcpyHostToDevice
            ));
            embed->forward(num_tokens, next_token_ids_d, loc_d);
            ncclResult_t result = ncclAllReduce(
                runtime_segment,
                runtime_segment,
                num_tokens * hidden_size,
                (std::is_same_v<DType, half> ? ncclFloat16 : ncclBfloat16),
                ncclSum,
                nccl_tp_comm_ref,
                0
            );
            if (result != ncclSuccess) {
                BZ_ERROR(
                    "NCCL AllReduce Error: {}",
                    ncclGetErrorString(result)
                );
            }
        }

        if (ragged_indptr_h) {
            //* Prefill phase *//
            prefill_attn_handle.plan(
                attn_float_buffer,
                float_buffer_size,
                attn_int_buffer,
                int_buffer_size,
                ragged_indptr_h,
                ragged_indptr_h,
                page_table.batch_size,
                num_qo_head,
                num_kv_head,
                head_size,
                page_table.page_size
            );
            CUDA_CHECK(cudaStreamSynchronize(0));
            CUDA_CHECK(cudaMemcpyAsync(
                ragged_indptr_d,
                ragged_indptr_h,
                (page_table.batch_size + 1) * sizeof(IdType),
                cudaMemcpyHostToDevice
            ));
            CUDA_CHECK(cudaStreamSynchronize(0));
            for (size_t i = 0; i < num_layers; ++i) {
                tfm_layers[i].forward(num_tokens, ragged_indptr_d, page_table);
                CUDA_CHECK(cudaStreamSynchronize(0));
                page_table.k_data +=
                    max_num_pages * (size_t)page_table.stride_page;
                page_table.v_data +=
                    max_num_pages * (size_t)page_table.stride_page;
            }
        } else {
            //* Decode phase *//
            decode_attn_handle.plan(
                attn_float_buffer,
                float_buffer_size,
                attn_int_buffer,
                int_buffer_size,
                page_table.indptr_h,
                page_table.last_page_len_h,
                page_table.batch_size,
                num_qo_head,
                num_kv_head,
                head_size,
                page_table.page_size
            );
            for (size_t i = 0; i < num_layers; ++i) {
                tfm_layers[i].forward(num_tokens, nullptr, page_table);
                page_table.k_data +=
                    max_num_pages * (size_t)page_table.stride_page;
                page_table.v_data +=
                    max_num_pages * (size_t)page_table.stride_page;
            }
        }

        //* predict logits *//
        // prepare for who's interested in logits
        DType* logits;
        blitz::kernel::rms_norm(
            runtime_segment,
            final_norm_weight,
            runtime_segment,
            num_tokens,
            hidden_size
        );
        lm_head->forward(
            &logits,
            num_tokens,
            page_table.batch_size,
            ragged_indptr_h ? ragged_indptr_d : nullptr
        );

        //* sampler *//
        // prepare for future usability
        blitz::kernel::batch_find_max(
            next_token_ids_d,
            logits,
            page_table.batch_size,
            vocab_size
        );
        CUDA_CHECK(cudaMemcpyAsync(
            next_token_ids_h,
            next_token_ids_d,
            page_table.batch_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));
    }
};

}  // namespace blitz::model
