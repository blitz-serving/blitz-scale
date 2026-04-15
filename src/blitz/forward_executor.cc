#include "forward_executor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "util/cuda_utils.h"

namespace blitz {

template<typename DType, typename IdType>
void ForwardExecutor<DType, IdType>::forward(
    forward::normal_t,
    Batch<DType, IdType>& batch,
    rank_t rank,
    uint32_t max_num_pages,
    uint32_t num_qo_head,
    uint32_t num_kv_head,
    uint32_t head_size,
    uint32_t hidden_size,
    uint32_t vocab_size,
    IdType* page_indptr_d,
    model::Llama<DType, IdType>& model,
    CacheManager<DType, IdType>& cache_manager
) {
    /// prepare page table
    auto error = cudaMemcpyAsync(
        page_indptr_d,
        batch.page_table.indices_h,
        (batch.page_table.indices.size() + batch.page_table.indptr.size()
         + batch.page_table.last_page_len.size())
            * sizeof(IdType),
        cudaMemcpyHostToDevice
    );
    if (error != cudaSuccess) {
        BZ_INFO(
            "indicies size: {}, indptr size: {}, last_page_len size: {}",
            batch.page_table.indices.size(),
            batch.page_table.indptr.size(),
            batch.page_table.last_page_len.size()
        );
        BZ_FATAL(
            "Rank<{}> Batch[{}] cudaMemcpyAsync failed: {}",
            rank,
            batch.id,
            cudaGetErrorString(error)
        );
    }
    batch.page_table.indices_d = page_indptr_d;
    batch.page_table.indptr_d = page_indptr_d + batch.page_table.indices.size();
    batch.page_table.last_page_len_d =
        batch.page_table.indptr_d + batch.page_table.indptr.size();

    /// prepare input
    uint32_t* next_token_buf = model.get_io_token_ids_buf();

    /// prefill
    if (batch.is_prefill()) {
        for (auto& tokens : batch.all_tokens) {
            std::copy(tokens.begin(), tokens.end(), next_token_buf);
            next_token_buf += tokens.size();
        }
        model.forward(
            batch.num_tokens,
            batch.ragged_indptr.data(),
            batch.page_table,
            cache_manager.get_total_block_num()
        );
        CUDA_CHECK(cudaStreamSynchronize(0));
        batch.ragged_indptr.clear();
        batch.num_tokens = 0;
        /// decode
    } else {
        assert((batch.all_tokens.size() == batch.size()));
        for (size_t i = 0; i < batch.all_tokens.size(); ++i) {
            next_token_buf[i] = batch.all_tokens[i].back();
        }
        model.forward(
            batch.batch_size,
            nullptr,
            batch.page_table,
            cache_manager.get_total_block_num()
        );
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // reset page table
    batch.page_table.k_data = cache_manager.get_base_ptr();
    batch.page_table.v_data =
        batch.page_table.k_data + batch.page_table.stride_page / 2;

    // retrive output
    std::vector<uint32_t> output_tokens {
        model.get_io_token_ids_buf(),
        model.get_io_token_ids_buf() + batch.batch_size
    };
    batch.update(output_tokens);

    // clean PageTable
    batch.page_table.indices_d = nullptr;
    batch.page_table.indptr_d = nullptr;
    batch.page_table.last_page_len_d = nullptr;
}

template<typename DType, typename IdType>
void ForwardExecutor<DType, IdType>::forward(
    forward::naivepp_t,
    Batch<DType, IdType>& batch,
    const generate::v2::PipeParaInfo& pp_info,
    const rank_t peer,
    generate::v2::PipeParaInfo& next_pp_info,
    rank_t rank,
    uint32_t num_layers,
    uint32_t max_num_pages,
    uint32_t num_qo_head,
    uint32_t num_kv_head,
    uint32_t head_size,
    uint32_t hidden_size,
    uint32_t vocab_size,
    IdType* page_indptr_d,
    model::Llama<DType, IdType>& model,
    CacheManager<DType, IdType>& cache_manager,
    ActivationManager<DType, IdType>& activation_manager,
    std::atomic_bool& embed_ready,
    std::atomic_uint32_t& tfm_layer_cnt,
    std::atomic_bool& lm_head_ready
) {
    assert(pp_info.has_tfm_layer());
    BZ_INFO(
        "Rank<{}> Batch[{}] do forward from layer#{}",
        rank,
        batch.id,
        pp_info.tfm_layer()
    );

    /// prepare page table
    CUDA_CHECK(cudaMemcpyAsync(
        page_indptr_d,
        batch.page_table.indices_h,
        (batch.page_table.indices.size() + batch.page_table.indptr.size()
         + batch.page_table.last_page_len.size())
            * sizeof(IdType),
        cudaMemcpyHostToDevice
    ));
    batch.page_table.indices_d = page_indptr_d;
    batch.page_table.indptr_d = page_indptr_d + batch.page_table.indices.size();
    batch.page_table.last_page_len_d =
        batch.page_table.indptr_d + batch.page_table.indptr.size();

    /// prepare input
    uint32_t* next_token_buf = model.get_io_token_ids_buf();
    assert((batch.is_prefill()));
    // retrieve from Embedding table
    if (pp_info.has_embedding_layer()) {
        BZ_DEBUG("Must embedding layer...");
        embed_ready.wait(false, std::memory_order::acquire);
        for (auto& tokens : batch.all_tokens) {
            std::copy(tokens.begin(), tokens.end(), next_token_buf);
            next_token_buf += tokens.size();
        }
        CUDA_CHECK(cudaMemcpyAsync(
            model.next_token_ids_d,
            model.next_token_ids_h,
            batch.num_tokens * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
        model.embed->forward(batch.num_tokens, model.next_token_ids_d);
        const_cast<generate::v2::PipeParaInfo&>(pp_info).set_tfm_layer(0);
        {
            CUDA_CHECK(cudaStreamSynchronize(0));
            BZ_DEBUG("Rank<{}> after embedding layer", rank);
        }
        // receive activation
    } else {
        BlitzTccl::TcclRecv(
            model.runtime_segment,
            batch.num_tokens * (size_t)hidden_size * sizeof(DType),
            peer,
            0
        );
        { BZ_DEBUG("Rank<{}> receive activation.", rank); }
    }

    // plan for attention
    model.prefill_attn_handle.plan(
        model.attn_float_buffer,
        model.float_buffer_size,
        model.attn_int_buffer,
        model.int_buffer_size,
        batch.ragged_indptr.data(),
        batch.ragged_indptr.data(),
        batch.page_table.batch_size,
        num_qo_head,
        num_kv_head,
        head_size,
        batch.page_table.page_size
    );
    CUDA_CHECK(cudaMemcpyAsync(
        model.ragged_indptr_d,
        batch.ragged_indptr.data(),
        (batch.page_table.batch_size + 1) * sizeof(IdType),
        cudaMemcpyHostToDevice
    ));

    // forward pass
    next_pp_info.CopyFrom(pp_info);
    if (!pp_info.has_lm_head()) {
        assert((pp_info.has_tfm_layer()));
        for (size_t layer_id = pp_info.tfm_layer(); layer_id
             < std::min(tfm_layer_cnt.load(std::memory_order::acquire),
                        num_layers);
             next_pp_info.set_tfm_layer(++layer_id)) {
            model.tfm_layers[layer_id].forward(
                batch.num_tokens,
                model.ragged_indptr_d,
                batch.page_table
            );
            CUDA_CHECK(cudaStreamSynchronize(0));
            batch.page_table.k_data +=
                max_num_pages * batch.page_table.stride_page;
            batch.page_table.v_data +=
                max_num_pages * batch.page_table.stride_page;
        }
        next_pp_info.add_num_layer_per_rank(
            next_pp_info.tfm_layer() - pp_info.tfm_layer()
        );
        BZ_DEBUG(
            "Rank<{}> forward tfm layer = {}",
            rank,
            *(next_pp_info.num_layer_per_rank().end() - 1)
        );
    } else {
        BZ_INFO("Rank<{}> no tfm layers to calculate", rank);
        next_pp_info.add_num_layer_per_rank(0);
    }
    //
    assert(
        (next_pp_info.num_layer_per_rank_size()
         == pp_info.num_layer_per_rank_size() + 1)
    );
    uint32_t next_forward_tfm_layer = next_pp_info.tfm_layer();
    if (next_forward_tfm_layer == num_layers) {
        assert(
            std::accumulate(
                next_pp_info.num_layer_per_rank().begin(),
                next_pp_info.num_layer_per_rank().end(),
                0u
            )
            == num_layers
        );
        next_pp_info.set_lm_head(1);
    }

    if (pp_info.has_lm_head()) {
        lm_head_ready.wait(false, std::memory_order::acquire);
    }
    if (lm_head_ready.load(std::memory_order::acquire)) {
        BZ_INFO("Rank<{}> LmHead Ready, do lm_head forward", rank);
        DType* logits;
        kernel::rms_norm(
            model.runtime_segment,
            model.final_norm_weight,
            model.runtime_segment,
            batch.num_tokens,
            hidden_size
        );
        model.lm_head->forward(
            &logits,
            batch.num_tokens,
            batch.page_table.batch_size,
            model.ragged_indptr_d
        );
        kernel::batch_find_max(
            model.next_token_ids_d,
            logits,
            batch.page_table.batch_size,
            vocab_size
        );
        CUDA_CHECK(cudaMemcpyAsync(
            model.next_token_ids_h,
            model.next_token_ids_d,
            batch.page_table.batch_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));
        next_pp_info.clear_start_layer();
    } else {
        assert((next_pp_info.has_tfm_layer() || next_pp_info.has_lm_head()));
        DType* ptr;
        activation_manager.acquire_activation_slot(&ptr);
        SendItem<DType> req = {
            ptr,
            batch.num_tokens * (size_t)hidden_size * sizeof(DType),
            peer,
            (uint32_t)0
        };
        BZ_INFO(
            "Rank<{}> add send request [{}:{}]",
            rank,
            ::fmt::ptr((void*)req.buffer),
            ::fmt::ptr((void*)((char*)req.buffer + req.length_in_bytes))
        );
        activation_manager.add_send(std::move(req));
    }

    batch.page_table.k_data = cache_manager.get_base_ptr();
    batch.page_table.v_data =
        batch.page_table.k_data + batch.page_table.stride_page / 2;

    if (next_pp_info.start_layer_case() == next_pp_info.START_LAYER_NOT_SET) {
        // retrive output
        std::vector<uint32_t> output_tokens {
            model.get_io_token_ids_buf(),
            model.get_io_token_ids_buf() + batch.batch_size
        };
        batch.update(output_tokens);
        /// \note \bug change state
        batch.ragged_indptr.clear();
        batch.num_tokens = 0;
    }
    // clean PageTable
    batch.page_table.indices_d = nullptr;
    batch.page_table.indptr_d = nullptr;
    batch.page_table.last_page_len_d = nullptr;
}

template<typename DType, typename IdType>
std::future<generate::v2::PipeParaInfo>
ForwardExecutor<DType, IdType>::add_zag_task(
    Batch<DType, IdType>& batch_mref,
    uint32_t peer,
    uint32_t seq_num,
    const generate::v2::PipeParaInfo& pp_info,
    rank_t rank,
    ActivationManager<DType, IdType>& activation_manager,
    AChannel<Task_t<DType, IdType>>& zag_task_queue,
    PriorityStepLocker<AChannel<Task_t<DType, IdType>>>& prio_step_locker
) {
    using T = zag::Task<DType, IdType>;
    Task_t<DType, IdType> task =
        std::make_unique<T>(batch_mref, peer, seq_num, pp_info);
    activation_manager.acquire_activation_slot(&(task->activation_slot));
    auto future = task->promise.get_future();
    {
        prio_step_locker.post_wait_lock(1);
        zag_task_queue._list.push_back(std::move(task));
        BZ_INFO(
            "Rank<{}> Task zag queue size: {}",
            rank,
            zag_task_queue._list.size()
        );
        zag_task_queue.unlock();
    }
    return future;
}

// Explicit template instantiation
template class ForwardExecutor<half, int32_t>;

}  // namespace blitz
