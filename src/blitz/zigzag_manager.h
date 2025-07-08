#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <thread>
#include <type_traits>

#include "model/llama.h"
#include "activation_manager.h"
#include "batch.h"
#include "generate.pb.h"
#include "include/atomic_channel.hpp"
#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "util/cuda_utils.h"

namespace blitz {

namespace zag {

template<typename DType, typename IdType>
struct Task {
    // work item
    ::blitz::Batch<DType, IdType>& batch;
    DType* activation_slot = nullptr;
    // peer info
    const int peer;
    const uint32_t seq_num;
    const generate::v2::PipeParaInfo& pp_info;
    generate::v2::PipeParaInfo next_pp_info;
    // local state
    std::uint32_t start_layer = 0;
    std::optional<uint32_t> curr_layer = std::nullopt;
    std::promise<decltype(next_pp_info)> promise;

    Task(
        ::blitz::Batch<DType, IdType>& batch,
        int peer,
        uint32_t seq_num,
        const generate::v2::PipeParaInfo& pp_info
    ) :
        batch(batch),
        peer(peer),
        seq_num(seq_num),
        pp_info(pp_info) {
        next_pp_info.CopyFrom(pp_info);
        start_layer = (pp_info.num_layer_per_rank().size() > 0)
            ? *pp_info.num_layer_per_rank().rbegin()
            : 0;
    }
};

// clang-format off
struct relay_t { explicit relay_t() = default; };

struct finish_t { explicit finish_t() = default; };

inline constexpr relay_t relay {};

inline constexpr finish_t finish {};
// clang-format on
} // namespace blitz::zag

template<typename DType, typename IdType>
using Task_t = std::unique_ptr<zag::Task<DType, IdType>>;

template<typename DType, typename IdType>
class ZigzagManager {
  public:
    ZigzagManager(
        const rank_t &rank_,
        const rank_t &device_,
        AChannel<Task_t<DType, IdType>>& task_queue_,
        std::atomic_bool& embed_ready_,
        std::atomic_uint32_t& tfm_layer_ready_,
        std::atomic_bool& lm_head_ready_,
        model::Llama<DType, IdType>& model_,
        ActivationManager<DType, IdType>& act,
        MonotonicStepLocker<uint32_t>& m_stp_lck,
        PriorityStepLocker<std::remove_reference_t<decltype(task_queue_)>>& p_stp_lck,
        IdType* page_indptr_d_
    ):
        rank(rank_),
        device(device_),
        interrupt_residual(rank),
        task_queue(task_queue_),
        embed_ready_ref(embed_ready_),
        lm_head_ready_ref(lm_head_ready_),
        model_ref(model_),
        act_ref(act),
        m_stp_lck_ref(m_stp_lck),
        p_stp_lck_ref(p_stp_lck),
        page_indptr_d(page_indptr_d_) {
        worker = std::thread(std::bind(&ZigzagManager::event_loop_inner, this));
    }

    ~ZigzagManager() {
        shutdown.store(true, std::memory_order::release);
        worker.join();
    }

    /// \brief switch on/off worker thread
    void awake() noexcept {
        awaken.store(true, std::memory_order::release);
        awaken.notify_one();
    }
    void asleep() noexcept {
        awaken.store(false, std::memory_order::release);
    }

    /**
     * \brief atomically rm head if w/ ownership
     *        otherwise doorbell event loop
     *
     * \return seq_num, -1 indicating finish
     */
    std::pair<int64_t, int32_t> interrupt_fst(int peer) {
        // task_queue.lock();
        p_stp_lck_ref.post_wait_lock(3);
        if (!task_queue._list.empty() && task_queue.front()) {
            /**
             * \brief no contention; atomically pop head task and doorbell activation sender
             *
             * \pre curr_task_idx > 0 && size(task_queue) > 1
             */
            using T = Task_t<DType, IdType>;
            T task = std::move(task_queue._list[0]);
            task_queue._list.pop_front();
            curr_task_idx--;
            assert((!task_queue._list[curr_task_idx]));
            task_queue.unlock();

            set_promise(task, peer);
            task->promise.set_value(std::move(task->next_pp_info));
            return {task->batch.id, task->seq_num};
        } else if (task_queue._list.empty()) {
            task_queue.unlock();
            return {-1, -1};
        } else {
            /**
             * \brief contention for head; event loop set seq num
             */
            interrupt_residual.store(peer);
            task_queue.unlock();
            interrupt_residual.wait(peer);
            return {
                relay_batch_id.exchange(-1, std::memory_order::acq_rel),
                relay_seq_num.exchange(-1, std::memory_order::acq_rel)
            };
        }
    }

    std::pair<int64_t, int32_t> interrupt_snd(int peer) {
        // task_queue.lock();
        p_stp_lck_ref.post_wait_lock(3);
        if (!task_queue._list.empty()) {
            assert((!task_queue._list[0] && curr_task_idx == 0));
            /**
             * \brief in refractory period, curr_task_idx == 0
             *
             * \pre curr_task_idx > 0 && size(task_queue) > 1
             */
            using T = Task_t<DType, IdType>;
            auto t = task_queue._list.begin() + 1;
            while (t != task_queue._list.end()
                   && (*t)->curr_layer.value_or(0) == model_ref.num_layers) {
                ++t;
            }
            if (t == task_queue._list.end()) {
                task_queue.unlock();
                BZ_INFO(
                    "Rank<{}> Task zag queue size: {}",
                    rank,
                    task_queue._list.size()
                );
                return {
                    relay_batch_id.exchange(-1, std::memory_order::acq_rel),
                    relay_seq_num.exchange(-1, std::memory_order::acq_rel)
                };
            }
            T task = std::move(*t);
            task_queue._list.erase(t);
            BZ_INFO(
                "Rank<{}> Task zag queue size: {}",
                rank,
                task_queue._list.size()
            );
            task_queue.unlock();

            set_promise(task, peer);
            task->promise.set_value(std::move(task->next_pp_info));
            return {task->batch.id, task->seq_num};
        } else {
            task_queue.unlock();
            BZ_INFO(
                "Rank<{}> Task zag queue size: {}",
                rank,
                task_queue._list.size()
            );
            return {
                relay_batch_id.exchange(-1, std::memory_order::acq_rel),
                relay_seq_num.exchange(-1, std::memory_order::acq_rel)
            };
        }
    }

  private:
    void set_promise(Task_t<DType, IdType>& task, rank_t const peer) {
        if (task->curr_layer.value_or(0) != 0) {
            task->next_pp_info.set_tfm_layer(*(task->curr_layer));
            task->next_pp_info.add_num_layer_per_rank(
                *(task->curr_layer) - task->start_layer
            );
            SendItem<DType> req = {
                task->activation_slot,
                task->batch.num_tokens * (size_t)model_ref.hidden_size
                    * sizeof(DType),
                peer,
                (uint32_t)0
            };
            act_ref.add_send(std::move(req));
        } else {
            task->next_pp_info.set_embedding_layer(1);
            SendItem<DType> req =
                {task->activation_slot, 0, -1, UINT32_MAX};
            act_ref.add_send(std::move(req));
        }
    }
    /**
     * \bug \todo too many side effects
     */
    void prologue(Task_t<DType, IdType>& task) {
        assert(task->batch.is_prefill());
        size_t num_tokens = task->batch.num_tokens;
        assert(num_tokens);

        uint32_t* io_token_ids_buf = model_ref.get_io_token_ids_buf();

        for (auto& tokens : task->batch.all_tokens) {
            std::copy(tokens.begin(), tokens.end(), io_token_ids_buf);
            io_token_ids_buf += tokens.size();
        }
        embed_ready_ref.wait(false, std::memory_order::acquire);
        CUDA_CHECK(cudaMemcpyAsync(
            model_ref.next_token_ids_d,
            model_ref.next_token_ids_h,
            num_tokens * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));

        /// \note copy to residual slot
        model_ref.embed->forward(num_tokens, model_ref.next_token_ids_d, task->activation_slot);
        const_cast<generate::v2::PipeParaInfo&>(task->pp_info).set_tfm_layer(0);
    }

    /**
     * \brief all tfm layers fwded, notify gRPC thrd to generate resp.  
     *
     * \note only called byworker within event loop
     *
     * \pre task != nullptr && task_queue[curr_task_idx] == nullptr
     */
    void epilogue(Task_t<DType, IdType>& task, zag::finish_t) {
        task->next_pp_info.clear_start_layer();

        BZ_WARN("Rank<{}> forced waiting lm head!", rank);
        lm_head_ready_ref.wait(false, std::memory_order::acquire);
        BZ_WARN("Rank<{}> receive lm head!", rank);

        DType* logits;
        /// \todo \bug is correct GPU memory?
        kernel::rms_norm(
            task->activation_slot,
            model_ref.final_norm_weight,
            model_ref.runtime_segment,
            task->batch.num_tokens,
            model_ref.hidden_size
        );
        model_ref.lm_head->forward(
            &logits,
            task->batch.num_tokens,
            task->batch.page_table.batch_size,
            model_ref.ragged_indptr_d
        );
        kernel::batch_find_max(
            model_ref.next_token_ids_d,
            logits,
            task->batch.page_table.batch_size,
            model_ref.vocab_size
        );
        CUDA_CHECK(cudaMemcpy(
            model_ref.next_token_ids_h,
            model_ref.next_token_ids_d,
            task->batch.page_table.batch_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));

        /// \todo \bug next_token_ids_h
        SendItem<DType> req {task->activation_slot, 0, -1, UINT32_MAX};
        act_ref.add_send(std::move(req));
        task->next_pp_info.add_num_layer_per_rank(*(task->curr_layer) - task->start_layer);
        task->promise.set_value(std::move(task->next_pp_info));

        // task_queue.lock();
        p_stp_lck_ref.post_wait_lock(2);
        assert((!task_queue._list[curr_task_idx]));
        auto iter = task_queue._list.begin() + curr_task_idx;
        *iter = std::move(task); // task := nullptr
        iter = task_queue._list.erase(iter); // call Deleter
        if (iter == task_queue._list.end())
            curr_task_idx = -1;
    }

    /**
     * \brief interrupt signal received, relay head task to OldPrefill
     *
     * \note only called by worker within event loop
     *
     * \pre interrupt_residual == peer &&
     *      task != nullptr && task_queue[curr_task_idx] == nullptr && 
     *      curr_task_idx == 0 
     * \post task == nullptr &&
     *       !task_queue.empty() => curr_task_idx == 0 &&
     *       task_queue.empty() => curr_task_idx == -1
     *       `relay_seq_num` is set
     */
    void epilogue(Task_t<DType, IdType>& task, int peer, zag::relay_t) {
        assert((curr_task_idx == 0));
        set_promise(task, peer);
        task->promise.set_value(std::move(task->next_pp_info));
        relay_batch_id.store(task->batch.id, std::memory_order::release);
        relay_seq_num.store(task->seq_num, std::memory_order::release);

        // task_queue.lock();
        p_stp_lck_ref.post_wait_lock(2);
        auto iter = task_queue._list.begin();
        *iter = std::move(task);  // reset task to nullptr
        iter = task_queue._list.erase(iter);  // call Deleter
        if (iter == task_queue._list.end())
            curr_task_idx = -1;
    }

    /**
     * \brief
     * 
     * \pre task != nullptr
     * \post
     */
    bool fwd_one_layer(Task_t<DType, IdType>& task, bool consecutive) {
        BZ_WARN("Rank<{}> batch.id={} fwd layer {}", rank, task->batch.id, *(task->curr_layer));
        assert((task->curr_layer.has_value()));
        /**
         * \pre interrupt not taken
         */
        // prepare page table
        if (!consecutive) {
            CUDA_CHECK(cudaMemcpyAsync(
                page_indptr_d,
                task->batch.page_table.indices_h,
                (task->batch.page_table.indices.size() + task->batch.page_table.indptr.size()
                    + task->batch.page_table.last_page_len.size())
                    * sizeof(IdType),
                cudaMemcpyHostToDevice
            ));
            task->batch.page_table.indices_d = page_indptr_d;
            task->batch.page_table.indptr_d = page_indptr_d + task->batch.page_table.indices.size();
            task->batch.page_table.last_page_len_d =
                task->batch.page_table.indptr_d + task->batch.page_table.indptr.size();
            
            // plan for attention
            model_ref.prefill_attn_handle.plan(
                model_ref.attn_float_buffer,
                model_ref.float_buffer_size,
                model_ref.attn_int_buffer,
                model_ref.int_buffer_size,
                task->batch.ragged_indptr.data(),
                task->batch.ragged_indptr.data(),
                task->batch.page_table.batch_size,
                model_ref.num_qo_head,
                model_ref.num_kv_head,
                model_ref.head_size,
                task->batch.page_table.page_size
            );
            CUDA_CHECK(cudaMemcpyAsync(
                model_ref.ragged_indptr_d,
                task->batch.ragged_indptr.data(),
                (task->batch.page_table.batch_size + 1) * sizeof(IdType),
                cudaMemcpyHostToDevice
            ));            
        }

        // BZ_INFO("Rank<{}> batch[{}] forward layer {}", rank, task->batch.id, *(task->curr_layer));
        auto& model_layer = model_ref.tfm_layers[*(task->curr_layer)];
        auto& self_attn_ref = model_layer.get_self_attn();
        auto& ffn_ref = model_layer.get_ffn();
        self_attn_ref.forward(
            task->batch.num_tokens,
            model_ref.ragged_indptr_d,
            task->batch.page_table,
            task->activation_slot
        );
        /// \note \post no side effect
        CUDA_CHECK(cudaStreamSynchronize(0));

        p_stp_lck_ref.post_wait_lock(2);
        if (int peer = interrupt_residual.load(); peer != rank) {
            task_queue.unlock();
            /**
             * \brief interrupt midway; abort and Redo
             *
             * \invariant let (intrpt = interrupt_residual != rank) in
             *            intrpt => task_queue.front() == nullptr
             */
            assert((!task_queue.front()));
            epilogue(task, peer, zag::relay);
            return false;
        } else {
            task_queue.unlock();
        }

        ffn_ref.forward(task->batch.num_tokens, task->activation_slot);
        /// \note \post no side effect
        CUDA_CHECK(cudaStreamSynchronize(0));
        return true;        
    }

    /**
     * \pre task_queue.lock() &&
     *      task == nullptr && curr_task_idx < len(task_queue)
     *
     * \post \retval True => (task != nullptr && task_queue[curr_task_idx] == nullptr)
     */
    bool sched_next_task(Task_t<DType, IdType>& next_task) {
        assert((!next_task));

        if (curr_task_idx == -1) {
            /**
             * \brief task_queue is likely empty, but new task may arrive
             */
            if (task_queue._list.empty()) {
                return false;
            } else {
                curr_task_idx = 0;
                next_task = std::move(task_queue._list.front());
                task_queue.unlock();
                return true;
            }
        } else {
            next_task = std::move(task_queue._list[curr_task_idx]);
            task_queue.unlock();
            return true;
        }
    }

    /**
     *
     *
     * \pre task_queue.lock() && task != nullptr
     *
     * \post task_queue.unlock() && task != nullptr &&
     *       curr_task_idx == 0
     */
    void preempt_curr_task(Task_t<DType, IdType>& task) {
        task_queue._list[curr_task_idx] = std::move(task);
        task = std::move(task_queue._list.front());
        curr_task_idx = 0;
        task_queue.unlock();
        assert(task);
    }

    void event_loop_inner() {
        CUDA_CHECK(cudaSetDevice(device));
        using T = Task_t<DType, IdType>;
        T task {nullptr};

        /// \todo
        uint32_t num_recv_layer;
        bool consecutive = false;

        while (!shutdown.load(std::memory_order::relaxed)) {
            awaken.wait(false, std::memory_order::acquire);
            
            // task_queue.lock();
            p_stp_lck_ref.post_wait_lock(2);
            while (sched_next_task(task)) {
                /**
                 * \post task_queue[curr_idx_task] == nullptr &&
                 *       task != nullptr
                 */
                fwd_eligible:
                BZ_INFO("Rank<{}> schedule :> batch.id {}", rank, task->batch.id);
                /// \brief new task; put embeddings into residual(activation) slot
                if (!task->curr_layer.has_value()) {
                    prologue(task);
                    task->curr_layer.emplace(0);
                }
                /// \brief haven't forward all transformer layers
                if (*task->curr_layer < model_ref.num_layers) {
                    /**
                     * \pre task_queue[curr_task_idx] == nullptr &&
                     *      task != nullptr
                     */
                    p_stp_lck_ref.post_wait_lock(2);
                    if (int peer = interrupt_residual.load(); peer != rank) {
                        task_queue.unlock();
                        /// \brief interrupt
                        epilogue(task, peer, zag::relay);
                        /// \post task == nullptr && task_queue.lock()
                        // finalize interrupt handling
                        interrupt_residual.store(rank, std::memory_order::release);
                        interrupt_residual.notify_one();
                        consecutive = false;
                        continue;
                    } else {
                        task_queue.unlock();
                    }
                    /**
                     * \pre task != nullptr; interrupt not taken onto head
                     */
                    if (fwd_one_layer(task, consecutive)) {
                        /**
                         * \post task != nullptr; interrupt not taken onto head
                         */
                        (*task->curr_layer)++;
                    } else {
                        /**
                         * \brief interrupt taken onto head
                         *
                         * \post task == nullptr && task_queue.lock()
                         *
                         * \remark epilogue(zag::relay) called inside fwd_one_layer
                         */
                        assert((task == nullptr));
                        interrupt_residual.store(rank, std::memory_order::release);
                        interrupt_residual.notify_one();
                        consecutive = false;
                        continue;
                    }
                }

                /// \brief unhandled interrupt residual signal
                p_stp_lck_ref.post_wait_lock(2);
                if (int peer = interrupt_residual.load(); peer != rank) {
                    task_queue.unlock();
                    if (*(task->curr_layer) < model_ref.num_layers) {
                        epilogue(task, peer, zag::relay);
                        /// \post relay_seq_num == task'->seq_num && 
                        ///       task == nullptr && task_queue.lock()
                        interrupt_residual.store(rank, std::memory_order::release);
                        interrupt_residual.notify_one();
                        consecutive = false;
                        continue;
                    } else {
                        /// \pre `relay_seq_num` == -1
                        interrupt_residual.store(rank, std::memory_order::release);
                        interrupt_residual.notify_one();
                        epilogue(task, zag::finish);
                        /// \post task == nullptr && task_queue.lock()
                        consecutive = false;
                        continue;
                    }
                }
                /// \note keep this lock to sequence transaction
                
                /**
                 * \brief all layers finished, indicate that model is ready
                 *
                 * \pre task != nullptr && no interruption
                 */
                if (*task->curr_layer == model_ref.num_layers) {
                    task_queue.unlock();
                    /// \brief fwd all transformer layers
                    epilogue(task, zag::finish);
                    /// \post task == nullptr && task_queue.lock()
                    /// \note curr_task_idx not changed, but points to the next task
                    consecutive = false;
                    continue;
                }

                num_recv_layer = m_stp_lck_ref.wait();
                // task_queue.lock();
                // p_stp_lck_ref.post_wait_lock(2);
                if (const T& head = task_queue.front();
                    head && num_recv_layer > *(head->curr_layer)) {
                    /**
                     * @brief preempt
                     *
                     * @note other thread may rm head, cmp w/ new head
                     */
                    BZ_INFO(
                        "Rank<{}> (preempt) load.layer={} :> task{}.layer={}",
                        rank,
                        num_recv_layer,
                        head->batch.id,
                        *(head->curr_layer)
                    );
                    assert((curr_task_idx > 0));
                    /// \pre task_queue.lock()
                    preempt_curr_task(task);
                    /// \post task != nullptr && task_queue.unlock()
                    consecutive = false;
                    goto fwd_eligible;
                } else if (num_recv_layer > *(task->curr_layer)) {
                    /**
                     * @brief continue
                     */
                    task_queue.unlock();
                    consecutive = true;
                    goto fwd_eligible;  
                } else {
                    /**
                     * \brief yield, schedule next task
                     *
                     * \pre task_queue.lock();
                     */
                    task_queue._list[curr_task_idx++] = std::move(task); // task := nullptr
                    if ((size_t)curr_task_idx == task_queue.size())
                        curr_task_idx = 0;
                    /// \post task_queue.lock() && task == nullptr
                    consecutive = false;
                }
            }
            if (int peer = interrupt_residual.load(); peer != rank) {
                /// \note notify w/ null value unchanged
                interrupt_residual.store(rank, std::memory_order::release);
                interrupt_residual.notify_one();
            }
            task_queue.unlock();
            std::this_thread::yield();
        }
    }
    std::atomic_bool awaken = false;
    std::atomic_bool shutdown = false;
    std::thread worker;
    // config
    const rank_t &rank;
    const rank_t &device;
    // internal doorbells
    std::atomic_int interrupt_residual;
    std::atomic<int64_t> relay_batch_id = -1;
    std::atomic<int32_t> relay_seq_num = -1;
    // external doorbells
    AChannel<Task_t<DType, IdType>>& task_queue;
    std::atomic_bool& embed_ready_ref;
    std::atomic_bool& lm_head_ready_ref;
    // internal state
    ssize_t curr_task_idx = -1;
    // handle
    model::Llama<DType, IdType>& model_ref;
    ActivationManager<DType, IdType>& act_ref;
    MonotonicStepLocker<uint32_t>& m_stp_lck_ref;
    PriorityStepLocker<std::remove_reference_t<decltype(task_queue)>>& p_stp_lck_ref; 
    IdType* const page_indptr_d;
};

} // namespace blitz