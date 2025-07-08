#pragma once

// clang-format on

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "activation_manager.h"
#include "batch.h"
#include "cache_manager.h"
#include "generate.pb.h"
#include "include/blitz_tccl.h"
#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "migration_manager.h"
#include "model/llama.h"
#include "model_loader.h"
#include "nccl_bcastor.h"
#include "util/cuda_utils.h"
#include "zigzag_manager.h"

/// \deprecated
#include "model/hyper.h"
#include "pickle.h"

namespace blitz {

/// \invariant
static std::atomic_bool param_guard = false;

/// \brief define types to distinguish overloaded funcitons
namespace forward {
// clang-format off
struct normal_t { explicit normal_t() = default; };

struct naivepp_t { explicit naivepp_t() = default; };

struct zag_t { explicit zag_t() = default; };

struct progress_t { explicit progress_t() = default; };

inline constexpr normal_t normal {};

inline constexpr naivepp_t naivepp {};

inline constexpr zag_t zag {};

inline constexpr progress_t progress {};
// clang-format on
}  // namespace forward

template<typename DType, typename IdType>
class Stub {
    constexpr static size_t kDop = 64;
    constexpr static size_t kNumSlot = 16;
  private:
    // stub config
    const rank_t world_size;
    const rank_t rank;
    const rank_t device;
    const char* ib_hca_name;
    // model config
    const uint32_t num_layers;
    const uint32_t num_qo_head;
    const uint32_t num_kv_head;
    const uint32_t head_size;
    const uint32_t hidden_size;
    const uint32_t inter_size;
    const uint32_t vocab_size;
    const float eps = 1e-5;
    const rank_t tp_size;
    const rank_t pp_size;
    // runtime config
    const uint32_t max_batch_size;
    const uint32_t max_batch_tokens;
    // kv_cache config
    const uint32_t page_size;
    const uint32_t max_num_pages;

    // managed runtime memory region
    IdType* page_indptr_d;

    // broadcast config
    MPI_Comm mpi_machine_comm;
    ncclUniqueId machine_comm_id;
    ncclComm_t nccl_machine_comm;
    cudaEvent_t nccl_send_recv_event;
    cudaStream_t send_stream, recv_stream;

    // parallel config
    MPI_Comm mpi_tp_comm;
    ncclUniqueId commId;
    ncclComm_t nccl_tp_comm;
    const rank_t tp_rank;
    const model::GptParallelismParam parallel_config;

    /// \brief parameter transfer
    std::thread param_thrd;

    /// \brief doorbells
    AChannel<Task_t<DType, IdType>> zag_task_queue;
    // model validity marker
    std::atomic_bool embed_ready = false, lm_head_ready = false;
    std::atomic_uint32_t tfm_layer_cnt = 0;

    // members
    std::unique_ptr<model::Llama<DType, IdType>> model;
    std::unique_ptr<CacheManager<DType, IdType>> cache_manager;
    std::unique_ptr<MigrationManager<DType, IdType>> migration_manager;
    std::unique_ptr<model::ModelLoader> model_loader;
    std::unique_ptr<MonotonicStepLocker<uint32_t>> mono_step_locker;
    std::unique_ptr<PriorityStepLocker<decltype(zag_task_queue)>>
        prio_step_locker;
    std::unique_ptr<NcclBcastor> nccl_bcastor;
    std::unique_ptr<ActivationManager<DType, IdType>> activation_manager;
    std::unique_ptr<ZigzagManager<DType, IdType>> zigzag_manager;

  public:
    Stub(
        bool active,
        rank_t world_size_,
        rank_t rank_,
        rank_t device_,
        const char* ib_,
        uint32_t layers_,
        uint32_t qo_head_,
        uint32_t kv_head_,
        uint32_t head_size_,
        uint32_t hidden_size_,
        uint32_t inter_,
        uint32_t vocab_,
        uint32_t batch_size_,
        uint32_t batch_tokens_,
        uint32_t page_size_,
        uint32_t max_num_pages_,
        MPI_Comm machine_comm_,
        MPI_Comm tp_comm_,
        rank_t tp_size_,
        rank_t tp_rank_,
        rank_t pp_size_,  // hasn't supported yet
        std::string model_path,
        std::string model_name
    ) :
        world_size(world_size_),
        rank(rank_),
        device(device_),
        ib_hca_name(ib_),
        num_layers(layers_),
        num_qo_head(qo_head_),
        num_kv_head(kv_head_),
        head_size(head_size_),
        hidden_size(hidden_size_),
        inter_size(inter_),
        vocab_size(vocab_),
        tp_size(tp_size_),
        pp_size(pp_size_),
        max_batch_size(batch_size_),
        max_batch_tokens(batch_tokens_),
        page_size(page_size_),
        max_num_pages(max_num_pages_),
        mpi_machine_comm(machine_comm_),
        mpi_tp_comm(tp_comm_),
        tp_rank(tp_rank_) {
        BZ_INFO(
            "Rank<{}> Tensor Parallelism: [{}/{}]",
            rank,
            this->tp_rank,
            this->tp_size
        );

        cudaSetDevice(device);
        // 64KB
        cudaMalloc(&page_indptr_d, 1024 * 1024);

        model = std::make_unique<model::Llama<DType, IdType>>(
            num_layers,
            num_qo_head,
            num_kv_head,
            head_size,
            hidden_size,
            inter_size,
            vocab_size,
            max_batch_size,
            max_batch_tokens,
            tp_size,
            nccl_tp_comm,
            tp_rank
        );
        cache_manager = std::make_unique<CacheManager<DType, IdType>>(
            num_kv_head / tp_size,
            num_layers,
            head_size,
            page_size,
            max_num_pages
        );
        migration_manager = std::make_unique<MigrationManager<DType, IdType>>(
            world_size,
            rank,
            page_size,
            num_kv_head / tp_size,
            head_size,
            max_num_pages,
            num_layers,
            cache_manager->get_base_ptr()
        );
        model_loader = std::make_unique<model::ModelLoader>(
            device,
            model->weight_segment,
            model->weight_segment_size_in_bytes
        );
        mono_step_locker = std::make_unique<MonotonicStepLocker<uint32_t>>(
            tfm_layer_cnt,
            mpi_tp_comm,
            tp_size,
            tp_rank
        );
        prio_step_locker =
            std::make_unique<PriorityStepLocker<decltype(zag_task_queue)>>(
                zag_task_queue,
                rank,
                mpi_tp_comm,
                tp_size,
                tp_rank
            );
        activation_manager = std::make_unique<ActivationManager<DType, IdType>>(
            rank,
            device,
            kNumSlot,
            model->hidden_size * max_batch_tokens
        );
        zigzag_manager = std::make_unique<ZigzagManager<DType, IdType>>(
            rank,
            device,
            zag_task_queue,
            embed_ready,
            tfm_layer_cnt,
            lm_head_ready,
            *model,
            *activation_manager,
            *mono_step_locker,
            *prio_step_locker,
            page_indptr_d
        );
        init_communicator();
        int machine_size, machine_rank;
        MPI_Comm_rank(mpi_machine_comm, &machine_rank);
        MPI_Comm_size(mpi_machine_comm, &machine_size);
        nccl_bcastor = std::make_unique<NcclBcastor>(
            mpi_machine_comm,
            machine_comm_id,
            rank,
            device,
            machine_rank,
            machine_size,
            num_layers,
            embed_ready,
            model->get_embed_size_in_bytes(),
            tfm_layer_cnt,
            model->get_layer_size_in_bytes(),
            lm_head_ready,
            model->get_lmhead_size_in_bytes(),
            *mono_step_locker
        );
        nccl_bcastor->set_weight_buffer(
            model->weight_segment,
            model->weight_segment_size_in_bytes
        );
        /// \post barrier; nccl_tp_comm != nullptr
        if (active) {
            BZ_INFO("Rank<{}> set status ready", rank);
            set_status_ready();
        }
        /// \todo dynamically start workers
        zigzag_manager->awake();
        activation_manager->awake();
    }

    // get stub's information
    int get_rank() const noexcept {
        return rank;
    }

    int get_device() const noexcept {
        return device;
    }

    uint32_t get_num_layers() const noexcept {
        return num_layers;
    }

    uint32_t get_page_size() const noexcept {
        return page_size;
    }

    uint32_t get_num_qo_heads() const noexcept {
        return num_qo_head;
    }

    uint32_t get_num_kv_heads() const noexcept {
        return num_kv_head;
    }

    uint32_t get_head_size() const noexcept {
        return head_size;
    }

    rank_t get_tp_rank() const noexcept {
        return tp_rank;
    }

    rank_t get_tp_size() const noexcept {
        return tp_size;
    }

    uint32_t get_max_position_embeddings() const noexcept {
        return this->model->get_max_position_embeddings();
    }

    uint32_t get_max_num_pages() const noexcept {
        return cache_manager->get_total_block_num();
    }

    // get stub's member
    DType* get_base_ptr() const {
        return cache_manager->get_base_ptr();
    }

    model::Llama<DType, IdType>& get_model() noexcept {
        return *model;
    }

    CacheManager<DType, IdType>& get_cache_manager() noexcept {
        return *cache_manager;
    }

    MigrationManager<DType, IdType>& get_migration_manager() noexcept {
        return *migration_manager;
    }

    // get stub's status
    bool is_ready_weak() const noexcept {
        return embed_ready.load(std::memory_order::relaxed)
            && tfm_layer_cnt.load(std::memory_order::relaxed) == num_layers
            && lm_head_ready.load(std::memory_order::relaxed);
    }

    bool is_ready_strong() const noexcept {
        return embed_ready.load(std::memory_order::acquire)
            && tfm_layer_cnt.load(std::memory_order::acquire) == num_layers
            && lm_head_ready.load(std::memory_order::acquire);
    }

    void report_status() const noexcept {
        BZ_INFO(
            "Rank<{}> embed {}, layer {}, lmhead {}",
            this->rank,
            embed_ready.load(),
            tfm_layer_cnt.load(),
            lm_head_ready.load()
        );
    }

    std::tuple<bool, uint32_t, bool> get_model_status() noexcept {
        return std::make_tuple(
            embed_ready.load(),
            tfm_layer_cnt.load(),
            lm_head_ready.load()
        );
    }

    /// \brief register MR for GDR && create NCCL comm for TP
    void init_communicator() {
        /// \brief register MR for GDR
        // register parameter
        BZ_INFO(
            "Rank<{}> register mr weight [{}:{}]...",
            rank,
            ::fmt::ptr((void*)model->weight_segment),
            ::fmt::ptr((void*)((char*)model->weight_segment
                               + model->weight_segment_size_in_bytes))
        );
        BlitzTccl::register_mr(
            model->weight_segment,
            model->weight_segment_size_in_bytes
        );
        // register kvcache
        auto [kvcache_buffer, kvcache_length_in_bytes] =
            cache_manager->get_kvcache_segment();
        BZ_INFO(
            "Rank<{}> register mr kv cache [{}:{}]...",
            rank,
            ::fmt::ptr((void*)kvcache_buffer),
            ::fmt::ptr((void*)((char*)kvcache_buffer + kvcache_length_in_bytes))
        );
        BlitzTccl::register_mr(kvcache_buffer, kvcache_length_in_bytes);
        // register runtime segment
        BZ_INFO(
            "Rank<{}> register mr runtime [{}:{}]",
            rank,
            ::fmt::ptr((void*)model->runtime_segment),
            ::fmt::ptr((void*)((char*)model->runtime_segment
                               + model->runtime_segment_size_in_bytes))
        );
        BlitzTccl::register_mr(
            model->runtime_segment,
            model->runtime_segment_size_in_bytes
        );
        // register activation segment
        BZ_INFO(
            "Rank<{}> register mr activation [{}:{}]",
            rank,
            ::fmt::ptr((void*)activation_manager->activation_segment),
            ::fmt::ptr(
                (void*)((char*)activation_manager->activation_segment
                        + activation_manager->num_activation_slot
                            * activation_manager->activation_slot_size_in_bytes)
            )
        );
        BlitzTccl::register_mr(
            activation_manager->activation_segment,
            activation_manager->num_activation_slot
                * activation_manager->activation_slot_size_in_bytes
        );
        /// create NCCL comm for braodcast
        int machine_rank;
        MPI_Comm_rank(mpi_machine_comm, &machine_rank);
        if (machine_rank == 0) {
            ncclGetUniqueId(&machine_comm_id);
        }
        MPI_Bcast(
            &machine_comm_id,
            NCCL_UNIQUE_ID_BYTES,
            MPI_BYTE,
            0,
            mpi_machine_comm
        );
        /// \brief create NCCL comm for TP
        int tp_rank, tp_size;
        MPI_Comm_rank(mpi_tp_comm, &tp_rank);
        MPI_Comm_size(mpi_tp_comm, &tp_size);
        if (tp_rank == 0) {
            ncclGetUniqueId(&commId);
        }
        MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_BYTE, 0, mpi_tp_comm);
        ncclCommInitRank(&nccl_tp_comm, tp_size, commId, tp_rank);
        MPI_Barrier(mpi_tp_comm);
    }

    void nccl_broadcast(std::vector<rank_t>&& item) {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        nccl_bcastor->broadcast(std::move(item), chain);
        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    void tanz_broadcast(std::vector<rank_t>&& item, size_t rt) {
        Assertion(
            (param_guard.load(std::memory_order::acquire)),
            flag::Param or flag::Tanz
        );
        nccl_bcastor->broadcast(std::move(item), tanz, rt);
    }

    /// \brief forward and update KVCache
    void forward(forward::normal_t, Batch<DType, IdType>& batch);

    std::tuple<void*, size_t> get_kvcache_segment() {
        return cache_manager->get_kvcache_segment();
    }

    std::tuple<void*, size_t> get_weight_segment() {
        return model->get_weight_segment();
    }

    /// \brief forward but migrate kvcache progressively
    ///        especially used in PD disaggregation
    void forward(
        forward::normal_t,
        Batch<DType, IdType>& batch,
        forward::progress_t
    );

    /// \brief forward according to naivepp policy
    /// \param pp_info is used to determine whih layer
    void forward(
        forward::naivepp_t,
        Batch<DType, IdType>& batch,
        const generate::v2::PipeParaInfo& pp_info,
        const rank_t peer,
        generate::v2::PipeParaInfo& next_pp_info
    );

    /// \brief forward according to naivepp policy
    ///        while migrate kv_cache progressively
    void forward(
        forward::naivepp_t,
        Batch<DType, IdType>& batch,
        const generate::v2::PipeParaInfo&,
        forward::progress_t
    );

    /// \brief awaken inner worker threads
    void awaken_inner_workers() noexcept {
        zigzag_manager->awake();
        activation_manager->awake();
    }

    /// \brief make inner workers asleep
    void asleep_inner_workers() noexcept {
        zigzag_manager->asleep();
        activation_manager->asleep();
    }

    /// \brief issue parameter transfer
    void send_params_to(const int dst) {
        Assertion((not param_thrd.joinable()), flag::Param);
        Assertion(
            (param_guard.load(std::memory_order::acquire) == false),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        auto __send = [this, dst] {
            CUDA_CHECK(cudaSetDevice(this->device));
            BZ_INFO("Rank<{}> send params to Rank<{}>", rank, dst);
            // std::vector<pickle::Handle> handles;
            size_t weight_length_in_bytes = model->weight_segment_size_in_bytes;
            size_t sent_length_in_bytes = 0;

            char* weight_buffer = (char*)model->weight_segment;
            size_t size_in_bytes = model->get_embed_size_in_bytes();

            // embed
            auto handle = BlitzTccl::TcclSend(
                (void*)weight_buffer,
                size_in_bytes,
                dst,
                2
            );
            handle.wait();
            sent_length_in_bytes += size_in_bytes;
            // layers
            for (size_t i = 0; i < model->get_layer_num(); i++) {
                size_in_bytes = model->get_layer_size_in_bytes();
                auto handle = BlitzTccl::TcclSend(
                    (void*)(weight_buffer + sent_length_in_bytes),
                    size_in_bytes,
                    dst,
                    2
                );
                handle.wait();
                sent_length_in_bytes += size_in_bytes;
            }
            // lm_head
            size_in_bytes = model->get_lmhead_size_in_bytes();
            handle = BlitzTccl::TcclSend(
                (void*)(weight_buffer + sent_length_in_bytes),
                size_in_bytes,
                dst,
                2
            );
            handle.wait();
            sent_length_in_bytes += size_in_bytes;

            assert((sent_length_in_bytes == weight_length_in_bytes));
            BZ_INFO("Rank<{}> parameter send done.", rank);
        };
        param_thrd = std::thread(__send);
    }

    void send_params_in_chain(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index
    ) {
        Assertion((not param_thrd.joinable()), flag::Param);
        Assertion(
            (param_guard.load(std::memory_order::acquire) == false),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        auto __send = [this, chain_index, ranks_in_chain] {
            CUDA_CHECK(cudaSetDevice(this->device));
            std::string chain_str =
                ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
            BZ_INFO(
                "Rank<{}> initiate params sending chain : {}",
                rank,
                chain_str
            );
            // std::vector<pickle::Handle> handles;
            size_t weight_length_in_bytes = model->weight_segment_size_in_bytes;
            size_t sent_length_in_bytes = 0;

            char* weight_buffer = (char*)model->weight_segment;
            size_t size_in_bytes = model->get_embed_size_in_bytes();

            // embed
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            sent_length_in_bytes += size_in_bytes;
            // layers
            for (size_t i = 0; i < model->get_layer_num(); i++) {
                size_in_bytes = model->get_layer_size_in_bytes();
                BlitzTccl::TcclChainMulticast(
                    (void*)weight_buffer,
                    size_in_bytes,
                    2,
                    ranks_in_chain,
                    chain_index
                );
                sent_length_in_bytes += size_in_bytes;
            }
            // lm_head
            size_in_bytes = model->get_lmhead_size_in_bytes();
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            sent_length_in_bytes += size_in_bytes;

            assert((sent_length_in_bytes == weight_length_in_bytes));
            BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
        };
        param_thrd = std::thread(__send);
    }

    /// \brief  issue parameter transfer in P2P RDMA
    ///
    /// \pre    model parameter is unloaded
    ///
    /// \post   model parameter is loaded
    void recv_params_from(const int src) {
        Assertion((not param_thrd.joinable()), flag::Param);
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        auto __recv = [this, src] {
            CUDA_CHECK(cudaSetDevice(this->device));
            BZ_INFO("Rank<{}> recv params from Rank<{}>", rank, src);
            // std::vector<pickle::Handle> handles;
            size_t weight_length_in_bytes = model->weight_segment_size_in_bytes;
            size_t sent_length_in_bytes = 0;

            char* weight_buffer = (char*)model->weight_segment;
            size_t size_in_bytes = model->get_embed_size_in_bytes();

            // embed
            auto handle = BlitzTccl::TcclRecv(
                (void*)weight_buffer,
                size_in_bytes,
                src,
                2
            );
            handle.wait();
            BZ_INFO("Rank<{}> embed ready.", rank);
            embed_ready.store(true, std::memory_order::release);
            embed_ready.notify_all();
            sent_length_in_bytes += size_in_bytes;
            // layers
            for (size_t i = 0; i < model->num_layers; i++) {
                auto begin_ts = std::chrono::system_clock::now();
                size_in_bytes = model->get_layer_size_in_bytes();
                auto handle = BlitzTccl::TcclRecv(
                    (void*)(weight_buffer + sent_length_in_bytes),
                    size_in_bytes,
                    src,
                    2
                );
                handle.wait();
                auto end_ts = std::chrono::system_clock::now();
                auto last_layer_cnt =
                    tfm_layer_cnt.fetch_add(1, std::memory_order::acq_rel);
                mono_step_locker->post();
                auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end_ts - begin_ts
                    );
                BZ_DEBUG(
                    "Rank<{}> tfm layer {} ready, elapse = {}ms",
                    rank,
                    last_layer_cnt + 1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        duration
                    )
                        .count()
                );
                sent_length_in_bytes += size_in_bytes;
                BZ_INFO("Rank<{}> loaded #layers = {}.", rank, last_layer_cnt);
            }
            // lm_head
            size_in_bytes = model->get_lmhead_size_in_bytes();
            handle = BlitzTccl::TcclRecv(
                (void*)(weight_buffer + sent_length_in_bytes),
                size_in_bytes,
                src,
                2
            );
            handle.wait();

            lm_head_ready.store(true, std::memory_order::release);
            lm_head_ready.notify_all();
            sent_length_in_bytes += size_in_bytes;

            assert((sent_length_in_bytes == weight_length_in_bytes));
            BZ_INFO("Rank<{}> parameter recv done.", device);
        };
        param_thrd = std::thread(__recv);
    }

    /// \brief  issue parameter transfer in RDMA chain
    ///
    /// \pre    model parameter is unloaded
    ///
    /// \post   model parameter is loaded
    void recv_params_in_chain(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index
    ) {
        Assertion((not param_thrd.joinable()), flag::Param);
        Assertion(
            (param_guard.load(std::memory_order::acquire) == false),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        auto __recv = [this, chain_index, ranks_in_chain] {
            CUDA_CHECK(cudaSetDevice(this->device));
            std::string chain_str =
                ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
            BZ_INFO("Rank<{}> join params casting chain : {}", rank, chain_str);
            // std::vector<pickle::Handle> handles;
            size_t weight_length_in_bytes = model->weight_segment_size_in_bytes;
            size_t sent_length_in_bytes = 0;

            char* weight_buffer = (char*)model->weight_segment;
            size_t size_in_bytes = model->get_embed_size_in_bytes();

            // embed
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            sent_length_in_bytes += size_in_bytes;
            BZ_INFO("Rank<{}> embed ready.", rank);
            embed_ready.store(true, std::memory_order::release);
            embed_ready.notify_all();
            // layers
            for (size_t i = 0; i < model->num_layers; i++) {
                auto begin_ts = std::chrono::system_clock::now();
                size_in_bytes = model->get_layer_size_in_bytes();
                BlitzTccl::TcclChainMulticast(
                    (void*)weight_buffer,
                    size_in_bytes,
                    2,
                    ranks_in_chain,
                    chain_index
                );
                auto end_ts = std::chrono::system_clock::now();
                auto last_layer_cnt =
                    tfm_layer_cnt.fetch_add(1, std::memory_order::acq_rel);
                mono_step_locker->post();
                auto duration =
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end_ts - begin_ts
                    );
                BZ_DEBUG(
                    "Rank<{}> tfm layer {} ready, elapse = {}ms",
                    rank,
                    last_layer_cnt + 1,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        duration
                    )
                        .count()
                );
                sent_length_in_bytes += size_in_bytes;
                BZ_INFO("Rank<{}> loaded #layers = {}.", rank, last_layer_cnt);
            }
            // lm_head
            size_in_bytes = model->get_lmhead_size_in_bytes();
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            lm_head_ready.store(true, std::memory_order::release);
            lm_head_ready.notify_all();
            sent_length_in_bytes += size_in_bytes;

            assert((sent_length_in_bytes == weight_length_in_bytes));
            BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
        };
        param_thrd = std::thread(__recv);
    }

    // clang-format off
    /// \brief  issue parameter transfer in RDMA chain
    void rdma_cast_params_in_chain_w_notify(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index,
        tanz_t,
        size_t const nlane,
        size_t const lane_id
    ) {
        Assertion((not param_thrd.joinable()), flag::Param);
        Assertion(
            (param_guard.load(std::memory_order::acquire) == true),
            flag::Param
        );
        auto __recv = [this, chain_index, ranks_in_chain, nlane, lane_id] {
            CUDA_CHECK(cudaSetDevice(this->device));
            std::string chain_str =
                ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
            BZ_INFO("Rank<{}> join params casting chain : {}", rank, chain_str);

            // clang-format on
            auto chain_multicast_notify_nccl = [this, nlane, lane_id](
                                                   void* buffer,
                                                   size_t length,
                                                   const std::vector<int>&
                                                       ranks_in_chain,
                                                   size_t chain_index
                                               ) {
                using namespace pickle;
                constexpr size_t strip_segment_size_in_bytes = 64 * 1024 * 1024;
                constexpr size_t rdma_chunk_bytes = 0x200000;
                constexpr size_t freq_notify =
                    strip_segment_size_in_bytes / rdma_chunk_bytes;
                constexpr size_t num_handles = 16;

                Assertion(
                    (strip_segment_size_in_bytes
                         % (num_handles * rdma_chunk_bytes)
                     == 0),
                    flag::Tanz
                );
                /// \note add stripping
                /// \note DON'T forget the last chunk
                /// \note primary Tanz chain <-> lane_id == 0 <-> fst_buf == buf
                size_t const segment_size_in_bytes =
                    strip_segment_size_in_bytes * nlane;
                size_t const num_segment = length / segment_size_in_bytes;
                size_t const remain_bytes =
                    length - num_segment * segment_size_in_bytes;
                char* const base = static_cast<char*>(buffer)
                    + lane_id * strip_segment_size_in_bytes;
                char* const buf_bound = static_cast<char*>(buffer) + length
                    - remain_bytes
                    - (nlane - lane_id - 1) * strip_segment_size_in_bytes;

                if (chain_index == 0) {
                    /// \brief SendingDecode as source
                    Assertion((rank == ranks_in_chain.front()), flag::Tanz);
                    std::vector<Handle> handles;
                    handles.reserve(num_handles);
                    char* buf = base;
                    size_t i = 0;
                    for (; i < num_handles; ++i) {
                        handles.emplace_back(BlitzTccl::TcclSend(
                            buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index + 1],
                            2
                        ));
                        buf += rdma_chunk_bytes;
                    }
                    Assertion((i < freq_notify), flag::Tanz);
                    while (buf < buf_bound) {
                        handles[i % num_handles].wait();
                        if ((i % freq_notify) == 0) {
                            /// \note \divisible rdma_chunk stripped_segment
                            ///         -> \devisible i freq_notify
                            ///         -> stripped_segment is sent
                            ///         -> step into stripped_segment in next segment
                            buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        handles[i % num_handles] = BlitzTccl::TcclSend(
                            buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index + 1],
                            2
                        );
                        buf += rdma_chunk_bytes;
                        i++;
                    }
                    // BZ_WARN("Rank<{}> (rdma) Hahaha.0, i = {}", rank, i);
                    if (lane_id == 0) {
                        /// \note process the remaining segment data
                        buf += (nlane - 1) * strip_segment_size_in_bytes;
                        // bug: base + length is not the end of the buffer
                        Assertion(
                            (buf + remain_bytes == base + length),
                            flag::Tanz
                        );
                        while (buf < base + length) {
                            handles[i % num_handles].wait();
                            size_t size = std::min(
                                (size_t)(base + length - buf),
                                rdma_chunk_bytes
                            );
                            handles[i % num_handles] = BlitzTccl::TcclSend(
                                buf,
                                size,
                                ranks_in_chain[chain_index + 1],
                                2
                            );
                            buf += rdma_chunk_bytes;
                            i++;
                        }
                    }
                    handles[(--i) % num_handles].wait();
                } else if (chain_index == ranks_in_chain.size() - 1) {
                    Assertion((rank == ranks_in_chain.back()), flag::Tanz);
                    std::vector<Handle> handles;
                    handles.reserve(num_handles);
                    size_t i = 0;
                    char* buf = base;
                    /// \note Receiver as initiator in RDMA comm impl
                    for (; i < num_handles; ++i) {
                        handles.emplace_back(BlitzTccl::TcclRecv(
                            buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index - 1],
                            2
                        ));
                        buf += rdma_chunk_bytes;
                    }
                    // no left RDMA semaphore wait/notify
                    Assertion((i < freq_notify), flag::Tanz);
                    while (buf < buf_bound) {
                        handles[i % num_handles].wait();
                        if ((i % freq_notify) == 0) {
                            this->nccl_bcastor->notify_by_rdma();
                            /// \note \divisible rdma_chunk stripped_segment
                            ///         -> \devisible i freq_notify
                            ///         -> stripped_segment is sent
                            ///         -> step into stripped_segment in next segment
                            buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        handles[i % num_handles] = BlitzTccl::TcclRecv(
                            buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index - 1],
                            2
                        );
                        buf += rdma_chunk_bytes;
                        i++;
                    }
                    Assertion(
                        (buf == buf_bound and i % freq_notify == 0),
                        flag::Tanz
                    );
                    this->nccl_bcastor->notify_by_rdma();
                    if (lane_id == 0) {
                        BZ_WARN("Rank<{}> lane_id <{}>", rank, lane_id);
                        buf += (nlane - 1) * strip_segment_size_in_bytes;
                        /// \note process the remaining segment data
                        Assertion(
                            (buf + remain_bytes == base + length),
                            flag::Tanz
                        );
                        while (buf < base + length) {
                            // BZ_WARN("Rank<{}> i = {}", rank, i);
                            handles[i % num_handles].wait();
                            size_t size = std::min(
                                static_cast<size_t>((base + length) - buf),
                                rdma_chunk_bytes
                            );
                            handles[i % num_handles] = BlitzTccl::TcclRecv(
                                buf,
                                size,
                                ranks_in_chain[chain_index - 1],
                                2
                            );
                            buf += size;
                            i++;
                        }
                        this->nccl_bcastor->notify_by_rdma();
                    }
                    // BZ_WARN("Rank<{}> (rdma) Hahaha.2 i = {}", rank, i);
                    /// \note the last emplaced handle -> the last handle
                    handles[(--i) % num_handles].wait();
                } else {
                    std::vector<Handle> handles;
                    Handle last_send_handle;
                    handles.reserve(num_handles);
                    size_t i = 0, j = 0;
                    char* recv_buf = base;
                    char* send_buf = base;
                    /// \note Receiver as initiator in RDMA comm impl
                    for (; i < num_handles; ++i) {
                        handles.emplace_back(BlitzTccl::TcclRecv(
                            recv_buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index - 1],
                            2
                        ));
                        recv_buf += rdma_chunk_bytes;
                    }
                    Assertion((i < freq_notify), flag::Tanz);
                    while (recv_buf < buf_bound) {
                        handles[i % num_handles].wait();
                        if ((i % freq_notify) == 0) {
                            this->nccl_bcastor->notify_by_rdma();
                            recv_buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        last_send_handle = BlitzTccl::TcclSend(
                            send_buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index + 1],
                            2
                        );
                        send_buf += rdma_chunk_bytes;
                        j++;
                        if ((j % freq_notify) == 0) {
                            send_buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        handles[i % num_handles] = BlitzTccl::TcclRecv(
                            recv_buf,
                            rdma_chunk_bytes,
                            ranks_in_chain[chain_index - 1],
                            2
                        );
                        recv_buf += rdma_chunk_bytes;
                        i++;
                    }
                    Assertion(
                        (recv_buf == buf_bound and i % freq_notify == 0),
                        flag::Tanz
                    );
                    this->nccl_bcastor->notify_by_rdma();
                    const size_t i0 = i;
                    if (lane_id == 0) {
                        recv_buf += (nlane - 1) * strip_segment_size_in_bytes;
                        /// \note process the remaining segment data
                        Assertion(
                            (recv_buf + remain_bytes == base + length),
                            flag::Tanz
                        );
                        while (recv_buf < base + length) {
                            handles[i % num_handles].wait();
                            size_t send_size = std::min(
                                (size_t)(base + length - send_buf),
                                rdma_chunk_bytes
                            );
                            last_send_handle = BlitzTccl::TcclSend(
                                send_buf,
                                send_size,
                                ranks_in_chain[chain_index + 1],
                                2
                            );
                            send_buf += send_size;
                            j++;
                            if ((j % freq_notify) == 0 && j <= i0) {
                                send_buf += segment_size_in_bytes
                                    - strip_segment_size_in_bytes;
                            }
                            size_t recv_size = std::min(
                                (size_t)(base + length - recv_buf),
                                rdma_chunk_bytes
                            );
                            handles[i % num_handles] = BlitzTccl::TcclRecv(
                                recv_buf,
                                recv_size,
                                ranks_in_chain[chain_index - 1],
                                2
                            );
                            recv_buf += recv_size;
                            i++;
                        }
                        // BZ_WARN(
                        //     "Rank<{}> (rdma) Hahaha.1 i = {}, j = {}",
                        //     rank,
                        //     i,
                        //     j
                        // );
                        this->nccl_bcastor->notify_by_rdma();
                    }
                    /// \pre j - i == num_handles
                    while (j < i) {
                        handles[j % num_handles].wait();
                        if ((j % freq_notify) == 0 && j <= i0) {
                            send_buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        size_t send_size = std::min(
                            (size_t)(base + length - send_buf),
                            rdma_chunk_bytes
                        );
                        last_send_handle = BlitzTccl::TcclSend(
                            send_buf,
                            send_size,
                            ranks_in_chain[chain_index + 1],
                            2
                        );
                        send_buf += send_size;
                        j++;
                    }
                    handles[(--i) % num_handles].wait();
                    last_send_handle.wait();
                }
            };
            // clang-format on

            /// \todo \bug forget to skip!
            char* weight_buffer = (char*)model->weight_segment;
            size_t weight_length_in_bytes = model->weight_segment_size_in_bytes;

            chain_multicast_notify_nccl(
                weight_buffer,
                weight_length_in_bytes,
                ranks_in_chain,
                chain_index
            );
            BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
        };
        param_thrd = std::thread(__recv);
    }

    bool join_params_thread() {
        if (param_thrd.joinable()) {
            param_thrd.join();
            Assertion(
                param_guard.load(std::memory_order::acquire),
                flag::Param
            );
            param_guard.store(false, std::memory_order::release);
            BZ_INFO("Rank<{}> (RDMA) parameter worker joined.", rank);
            return true;
        } else {
            return false;
        }
    }

    void reset_status() noexcept {
        /// \pre model parameter is loaded
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        Assertion((embed_ready.load(std::memory_order::acquire)), flag::Param);
        Assertion(
            (tfm_layer_cnt.load(std::memory_order::acquire) == num_layers),
            flag::Param
        );
        Assertion(
            (lm_head_ready.load(std::memory_order::acquire)),
            flag::Param
        );

        embed_ready.store(false, std::memory_order::release);
        tfm_layer_cnt.store(0, std::memory_order::release);
        lm_head_ready.store(false, std::memory_order::release);
        mono_step_locker->update();
        Assertion(nccl_bcastor->is_empty(tanz), flag::Param && flag::Tanz);
    }

    /// \brief calculate model parameter status invariant
    bool status_invariant() noexcept {
        bool embedding_layer = embed_ready.load(std::memory_order::acquire);
        auto num_transformer_layers =
            tfm_layer_cnt.load(std::memory_order::acquire);
        bool lm_head = lm_head_ready.load(std::memory_order::acquire);
        bool loaded = {
            lm_head && num_transformer_layers == this->num_layers
            && embedding_layer
        };
        bool unloaded = {
            !lm_head
            && ((!embedding_layer && num_transformer_layers == 0)
                || (num_transformer_layers > 0))
        };
        return num_transformer_layers <= this->num_layers
            && (loaded || unloaded) && !(loaded && unloaded);
    }

    /// \pre model parameter is unloaded
    void load_params_from_disk(std::string model_path = "", int device = 0) {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        assert((model_path != ""));
        BZ_INFO("Rank<{}> load model from disk: {}", rank, model_path);
        param_guard.store(true, std::memory_order::release);
        model_loader->load_model_host_fast(model_path);
        model_loader->load_model_gpu();

        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    void load_params_from_host_memory() {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        param_guard.store(true, std::memory_order::release);
        model_loader->load_model_gpu();

        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    /// \pre model parameter is unloaded
    void set_status_ready() noexcept {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        this->embed_ready.store(true, std::memory_order::release);
        this->tfm_layer_cnt.store(num_layers, std::memory_order::release);
        this->lm_head_ready.store(true, std::memory_order::release);
        this->mono_step_locker->update();
        Assertion(nccl_bcastor->is_empty(tanz), flag::Param && flag::Tanz);
    }

    /// \brief acquire the param guard, used in Tanz
    void get_status_mutex() noexcept {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
    }

    /// \brief used by gRPC thread to add task
    std::future<generate::v2::PipeParaInfo> add_zag_task(
        Batch<DType, IdType>& batch_mref,
        uint32_t peer,
        uint32_t seq_num,
        const generate::v2::PipeParaInfo& pp_info
    );

    std::pair<int64_t, int32_t> relay_zag_task(int peer, bool relax_head) {
        if (!relax_head) {
            return zigzag_manager->interrupt_fst(peer);
        } else {
            return zigzag_manager->interrupt_snd(peer);
        }
    }
};

template<typename DType, typename IdType>
void Stub<DType, IdType>::forward(
    forward::normal_t,
    Batch<DType, IdType>& batch
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
    uint32_t* next_token_buf = model->get_io_token_ids_buf();

    /// prefill
    if (batch.is_prefill()) {
        for (auto& tokens : batch.all_tokens) {
            std::copy(tokens.begin(), tokens.end(), next_token_buf);
            next_token_buf += tokens.size();
        }
        model->forward(
            batch.num_tokens,
            batch.ragged_indptr.data(),
            batch.page_table,
            cache_manager->get_total_block_num()
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
        model->forward(
            batch.batch_size,
            nullptr,
            batch.page_table,
            cache_manager->get_total_block_num()
        );
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    // reset page table
    batch.page_table.k_data = cache_manager->get_base_ptr();
    batch.page_table.v_data =
        batch.page_table.k_data + batch.page_table.stride_page / 2;

    // retrive output
    std::vector<uint32_t> output_tokens {
        model->get_io_token_ids_buf(),
        model->get_io_token_ids_buf() + batch.batch_size
    };
    batch.update(output_tokens);

    // clean PageTable
    batch.page_table.indices_d = nullptr;
    batch.page_table.indptr_d = nullptr;
    batch.page_table.last_page_len_d = nullptr;
}

template<typename DType, typename IdType>
void Stub<DType, IdType>::forward(
    forward::naivepp_t,
    Batch<DType, IdType>& batch,
    const generate::v2::PipeParaInfo& pp_info,
    const rank_t peer,
    generate::v2::PipeParaInfo& next_pp_info
) {
    assert(pp_info.has_tfm_layer());
    BZ_INFO(
        "Rank<{}> Batch[{}] do forward from layer#{}",
        rank,
        batch.id,
        pp_info.tfm_layer()
    );
    /// \bug \todo Xia JB Suan
    ///            fix after submission

    /// \todo unfinished send

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
    uint32_t* next_token_buf = model->get_io_token_ids_buf();
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
            model->next_token_ids_d,
            model->next_token_ids_h,
            batch.num_tokens * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));
        model->embed->forward(batch.num_tokens, model->next_token_ids_d);
        const_cast<generate::v2::PipeParaInfo&>(pp_info).set_tfm_layer(0);
        {
            CUDA_CHECK(cudaStreamSynchronize(0));
            BZ_DEBUG("Rank<{}> after embedding layer", rank);
        }
        // receive activation
    } else {
        BlitzTccl::TcclRecv(
            model->runtime_segment,
            batch.num_tokens * (size_t)hidden_size * sizeof(DType),
            peer,
            0
        );
        { BZ_DEBUG("Rank<{}> receive activation.", rank); }
    }

    // plan for attention
    model->prefill_attn_handle.plan(
        model->attn_float_buffer,
        model->float_buffer_size,
        model->attn_int_buffer,
        model->int_buffer_size,
        batch.ragged_indptr.data(),
        batch.ragged_indptr.data(),
        batch.page_table.batch_size,
        num_qo_head,
        num_kv_head,
        head_size,
        batch.page_table.page_size
    );
    CUDA_CHECK(cudaMemcpyAsync(
        model->ragged_indptr_d,
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
            model->tfm_layers[layer_id].forward(
                batch.num_tokens,
                model->ragged_indptr_d,
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
            model->runtime_segment,
            model->final_norm_weight,
            model->runtime_segment,
            batch.num_tokens,
            hidden_size
        );
        model->lm_head->forward(
            &logits,
            batch.num_tokens,
            batch.page_table.batch_size,
            model->ragged_indptr_d
        );
        kernel::batch_find_max(
            model->next_token_ids_d,
            logits,
            batch.page_table.batch_size,
            vocab_size
        );
        CUDA_CHECK(cudaMemcpyAsync(
            model->next_token_ids_h,
            model->next_token_ids_d,
            batch.page_table.batch_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));
        next_pp_info.clear_start_layer();
    } else {
        assert((next_pp_info.has_tfm_layer() || next_pp_info.has_lm_head()));
        /// \todo put activations to activation slot
        DType* ptr;
        activation_manager->acquire_activation_slot(&ptr);
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
        activation_manager->add_send(std::move(req));
    }

    batch.page_table.k_data = cache_manager->get_base_ptr();
    batch.page_table.v_data =
        batch.page_table.k_data + batch.page_table.stride_page / 2;

    if (next_pp_info.start_layer_case() == next_pp_info.START_LAYER_NOT_SET) {
        // retrive output
        std::vector<uint32_t> output_tokens {
            model->get_io_token_ids_buf(),
            model->get_io_token_ids_buf() + batch.batch_size
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
std::future<generate::v2::PipeParaInfo> Stub<DType, IdType>::add_zag_task(
    Batch<DType, IdType>& batch_mref,
    uint32_t peer,
    uint32_t seq_num,
    const generate::v2::PipeParaInfo& pp_info
) {
    using T = zag::Task<DType, IdType>;
    Task_t<DType, IdType> task =
        std::make_unique<T>(batch_mref, peer, seq_num, pp_info);
    activation_manager->acquire_activation_slot(&(task->activation_slot));
    auto future = task->promise.get_future();
    // zag_task_queue.push_back(std::move(task));
    {
        prio_step_locker->post_wait_lock(1);
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

}  // namespace blitz

// clang-format on