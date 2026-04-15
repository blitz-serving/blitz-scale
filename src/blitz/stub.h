#pragma once

// clang-format on

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "activation_manager.h"
#include "batch.h"
#include "cache_manager.h"
#include "comm_bootstrap.h"
#include "blitz_proto.h"
#include "forward_executor.h"
#include "include/blitz_tccl.h"
#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "migration_manager.h"
#include "model/llama.h"
#include "model_loader.h"
#include "model_status.h"
#include "nccl_bcastor.h"
#include "param_transfer.h"
#include "stub_config.h"
#include "util/cuda_utils.h"
#include "zigzag_manager.h"

/// \deprecated
#include "model/hyper.h"
#include "pickle.h"

namespace blitz {

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

    // parallel config
    const rank_t tp_rank;
    const model::GptParallelismParam parallel_config;

    /// \brief parameter transfer thread
    std::thread param_thrd;

    /// \brief doorbells
    AChannel<Task_t<DType, IdType>> zag_task_queue;

    // sub-components
    ModelStatus<DType, IdType> model_status;
    CommBootstrap<DType, IdType> comm;
    ParamTransfer<DType, IdType> param_transfer;
    ForwardExecutor<DType, IdType> fwd_executor;

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
    /// \brief Construct from StubConfig
    explicit Stub(const StubConfig& cfg) :
        world_size(cfg.world_size),
        rank(cfg.rank),
        device(cfg.device),
        ib_hca_name(cfg.ib_hca_name),
        num_layers(cfg.num_layers),
        num_qo_head(cfg.num_qo_head),
        num_kv_head(cfg.num_kv_head),
        head_size(cfg.head_size),
        hidden_size(cfg.hidden_size),
        inter_size(cfg.inter_size),
        vocab_size(cfg.vocab_size),
        tp_size(cfg.tp_size),
        pp_size(cfg.pp_size),
        max_batch_size(cfg.max_batch_size),
        max_batch_tokens(cfg.max_batch_tokens),
        page_size(cfg.page_size),
        max_num_pages(cfg.max_num_page),
        tp_rank(cfg.tp_rank) {
        BZ_INFO(
            "Rank<{}> Tensor Parallelism: [{}/{}]",
            rank,
            this->tp_rank,
            this->tp_size
        );

        cudaSetDevice(device);
        // 64KB
        cudaMalloc(&page_indptr_d, 1024 * 1024);

        // Initialize comm MPI/NCCL comms (stored in CommBootstrap)
        comm.mpi_machine_comm = cfg.machine_comm;
        comm.mpi_tp_comm = cfg.tp_comm;

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
            comm.nccl_tp_comm,
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
            model_status.tfm_layer_cnt,
            comm.mpi_tp_comm,
            tp_size,
            tp_rank
        );
        prio_step_locker =
            std::make_unique<PriorityStepLocker<decltype(zag_task_queue)>>(
                zag_task_queue,
                rank,
                comm.mpi_tp_comm,
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
            model_status.embed_ready,
            model_status.tfm_layer_cnt,
            model_status.lm_head_ready,
            *model,
            *activation_manager,
            *mono_step_locker,
            *prio_step_locker,
            page_indptr_d
        );
        comm.init_communicator(rank, *model, *cache_manager, *activation_manager);
        int machine_size, machine_rank;
        MPI_Comm_rank(comm.mpi_machine_comm, &machine_rank);
        MPI_Comm_size(comm.mpi_machine_comm, &machine_size);
        nccl_bcastor = std::make_unique<NcclBcastor>(
            comm.mpi_machine_comm,
            comm.machine_comm_id,
            rank,
            device,
            machine_rank,
            machine_size,
            num_layers,
            model_status.embed_ready,
            model->get_embed_size_in_bytes(),
            model_status.tfm_layer_cnt,
            model->get_layer_size_in_bytes(),
            model_status.lm_head_ready,
            model->get_lmhead_size_in_bytes(),
            *mono_step_locker
        );
        nccl_bcastor->set_weight_buffer(
            model->weight_segment,
            model->weight_segment_size_in_bytes
        );
        /// \post barrier; nccl_tp_comm != nullptr
        if (cfg.is_active) {
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

    // --- Model status delegation ---

    bool is_ready_weak() const noexcept {
        return model_status.is_ready_weak(num_layers);
    }

    bool is_ready_strong() const noexcept {
        return model_status.is_ready_strong(num_layers);
    }

    void report_status() const noexcept {
        model_status.report_status(rank);
    }

    std::tuple<bool, uint32_t, bool> get_model_status() noexcept {
        return model_status.get_model_status();
    }

    void load_params_from_disk(std::string model_path = "", int device = 0) {
        model_status.load_params_from_disk(rank, *model_loader, model_path, device);
    }

    void load_params_from_host_memory() {
        model_status.load_params_from_host_memory(*model_loader);
    }

    void set_status_ready() noexcept {
        model_status.set_status_ready(num_layers, *mono_step_locker, *nccl_bcastor);
    }

    void reset_status() noexcept {
        model_status.reset_status(num_layers, *mono_step_locker, *nccl_bcastor);
    }

    bool status_invariant() noexcept {
        return model_status.status_invariant(num_layers);
    }

    void get_status_mutex() noexcept {
        model_status.get_status_mutex();
    }

    // --- Communication delegation ---

    void init_communicator() {
        comm.init_communicator(rank, *model, *cache_manager, *activation_manager);
    }

    void nccl_broadcast(std::vector<rank_t>&& item) {
        comm.nccl_broadcast(
            std::move(item), *nccl_bcastor, model_status.get_param_guard()
        );
    }

    void tanz_broadcast(std::vector<rank_t>&& item, size_t rt) {
        comm.tanz_broadcast(
            std::move(item), rt, *nccl_bcastor, model_status.get_param_guard()
        );
    }

    // --- Parameter transfer delegation ---

    void send_params_to(const int dst) {
        param_transfer.send_params_to(
            rank, device, dst, *model, param_thrd,
            model_status.get_param_guard()
        );
    }

    void send_params_in_chain(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index
    ) {
        param_transfer.send_params_in_chain(
            rank, device, ranks_in_chain, chain_index, *model, param_thrd,
            model_status.get_param_guard()
        );
    }

    void recv_params_from(const int src) {
        param_transfer.recv_params_from(
            rank, device, src, *model,
            model_status.embed_ready, model_status.tfm_layer_cnt,
            model_status.lm_head_ready, *mono_step_locker,
            param_thrd, model_status.get_param_guard()
        );
    }

    void recv_params_in_chain(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index
    ) {
        param_transfer.recv_params_in_chain(
            rank, device, ranks_in_chain, chain_index, *model,
            model_status.embed_ready, model_status.tfm_layer_cnt,
            model_status.lm_head_ready, *mono_step_locker,
            param_thrd, model_status.get_param_guard()
        );
    }

    // clang-format off
    void rdma_cast_params_in_chain_w_notify(
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index,
        tanz_t,
        size_t const nlane,
        size_t const lane_id
    ) {
        param_transfer.rdma_cast_params_in_chain_w_notify(
            rank, device, ranks_in_chain, chain_index, tanz,
            nlane, lane_id, *model, *nccl_bcastor,
            param_thrd, model_status.get_param_guard()
        );
    }
    // clang-format on

    bool join_params_thread() {
        return param_transfer.join_params_thread(
            rank, param_thrd, model_status.get_param_guard()
        );
    }

    // --- Forward execution delegation ---

    void forward(forward::normal_t, Batch<DType, IdType>& batch) {
        fwd_executor.forward(
            forward::normal, batch, rank, max_num_pages,
            num_qo_head, num_kv_head, head_size, hidden_size, vocab_size,
            page_indptr_d, *model, *cache_manager
        );
    }

    std::tuple<void*, size_t> get_kvcache_segment() {
        return cache_manager->get_kvcache_segment();
    }

    std::tuple<void*, size_t> get_weight_segment() {
        return model->get_weight_segment();
    }

    void forward(
        forward::normal_t,
        Batch<DType, IdType>& batch,
        forward::progress_t
    );

    void forward(
        forward::naivepp_t,
        Batch<DType, IdType>& batch,
        const generate::v2::PipeParaInfo& pp_info,
        const rank_t peer,
        generate::v2::PipeParaInfo& next_pp_info
    ) {
        fwd_executor.forward(
            forward::naivepp, batch, pp_info, peer, next_pp_info,
            rank, num_layers, max_num_pages,
            num_qo_head, num_kv_head, head_size, hidden_size, vocab_size,
            page_indptr_d, *model, *cache_manager, *activation_manager,
            model_status.embed_ready, model_status.tfm_layer_cnt,
            model_status.lm_head_ready
        );
    }

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

    /// \brief used by gRPC thread to add task
    std::future<generate::v2::PipeParaInfo> add_zag_task(
        Batch<DType, IdType>& batch_mref,
        uint32_t peer,
        uint32_t seq_num,
        const generate::v2::PipeParaInfo& pp_info
    ) {
        return fwd_executor.add_zag_task(
            batch_mref, peer, seq_num, pp_info, rank,
            *activation_manager, zag_task_queue, *prio_step_locker
        );
    }

    std::pair<int64_t, int32_t> relay_zag_task(int peer, bool relax_head) {
        return fwd_executor.relay_zag_task(peer, relax_head, *zigzag_manager);
    }
};

}  // namespace blitz

// clang-format on
