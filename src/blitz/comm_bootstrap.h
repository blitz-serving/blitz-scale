#pragma once

#include <mpi.h>
#include <nccl.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "activation_manager.h"
#include "cache_manager.h"
#include "include/blitz_tccl.h"
#include "include/logger.hpp"
#include "include/types.hpp"
#include "model/llama.h"
#include "nccl_bcastor.h"

namespace blitz {

/// \brief Handles MPI/NCCL communication initialization and broadcast
///        operations for both TP (tensor parallelism) and machine-level comms.
template<typename DType, typename IdType>
class CommBootstrap {
  public:
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

    CommBootstrap() = default;

    /// \brief register MR for GDR && create NCCL comm for TP
    void init_communicator(
        rank_t rank,
        model::Llama<DType, IdType>& model,
        CacheManager<DType, IdType>& cache_manager,
        ActivationManager<DType, IdType>& activation_manager
    ) {
        /// \brief register MR for GDR
        // register parameter
        BZ_INFO(
            "Rank<{}> register mr weight [{}:{}]...",
            rank,
            ::fmt::ptr((void*)model.weight_segment),
            ::fmt::ptr((void*)((char*)model.weight_segment
                               + model.weight_segment_size_in_bytes))
        );
        BlitzTccl::register_mr(
            model.weight_segment,
            model.weight_segment_size_in_bytes
        );
        // register kvcache
        auto [kvcache_buffer, kvcache_length_in_bytes] =
            cache_manager.get_kvcache_segment();
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
            ::fmt::ptr((void*)model.runtime_segment),
            ::fmt::ptr((void*)((char*)model.runtime_segment
                               + model.runtime_segment_size_in_bytes))
        );
        BlitzTccl::register_mr(
            model.runtime_segment,
            model.runtime_segment_size_in_bytes
        );
        // register activation segment
        BZ_INFO(
            "Rank<{}> register mr activation [{}:{}]",
            rank,
            ::fmt::ptr((void*)activation_manager.activation_segment),
            ::fmt::ptr(
                (void*)((char*)activation_manager.activation_segment
                        + activation_manager.num_activation_slot
                            * activation_manager.activation_slot_size_in_bytes)
            )
        );
        BlitzTccl::register_mr(
            activation_manager.activation_segment,
            activation_manager.num_activation_slot
                * activation_manager.activation_slot_size_in_bytes
        );
        /// create NCCL comm for broadcast
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

    void nccl_broadcast(
        std::vector<rank_t>&& item,
        NcclBcastor& nccl_bcastor,
        std::atomic_bool& param_guard
    ) {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
        nccl_bcastor.broadcast(std::move(item), chain);
        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    void tanz_broadcast(
        std::vector<rank_t>&& item,
        size_t rt,
        NcclBcastor& nccl_bcastor,
        std::atomic_bool& param_guard
    ) {
        Assertion(
            (param_guard.load(std::memory_order::acquire)),
            flag::Param or flag::Tanz
        );
        nccl_bcastor.broadcast(std::move(item), tanz, rt);
    }
};

}  // namespace blitz
