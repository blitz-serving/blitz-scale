#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "include/blitz_tccl.h"
#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "model/llama.h"
#include "nccl_bcastor.h"

/// \deprecated
#include "pickle.h"

namespace blitz {

/// \brief Handles all parameter migration operations including P2P RDMA
///        send/recv, chain multicast, and the Tanz striped RDMA protocol.
template<typename DType, typename IdType>
class ParamTransfer {
  public:
    ParamTransfer() = default;

    /// \brief issue parameter transfer (P2P send)
    void send_params_to(
        rank_t rank,
        rank_t device,
        const int dst,
        model::Llama<DType, IdType>& model,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );

    /// \brief issue parameter transfer (chain multicast, sender)
    void send_params_in_chain(
        rank_t rank,
        rank_t device,
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index,
        model::Llama<DType, IdType>& model,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );

    /// \brief issue parameter transfer in P2P RDMA
    /// \pre   model parameter is unloaded
    /// \post  model parameter is loaded
    void recv_params_from(
        rank_t rank,
        rank_t device,
        const int src,
        model::Llama<DType, IdType>& model,
        std::atomic_bool& embed_ready,
        std::atomic_uint32_t& tfm_layer_cnt,
        std::atomic_bool& lm_head_ready,
        MonotonicStepLocker<uint32_t>& mono_step_locker,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );

    /// \brief issue parameter transfer in RDMA chain
    /// \pre   model parameter is unloaded
    /// \post  model parameter is loaded
    void recv_params_in_chain(
        rank_t rank,
        rank_t device,
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index,
        model::Llama<DType, IdType>& model,
        std::atomic_bool& embed_ready,
        std::atomic_uint32_t& tfm_layer_cnt,
        std::atomic_bool& lm_head_ready,
        MonotonicStepLocker<uint32_t>& mono_step_locker,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );

    /// \brief issue parameter transfer in RDMA chain with Tanz protocol
    void rdma_cast_params_in_chain_w_notify(
        rank_t rank,
        rank_t device,
        const std::vector<int>& ranks_in_chain,
        const size_t chain_index,
        tanz_t,
        size_t const nlane,
        size_t const lane_id,
        model::Llama<DType, IdType>& model,
        NcclBcastor& nccl_bcastor,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );

    /// \brief join the background parameter thread
    bool join_params_thread(
        rank_t rank,
        std::thread& param_thrd,
        std::atomic_bool& param_guard
    );
};

}  // namespace blitz
