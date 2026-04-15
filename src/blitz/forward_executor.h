#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "activation_manager.h"
#include "batch.h"
#include "cache_manager.h"
#include "blitz_proto.h"
#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "model/llama.h"
#include "zigzag_manager.h"

namespace blitz {

/// \brief define types to distinguish overloaded functions
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

/// \brief Executes all forward pass variants (normal, progress, naivepp)
///        and manages ZigZag task submission/relay.
template<typename DType, typename IdType>
class ForwardExecutor {
  public:
    ForwardExecutor() = default;

    /// \brief forward and update KVCache
    void forward(
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
    );

    /// \brief forward but migrate kvcache progressively
    ///        especially used in PD disaggregation
    void forward(
        forward::normal_t,
        Batch<DType, IdType>& batch,
        forward::progress_t
    );

    /// \brief forward according to naivepp policy
    /// \param pp_info is used to determine which layer
    void forward(
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
    );

    /// \brief forward according to naivepp policy
    ///        while migrate kv_cache progressively
    void forward(
        forward::naivepp_t,
        Batch<DType, IdType>& batch,
        const generate::v2::PipeParaInfo&,
        forward::progress_t
    );

    /// \brief used by gRPC thread to add task
    std::future<generate::v2::PipeParaInfo> add_zag_task(
        Batch<DType, IdType>& batch_mref,
        uint32_t peer,
        uint32_t seq_num,
        const generate::v2::PipeParaInfo& pp_info,
        rank_t rank,
        ActivationManager<DType, IdType>& activation_manager,
        AChannel<Task_t<DType, IdType>>& zag_task_queue,
        PriorityStepLocker<AChannel<Task_t<DType, IdType>>>& prio_step_locker
    );

    std::pair<int64_t, int32_t> relay_zag_task(
        int peer,
        bool relax_head,
        ZigzagManager<DType, IdType>& zigzag_manager
    ) {
        if (!relax_head) {
            return zigzag_manager.interrupt_fst(peer);
        } else {
            return zigzag_manager.interrupt_snd(peer);
        }
    }
};

}  // namespace blitz
