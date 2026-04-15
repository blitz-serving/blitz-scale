#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <tuple>

#include "include/logger.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"
#include "model_loader.h"
#include "nccl_bcastor.h"

namespace blitz {

/// \brief Tracks model loading status via atomic flags for embed, transformer
///        layers, and lm_head. Provides readiness queries, disk/host loading,
///        and status management.
template<typename DType, typename IdType>
class ModelStatus {
  public:
    // Model validity markers
    std::atomic_bool embed_ready = false;
    std::atomic_bool lm_head_ready = false;
    std::atomic_uint32_t tfm_layer_cnt = 0;

    ModelStatus() = default;

    bool is_ready_weak(uint32_t num_layers) const noexcept {
        return embed_ready.load(std::memory_order::relaxed)
            && tfm_layer_cnt.load(std::memory_order::relaxed) == num_layers
            && lm_head_ready.load(std::memory_order::relaxed);
    }

    bool is_ready_strong(uint32_t num_layers) const noexcept {
        return embed_ready.load(std::memory_order::acquire)
            && tfm_layer_cnt.load(std::memory_order::acquire) == num_layers
            && lm_head_ready.load(std::memory_order::acquire);
    }

    void report_status(rank_t rank) const noexcept {
        BZ_INFO(
            "Rank<{}> embed {}, layer {}, lmhead {}",
            rank,
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

    /// \pre model parameter is unloaded
    void load_params_from_disk(
        rank_t rank,
        model::ModelLoader& model_loader,
        std::string model_path = "",
        int device = 0
    ) {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        assert((model_path != ""));
        BZ_INFO("Rank<{}> load model from disk: {}", rank, model_path);
        param_guard.store(true, std::memory_order::release);
        model_loader.load_model_host_fast(model_path);
        model_loader.load_model_gpu();

        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    void load_params_from_host_memory(model::ModelLoader& model_loader) {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        param_guard.store(true, std::memory_order::release);
        model_loader.load_model_gpu();

        Assertion((param_guard.load(std::memory_order::acquire)), flag::Param);
        param_guard.store(false, std::memory_order::release);
    }

    /// \pre model parameter is unloaded
    void set_status_ready(
        uint32_t num_layers,
        MonotonicStepLocker<uint32_t>& mono_step_locker,
        NcclBcastor& nccl_bcastor
    ) noexcept {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );

        embed_ready.store(true, std::memory_order::release);
        tfm_layer_cnt.store(num_layers, std::memory_order::release);
        lm_head_ready.store(true, std::memory_order::release);
        mono_step_locker.update();
        Assertion(nccl_bcastor.is_empty(tanz), flag::Param && flag::Tanz);
    }

    void reset_status(
        uint32_t num_layers,
        MonotonicStepLocker<uint32_t>& mono_step_locker,
        NcclBcastor& nccl_bcastor
    ) noexcept {
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
        mono_step_locker.update();
        Assertion(nccl_bcastor.is_empty(tanz), flag::Param && flag::Tanz);
    }

    /// \brief calculate model parameter status invariant
    bool status_invariant(uint32_t num_layers) noexcept {
        bool embedding_layer = embed_ready.load(std::memory_order::acquire);
        auto num_transformer_layers =
            tfm_layer_cnt.load(std::memory_order::acquire);
        bool lm_head = lm_head_ready.load(std::memory_order::acquire);
        bool loaded = {
            lm_head && num_transformer_layers == num_layers
            && embedding_layer
        };
        bool unloaded = {
            !lm_head
            && ((!embedding_layer && num_transformer_layers == 0)
                || (num_transformer_layers > 0))
        };
        return num_transformer_layers <= num_layers
            && (loaded || unloaded) && !(loaded && unloaded);
    }

    /// \brief acquire the param guard, used in Tanz
    void get_status_mutex() noexcept {
        Assertion(
            (not param_guard.load(std::memory_order::acquire)),
            flag::Param
        );
        param_guard.store(true, std::memory_order::release);
    }

    /// \brief reference to the global param_guard
    std::atomic_bool& get_param_guard() noexcept {
        return param_guard;
    }

  private:
    /// \invariant
    std::atomic_bool param_guard = false;
};

}  // namespace blitz
