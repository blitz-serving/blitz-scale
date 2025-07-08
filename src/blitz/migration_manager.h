#pragma once

#include <fmt/format.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "batch.h"
#include "include/blitz_tccl.h"
#include "include/logger.hpp"
#include "include/types.hpp"

namespace blitz {
// clang-format off
struct fst_half_t { explicit fst_half_t() = default; };

struct snd_half_t { explicit snd_half_t() = default; };

struct all_layer_t { explicit all_layer_t() = default; };

inline constexpr fst_half_t fst_half {};

inline constexpr snd_half_t snd_half {};

inline constexpr all_layer_t all_layer {};
// clang-format on

template<typename DType, typename IdType>
class MigrationManager {
    const static size_t kDop = 64;
    using B = Batch<DType, IdType>;
    static constexpr uint32_t BCCL_KVCACHE_STREAM = 1;
  public:
    MigrationManager(
        const rank_t &world_size,
        const rank_t &rank,
        const uint32_t &page_size_,
        const uint32_t local_num_kv_heads,
        const uint32_t &num_head_dim,
        const uint32_t &max_num_pages,
        const uint32_t &local_num_layers,
        DType* ptr
    );

    ~MigrationManager();

    inline std::vector<cudaEvent_t>& get_layer_events_ref() {
        return layer_events;
    }

    inline std::vector<cudaEvent_t>* get_layer_events_ptr() {
        return &layer_events;
    }

    inline bool empty() const {
        bool res = true;
        for (auto it = qp_guards.cbegin(); res && it != qp_guards.cend();
             ++it) {
            res = it->load(std::memory_order::relaxed) == -1;
        }
        return res;
    }

    /**
     * \brief synchronous send function
     */
    bool send_kv_cache(const B&, all_layer_t, rank_t dst);
    bool send_kv_cache(const B&, fst_half_t, rank_t dst, uint32_t num_layers);
    bool send_kv_cache(const B&, snd_half_t, rank_t dst, uint32_t num_layers);
    /**
     * \brief synchronous receive function
     */
    bool recv_kv_cache(B&, all_layer_t, rank_t src);
    bool
    recv_kv_cache(B& batch_mref, fst_half_t, rank_t src, uint32_t num_layers);
    bool
    recv_kv_cache(B& batch_mref, snd_half_t, rank_t src, uint32_t num_layers);

    inline DType*
    get_page_ptr(const B& batch, const size_t block_id, const size_t curr_layer)
        const noexcept {
        DType* layer_base =
            kv_data + curr_layer * max_num_pages * batch.page_table.stride_page;
        return layer_base + block_id * batch.page_table.stride_page;
    }

  private:
    /// \brief router specific debug
    std::pair<std::atomic_int64_t, std::atomic_int64_t> occupied;
    // model config
    const uint32_t local_num_kv_heads;
    const uint32_t &local_num_layers;
    const uint32_t &head_dim;
    // comm
    const rank_t &world_size;
    const rank_t &rank;
    std::vector<std::atomic_int64_t> qp_guards;
    // kvcache config
    const uint32_t &page_size;
    const uint32_t &max_num_pages;
    DType* kv_data;
    // layer by layer
    std::vector<cudaEvent_t> layer_events;

    void send_multi_layer_fast(
        const B& batch_ref,
        const size_t start_layer,
        const size_t layer_cnt,
        const rank_t dst,
        uint32_t stream_id
    );
    void receive_multi_layer_fast(
        B& batch,
        const size_t start_layer,
        const size_t layer_cnt,
        const rank_t src,
        uint32_t stream_id
    );

    bool is_empty() const;
};

}  // namespace blitz

/*-------------------IMPLEMENTATION------------------*/

namespace blitz {
template<typename DType, typename IdType>
MigrationManager<DType, IdType>::MigrationManager(
    const rank_t &world_size,
    const rank_t &rank,
    const uint32_t &page_size_,
    const uint32_t local_num_kv_heads,
    const uint32_t &num_head_dim,
    const uint32_t &max_num_pages,
    const uint32_t &local_num_layers,
    DType* ptr
) :
    occupied {-1, -1},
    local_num_kv_heads(local_num_kv_heads),
    local_num_layers(local_num_layers),
    head_dim(num_head_dim),
    world_size(world_size),
    rank(rank),
    qp_guards(world_size),
    page_size(page_size_),
    max_num_pages(max_num_pages),
    kv_data(ptr),
    layer_events(local_num_layers) {
    // QP guard
    std::for_each(qp_guards.begin(), qp_guards.end(), [](auto& g) {
        g.store(-1, std::memory_order::release);
    });
    // used for layer-by-layer transfer
    for (cudaEvent_t& event : layer_events) {
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    }
}

template<typename DType, typename IdType>
MigrationManager<DType, IdType>::~MigrationManager() {
    for (size_t i = 0; i < local_num_layers; ++i) {
        cudaEventDestroy(layer_events[i]);
    }
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::send_kv_cache(
    const Batch<DType, IdType>& batch_ref,
    all_layer_t,
    rank_t dst
) {
    // debug guard
    if (int64_t ocp =
            occupied.first.exchange(batch_ref.id, std::memory_order::acquire);
        ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently migrating",
            rank,
            ocp,
            batch_ref.id
        );
    }
    assert((dst != rank));
    // parse args
    if (int64_t g = qp_guards[dst].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> -~> <{}>",
            rank,
            g,
            batch_ref.id,
            rank,
            dst
        );
        return false;
    }
    qp_guards[dst].store(batch_ref.id, std::memory_order::release);
    send_multi_layer_fast(
        batch_ref,
        0,
        local_num_layers,
        dst,
        BCCL_KVCACHE_STREAM
    );
    qp_guards[dst].store(-1, std::memory_order::release);

    occupied.first.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::send_kv_cache(
    const B& batch_ref,
    fst_half_t,
    rank_t dst,
    uint32_t num_layers
) {
    // debug guard
    int64_t ocp =
        occupied.first.exchange(batch_ref.id, std::memory_order::acquire);
    if (ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently migrating",
            rank,
            ocp,
            batch_ref.id
        );
    }
    assert((dst != rank));
    assert((num_layers <= local_num_layers));
    if (int64_t g = qp_guards[dst].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> -~> <{}>",
            rank,
            g,
            batch_ref.id,
            rank,
            dst
        );
        return false;
    }
    qp_guards[dst].store(batch_ref.id, std::memory_order::release);
    send_multi_layer_fast(batch_ref, 0, num_layers, dst, BCCL_KVCACHE_STREAM);
    qp_guards[dst].store(-1, std::memory_order::release);

    occupied.first.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::send_kv_cache(
    const B& batch_ref,
    snd_half_t,
    rank_t dst,
    uint32_t num_layers
) {
    // debug guard
    int64_t ocp =
        occupied.first.exchange(batch_ref.id, std::memory_order::acquire);
    if (ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently migrating",
            rank,
            ocp,
            batch_ref.id
        );
    }
    assert((dst != rank));
    assert((num_layers <= local_num_layers));
    if (int64_t g = qp_guards[dst].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> -~> <{}>",
            rank,
            g,
            batch_ref.id,
            rank,
            dst
        );
        return false;
    }
    qp_guards[dst].store(batch_ref.id, std::memory_order::release);
    send_multi_layer_fast(
        batch_ref,
        num_layers,
        local_num_layers - num_layers,
        dst,
        BCCL_KVCACHE_STREAM
    );
    qp_guards[dst].store(-1, std::memory_order::release);

    occupied.first.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::recv_kv_cache(
    Batch<DType, IdType>& batch_mref,
    all_layer_t,
    rank_t src
) {
    // debug guard
    if (int64_t ocp =
            occupied.second.exchange(batch_mref.id, std::memory_order::acquire);
        ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently immigrating",
            rank,
            ocp,
            batch_mref.id
        );
    }
    assert((src != rank));
    if (int64_t g = qp_guards[src].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> <~- <{}>",
            rank,
            g,
            batch_mref.id,
            rank,
            src
        );
        return false;
    }
    qp_guards[src].store(batch_mref.id, std::memory_order::release);
    receive_multi_layer_fast(
        batch_mref,
        0,
        local_num_layers,
        src,
        BCCL_KVCACHE_STREAM
    );
    qp_guards[src].store(-1, std::memory_order::release);

    occupied.second.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::recv_kv_cache(
    Batch<DType, IdType>& batch_mref,
    fst_half_t,
    rank_t src,
    uint32_t num_layers
) {
    // debug guard
    int64_t ocp =
        occupied.second.exchange(batch_mref.id, std::memory_order::acquire);
    if (ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently immigrating",
            rank,
            ocp,
            batch_mref.id
        );
    }
    assert((src != rank));
    assert((num_layers <= local_num_layers));
    if (auto g = qp_guards[src].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> <~- <{}>",
            rank,
            g,
            batch_mref.id,
            rank,
            src
        );
        return false;
    }
    qp_guards[src].store(batch_mref.id, std::memory_order::release);
    receive_multi_layer_fast(
        batch_mref,
        0,
        num_layers,
        src,
        BCCL_KVCACHE_STREAM
    );
    qp_guards[src].store(-1, std::memory_order::release);

    occupied.second.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
bool MigrationManager<DType, IdType>::recv_kv_cache(
    Batch<DType, IdType>& batch_mref,
    snd_half_t,
    rank_t src,
    uint32_t num_layers
) {
    // debug guard
    int64_t ocp =
        occupied.second.exchange(batch_mref.id, std::memory_order::acquire);
    if (ocp != -1) {
        BZ_WARN(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently immigrating",
            rank,
            ocp,
            batch_mref.id
        );
    }
    assert((src != rank));
    assert((num_layers <= local_num_layers));
    if (auto g = qp_guards[src].load(std::memory_order::acquire); g != -1) {
        BZ_ERROR(
            "Rank<{}> [{{MigrationManager}}] Batch[{}] & Batch[{}] concurrently using QP <{}> <~- <{}>",
            rank,
            ocp,
            batch_mref.id,
            rank,
            src
        );
        return false;
    }
    qp_guards[src].store(batch_mref.id, std::memory_order::release);
    receive_multi_layer_fast(
        batch_mref,
        num_layers,
        local_num_layers - num_layers,
        src,
        BCCL_KVCACHE_STREAM
    );
    qp_guards[src].store(-1, std::memory_order::release);

    occupied.second.store(-1, std::memory_order::release);
    return true;
}

template<typename DType, typename IdType>
void MigrationManager<DType, IdType>::send_multi_layer_fast(
    const Batch<DType, IdType>& batch_ref,
    const size_t start_layer,
    const size_t layer_cnt,
    const rank_t dst,
    uint32_t stream_id
) {
    size_t cnt = 0;
    std::vector<pickle::Handle> handles {};
    uint64_t k = 0;
    BZ_DEBUG("Rank<{}> -~> Rank<{}> Batch[{}]", rank, dst, batch_ref.id);
    auto s_time = std::chrono::steady_clock::now();
    for (const auto& blocks : batch_ref.indices_2d) {
        size_t prefill_needed_block = batch_ref.num_prompt_blocks[cnt++];
        assert(
            (blocks.size() == prefill_needed_block
             || (blocks.size() - 1) == prefill_needed_block)
        );
        /// \todo support layer by layer
        for (size_t i = 0; i < prefill_needed_block; ++i) {
            auto block_id = blocks[i];
            auto size_in_bytes =
                batch_ref.page_table.stride_page * sizeof(DType);
            for (auto layer = start_layer; layer < start_layer + layer_cnt;
                 layer++) {
                k++;
                handles.push_back(BlitzTccl::TcclSend(
                    (void*)get_page_ptr(batch_ref, block_id, layer),
                    size_in_bytes,
                    dst,
                    stream_id
                ));
                if (k % kDop == 0) {
                    handles.back().wait();
                    handles.clear();
                }
            }
        }
    }
    auto e_time = std::chrono::steady_clock::now();
    auto elapse =
        std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)
            .count();
    BZ_INFO(
        "Batch[{}] Rank<{}> -~> Rank<{}> sent {} iters, bandwidth={}GiBps",
        batch_ref.id,
        rank,
        dst,
        k,
        (double)(k)*batch_ref.page_table.stride_page * sizeof(DType) / 1024
            / 1024 / 1024 * 1000 / elapse
    );
}

template<typename DType, typename IdType>
void MigrationManager<DType, IdType>::receive_multi_layer_fast(
    Batch<DType, IdType>& batch_mref,
    const size_t start_layer,
    const size_t layer_cnt,
    const rank_t src,
    uint32_t stream_id
) {
    size_t cnt = 0;
    std::vector<pickle::Handle> handles {};
    uint64_t k = 0;
    BZ_DEBUG("Rank<{}> <~- Rank<{}> Batch[{}]", rank, src, batch_mref.id);
    for (const auto& blocks : batch_mref.indices_2d) {
        size_t prefill_needed_block = batch_mref.num_prompt_blocks[cnt++];
        assert(
            (blocks.size() == prefill_needed_block)
            || (blocks.size() - 1 == prefill_needed_block)
        );
        /// \todo support layer by layer
        for (size_t i = 0; i < prefill_needed_block; ++i) {
            auto block_id = blocks[i];
            auto size_in_bytes =
                batch_mref.page_table.stride_page * sizeof(DType);
            for (auto layer = start_layer; layer < start_layer + layer_cnt;
                 layer++) {
                k++;
                handles.emplace_back(BlitzTccl::TcclRecv(
                    (char*)get_page_ptr(batch_mref, block_id, layer),
                    size_in_bytes,
                    src,
                    stream_id
                ));
                if (k % kDop == 0) {
                    handles.back().wait();
                    handles.clear();
                }
            }
        }
    }
    BZ_INFO(
        "Batch[{}] Rank<{}> <~- Rank<{}> received {} iters.",
        batch_mref.id,
        rank,
        src,
        k
    );
}

}  // namespace blitz
