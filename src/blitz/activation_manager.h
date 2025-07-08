#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <thread>

#include "include/blitz_tccl.h"
#include "include/spinlock.hpp"
#include "include/logger.hpp"
#include "include/types.hpp"
#include "util/cuda_utils.h"

namespace blitz {

template<typename DType, typename IdType>
class Stub;

template<typename DType>
struct SendItem {
    DType* buffer;
    uint64_t length_in_bytes;
    rank_t send_to;
    uint32_t stream_id;
};

template<typename DType, typename IdType>
class ActivationManager {
    friend class Stub<DType, IdType>;
  public:
    ActivationManager(
        const rank_t& rank,
        const rank_t& device,
        uint32_t nslot,
        size_t slot_size
    ) :
        num_activation_slot(nslot),
        activation_slot_size(slot_size),
        activation_slot_size_in_bytes(slot_size * sizeof(DType)),
        rank(rank),
        device(device) {
        CUDA_CHECK(cudaSetDevice(device));
        cudaMalloc(
            &activation_segment,
            num_activation_slot * activation_slot_size_in_bytes
        );
        for (size_t i = 0; i < num_activation_slot; ++i) {
            free.push_back(activation_segment + i * activation_slot_size);
        }
        worker = std::thread(
            std::bind(&ActivationManager::activation_worker_inner, this)
        );
    }

    ~ActivationManager() {
        shutdown.store(true, std::memory_order::release);
        worker.join();
        CUDA_CHECK(cudaFree(activation_segment));
    }

    /// \brief add activation transfer
    void add_send(SendItem<DType>&& req) {
        wq_lck.lock();
        unfinished_sends.push_back(std::move(req));
        wq_lck.unlock();
    }

    bool empty() noexcept {
        wq_lck.lock();
        bool ret = unfinished_sends.empty();
        wq_lck.unlock();
        return ret;
    }

    /// \brief switch on/off worker thread
    void awake() noexcept {
        awaken.store(true, std::memory_order::release);
        awaken.notify_one();
    }

    void asleep() noexcept {
        awaken.store(false, std::memory_order::release);
    }

    /// \brief acquire a slot for residual before inference
    void acquire_activation_slot(DType** ptr) {
        full.wait(true, std::memory_order::acquire);
        rb_lck.lock();
        *ptr = free.back();
        free.pop_back();
        used.push_back(*ptr);
        full.store(free.empty(), std::memory_order::release);
        rb_lck.unlock();
    }

    void release_activation_slot(DType* ptr) {
        rb_lck.lock();
        auto iter = std::find(used.begin(), used.end(), ptr);
        assert((iter != used.end()));
        used.erase(iter);
        free.push_back(ptr);
        full.store(false, std::memory_order::release);
        full.notify_one();
        rb_lck.unlock();
    }

  private:
    // worker
    void activation_worker_inner() {
        cudaSetDevice(device);
        typename decltype(unfinished_sends)::value_type item;
        // item :: (buffer, length, send_to, priority)
        while (!shutdown.load(std::memory_order::relaxed)) {
            // sleep to save CPU
            awaken.wait(false, std::memory_order::acquire);
            // working...
            wq_lck.lock();
            auto iter = unfinished_sends.begin();
            for (; iter != unfinished_sends.end();) {
                item = *iter;
                wq_lck.unlock();
                if (item.length_in_bytes) {
                    // actual send
                    BZ_INFO(
                        "Rank<{}> send slot {} activation buffer: [{}:{}] :> Rank<{}>...",
                        rank,
                        ((char*)item.buffer - (char*)activation_segment)
                            / activation_slot_size_in_bytes,
                        ::fmt::ptr((void*)item.buffer),
                        ::fmt::ptr((void*)((char*)item.buffer
                                           + item.length_in_bytes)),
                        item.send_to
                    );
                    auto handle = BlitzTccl::TcclSend(
                        item.buffer,
                        item.length_in_bytes,
                        item.send_to,
                        item.stream_id
                    );
                    handle.wait();
                    BZ_INFO(
                        "Rank<{}> Send slot {} done.",
                        rank,
                        ((char*)item.buffer - (char*)activation_segment)
                            / activation_slot_size_in_bytes
                    );
                } else {
                    BZ_INFO(
                        "Rank<{}> recycle slot {} activation buffer: @ {}",
                        rank,
                        ((char*)item.buffer - (char*)activation_segment)
                            / activation_slot_size_in_bytes,
                        ::fmt::ptr((void*)item.buffer)
                    );
                    // otherwise, placeholder
                    assert((item.send_to == -1));
                    assert((item.stream_id == UINT32_MAX));
                }
                release_activation_slot(item.buffer);
                wq_lck.lock();
                unfinished_sends.pop_front();
                iter = unfinished_sends.begin();
            }
            wq_lck.unlock();
        }
    }

    std::atomic_bool awaken = false;
    std::atomic_bool shutdown = false;
    std::thread worker;

    // ring buffer
    DType* activation_segment;
    Spinlock rb_lck;
    std::deque<DType*> used = {};
    std::deque<DType*> free = {};
    std::atomic_bool full = false;

    // config
    uint32_t num_activation_slot = 16;
    size_t activation_slot_size = 8192 * 4096;  // [max_batch_size, hidden_size]
    size_t activation_slot_size_in_bytes;
    const rank_t& rank;
    const rank_t& device;

    // work queue, main thread put request into this queue
    std::deque<SendItem<DType>> unfinished_sends;
    Spinlock wq_lck;
};

}  // namespace blitz