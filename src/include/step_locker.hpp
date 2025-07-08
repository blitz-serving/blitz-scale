#pragma once

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "logger.hpp"
#include "types.hpp"

template<std::totally_ordered T>
class MonotonicStepLocker {
    using rank_t = ::blitz::rank_t;
  public:
    MonotonicStepLocker(
        std::atomic<T>& var,
        MPI_Comm comm,
        rank_t comm_size,
        rank_t comm_rank
    ) :
        var_ref(var),
        tp_comm(comm),
        comm_size(comm_size),
        comm_rank(comm_rank) {
        val_sync.store(var.load(std::memory_order::acquire));
        T val = val_sync.load(std::memory_order::relaxed);
        sync_buf.assign(comm_size, val);
        worker =
            std::thread(std::bind(&MonotonicStepLocker::step_locker_inner, this)
            );
    }

    ~MonotonicStepLocker() {
        shutdown.store(true, std::memory_order::release);
        worker.join();
    }

    /// \brief one sided operation
    /// \remark MonotonicStepLocker must step one
    void post() noexcept {
        awaken.fetch_add(1, std::memory_order::acq_rel);
        awaken.notify_one();
    }

    T wait() {
        T val = val_sync.load(std::memory_order::acquire);
        sync_data(&val);
        auto val_it = std::max_element(sync_buf.begin(), sync_buf.end());
        return *val_it;
    }

    void update() noexcept {
        val_sync.store(var_ref.load());
    }

  private:
    void sync_data(const T* data) {
        MPI_Allgather(
            data,
            sizeof(T),
            MPI_BYTE,
            sync_buf.data(),
            sizeof(T),
            MPI_BYTE,
            tp_comm
        );
    }

    void step_locker_inner() {
        while (!shutdown.load(std::memory_order::relaxed)) {
            awaken.wait(0, std::memory_order::acquire);
            MPI_Barrier(tp_comm);
            T val = var_ref.load(std::memory_order::acquire);
            val_sync.store(val, std::memory_order::release);
            awaken.fetch_sub(1, std::memory_order::acq_rel);
        }
    }

    std::atomic<T>& var_ref;
    std::atomic<T> val_sync;
    std::vector<T> sync_buf;

    MPI_Comm tp_comm;
    const rank_t comm_size;
    const rank_t comm_rank;

    std::atomic_size_t awaken = 0;
    std::atomic_bool shutdown = false;
    std::thread worker;
};

template<typename LOCK>
class PriorityStepLocker {
    using rank_t = ::blitz::rank_t;
  public:
    PriorityStepLocker(
        LOCK& lck,
        rank_t rank,
        MPI_Comm comm,
        rank_t comm_size,
        rank_t comm_rank
    ) :
        lck(lck),
        rank(rank),
        comm(comm),
        comm_size(comm_size),
        comm_rank(comm_rank) {
        sync_buf.assign(comm_size, 0);
        worker =
            std::thread(std::bind(&PriorityStepLocker::step_locker_inner, this)
            );
    }

    ~PriorityStepLocker() {
        shutdown.store(true, std::memory_order::release);
        worker.join();
    }

    void post_wait_lock(size_t priority) noexcept {
        size_t prio = 1 << (priority - 1);
        awaken.fetch_or(prio, std::memory_order::seq_cst);
        awaken.notify_one();
        wait(priority);
        awaken.fetch_xor(prio, std::memory_order::acq_rel);
        _wait.store(0, std::memory_order::seq_cst);
    }

  private:
    void sync_data(const size_t* data) {
        MPI_Allgather(
            data,
            1,
            MPI_UINT64_T,
            sync_buf.data(),
            1,
            MPI_UINT64_T,
            comm
        );
    }

    void step_locker_inner() {
        while (!shutdown.load(std::memory_order::relaxed)) {
            awaken.wait(0, std::memory_order::seq_cst);
            /// \post a new round begin
            const size_t val = awaken.load(std::memory_order::acquire);
            sync_data(&val);
            size_t var = 0;
            for (const auto& val : sync_buf) {
                var |= val;
            }
            /// \post gather threads want this lock in bit mask
            while (var != 0) {
                size_t priority = std::bit_width(var);
                /// \remark adopt lock to acquirer
                lck.lock();
                size_t prio = 1 << (priority - 1);
                var ^= prio;
                _wait.store(priority, std::memory_order::release);
                wait(0);
                // BZ_INFO("Rank<{}>::<{}> grant lock to priority {} var={:b}", rank, comm_rank, priority, var);
            }
        }
    }

    void wait(const size_t val) const noexcept {
        for (;;) {
            // conform to MOESI
            if (_wait.load(std::memory_order::acquire) == val) {
                break;
            }
            while (_wait.load(std::memory_order::relaxed) != val) {
                // avoid continuously loading
                __builtin_ia32_pause();
            }
        }
    }

    LOCK& lck;

    std::atomic_size_t _wait = {0};
    std::vector<size_t> sync_buf;

    const rank_t rank;
    MPI_Comm comm;
    const rank_t comm_size;
    const rank_t comm_rank;

    std::atomic_size_t awaken = 0;
    std::atomic_bool shutdown = false;
    std::thread worker;
};
