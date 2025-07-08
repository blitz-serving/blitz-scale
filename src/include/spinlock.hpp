/**
    Spinlock borrowed from https://rigtorp.se/spinlock/
 */
#pragma once

#include <atomic>
#include <optional>
#include <type_traits>

struct Spinlock {
    std::atomic_bool _lock {false};

    void lock() noexcept {
        for (;;) {
            // conform to MOESI
            if (!_lock.exchange(true, std::memory_order::acquire)) {
                break;
            }
            while (_lock.load(std::memory_order::relaxed)) {
                // avoid continuously loading
                __builtin_ia32_pause();
            }
        }
    }

    void unlock() noexcept {
        _lock.store(false, std::memory_order::release);
    }
};

template<
    typename T,
    typename = std::enable_if_t<std::is_copy_constructible_v<T>>>
struct OneshotSemaphore {
    std::atomic_bool _lock {true};
    std::optional<T> var {std::nullopt};

    T wait() {
        for (;;) {
            // conform to MOESI
            if (!_lock.exchange(true, std::memory_order::acquire)) {
                break;
            }
            while (_lock.load(std::memory_order::relaxed)) {
                // avoid continuously loading
                __builtin_ia32_pause();
            }
        }
        T val = *var;
        var.reset();
        return val;
    }

    void notify_one(const T& val) {
        var = val;
        _lock.store(false, std::memory_order::release);
    }
};

struct CountingSemaphore {
    /// \note "free" <-> lock >= 0 && "used out" <-> lock < 0
    std::atomic_int64_t _lock {0};

    void wait() {
        /// \invariant lock >= 0
        if (_lock.fetch_sub(1, std::memory_order::acq_rel) > 0) {
            return;
        };

        while (1) {
            if (_lock.load(std::memory_order::acquire) >= 0) {
                break;
            }
            while (_lock.load(std::memory_order::relaxed) < 0) {
                __builtin_ia32_pause();
            }
        }
    }

    void notify_one() {
        _lock.fetch_add(1, std::memory_order::acq_rel);
    }
};