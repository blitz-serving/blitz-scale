#pragma once

#include <deque>
#include <optional>
#include "spinlock.hpp"

template<typename T, typename Container = std::deque<T>>
struct AChannel {
    Spinlock _lck;
    Container _list;
  public:
    explicit AChannel() = default;
    bool empty() const noexcept { return _list.empty(); }
    std::size_t size() const noexcept { return _list.size(); }
    void lock() noexcept { _lck.lock(); }
    void unlock() noexcept { _lck.unlock(); }
    // push_back(value_type&& __x)
    // while emplace should take __Args
    // T isn't deduced here
    void push_back(T&& x) {
        _lck.lock();
        _list.emplace_back(std::move(x));
        _lck.unlock();
    }

    void pop_front() {
        _lck.lock();
        _list.pop_front();
        _lck.unlock();
    }

    std::optional<T> try_pop_front() {
        std::optional<T> res {};
        _lck.lock();
        if (!_list.empty()) {
            res.emplace(std::move(_list.front()));
            _list.pop_front();
        }
        _lck.unlock();
        return res;
    }

    std::optional<T> try_move_front() {
        std::optional<T> res {};
        _lck.lock();
        if (!_list.empty()) {
            res.emplace(std::move(_list.front()));
        }
        _lck.unlock();
        return res;
    }

    decltype(auto) front() {
        return _list.front();
    }

    static void swap(AChannel<T, Container> &x, AChannel<T, Container> &y) {
        x._lck.lock();
        y._lck.lock();
        std::swap(x._list, y._list);
        y._lck.unlock();
        x._lck.unlock();
    }
};
