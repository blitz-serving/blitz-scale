#pragma once

#include <mpi.h>
#include <numa.h>
#include <numaif.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

#include "logger.hpp"
#include "pickle.h"
#include "rdma_util.h"
#include <string>

namespace BlitzTccl {

class ThreadContoller {
public:
    inline void add_thread(std::thread t, std::function<void()> stop_function) {
        background_threads_.emplace_back(std::move(t));
        stop_functions_.emplace_back(std::move(stop_function));
    }

    ThreadContoller() = default;

    ~ThreadContoller() {
        for (auto& stop_function : stop_functions_) {
            if (stop_function) {
                stop_function();
            }
        }

        for (auto& t : background_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    ThreadContoller& operator=(const ThreadContoller&) = delete;
    ThreadContoller(const ThreadContoller&) = delete;
    ThreadContoller& operator=(ThreadContoller&&) = delete;
    ThreadContoller(ThreadContoller&&) = delete;

    std::vector<std::thread> background_threads_;
    std::vector<std::function<void()>> stop_functions_;
};

extern ThreadContoller gThreadController;

extern std::shared_ptr<rdma_util::ProtectionDomain> gProtectionDomain;
extern std::shared_ptr<pickle::Flusher> gFlusher;

extern std::vector<std::shared_ptr<rdma_util::MemoryRegion>> gMemoryRegionList;
extern std::vector<std::shared_ptr<pickle::PickleSender>> gPickleSenderList;
extern std::vector<std::shared_ptr<pickle::PickleRecver>> gPickleRecverList;

extern uint32_t gWorldSize;
extern uint32_t gWorldRank;
extern ibv_rate gRate;

struct ExchangeData {
    uint32_t id;
    uint32_t role;
    rdma_util::HandshakeData data;
};

inline uint32_t lkey(uint64_t addr, uint64_t length = 0) {
    for (const auto& mr : gMemoryRegionList) {
        if (addr >= uint64_t(mr->get_addr())
            && addr < uint64_t(mr->get_addr()) + mr->get_length()) {
        if (addr + length > uint64_t(mr->get_addr()) + mr->get_length()) {
            BZ_ERROR("lkey [{}:{}] exceeds MR [{}:{}]",
                ::fmt::ptr((void*)addr), 
                ::fmt::ptr((void*)(addr + length)),
                ::fmt::ptr((void*)mr->get_addr()), 
                ::fmt::ptr((void*)((char*)mr->get_addr() + mr->get_length())) 
            );
        }
            assert(
                (addr + length <= uint64_t(mr->get_addr()) + mr->get_length())
            );
            return mr->get_lkey();
        }
    }
    BZ_ERROR("Invalid address {}", ::fmt::ptr((char*)addr));
    throw std::runtime_error(
        ::fmt::format("Invalid address {}", ::fmt::ptr((char*)addr))
    );
}

inline uint32_t rkey(uint64_t addr, uint64_t length = 0) {
    for (const auto& mr : gMemoryRegionList) {
        if (addr >= uint64_t(mr->get_addr())
            && addr < uint64_t(mr->get_addr()) + mr->get_length()) {
            assert(
                (addr + length <= uint64_t(mr->get_addr()) + mr->get_length())
            );
            return mr->get_rkey();
        }
    }
    BZ_ERROR("Invalid address {}", ::fmt::ptr((char*)addr));
    throw std::runtime_error(
        ::fmt::format("Invalid address {}", ::fmt::ptr((char*)addr))
    );
}

inline std::vector<uint32_t> parse_cuda_visible_devices() {
    std::vector<uint32_t> devices;
    const char* env_p = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!env_p) {
        return devices;
    }

    std::string env_str(env_p);
    std::stringstream ss(env_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        devices.push_back(static_cast<uint32_t>(std::stoul(token)));
    }
    return devices;
}

inline const char* rank_to_rnic_name(int rank) {
    auto cuda_decives = parse_cuda_visible_devices();
    
    std::string base_env = "RNIC_NAMES_FOR_RANK_";
    std::string full_env = base_env + std::to_string(cuda_decives[rank % cuda_decives.size()]);
    const char* rnic_name = std::getenv(full_env.c_str());

    if(rnic_name == nullptr){
        BZ_ERROR("No rnic found for rank {}", rank)
        throw std::runtime_error("No matching rnic for rank");
    }
    return rnic_name;
}

void mpi_initialize_tccl_context(
    uint32_t world_size,
    uint32_t world_rank
) noexcept(false);

void bind_node(int node_id);

void register_mr(void* buffer, uint64_t length) noexcept(false);

inline pickle::Handle TcclSend(
    void* buffer,
    uint64_t length,
    uint32_t send_to,
    uint32_t stream_id
) noexcept(false) {
    assert(send_to < gWorldSize && send_to != gWorldRank);
    return gPickleSenderList[send_to > gWorldRank ? send_to - 1 : send_to]
        ->send(
            stream_id,
            uint64_t(buffer),
            length,
            lkey(uint64_t(buffer), length)
        );
}

inline pickle::Handle TcclRecv(
    void* buffer,
    uint64_t length,
    uint32_t recv_from,
    uint32_t stream_id
) noexcept(false) {
    assert(recv_from < gWorldSize && recv_from != gWorldRank);
    return gPickleRecverList[recv_from > gWorldRank ? recv_from - 1 : recv_from]
        ->recv(
            stream_id,
            uint64_t(buffer),
            length,
            rkey(uint64_t(buffer), length)
        );
}

inline void TcclChainMulticast(
    void* buffer,
    size_t length,
    uint32_t stream_id,
    const std::vector<int>& ranks_in_chain,
    size_t chain_index,
    size_t chunk_size = 0x100000
) noexcept(false) {
    BZ_DEBUG(
        "Rank<{}> Multicast group: [{}]",
        gWorldRank,
        fmt::join(ranks_in_chain, ",")
    );
    assert(ranks_in_chain.size() >= 2 && chain_index < ranks_in_chain.size());
    if (ranks_in_chain.size() == 2) {
        if (chain_index == 0) {
            TcclSend(buffer, length, ranks_in_chain[1], stream_id).wait();
        } else {
            TcclRecv(buffer, length, ranks_in_chain[0], stream_id).wait();
        }
        return;
    }

    if (chain_index == 0) {
        uint64_t chunks = (length + chunk_size - 1) / chunk_size;
        for (uint64_t i = 0; i < chunks; ++i) {
            uint64_t size = std::min(length - i * chunk_size, chunk_size);
            TcclSend(
                (char*)buffer + i * chunk_size,
                size,
                ranks_in_chain[chain_index + 1],
                stream_id
            )
                .wait();
            BZ_DEBUG(
                "Rank<{}> sent {} bytes to Rank<{}>",
                ranks_in_chain[chain_index],
                size,
                ranks_in_chain[chain_index + 1]
            );
        }
    } else if (chain_index == ranks_in_chain.size() - 1) {
        std::vector<pickle::Handle> handles;
        pickle::Handle handle;
        uint64_t chunks = (length + chunk_size - 1) / chunk_size;
        for (uint64_t i = 0; i < chunks; ++i) {
            uint64_t size = std::min(length - i * chunk_size, chunk_size);
            handles.push_back(TcclRecv(
                (char*)buffer + i * chunk_size,
                size,
                ranks_in_chain[chain_index - 1],
                stream_id
            ));
        }
        for (uint64_t i = 0; i < chunks; ++i) {
            handles[i].wait();
            BZ_DEBUG(
                "Rank<{}> received {} bytes from Rank<{}>",
                ranks_in_chain[chain_index],
                std::min(length - i * chunk_size, chunk_size),
                ranks_in_chain[chain_index - 1]
            );
        }
    } else {
        std::vector<pickle::Handle> handles;
        pickle::Handle handle;
        uint64_t chunks = (length + chunk_size - 1) / chunk_size;
        for (uint64_t i = 0; i < chunks; ++i) {
            uint64_t size = std::min(length - i * chunk_size, chunk_size);
            handles.push_back(TcclRecv(
                (char*)buffer + i * chunk_size,
                size,
                ranks_in_chain[chain_index - 1],
                stream_id
            ));
        }
        for (uint64_t i = 0; i < chunks; ++i) {
            uint64_t size = std::min(length - i * chunk_size, chunk_size);
            handles[i].wait();
            handle = TcclSend(
                (char*)buffer + i * chunk_size,
                size,
                ranks_in_chain[chain_index + 1],
                stream_id
            );
            BZ_DEBUG(
                "Rank<{}> forwarded {} bytes to Rank<{}>",
                ranks_in_chain[chain_index],
                size,
                ranks_in_chain[chain_index + 1]
            );
        }
    }
}

[[deprecated]]
inline pickle::Handle TcclSend(
    uint64_t addr,
    uint64_t length,
    uint32_t send_to,
    uint32_t stream_id
) noexcept(false) {
    assert(send_to < gWorldSize && send_to != gWorldRank);
    return gPickleSenderList[send_to > gWorldRank ? send_to - 1 : send_to]
        ->send(stream_id, addr, length, lkey(addr, length));
}

[[deprecated]]
inline pickle::Handle TcclRecv(
    uint64_t addr,
    uint64_t length,
    uint32_t recv_from,
    uint32_t stream_id
) noexcept(false) {
    assert(recv_from < gWorldSize && recv_from != gWorldRank);
    return gPickleRecverList[recv_from > gWorldRank ? recv_from - 1 : recv_from]
        ->recv(stream_id, addr, length, rkey(addr, length));
}

}  // namespace BlitzTccl