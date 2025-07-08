#include "blitz_tccl.h"

#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>

#include <cassert>
#include <cstdint>
#include <thread>

#include "include/logger.hpp"

namespace BlitzTccl {

ThreadContoller gThreadController = ThreadContoller();

std::shared_ptr<rdma_util::ProtectionDomain> gProtectionDomain = nullptr;
std::shared_ptr<pickle::Flusher> gFlusher = nullptr;

std::vector<std::shared_ptr<rdma_util::MemoryRegion>> gMemoryRegionList {};
std::vector<std::shared_ptr<pickle::PickleSender>> gPickleSenderList {};
std::vector<std::shared_ptr<pickle::PickleRecver>> gPickleRecverList {};

uint32_t gWorldSize = 0;
uint32_t gWorldRank = 0;

ibv_rate gRate = ibv_rate::IBV_RATE_MAX;
uint32_t gGidIndex = 3;

void bind_node(int node_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    struct bitmask* cpumask = numa_allocate_cpumask();
    numa_node_to_cpus(node_id, cpumask);
    for (uint64_t i = 0; i < cpumask->size; ++i) {
        if (numa_bitmask_isbitset(cpumask, i)) {
            CPU_SET(i, &cpuset);
        }
    }
    pthread_t thread = pthread_self();
    assert(pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) == 0);
    numa_free_cpumask(cpumask);

    bitmask* bm = numa_allocate_nodemask();
    numa_bitmask_setbit(bm, node_id);
    assert(set_mempolicy(MPOL_BIND, bm->maskp, bm->size + 1) == 0);
    numa_free_nodemask(bm);
}

void mpi_initialize_tccl_context(
    uint32_t world_size,
    uint32_t world_rank
) noexcept(false) {
    gWorldSize = world_size;
    gWorldRank = world_rank;

    auto name = rank_to_rnic_name(world_rank);
    BZ_INFO("Rank<{}> use RNIC {}", world_rank, name);

    gProtectionDomain =
        rdma_util::ProtectionDomain::create(rdma_util::Context::create(name));
    gFlusher = pickle::Flusher::create(gProtectionDomain);

    std::vector<std::unique_ptr<rdma_util::RcQueuePair>> sender_qp_list,
        recver_qp_list;

    for (uint32_t i = 0; i < world_size - 1; ++i) {
        sender_qp_list.push_back(
            rdma_util::RcQueuePair::create(gProtectionDomain)
        );
        recver_qp_list.push_back(
            rdma_util::RcQueuePair::create(gProtectionDomain)
        );
    }

    std::vector<ExchangeData> sender_scatter_data, recver_scatter_data;
    std::vector<ExchangeData> sender_gather_data(world_size),
        recver_gather_data(world_size);

    // prepare rdma_util::HandshakeData
    // TODO @wht: RoCE specific gid_index. 0 for Infiniband.
    for (const auto& qp : sender_qp_list) {
        sender_scatter_data.push_back({world_rank, 1, qp->get_handshake_data(3)});
    }
    for (const auto& qp : recver_qp_list) {
        recver_scatter_data.push_back({world_rank, 2, qp->get_handshake_data(3)});
    }

    // emplace a place holder to `self`
    sender_scatter_data.insert(
        sender_scatter_data.begin() + world_rank,
        {0, 0, rdma_util::HandshakeData()}
    );
    recver_scatter_data.insert(
        recver_scatter_data.begin() + world_rank,
        {0, 0, rdma_util::HandshakeData()}
    );

    assert(
        (sender_scatter_data.size() == world_size
         && recver_scatter_data.size() == world_size)
    );

    MPI_Barrier(MPI_COMM_WORLD);

    // All2All exchange data
    for (uint32_t root = 0; root < world_size; ++root) {
        // root scatter designated rdma_util::HandshakeData
        MPI_Scatter(
            sender_scatter_data.data(),
            sizeof(ExchangeData),
            MPI_BYTE,
            &sender_gather_data[root],
            sizeof(ExchangeData),
            MPI_BYTE,
            root,
            MPI_COMM_WORLD
        );
        MPI_Barrier(MPI_COMM_WORLD);
    }

    for (uint32_t root = 0; root < world_size; ++root) {
        // root scatter designated rdma_util::HandshakeData
        MPI_Scatter(
            recver_scatter_data.data(),
            sizeof(ExchangeData),
            MPI_BYTE,
            &recver_gather_data[root],
            sizeof(ExchangeData),
            MPI_BYTE,
            root,
            MPI_COMM_WORLD
        );
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // remove place holder
    sender_gather_data.erase(sender_gather_data.begin() + world_rank);
    recver_gather_data.erase(recver_gather_data.begin() + world_rank);

    // qp_list[i] && gathered_data[i] are in pair
    for (uint32_t j = 0; j < world_size - 1; j++) {
        BZ_DEBUG("Rank<{}> sender connected with Rank<{}> recver", world_rank, recver_gather_data[j].id);
        BZ_DEBUG("Rank<{}> recver connected with Rank<{}> sender", world_rank, sender_gather_data[j].id);
        // TODO @wht: RoCE specific gid_index. 0 for Infiniband.
        sender_qp_list[j]->bring_up(recver_gather_data[j].data, 3, gRate);
        recver_qp_list[j]->bring_up(sender_gather_data[j].data, 3, gRate);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (uint32_t i = 0; i < world_size - 1; ++i) {
        gPickleSenderList.push_back(pickle::PickleSender::create(
            std::move(sender_qp_list[i]),
            256 * 1024
        ));
        gPickleRecverList.push_back(
            pickle::PickleRecver::create(std::move(recver_qp_list[i]), gFlusher)
        );
    }

    int node_id;
    auto cuda_decives = parse_cuda_visible_devices();
    switch (cuda_decives[world_rank % cuda_decives.size()]) {
        case 0:
        case 1:
        case 2:
        case 3:
            node_id = 0;
            break;
        case 4:
        case 5:
        case 6:
        case 7:
            node_id = 1;
            break;
        default:
            throw std::runtime_error("Invalid rank");
    }

    std::shared_ptr<std::atomic<bool>> flag =
        std::make_shared<std::atomic<bool>>(true);
    std::function<void()> stop_function = [flag]() mutable {
        flag->store(false);
    };
    std::thread polling_thread([node_id, flag]() mutable {
        bind_node(node_id);
        auto flusher = gFlusher;
        auto senders = gPickleSenderList;
        auto recvers = gPickleRecverList;

        while (flag->load(std::memory_order_relaxed)) {
            flusher->poll();
            for (const auto& sender : senders) {
                sender->poll();
            }
            for (const auto& recver : recvers) {
                recver->poll();
            }
            std::this_thread::yield();
        }
    });

    gThreadController.add_thread(
        std::move(polling_thread),
        std::move(stop_function)
    );
}

void register_mr(void* buffer, uint64_t length) noexcept(false) {
    if (gProtectionDomain.get() == nullptr) {
        throw std::runtime_error("RDMA resources have not been initialized.");
    }
    gMemoryRegionList.push_back(
        rdma_util::MemoryRegion::create(gProtectionDomain, buffer, length)
    );
    BZ_INFO("Rank<{}> has registered a memory region", gWorldRank);
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace BlitzTccl