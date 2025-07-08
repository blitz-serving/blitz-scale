#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "include/logger.hpp"
#include "include/spinlock.hpp"
#include "include/step_locker.hpp"
#include "include/types.hpp"

namespace blitz {
// clang-format off
struct chain_t { explicit chain_t() = default; };

struct tanz_t { explicit tanz_t() = default; };

inline constexpr chain_t chain {};

inline constexpr tanz_t tanz {};

// clang-format on

class NcclBcastor {
  private:
    MPI_Comm mpi_comm;
    ncclComm_t comm;
    cudaStream_t fst_stream, snd_stream;

    std::atomic_bool awaken = false;
    std::atomic_bool shutdown = false;
    std::thread worker;

    char* weight_segment = nullptr;
    size_t weight_segment_size_in_bytes = 0;
    const uint32_t& num_transformer_layers;
    std::atomic_bool& embed_ready;
    std::atomic_uint32_t& tfm_layer_cnt;
    std::atomic_bool& lm_head_ready;
    const size_t embed_size_in_bytes;
    const size_t tfm_layer_size_in_bytes;
    const size_t lm_head_size_in_bytes;
    MonotonicStepLocker<uint32_t>& stp_lck_ref;

    const rank_t& rank;
    const rank_t& device;
    rank_t const machine_rank;
    rank_t const machine_size;

    struct WorkItem {
        size_t nlane = 0;
        // std::vector<rank_t> src_ranks;
        std::vector<rank_t> dst_ranks;
    } work_item;

    CountingSemaphore rdma_sem;

    void init_p2p() {
        /// \brief create ALL P2P connects ahead
        void* buf;
        cudaMalloc(&buf, 1024 * 1024);
        for (int dist = 1; dist < machine_size; ++dist) {
            int prev = (machine_rank + machine_size - dist) % machine_size;
            int succ = (machine_rank + machine_size + dist) % machine_size;
            ncclGroupStart();
            if (machine_rank & 0x1) {
                ncclRecv(buf, 1024 * 1024, ncclChar, prev, comm, snd_stream);
                ncclSend(buf, 1024 * 1024, ncclChar, succ, comm, fst_stream);
            } else {
                ncclSend(buf, 1024 * 1024, ncclChar, succ, comm, fst_stream);
                ncclRecv(buf, 1024 * 1024, ncclChar, prev, comm, snd_stream);
            }
            ncclGroupEnd();
            MPI_Barrier(mpi_comm);
        }
        cudaFree(buf);
    }

    template<typename T>
    typename std::enable_if<std::is_pointer<T>::value>::type
    wrap_around_sub(T& self, T const& base, const size_t sub, const size_t wrap)
        const noexcept {
        uintptr_t a = reinterpret_cast<uintptr_t>(self);
        uintptr_t b = reinterpret_cast<uintptr_t>(base);
        Precondition((a >= b), flag::Tanz);
        uintptr_t res = ((a - b) < sub) ? (a + wrap - sub) : (a - sub);
        Postcondition((res >= b && res <= b + wrap), flag::Tanz);
        self = (T)res;
    }

    /// \note last & next are both machine ranks
    void nccl_bcastor_edge_inner(
        rank_t last,
        rank_t next,
        tanz_t,
        size_t const lane_id
    ) {
        const size_t n = work_item.nlane;
        const size_t nedges = n - 1;
        constexpr size_t strip_chunk_size_in_bytes = 64 * 1024 * 1024;
        const size_t chunk_size_in_bytes = n * strip_chunk_size_in_bytes;

        BZ_WARN(
            "Rank<{}>:({}) (nccl) last = ({}) next = ({})",
            rank,
            machine_rank,
            last,
            next
        );

        char* buf = weight_segment;
        size_t const remain_bytes =
            weight_segment_size_in_bytes % chunk_size_in_bytes;
        char* const buf_bound =
            buf + weight_segment_size_in_bytes - remain_bytes;
        Assertion((buf != nullptr), flag::Cuda);

        char* fst_buf = buf + lane_id * strip_chunk_size_in_bytes;
        char* snd_buf = fst_buf;
        wrap_around_sub(
            snd_buf,
            buf,
            strip_chunk_size_in_bytes,
            chunk_size_in_bytes
        );
        size_t recv_data_in_bytes = 0;

        /// \note total correctness
        // can be optimized
        while (buf < buf_bound) {
            /// \pre primary buffer points to newly received chunk from other rank (Nvl or Rdma)
            auto const last_fst = fst_buf;
            auto const last_snd = snd_buf;
            rdma_sem.wait();
            recv_data_in_bytes += strip_chunk_size_in_bytes;
            for (size_t e = 0; e < nedges; ++e) {
                /// \invariant fst_buf is primary buffer
                ncclGroupStart();
                ncclSend(
                    fst_buf,
                    strip_chunk_size_in_bytes,
                    ncclChar,
                    next,
                    comm,
                    fst_stream
                );
                ncclRecv(
                    snd_buf,
                    strip_chunk_size_in_bytes,
                    ncclChar,
                    last,
                    comm,
                    snd_stream
                );
                ncclGroupEnd();
                /// \post buf is unchanged
                recv_data_in_bytes += strip_chunk_size_in_bytes;
                /// \pre fst_buf is sent; wrap_sub sent buffer
                wrap_around_sub(
                    fst_buf,
                    buf,
                    2 * strip_chunk_size_in_bytes,
                    chunk_size_in_bytes
                );
                std::swap(fst_buf, snd_buf);
                std::swap(fst_stream, snd_stream);
                /// \post rename fst/snd buffer -> \invariant fst_buf is primary
            }
            /**
             *  \brief tackle loop \invariant
             *
             *  \note this is a Cyclic Group of order `n` !
             *
             *  \invariant snd_buf == fst_buf --
             *  \remark surfix "++" := warp_add(a, 1);
             *  \remark surfix "--" := warp_sub(a, 1);
             *  \remark snd == fst -- ;; fst <- (fst --) -- ;; swap fst snd -> snd == fst --
             *
             *  \remark additional Group Op will make fst_buf to its origin location
             *  \remark can't simply add one, because of Fermat's little law, n maybe even
             */
            wrap_around_sub(
                fst_buf,
                buf,
                strip_chunk_size_in_bytes,
                chunk_size_in_bytes
            );
            wrap_around_sub(
                snd_buf,
                buf,
                strip_chunk_size_in_bytes,
                chunk_size_in_bytes
            );
            Postcondition(
                (fst_buf == last_fst and snd_buf == last_snd),
                flag::Tanz
            );
            /// \post fst_buf in it's origin position
            /// \note tackle loop \invariant
            fst_buf += chunk_size_in_bytes;
            snd_buf += chunk_size_in_bytes;
            buf += chunk_size_in_bytes;
            std::swap(fst_stream, snd_stream);
            cudaStreamSynchronize(fst_stream);
            /// \brief notify zigzag thread
            if (not embed_ready.load(std::memory_order::relaxed)
                && recv_data_in_bytes >= embed_size_in_bytes) {
                embed_ready.store(true, std::memory_order::release);
                embed_ready.notify_one();
                continue;
            } else if (auto i = tfm_layer_cnt.load(std::memory_order::relaxed);
                       i < num_transformer_layers
                       && recv_data_in_bytes >= embed_size_in_bytes
                               + (i + 1) * tfm_layer_size_in_bytes) {
                tfm_layer_cnt.store(i + 1, std::memory_order::release);
                stp_lck_ref.post();
            }
        }
        Postcondition(
            (recv_data_in_bytes + remain_bytes == weight_segment_size_in_bytes),
            flag::Tanz
        );
        if (not(recv_data_in_bytes == weight_segment_size_in_bytes)) {
            char* res_buf = weight_segment + recv_data_in_bytes;
            size_t const strip_nbytes =
                weight_segment_size_in_bytes - recv_data_in_bytes;
            /// \note unique rank within Tanz
            if (fst_buf == buf) {
                rdma_sem.wait();
                ncclSend(
                    res_buf,
                    strip_nbytes,
                    ncclChar,
                    last,
                    comm,
                    fst_stream
                );
            } else if (snd_buf == buf) {
                ncclRecv(
                    res_buf,
                    strip_nbytes,
                    ncclChar,
                    next,
                    comm,
                    fst_stream
                );
            } else {
                ncclRecv(
                    res_buf,
                    strip_nbytes,
                    ncclChar,
                    next,
                    comm,
                    fst_stream
                );
                ncclSend(
                    res_buf,
                    strip_nbytes,
                    ncclChar,
                    last,
                    comm,
                    fst_stream
                );
            }
        }
        /// \brief all parameters loaded
        lm_head_ready.store(true, std::memory_order::release);
        lm_head_ready.notify_one();
        Postcondition((rdma_sem._lock == 0), flag::Tanz);
    }

    /// \note last & next are both machine ranks
    void nccl_bcastor_edge_inner(rank_t last, rank_t next, chain_t) {
        BZ_WARN(
            "Rank<{}>:({}) last = {} next = {}",
            rank,
            machine_rank,
            last,
            next
        );
        if (last == -1) {
            /// \brief src
            constexpr size_t chunk_size_in_bytes = 256 * 1024 * 1024;
            char* buf = weight_segment;
            size_t sent_bytes = 0;
            size_t nbytes = 0;
            Assertion((buf != nullptr), flag::Cuda);
            while (sent_bytes < weight_segment_size_in_bytes) {
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclSend(buf, nbytes, ncclChar, next, comm, fst_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(snd_stream);
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclSend(buf, nbytes, ncclChar, next, comm, snd_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(fst_stream);
            }
        } else if (next == -1) {
            /// \brief last dst
            constexpr size_t chunk_size_in_bytes = 256 * 1024 * 1024;
            char* buf = weight_segment;
            size_t sent_bytes = 0;
            size_t nbytes = 0;
            Assertion((buf != nullptr), flag::Cuda);
            while (sent_bytes < weight_segment_size_in_bytes) {
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclRecv(buf, nbytes, ncclChar, last, comm, fst_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(snd_stream);
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclRecv(buf, nbytes, ncclChar, last, comm, snd_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(fst_stream);
            }
        } else {
            /// \brief intermediate
            constexpr size_t chunk_size_in_bytes = 256 * 1024 * 1024;
            char* buf = weight_segment;
            size_t sent_bytes = 0;
            size_t nbytes = 0;
            Assertion((buf != nullptr), flag::Cuda);
            /// \invariant event is available
            while (sent_bytes < weight_segment_size_in_bytes) {
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclRecv(buf, nbytes, ncclChar, last, comm, fst_stream);
                ncclSend(buf, nbytes, ncclChar, next, comm, fst_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(snd_stream);
                nbytes = std::min(
                    chunk_size_in_bytes,
                    weight_segment_size_in_bytes - sent_bytes
                );
                ncclRecv(buf, nbytes, ncclChar, last, comm, snd_stream);
                ncclSend(buf, nbytes, ncclChar, next, comm, snd_stream);
                buf += nbytes;
                sent_bytes += nbytes;
                cudaStreamSynchronize(fst_stream);
            }
        }
    }

    void nccl_bcastor_worker_inner() {
        cudaSetDevice(device);
        init_p2p();

        while (!shutdown.load(std::memory_order::relaxed)) {
            awaken.wait(false, std::memory_order::acquire);
            auto start = std::chrono::steady_clock::now();
            rank_t const world_to_machine = rank - machine_rank;

            /// FIXME \todo
            if (work_item.nlane == 0) {
                if (rank == work_item.dst_ranks[0]) {
                    // src : send only
                    nccl_bcastor_edge_inner(
                        -1,
                        work_item.dst_ranks[1] - world_to_machine,
                        chain
                    );
                } else if (rank == work_item.dst_ranks.back()) {
                    // far end of broadcast chain
                    nccl_bcastor_edge_inner(
                        *(work_item.dst_ranks.end() - 2) - world_to_machine,
                        -1,
                        chain
                    );
                } else {
                    // intermediate
                    auto it = std::find(
                        work_item.dst_ranks.begin(),
                        work_item.dst_ranks.end() - 1,
                        rank
                    );
                    nccl_bcastor_edge_inner(
                        *(it - 1) - world_to_machine,
                        *(it + 1) - world_to_machine,
                        chain
                    );
                }
            } else {
                /**
                 *  \note form subgroup, according to Lagrange's thm
                 *
                 *  \remark can't add 2, since {0, 2, 4} and {1, 3, 5}
                 *  \remark dst_ranks {0, 1, 2, 3, 4, 5} => shuffled_ranks {5, 0, 2, 4, 1, 3, 5, 0}
                 */
                std::vector<rank_t> const dst_ranks =
                    std::move(work_item.dst_ranks);
                std::vector<rank_t> wrapped_ranks = {
                    dst_ranks.back() - world_to_machine
                };
                for (auto it = dst_ranks.cbegin(); it < dst_ranks.cend();
                     it += 1) {
                    wrapped_ranks.push_back((*it) - world_to_machine);
                }
                wrapped_ranks.push_back(dst_ranks.front() - world_to_machine);
                auto it = std::find(
                    wrapped_ranks.begin() + 1,
                    wrapped_ranks.end() - 1,
                    machine_rank
                );
                // iterator does not exceed boundary
                nccl_bcastor_edge_inner(
                    *(it - 1),
                    *(it + 1),
                    tanz,
                    it - (wrapped_ranks.begin() + 1)
                );
            }

            auto end = std::chrono::steady_clock::now();
            auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end - start
            )
                              .count();

            awaken.store(false, std::memory_order::release);
            awaken.notify_one();
            BZ_INFO(
                "Rank<{}> Broadcast Src elaspse: {}ms, Bandwidth={}GBps",
                rank,
                elapse,
                (double)weight_segment_size_in_bytes * 1000 / 1024 / 1024 / 1024
                    / elapse
            );
        }
    }

  public:
    NcclBcastor(
        MPI_Comm mpi_comm,
        ncclUniqueId commId,
        const rank_t& rank,
        const rank_t& device,
        rank_t machine_rank,
        rank_t machine_size,
        const uint32_t& num_layers,
        std::atomic_bool& embed_ready,
        size_t const embed_size_in_bytes,
        std::atomic_uint32_t& tfm_layer_cnt,
        size_t const tfm_layer_size_in_bytes,
        std::atomic_bool& lm_head_ready,
        size_t const lm_head_size_in_bytes,
        MonotonicStepLocker<uint32_t>& stp_lck
    ) :
        mpi_comm(mpi_comm),
        num_transformer_layers(num_layers),
        embed_ready(embed_ready),
        tfm_layer_cnt(tfm_layer_cnt),
        lm_head_ready(lm_head_ready),
        embed_size_in_bytes(embed_size_in_bytes),
        tfm_layer_size_in_bytes(tfm_layer_size_in_bytes),
        lm_head_size_in_bytes(lm_head_size_in_bytes),
        stp_lck_ref(stp_lck),
        rank(rank),
        device(device),
        machine_rank(machine_rank),
        machine_size(machine_size) {
        /// \pre constructor on current device
        ncclCommInitRank(&comm, machine_size, commId, machine_rank);
        cudaStreamCreateWithFlags(&fst_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&snd_stream, cudaStreamNonBlocking);
        worker =
            std::thread(std::bind(&NcclBcastor::nccl_bcastor_worker_inner, this)
            );
    }

    ~NcclBcastor() {
        shutdown.store(true, std::memory_order::release);
        worker.join();
        ncclCommDestroy(comm);
        cudaStreamDestroy(fst_stream);
        cudaStreamDestroy(snd_stream);
    }

    void set_weight_buffer(void* buf, size_t nbytes) noexcept {
        weight_segment = (char*)buf;
        weight_segment_size_in_bytes = nbytes;
    }

    void broadcast(std::vector<rank_t>&& item, chain_t) {
        assert(
            (rank - item.front() < machine_size
             && item.back() - rank < machine_size)
        );
        work_item.nlane = 0;
        work_item.dst_ranks = std::vector<rank_t> {
            std::make_move_iterator(item.begin()),
            std::make_move_iterator(item.end())
        };

        awaken.store(true, std::memory_order::release);
        awaken.notify_one();
        awaken.wait(true, std::memory_order::acquire);
    }

    void broadcast(std::vector<rank_t>&& item, tanz_t, size_t const rt) {
        /// \note rt := subscription ratio, #GPU/#NIC
        ///
        /// \remark if rt == 2, choose from 1 of consecutive ranks
        work_item.nlane = item.size();
        work_item.dst_ranks = std::move(item);

        awaken.store(true, std::memory_order::release);
        awaken.notify_one();
        awaken.wait(true, std::memory_order::acquire);
    }

    void notify_by_rdma() noexcept {
        rdma_sem.notify_one();
    }

    bool is_empty(tanz_t) noexcept {
        return rdma_sem._lock.load(std::memory_order::acquire) == 0;
    }
};

}  // namespace blitz