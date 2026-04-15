#include "param_transfer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "util/cuda_utils.h"

namespace blitz {

template<typename DType, typename IdType>
void ParamTransfer<DType, IdType>::send_params_to(
    rank_t rank,
    rank_t device,
    const int dst,
    model::Llama<DType, IdType>& model,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    Assertion((not param_thrd.joinable()), flag::Param);
    Assertion(
        (param_guard.load(std::memory_order::acquire) == false),
        flag::Param
    );
    param_guard.store(true, std::memory_order::release);
    auto __send = [rank, device, dst, &model] {
        CUDA_CHECK(cudaSetDevice(device));
        BZ_INFO("Rank<{}> send params to Rank<{}>", rank, dst);
        size_t weight_length_in_bytes = model.weight_segment_size_in_bytes;
        size_t sent_length_in_bytes = 0;

        char* weight_buffer = (char*)model.weight_segment;
        size_t size_in_bytes = model.get_embed_size_in_bytes();

        // embed
        auto handle = BlitzTccl::TcclSend(
            (void*)weight_buffer,
            size_in_bytes,
            dst,
            2
        );
        handle.wait();
        sent_length_in_bytes += size_in_bytes;
        // layers
        for (size_t i = 0; i < model.get_layer_num(); i++) {
            size_in_bytes = model.get_layer_size_in_bytes();
            auto handle = BlitzTccl::TcclSend(
                (void*)(weight_buffer + sent_length_in_bytes),
                size_in_bytes,
                dst,
                2
            );
            handle.wait();
            sent_length_in_bytes += size_in_bytes;
        }
        // lm_head
        size_in_bytes = model.get_lmhead_size_in_bytes();
        handle = BlitzTccl::TcclSend(
            (void*)(weight_buffer + sent_length_in_bytes),
            size_in_bytes,
            dst,
            2
        );
        handle.wait();
        sent_length_in_bytes += size_in_bytes;

        assert((sent_length_in_bytes == weight_length_in_bytes));
        BZ_INFO("Rank<{}> parameter send done.", rank);
    };
    param_thrd = std::thread(__send);
}

template<typename DType, typename IdType>
void ParamTransfer<DType, IdType>::send_params_in_chain(
    rank_t rank,
    rank_t device,
    const std::vector<int>& ranks_in_chain,
    const size_t chain_index,
    model::Llama<DType, IdType>& model,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    Assertion((not param_thrd.joinable()), flag::Param);
    Assertion(
        (param_guard.load(std::memory_order::acquire) == false),
        flag::Param
    );
    param_guard.store(true, std::memory_order::release);
    auto __send = [rank, device, chain_index, ranks_in_chain, &model] {
        CUDA_CHECK(cudaSetDevice(device));
        std::string chain_str =
            ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
        BZ_INFO(
            "Rank<{}> initiate params sending chain : {}",
            rank,
            chain_str
        );
        size_t weight_length_in_bytes = model.weight_segment_size_in_bytes;
        size_t sent_length_in_bytes = 0;

        char* weight_buffer = (char*)model.weight_segment;
        size_t size_in_bytes = model.get_embed_size_in_bytes();

        // embed
        BlitzTccl::TcclChainMulticast(
            (void*)weight_buffer,
            size_in_bytes,
            2,
            ranks_in_chain,
            chain_index
        );
        sent_length_in_bytes += size_in_bytes;
        // layers
        for (size_t i = 0; i < model.get_layer_num(); i++) {
            size_in_bytes = model.get_layer_size_in_bytes();
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            sent_length_in_bytes += size_in_bytes;
        }
        // lm_head
        size_in_bytes = model.get_lmhead_size_in_bytes();
        BlitzTccl::TcclChainMulticast(
            (void*)weight_buffer,
            size_in_bytes,
            2,
            ranks_in_chain,
            chain_index
        );
        sent_length_in_bytes += size_in_bytes;

        assert((sent_length_in_bytes == weight_length_in_bytes));
        BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
    };
    param_thrd = std::thread(__send);
}

template<typename DType, typename IdType>
void ParamTransfer<DType, IdType>::recv_params_from(
    rank_t rank,
    rank_t device,
    const int src,
    model::Llama<DType, IdType>& model,
    std::atomic_bool& embed_ready,
    std::atomic_uint32_t& tfm_layer_cnt,
    std::atomic_bool& lm_head_ready,
    MonotonicStepLocker<uint32_t>& mono_step_locker,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    Assertion((not param_thrd.joinable()), flag::Param);
    Assertion(
        (not param_guard.load(std::memory_order::acquire)),
        flag::Param
    );
    param_guard.store(true, std::memory_order::release);
    auto __recv = [rank, device, src, &model, &embed_ready, &tfm_layer_cnt,
                   &lm_head_ready, &mono_step_locker] {
        CUDA_CHECK(cudaSetDevice(device));
        BZ_INFO("Rank<{}> recv params from Rank<{}>", rank, src);
        size_t weight_length_in_bytes = model.weight_segment_size_in_bytes;
        size_t sent_length_in_bytes = 0;

        char* weight_buffer = (char*)model.weight_segment;
        size_t size_in_bytes = model.get_embed_size_in_bytes();

        // embed
        auto handle = BlitzTccl::TcclRecv(
            (void*)weight_buffer,
            size_in_bytes,
            src,
            2
        );
        handle.wait();
        BZ_INFO("Rank<{}> embed ready.", rank);
        embed_ready.store(true, std::memory_order::release);
        embed_ready.notify_all();
        sent_length_in_bytes += size_in_bytes;
        // layers
        for (size_t i = 0; i < model.num_layers; i++) {
            auto begin_ts = std::chrono::system_clock::now();
            size_in_bytes = model.get_layer_size_in_bytes();
            auto handle = BlitzTccl::TcclRecv(
                (void*)(weight_buffer + sent_length_in_bytes),
                size_in_bytes,
                src,
                2
            );
            handle.wait();
            auto end_ts = std::chrono::system_clock::now();
            auto last_layer_cnt =
                tfm_layer_cnt.fetch_add(1, std::memory_order::acq_rel);
            mono_step_locker.post();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_ts - begin_ts
                );
            BZ_DEBUG(
                "Rank<{}> tfm layer {} ready, elapse = {}ms",
                rank,
                last_layer_cnt + 1,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    duration
                )
                    .count()
            );
            sent_length_in_bytes += size_in_bytes;
            BZ_INFO("Rank<{}> loaded #layers = {}.", rank, last_layer_cnt);
        }
        // lm_head
        size_in_bytes = model.get_lmhead_size_in_bytes();
        handle = BlitzTccl::TcclRecv(
            (void*)(weight_buffer + sent_length_in_bytes),
            size_in_bytes,
            src,
            2
        );
        handle.wait();

        lm_head_ready.store(true, std::memory_order::release);
        lm_head_ready.notify_all();
        sent_length_in_bytes += size_in_bytes;

        assert((sent_length_in_bytes == weight_length_in_bytes));
        BZ_INFO("Rank<{}> parameter recv done.", device);
    };
    param_thrd = std::thread(__recv);
}

template<typename DType, typename IdType>
void ParamTransfer<DType, IdType>::recv_params_in_chain(
    rank_t rank,
    rank_t device,
    const std::vector<int>& ranks_in_chain,
    const size_t chain_index,
    model::Llama<DType, IdType>& model,
    std::atomic_bool& embed_ready,
    std::atomic_uint32_t& tfm_layer_cnt,
    std::atomic_bool& lm_head_ready,
    MonotonicStepLocker<uint32_t>& mono_step_locker,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    Assertion((not param_thrd.joinable()), flag::Param);
    Assertion(
        (param_guard.load(std::memory_order::acquire) == false),
        flag::Param
    );
    param_guard.store(true, std::memory_order::release);
    auto __recv = [rank, device, chain_index, ranks_in_chain, &model,
                   &embed_ready, &tfm_layer_cnt, &lm_head_ready,
                   &mono_step_locker] {
        CUDA_CHECK(cudaSetDevice(device));
        std::string chain_str =
            ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
        BZ_INFO("Rank<{}> join params casting chain : {}", rank, chain_str);
        size_t weight_length_in_bytes = model.weight_segment_size_in_bytes;
        size_t sent_length_in_bytes = 0;

        char* weight_buffer = (char*)model.weight_segment;
        size_t size_in_bytes = model.get_embed_size_in_bytes();

        // embed
        BlitzTccl::TcclChainMulticast(
            (void*)weight_buffer,
            size_in_bytes,
            2,
            ranks_in_chain,
            chain_index
        );
        sent_length_in_bytes += size_in_bytes;
        BZ_INFO("Rank<{}> embed ready.", rank);
        embed_ready.store(true, std::memory_order::release);
        embed_ready.notify_all();
        // layers
        for (size_t i = 0; i < model.num_layers; i++) {
            auto begin_ts = std::chrono::system_clock::now();
            size_in_bytes = model.get_layer_size_in_bytes();
            BlitzTccl::TcclChainMulticast(
                (void*)weight_buffer,
                size_in_bytes,
                2,
                ranks_in_chain,
                chain_index
            );
            auto end_ts = std::chrono::system_clock::now();
            auto last_layer_cnt =
                tfm_layer_cnt.fetch_add(1, std::memory_order::acq_rel);
            mono_step_locker.post();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end_ts - begin_ts
                );
            BZ_DEBUG(
                "Rank<{}> tfm layer {} ready, elapse = {}ms",
                rank,
                last_layer_cnt + 1,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    duration
                )
                    .count()
            );
            sent_length_in_bytes += size_in_bytes;
            BZ_INFO("Rank<{}> loaded #layers = {}.", rank, last_layer_cnt);
        }
        // lm_head
        size_in_bytes = model.get_lmhead_size_in_bytes();
        BlitzTccl::TcclChainMulticast(
            (void*)weight_buffer,
            size_in_bytes,
            2,
            ranks_in_chain,
            chain_index
        );
        lm_head_ready.store(true, std::memory_order::release);
        lm_head_ready.notify_all();
        sent_length_in_bytes += size_in_bytes;

        assert((sent_length_in_bytes == weight_length_in_bytes));
        BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
    };
    param_thrd = std::thread(__recv);
}

// clang-format off
template<typename DType, typename IdType>
void ParamTransfer<DType, IdType>::rdma_cast_params_in_chain_w_notify(
    rank_t rank,
    rank_t device,
    const std::vector<int>& ranks_in_chain,
    const size_t chain_index,
    tanz_t,
    size_t const nlane,
    size_t const lane_id,
    model::Llama<DType, IdType>& model,
    NcclBcastor& nccl_bcastor,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    Assertion((not param_thrd.joinable()), flag::Param);
    Assertion(
        (param_guard.load(std::memory_order::acquire) == true),
        flag::Param
    );
    auto __recv = [rank, device, chain_index, ranks_in_chain, nlane, lane_id,
                   &model, &nccl_bcastor] {
        CUDA_CHECK(cudaSetDevice(device));
        std::string chain_str =
            ::fmt::format("{}", ::fmt::join(ranks_in_chain, "~+~"));
        BZ_INFO("Rank<{}> join params casting chain : {}", rank, chain_str);

        // clang-format on
        auto chain_multicast_notify_nccl = [rank, nlane, lane_id, &nccl_bcastor](
                                               void* buffer,
                                               size_t length,
                                               const std::vector<int>&
                                                   ranks_in_chain,
                                               size_t chain_index
                                           ) {
            using namespace pickle;
            constexpr size_t strip_segment_size_in_bytes = 64 * 1024 * 1024;
            constexpr size_t rdma_chunk_bytes = 0x200000;
            constexpr size_t freq_notify =
                strip_segment_size_in_bytes / rdma_chunk_bytes;
            constexpr size_t num_handles = 16;

            Assertion(
                (strip_segment_size_in_bytes
                     % (num_handles * rdma_chunk_bytes)
                 == 0),
                flag::Tanz
            );
            /// \note add stripping
            /// \note DON'T forget the last chunk
            /// \note primary Tanz chain <-> lane_id == 0 <-> fst_buf == buf
            size_t const segment_size_in_bytes =
                strip_segment_size_in_bytes * nlane;
            size_t const num_segment = length / segment_size_in_bytes;
            size_t const remain_bytes =
                length - num_segment * segment_size_in_bytes;
            char* const base = static_cast<char*>(buffer)
                + lane_id * strip_segment_size_in_bytes;
            char* const buf_bound = static_cast<char*>(buffer) + length
                - remain_bytes
                - (nlane - lane_id - 1) * strip_segment_size_in_bytes;

            if (chain_index == 0) {
                /// \brief SendingDecode as source
                Assertion((rank == ranks_in_chain.front()), flag::Tanz);
                std::vector<Handle> handles;
                handles.reserve(num_handles);
                char* buf = base;
                size_t i = 0;
                for (; i < num_handles; ++i) {
                    handles.emplace_back(BlitzTccl::TcclSend(
                        buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index + 1],
                        2
                    ));
                    buf += rdma_chunk_bytes;
                }
                Assertion((i < freq_notify), flag::Tanz);
                while (buf < buf_bound) {
                    handles[i % num_handles].wait();
                    if ((i % freq_notify) == 0) {
                        buf += segment_size_in_bytes
                            - strip_segment_size_in_bytes;
                    }
                    handles[i % num_handles] = BlitzTccl::TcclSend(
                        buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index + 1],
                        2
                    );
                    buf += rdma_chunk_bytes;
                    i++;
                }
                if (lane_id == 0) {
                    buf += (nlane - 1) * strip_segment_size_in_bytes;
                    Assertion(
                        (buf + remain_bytes == base + length),
                        flag::Tanz
                    );
                    while (buf < base + length) {
                        handles[i % num_handles].wait();
                        size_t size = std::min(
                            (size_t)(base + length - buf),
                            rdma_chunk_bytes
                        );
                        handles[i % num_handles] = BlitzTccl::TcclSend(
                            buf,
                            size,
                            ranks_in_chain[chain_index + 1],
                            2
                        );
                        buf += rdma_chunk_bytes;
                        i++;
                    }
                }
                handles[(--i) % num_handles].wait();
            } else if (chain_index == ranks_in_chain.size() - 1) {
                Assertion((rank == ranks_in_chain.back()), flag::Tanz);
                std::vector<Handle> handles;
                handles.reserve(num_handles);
                size_t i = 0;
                char* buf = base;
                for (; i < num_handles; ++i) {
                    handles.emplace_back(BlitzTccl::TcclRecv(
                        buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index - 1],
                        2
                    ));
                    buf += rdma_chunk_bytes;
                }
                Assertion((i < freq_notify), flag::Tanz);
                while (buf < buf_bound) {
                    handles[i % num_handles].wait();
                    if ((i % freq_notify) == 0) {
                        nccl_bcastor.notify_by_rdma();
                        buf += segment_size_in_bytes
                            - strip_segment_size_in_bytes;
                    }
                    handles[i % num_handles] = BlitzTccl::TcclRecv(
                        buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index - 1],
                        2
                    );
                    buf += rdma_chunk_bytes;
                    i++;
                }
                Assertion(
                    (buf == buf_bound and i % freq_notify == 0),
                    flag::Tanz
                );
                nccl_bcastor.notify_by_rdma();
                if (lane_id == 0) {
                    BZ_WARN("Rank<{}> lane_id <{}>", rank, lane_id);
                    buf += (nlane - 1) * strip_segment_size_in_bytes;
                    Assertion(
                        (buf + remain_bytes == base + length),
                        flag::Tanz
                    );
                    while (buf < base + length) {
                        handles[i % num_handles].wait();
                        size_t size = std::min(
                            static_cast<size_t>((base + length) - buf),
                            rdma_chunk_bytes
                        );
                        handles[i % num_handles] = BlitzTccl::TcclRecv(
                            buf,
                            size,
                            ranks_in_chain[chain_index - 1],
                            2
                        );
                        buf += size;
                        i++;
                    }
                    nccl_bcastor.notify_by_rdma();
                }
                handles[(--i) % num_handles].wait();
            } else {
                std::vector<Handle> handles;
                Handle last_send_handle;
                handles.reserve(num_handles);
                size_t i = 0, j = 0;
                char* recv_buf = base;
                char* send_buf = base;
                for (; i < num_handles; ++i) {
                    handles.emplace_back(BlitzTccl::TcclRecv(
                        recv_buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index - 1],
                        2
                    ));
                    recv_buf += rdma_chunk_bytes;
                }
                Assertion((i < freq_notify), flag::Tanz);
                while (recv_buf < buf_bound) {
                    handles[i % num_handles].wait();
                    if ((i % freq_notify) == 0) {
                        nccl_bcastor.notify_by_rdma();
                        recv_buf += segment_size_in_bytes
                            - strip_segment_size_in_bytes;
                    }
                    last_send_handle = BlitzTccl::TcclSend(
                        send_buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index + 1],
                        2
                    );
                    send_buf += rdma_chunk_bytes;
                    j++;
                    if ((j % freq_notify) == 0) {
                        send_buf += segment_size_in_bytes
                            - strip_segment_size_in_bytes;
                    }
                    handles[i % num_handles] = BlitzTccl::TcclRecv(
                        recv_buf,
                        rdma_chunk_bytes,
                        ranks_in_chain[chain_index - 1],
                        2
                    );
                    recv_buf += rdma_chunk_bytes;
                    i++;
                }
                Assertion(
                    (recv_buf == buf_bound and i % freq_notify == 0),
                    flag::Tanz
                );
                nccl_bcastor.notify_by_rdma();
                const size_t i0 = i;
                if (lane_id == 0) {
                    recv_buf += (nlane - 1) * strip_segment_size_in_bytes;
                    Assertion(
                        (recv_buf + remain_bytes == base + length),
                        flag::Tanz
                    );
                    while (recv_buf < base + length) {
                        handles[i % num_handles].wait();
                        size_t send_size = std::min(
                            (size_t)(base + length - send_buf),
                            rdma_chunk_bytes
                        );
                        last_send_handle = BlitzTccl::TcclSend(
                            send_buf,
                            send_size,
                            ranks_in_chain[chain_index + 1],
                            2
                        );
                        send_buf += send_size;
                        j++;
                        if ((j % freq_notify) == 0 && j <= i0) {
                            send_buf += segment_size_in_bytes
                                - strip_segment_size_in_bytes;
                        }
                        size_t recv_size = std::min(
                            (size_t)(base + length - recv_buf),
                            rdma_chunk_bytes
                        );
                        handles[i % num_handles] = BlitzTccl::TcclRecv(
                            recv_buf,
                            recv_size,
                            ranks_in_chain[chain_index - 1],
                            2
                        );
                        recv_buf += recv_size;
                        i++;
                    }
                    nccl_bcastor.notify_by_rdma();
                }
                while (j < i) {
                    handles[j % num_handles].wait();
                    if ((j % freq_notify) == 0 && j <= i0) {
                        send_buf += segment_size_in_bytes
                            - strip_segment_size_in_bytes;
                    }
                    size_t send_size = std::min(
                        (size_t)(base + length - send_buf),
                        rdma_chunk_bytes
                    );
                    last_send_handle = BlitzTccl::TcclSend(
                        send_buf,
                        send_size,
                        ranks_in_chain[chain_index + 1],
                        2
                    );
                    send_buf += send_size;
                    j++;
                }
                handles[(--i) % num_handles].wait();
                last_send_handle.wait();
            }
        };
        // clang-format on

        char* weight_buffer = (char*)model.weight_segment;
        size_t weight_length_in_bytes = model.weight_segment_size_in_bytes;

        chain_multicast_notify_nccl(
            weight_buffer,
            weight_length_in_bytes,
            ranks_in_chain,
            chain_index
        );
        BZ_INFO("Rank<{}> params chain {} done.", rank, chain_str);
    };
    param_thrd = std::thread(__recv);
}

template<typename DType, typename IdType>
bool ParamTransfer<DType, IdType>::join_params_thread(
    rank_t rank,
    std::thread& param_thrd,
    std::atomic_bool& param_guard
) {
    if (param_thrd.joinable()) {
        param_thrd.join();
        Assertion(
            param_guard.load(std::memory_order::acquire),
            flag::Param
        );
        param_guard.store(false, std::memory_order::release);
        BZ_INFO("Rank<{}> (RDMA) parameter worker joined.", rank);
        return true;
    } else {
        return false;
    }
}

// Explicit template instantiation
template class ParamTransfer<half, int32_t>;

}  // namespace blitz
