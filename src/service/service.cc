#include "service.h"

#include <grpcpp/support/status.h>
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "blitz/batch.h"
#include "blitz/stub.h"
#include "fmt/core.h"
#include "fmt/format.h"
#include "generate.pb.h"
#include "include/logger.hpp"
#include "include/types.hpp"
#include "migration_manager.h"
#include "nccl_bcastor.h"
#include "util/cuda_utils.h"

#define HANDLE_GRPC_ERROR                                              \
    catch (const std::exception& e) {                                  \
        BZ_ERROR(e.what());                                            \
        return ::grpc::Status(::grpc::StatusCode::INTERNAL, e.what()); \
    }                                                                  \
    catch (...) {                                                      \
        BZ_ERROR("Unknown exception");                                 \
        return ::grpc::Status(                                         \
            ::grpc::StatusCode::INTERNAL,                              \
            "Unknown exception"                                        \
        );                                                             \
    }

#define CHRONO_TIMEPOINT_TO_NANOSECONDS(x)                                   \
    (std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(x) \
         .count())

template<typename DType, typename IdType>
TextGenerationServiceImpl<DType, IdType>::TextGenerationServiceImpl(
    const int world_size,
    const int rank,
    const int device,
    const char* ib_hca_name,
    std::unique_ptr<blitz::Stub<DType, IdType>> stub,
    blitz::model::GptHyperParam& hyper_param,
    blitz::HuggingfaceTokenizer&& tokenizer,
    std::vector<std::string>&& _server_urls
) :
    blitz_stub(std::move(stub)),
    next_zag_seq(0),
    hf_tokenizer(tokenizer),
    cuda_device(device),
    rank(rank),
    server_urls(std::move(_server_urls)) {
    BZ_DEBUG("TextGenerationServiceImpl constructed");
}

// Model Info
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Info(
    ::grpc::ServerContext* context,
    const ::generate::v2::InfoRequest* request,
    ::generate::v2::InfoResponse* response
) {
    return ::grpc::Status(grpc::UNIMPLEMENTED, "Return what info?");
}

// Service discovery
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::ServiceDiscovery(
    ::grpc::ServerContext* context,
    const ::generate::v2::ServiceDiscoveryRequest* request,
    ::generate::v2::ServiceDiscoveryResponse* response
) {
    try {
        BZ_DEBUG("In service discovery.");
        BZ_DEBUG(::fmt::join(server_urls, ","));
        for (const auto& url : server_urls) {
            response->add_urls(url);
        }
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

// Empties batch cache
/// \note: done
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::ClearCache(
    ::grpc::ServerContext* context,
    const ::generate::v2::ClearCacheRequest* request,
    ::generate::v2::ClearCacheResponse* response
) {
    BZ_INFO("Rank<{}> RPC ClearCache", rank);

    if (request->has_id()) {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        cached_batches_lck.lock();
        auto iter = cached_batches.find(request->id());
        cached_batches_lck.unlock();
        if (iter == cached_batches.end()) {
            std::string error_message = ::fmt::format(
                "Batch[{}] not found in server cache.",
                request->id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::OUT_OF_RANGE,
                error_message
            );
        }
        auto batch = std::move(iter->second);

        BZ_INFO(
            "Rank<{}> Batch[{}] Requests[{}] Cleared",
            rank,
            request->id(),
            ::fmt::join(batch->request_ids, ",")
        );

        for (const auto& block_table : batch->indices_2d) {
            batch->cache_manager->free(block_table);
        }
        batch->cache_manager->free_block_meta(batch->pinned_memory_idx);
        cached_batches_lck.lock();
        cached_batches.erase(iter);
        cached_batches_lck.unlock();
    }
    return ::grpc::Status::OK;
}

// Remove requests from a cached batch
/// \note: done
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::FilterBatch(
    ::grpc::ServerContext* context,
    const ::generate::v2::FilterBatchRequest* request,
    ::generate::v2::FilterBatchResponse* response
) {
    BZ_INFO("Rank<{}> RPC FilterBatch", rank);
    if (request->request_ids_size() != 0) {
        cudaSetDevice(cuda_device);
        cached_batches_lck.lock();
        auto iter = cached_batches.find(request->batch_id());
        cached_batches_lck.unlock();
        if (iter == cached_batches.end()) {
            std::string error_message = ::fmt::format(
                "Batch[{}] not found in server cache.",
                request->batch_id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::OUT_OF_RANGE,
                error_message
            );
        }
        auto batch = std::move(iter->second);
        CUDA_CHECK(cudaStreamSynchronize(0));

        uint32_t before = batch->get_max_tokens();
        batch->filter(request->request_ids());
        uint32_t after = batch->get_max_tokens();
        if (int32_t(after) < 0) {
            std::string error_message = ::fmt::format(
                "Underflow in max tokens {} {}",
                before,
                uint32_t(after)
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(::grpc::StatusCode::INTERNAL, error_message);
        }

        // Prepare Response
        ::generate::v2::CachedBatch pb_cached_batch;
        pb_cached_batch.set_id(batch->id);
        pb_cached_batch.set_size(batch->size());
        pb_cached_batch.set_max_tokens(batch->max_tokens);
        *pb_cached_batch.mutable_request_ids(
        ) = {batch->request_ids.begin(), batch->request_ids.end()};
        *response->mutable_batch() = pb_cached_batch;

        iter->second = std::move(batch);
    } else {
        std::string error_message = "Filter batch with no request ids";
        BZ_ERROR(error_message);
        return ::grpc::Status(
            ::grpc::StatusCode::INVALID_ARGUMENT,
            error_message
        );
    }
    BZ_INFO("Rank<{}> RPC FilterBatch Batch[{}]", rank, request->batch_id());

    return ::grpc::Status::OK;
}

// Warmup the model and compute max cache size
/// \todo: check
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Warmup(
    ::grpc::ServerContext* context,
    const ::generate::v2::WarmupRequest* request,
    ::generate::v2::WarmupResponse* response
) {
    BZ_TRACE("Rank<{}> RPC Warmup...", rank);
    try {
        cudaSetDevice(cuda_device);
        // clean KVCache
        // 1 / pagedattn.blocksize total memory
        // int num_tokens = model->hyper_param.max_position_embeddings;
        int num_tokens = blitz_stub->get_max_position_embeddings();
        BZ_INFO("Rank<{}> Warmup with num tokens={}", rank, num_tokens);
        std::vector<IdType> ragged_indptr = {0, num_tokens};
        std::vector<uint32_t> tmp1(num_tokens, 42), tmp2 {0};
        std::vector<std::vector<uint32_t>> tmptmp1 = {std::move(tmp1)};
        uint32_t* next_token_buf =
            (uint32_t*)blitz_stub->get_model().get_io_token_ids_buf();
        for (auto& tokens : tmptmp1) {
            std::copy(tokens.begin(), tokens.end(), next_token_buf);
            next_token_buf += tokens.size();
        }
        auto page_table = blitz::kernel::PageTable<DType, IdType>(
            1,
            blitz_stub->get_page_size(),
            blitz_stub->get_num_kv_heads() / blitz_stub->get_tp_size(),
            blitz_stub->get_head_size(),
            blitz_stub->get_base_ptr()
        );
        // todo construct a batch
        // 4096 = 256 * 16
        for (size_t i = 0; i < 256; ++i) {
            page_table.indices.push_back(i);
        }
        page_table.indices.insert(page_table.indices.end(), {0, 256, 16});
        page_table.indices_h = page_table.indices.data();
        page_table.indptr_h = page_table.indices_h + 256;
        page_table.last_page_len_h = page_table.indptr_h + 2;
        CUDA_CHECK(cudaMalloc(&page_table.indices_d, 64 * 1024));
        CUDA_CHECK(cudaMemcpy(
            page_table.indices_d,
            page_table.indices_h,
            page_table.indices.size() * sizeof(IdType),
            cudaMemcpyHostToDevice
        ));
        page_table.indptr_d = page_table.indices_d + 256;
        page_table.last_page_len_d = page_table.indptr_d + 2;

        // debug
        CUDA_CHECK(cudaStreamSynchronize(0));

        blitz_stub->get_model().forward(
            num_tokens,
            ragged_indptr.data(),
            page_table,
            blitz_stub->get_max_num_pages()
        );
        CUDA_CHECK(cudaFree(page_table.indices_d));
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

// Prefill batch and decode first token
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Prefill(
    ::grpc::ServerContext* context,
    const ::generate::v2::PrefillRequest* request,
    ::generate::v2::PrefillResponse* response
) {
    try {
        return ::grpc::Status(
            grpc::DO_NOT_USE,
            "Call Prefillv2 instead of Prefill"
        );
    }
    HANDLE_GRPC_ERROR
}

// Decode token for a list of prefilled batches
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Decode(
    ::grpc::ServerContext* context,
    const ::generate::v2::DecodeRequest* request,
    ::generate::v2::DecodeResponse* response
) {
    try {
        return ::grpc::Status(
            grpc::DO_NOT_USE,
            "Call DecodeV2 instead of Decode"
        );
    }
    HANDLE_GRPC_ERROR
}

// Health check
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Health(
    ::grpc::ServerContext* context,
    const ::generate::v2::HealthRequest* request,
    ::generate::v2::HealthResponse* response
) {
    try {
        return ::grpc::Status(grpc::UNIMPLEMENTED, "How to define Health");
    }
    HANDLE_GRPC_ERROR
}

// Send parameters to dst rank
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::SendParams(
    ::grpc::ServerContext* context,
    const ::generate::v2::SendParamsRequest* request,
    ::generate::v2::SendParamsResponse* response
) {
    using namespace blitz::flag;
    Precondition((blitz_stub->is_ready_strong()), Param);
    /// -> \invariant "param_status"
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        blitz_stub->send_params_to(request->dst());
        Postcondition((blitz_stub->is_ready_strong()), false);
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

// Receive parameters from src rank
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::RecvParams(
    ::grpc::ServerContext* context,
    const ::generate::v2::RecvParamsRequest* request,
    ::generate::v2::RecvParamsResponse* response
) {
    using namespace blitz::flag;
    Precondition((not blitz_stub->is_ready_strong()), Param);
    /// -> \invariant "param_status"
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        blitz_stub->recv_params_from(request->src());
        Postcondition((blitz_stub->is_ready_strong()), false);
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::RdmaBroadcast(
    ::grpc::ServerContext* context,
    const ::generate::v2::BroadcastRequest* request,
    ::generate::v2::BroadcastResponse* response
) {
    using namespace blitz::flag;
    Invariant(blitz_stub->status_invariant(), Param);
    try {
        using rank_t = blitz::rank_t;
        BZ_INFO("Rank<{}> Rdma Broadcasting...", rank);
        const rank_t tp_size = blitz_stub->get_tp_size();
        const rank_t tp_rank = blitz_stub->get_tp_rank();
        std::vector<rank_t> ranks_in_chain {request->src_ranks(tp_rank)};
        for (auto it = request->dst_ranks().begin() + tp_rank;
             it < request->dst_ranks().end();
             it += tp_size) {
            ranks_in_chain.push_back(*it);
        }
        Assertion((ranks_in_chain.size() >= 2), Param or Rdma);
        if (auto it =
                std::find(ranks_in_chain.begin(), ranks_in_chain.end(), rank);
            it == ranks_in_chain.end()) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Rdma Broadcasting w/ invalid arguments: src_ranks=<{}>; dst_ranks=<{}>!",
                rank,
                ::fmt::join(request->src_ranks(), ","),
                ::fmt::join(request->dst_ranks(), ",")
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::INVALID_ARGUMENT,
                error_message
            );
        }
        if (rank == ranks_in_chain[0]) {
            blitz_stub->send_params_in_chain(ranks_in_chain, 0);
        } else {
            size_t index = 1;
            for (;
                 ranks_in_chain[index] != rank && index < ranks_in_chain.size();
                 ++index) {}
            blitz_stub->recv_params_in_chain(ranks_in_chain, index);
        }
        Postcondition((blitz_stub->is_ready_strong()), Param);
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

/// \todo Define action of LoadParams
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::LoadParams(
    ::grpc::ServerContext* context,
    const ::generate::v2::LoadParamsRequest* request,
    ::generate::v2::LoadParamsResponse* response
) {
    using namespace blitz::flag;
    Precondition((not blitz_stub->is_ready_strong()), Param);
    Invariant((true && not Rdma), Param);
    try {
        BZ_INFO("Rank<{}> RPC LoadParams", rank);
        cudaSetDevice(cuda_device);
        switch (request->load_case()) {
            using namespace generate::v2;
            case LOAD_FROM_DISK: {
                BZ_INFO("Rank<{}> load parameter from disk...", rank);
                assert(request->has_model_path());
                std::filesystem::path model_path = request->model_path();
                std::filesystem::path document = fmt::format(
                    "dangertensors.{}.bin",
                    blitz_stub->get_tp_rank()
                );
                blitz_stub->load_params_from_disk(
                    (model_path / document).string(),
                    rank
                );
                break;
            }
            case LOAD_FROM_HOST_MEM: {
                BZ_INFO("Rank<{}> load parameter from host memory...", rank);
                blitz_stub->load_params_from_host_memory();
                break;
            }
        }
        blitz_stub->set_status_ready();
        Postcondition((blitz_stub->is_ready_strong()), false);
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::PrefillV2(
    ::grpc::ServerContext* context,
    const ::generate::v2::PrefillV2Request* request,
    ::generate::v2::PrefillV2Response* response
) {
    std::vector<uint64_t> req_ids;
    for (const auto& req : request->batch().requests()) {
        req_ids.push_back(req.id());
    }
    BZ_INFO(
        "Rank<{}> RPC PrefillV2 Batch[{}] Requests[{}] prefillv2 case {}, max_num_tokens={}",
        rank,
        request->batch().id(),
        ::fmt::join(req_ids, ","),
        request->forward_case(),
        request->batch().max_tokens()
    );

    try {
        auto start_time = std::chrono::steady_clock::now();

        cudaSetDevice(cuda_device);
        auto pb_batch = request->batch();
        Batch_t<DType, IdType> batch;
        uint32_t f_case = request->forward_case();

        std::vector<int64_t> cur_iter_output_tokens;
        std::optional<std::future<void>> maybe_unfinished_nccl_send;
        typename decltype(cached_batches)::iterator iter;

        std::vector<uint64_t> rids {};
        for (const auto& r : request->batch().requests()) {
            rids.push_back(r.id());
        }

        switch (f_case) {
            case generate::v2::PrefillCase::PREFILL_CASE_NORMAL: {
                batch = std::make_unique<blitz::Batch<DType, IdType>>(
                    blitz::batch::prefill,
                    pb_batch,
                    &blitz_stub->get_cache_manager(),
                    hf_tokenizer
                );
                BZ_DEBUG(
                    "Rank<{}> Batch[{}] Normal",
                    rank,
                    request->batch().id()
                );
                if (!blitz_stub->is_ready_weak()
                    && !blitz_stub->is_ready_strong()) {
                    auto tmp = blitz_stub->get_model_status();
                    std::string error_message = ::fmt::format(
                        "Rank<{}> illegal call with CASE_NORMAL: blitz instance not ready ({}::{}::{})",
                        blitz_stub->get_rank(),
                        std::get<0>(tmp),
                        std::get<1>(tmp),
                        std::get<2>(tmp)
                    );
                    BZ_ERROR(error_message);
                    return ::grpc::Status(
                        ::grpc::StatusCode::FAILED_PRECONDITION,
                        error_message
                    );
                }
                blitz_stub->forward(blitz::forward::normal, *batch);
                // put batch back
                bool is_new_batch_id;
                cached_batches_lck.lock();
                std::tie(iter, is_new_batch_id) =
                    cached_batches.try_emplace(pb_batch.id(), std::move(batch));
                cached_batches_lck.unlock();
                assert(is_new_batch_id);
                break;
            }
            // [cfg=sche_naive_pp || sche_zag] OldPrefill replica
            case (
                generate::v2::PrefillCase::PREFILL_CASE_IMMIGRATE
                | generate::v2::PrefillCase::PREFILL_CASE_NAIVE_PP
            ): {
                batch = std::make_unique<blitz::Batch<DType, IdType>>(
                    blitz::batch::prefill,
                    pb_batch,
                    &blitz_stub->get_cache_manager(),
                    hf_tokenizer
                );
                BZ_INFO(
                    "Rank<{}> Batch[{}] ZagPrefill(OldReplica) peer={}",
                    rank,
                    request->batch().id(),
                    request->pipe_peer(blitz_stub->get_tp_rank())
                );
                // update batch inside
                generate::v2::PipeParaInfo next_pp_info;
                blitz_stub->forward(
                    blitz::forward::naivepp,
                    *batch,
                    request->pp_info(),
                    request->pipe_peer(blitz_stub->get_tp_rank()),
                    next_pp_info
                );
                // put batch back
                bool is_new_batch_id;
                cached_batches_lck.lock();
                std::tie(iter, is_new_batch_id) =
                    cached_batches.try_emplace(pb_batch.id(), std::move(batch));
                cached_batches_lck.unlock();
                assert(is_new_batch_id);
                BZ_INFO(
                    "Rank<{}> Batch[{}] NaivePP(OldReplica) ready to return",
                    rank,
                    request->batch().id()
                );
                *response->mutable_pp_info() = next_pp_info;
                // no need to put batch back
                break;
            }
            default: {
                BZ_ERROR("Unimplemented PrefillV2 case [{}]", f_case);
            }
        }

        auto stop_time = std::chrono::steady_clock::now();

        // Prepare response
        const auto& batch_ref = *iter->second;

        ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>
            pb_generations;
        ::generate::v2::CachedBatch pb_cached_batch;

        /// \todo updated
        std::tie(pb_generations, pb_cached_batch) =
            batch_ref.to_pb(hf_tokenizer);
        *response->mutable_generations() = std::move(pb_generations);
        *response->mutable_batch() = std::move(pb_cached_batch);

        response->set_total_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );
        response->set_forward_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );
        response->set_decode_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );

        BZ_INFO(
            "Rank<{}> Batch[{}] prefillv2 case {} done",
            rank,
            batch_ref.id,
            request->forward_case()
        );

        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

// Decode token for a list of prefilled batches
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::DecodeV2(
    ::grpc::ServerContext* context,
    const ::generate::v2::DecodeV2Request* request,
    ::generate::v2::DecodeV2Response* response
) {
    std::vector<uint64_t> s_ids;
    for (const auto& b : request->batches()) {
        s_ids.push_back(b.id());
    }
    // BZ_INFO(
    //     "Rank<{}> RPC DecodeV2 Batches[{}]",
    //     rank,
    //     ::fmt::join(s_ids, ",")
    // );
    if (!request->last_iter_tokens().empty()) {
        // assert((!parallel_config.is_last_stage()
        //         || migration_manager->is_decode_node()));
    }
    try {
        auto start_time = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaSetDevice(cuda_device));
        std::vector<std::reference_wrapper<blitz::Batch<DType, IdType>>>
            batch_refs;
        Batch_t<DType, IdType> extend_batch = nullptr;

        auto last_iter_output_tokens = request->last_iter_tokens().begin();
        std::vector<typename decltype(cached_batches)::iterator> iter_vec;

        for (auto b : request->batches()) {
            cached_batches_lck.lock();
            auto iter = cached_batches.find(b.id());
            cached_batches_lck.unlock();
            if (iter == cached_batches.end()) {
                std::string error_message = ::fmt::format(
                    "Rank<{}> Batch[{}] not found in server cache",
                    rank,
                    b.id()
                );
                BZ_ERROR(error_message);
                return ::grpc::Status(
                    ::grpc::StatusCode::OUT_OF_RANGE,
                    error_message
                );
            }
            blitz::Batch<DType, IdType>& batch_mref = *iter->second;
            // Update Decode tokens
            if (!request->last_iter_tokens().empty()) {
                if (!last_iter_output_tokens->ids().empty()) {
                    uint32_t before = batch_mref.get_max_tokens();
                    batch_mref.update(
                        {last_iter_output_tokens->ids().begin(),
                         last_iter_output_tokens->ids().end()}
                    );
                    uint32_t after = batch_mref.get_max_tokens();
                    if (int32_t(after) < 0) {
                        BZ_ERROR(
                            "Rank<{}> Underflow in max tokens {} {}",
                            rank,
                            before,
                            uint32_t(after)
                        );
                        return ::grpc::Status(
                            ::grpc::StatusCode::OUT_OF_RANGE,
                            "Underflow in max tokens"
                        );
                    }
                }
                last_iter_output_tokens++;
            }
            if (!extend_batch) {
                extend_batch = std::move(iter->second);
            } else {
                batch_refs.emplace_back(batch_mref);
                iter_vec.emplace_back(std::move(iter));
            }
        }
        assert((last_iter_output_tokens == request->last_iter_tokens().end()));

        Batch_t<DType, IdType> batch;

        if (batch_refs.size() > 0) {
            uint32_t before = extend_batch->get_max_tokens();
            extend_batch->extend(batch_refs);
            uint32_t after = extend_batch->get_max_tokens();
            if (int32_t(after) < 0) {
                BZ_ERROR(
                    "Rank<{}> Underflow in max tokens {} {}",
                    rank,
                    before,
                    uint32_t(after)
                );
                return ::grpc::Status(
                    ::grpc::StatusCode::OUT_OF_RANGE,
                    "Underflow in max tokens"
                );
            }

            batch = std::move(extend_batch);
            BZ_DEBUG("RANK<{}> new batch size {}", rank, batch->size());
            for (auto it : iter_vec) {
                cached_batches_lck.lock();
                cached_batches.erase(it);
                cached_batches_lck.unlock();
            }
        } else {
            batch = std::move(extend_batch);
        }

        // forward
        if (blitz_stub->is_ready_weak()) {
            blitz_stub->forward(blitz::forward::normal, *batch);
        } else if (blitz_stub->is_ready_strong()) {
            blitz_stub->forward(blitz::forward::normal, *batch);
        } else {
            throw std::runtime_error(
                "Try decoding in an unparameterised Blitz instance!"
            );
        }

        // put the unique_ptr back
        typename decltype(cached_batches)::iterator iter;
        bool is_new_batch_id;
        cached_batches_lck.lock();
        std::tie(iter, is_new_batch_id) =
            cached_batches.insert_or_assign(batch->id, std::move(batch));
        cached_batches_lck.unlock();
        assert((!is_new_batch_id));

        auto stop_time = std::chrono::steady_clock::now();

        // Prepare response
        const blitz::Batch<DType, IdType>& batch_ref = *(iter->second);
        ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>
            pb_generations;
        ::generate::v2::CachedBatch pb_cached_batch;
        std::tie(pb_generations, pb_cached_batch) =
            batch_ref.to_pb(hf_tokenizer);
        *response->mutable_generations() = std::move(pb_generations);
        *response->mutable_batch() = std::move(pb_cached_batch);
        response->set_total_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );

        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::ZagPrefill(
    ::grpc::ServerContext* context,
    const ::generate::v2::ZagPrefillRequest* request,
    ::generate::v2::PrefillV2Response* response
) {
    std::vector<uint64_t> req_ids;
    for (const auto& req : request->batch().requests()) {
        req_ids.push_back(req.id());
    }
    BZ_INFO(
        "Rank<{}> RPC ZagPrefill Batch[{}] Requests[{}] seq_num={}",
        rank,
        request->batch().id(),
        ::fmt::join(req_ids, ","),
        request->zag_seq_num()
    );

    try {
        auto start_time = std::chrono::steady_clock::now();
        cudaSetDevice(cuda_device);
        const auto& pb_batch = request->batch();

        Batch_t<DType, IdType> batch =
            std::make_unique<blitz::Batch<DType, IdType>>(
                blitz::batch::prefill,
                pb_batch,
                &blitz_stub->get_cache_manager(),
                hf_tokenizer
            );
        const uint32_t f_case = request->forward_case();
        uint32_t zag_seq_num = request->zag_seq_num();

        generate::v2::PipeParaInfo next_pp_info;
        std::optional<std::future<void>> maybe_unfinished_nccl_send;
        std::future<generate::v2::PipeParaInfo> future;

        assert((f_case == generate::v2::PrefillCase::PREFILL_CASE_NAIVE_PP));
        /// @note serialize ZagPrefill RPC
        {
            uint32_t tmp;
            for (;;) {
                if ((tmp = this->next_zag_seq.load(std::memory_order::acquire))
                    == zag_seq_num) {
                    break;
                }
                while ((tmp =
                            this->next_zag_seq.load(std::memory_order::relaxed))
                       != zag_seq_num) {
                    __builtin_ia32_pause();
                }
            }
            assert((tmp == zag_seq_num));
        }
        // protect by this counter lock
        future =
            blitz_stub
                ->add_zag_task(*batch, -1, zag_seq_num, request->pp_info());
        // release this counter lock
        this->next_zag_seq.store(zag_seq_num + 1, std::memory_order::release);

        /// \remark wait here...
        next_pp_info = future.get();
        BZ_INFO(
            "Rank<{}> ZagPrefill Batch[{}] ready to return w/ PPInfo {} TfmLayer {}",
            rank,
            batch->id,
            int(next_pp_info.start_layer_case()),
            next_pp_info.tfm_layer()
        );

        /// \todo delegate `update` to Stub?
        *response->mutable_pp_info() = next_pp_info;

        /// \brief epilogue
        typename decltype(cached_batches)::iterator iter;
        bool is_new_batch_id;
        cached_batches_lck.lock();
        std::tie(iter, is_new_batch_id) =
            cached_batches.try_emplace(pb_batch.id(), std::move(batch));
        cached_batches_lck.unlock();
        assert(is_new_batch_id);

        auto stop_time = std::chrono::steady_clock::now();

        // Prepare response
        const blitz::Batch<DType, IdType>& batch_ref = *(iter->second);
        ::google::protobuf::RepeatedPtrField<::generate::v2::Generation>
            pb_generations;
        ::generate::v2::CachedBatch pb_cached_batch;

        // If add layers are done
        // commonly, there shouldn't be
        uint32_t exec_tfm_layers = std::accumulate(
            next_pp_info.num_layer_per_rank().begin(),
            next_pp_info.num_layer_per_rank().end(),
            0u
        );
        if (exec_tfm_layers == blitz_stub->get_num_layers()) {
            assert(
                (next_pp_info.start_layer_case()
                 == next_pp_info.START_LAYER_NOT_SET)
            );
            if (!blitz_stub->is_ready_strong()) {
                blitz_stub->report_status();
            }
            assert((blitz_stub->is_ready_strong()));
            std::tie(pb_generations, pb_cached_batch) =
                batch_ref.to_pb(hf_tokenizer);
            *response->mutable_generations() = std::move(pb_generations);
            *response->mutable_batch() = std::move(pb_cached_batch);
        }
        response->set_total_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );
        response->set_forward_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );
        response->set_decode_ns(
            CHRONO_TIMEPOINT_TO_NANOSECONDS(stop_time - start_time)
        );

        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Migrate(
    ::grpc::ServerContext* context,
    const ::generate::v2::MigrateRequest* request,
    ::generate::v2::MigrateResponse* response
) {
    assert((request->src(blitz_stub->get_tp_rank()) == rank));
    blitz::rank_t peer = request->dst(blitz_stub->get_tp_rank());
    BZ_INFO(
        "Rank<{}> Migrate Batch[{}] -~> Rank<{}>",
        rank,
        request->batch().id(),
        peer
    );
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        auto migrate_start_time = std::chrono::steady_clock::now();

        cached_batches_lck.lock();
        auto iter = cached_batches.find(request->batch().id());
        cached_batches_lck.unlock();
        if (iter == cached_batches.end()) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Batch[{}] not found in server cache",
                rank,
                request->batch().id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::OUT_OF_RANGE,
                error_message
            );
        }

        auto& batch_ref = *(iter->second);
        auto& mm = blitz_stub->get_migration_manager();
        /// \todo support TP
        bool success = mm.send_kv_cache(batch_ref, blitz::all_layer, peer);
        if (!success) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Batch[{}] failed to send kv_cache",
                rank,
                request->batch().id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::FAILED_PRECONDITION,
                error_message
            );
        }

        // clear batch kvcache
        batch_ref.cache_manager->free(batch_ref.page_table.indices);
        batch_ref.cache_manager->free_block_meta(batch_ref.pinned_memory_idx);

        cached_batches_lck.lock();
        cached_batches.erase(iter);
        cached_batches_lck.unlock();
        auto migrate_end_time = std::chrono::steady_clock::now();
        BZ_INFO(
            "Rank<{}> (migrate) Batch[{}] -~> Rank<{}> elapsed {}ms",
            rank,
            request->batch().id(),
            peer,
            std::chrono::duration<double, std::milli>(
                migrate_end_time - migrate_start_time
            )
                .count()
        );
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Immigrate(
    ::grpc::ServerContext* context,
    const ::generate::v2::ImmigrateRequest* request,
    ::generate::v2::ImmigrateResponse* response
) {
    assert((request->dst(blitz_stub->get_tp_rank()) == rank));
    blitz::rank_t peer = request->src(blitz_stub->get_tp_rank());
    BZ_INFO(
        "Rank<{}> Immigrate Batch[{}] <~- Rank<{}>",
        rank,
        request->batch().id(),
        peer
    );
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        // create new batch
        const auto& pb_batch = request->batch();
        typename decltype(cached_batches)::iterator iter;
        typedef blitz::Batch<DType, IdType> B;
        bool is_new_batch_id;
        auto& cache_manager = blitz_stub->get_cache_manager();
        cached_batches_lck.lock();
        /// \note use prefill Batch constructor
        ///       to ensure num_prompt_blocks consistent
        std::tie(iter, is_new_batch_id) = cached_batches.try_emplace(
            pb_batch.id(),
            std::make_unique<B>(
                blitz::batch::prefill,
                pb_batch,
                &cache_manager,
                hf_tokenizer
            )
        );
        cached_batches_lck.unlock();
        if (!is_new_batch_id) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Batch[{}] already exists",
                rank,
                pb_batch.id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::ALREADY_EXISTS,
                error_message
            );
        }

        auto& mm = blitz_stub->get_migration_manager();
        blitz::Batch<DType, IdType>& batch_mref = *(iter->second);
        bool success = mm.recv_kv_cache(batch_mref, blitz::all_layer, peer);
        if (!success) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Batch[{}] failed to receive kv_cache",
                rank,
                request->batch().id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::FAILED_PRECONDITION,
                error_message
            );
        }
        /// \bug HACK
        batch_mref.num_tokens = 0;
        batch_mref.ragged_indptr.clear();
        auto pb_cached_batch = batch_mref.to_pb();
        *response->mutable_batch() = std::move(pb_cached_batch);
        BZ_INFO(
            "Rank<{}> Imigrate Batch[{}] <~- Rank<{}> done.",
            rank,
            request->batch().id(),
            peer
        );
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

// Service discovery
template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::WaitRdmaDone(
    ::grpc::ServerContext* context,
    const ::generate::v2::WaitRdmaDoneRequest* request,
    ::generate::v2::WaitRdmaDoneResponse* response
) {
    try {
        if (blitz_stub->join_params_thread()) {
            Postcondition((blitz_stub->is_ready_strong()), blitz::flag::Rdma);
            return ::grpc::Status::OK;
        } else {
            std::string error_message =
                "How can you wait for an unstarted worker?";
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::INVALID_ARGUMENT,
                error_message
            );
        }
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::ResetStatus(
    ::grpc::ServerContext* context,
    const ::generate::v2::ResetStatusRequest* request,
    ::generate::v2::ResetStatusResponse* response
) {
    using namespace blitz::flag;
    BZ_INFO("Rank<{}> reset status", blitz_stub->get_rank());
    /// \invariant "param_status"
    Invariant(blitz_stub->status_invariant(), Param);
    // noexcept
    blitz_stub->reset_status();
    this->next_zag_seq.store(0, std::memory_order::release);
    return ::grpc::Status::OK;
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::SetStatusReady(
    ::grpc::ServerContext* context,
    const ::generate::v2::SetStatusReadyRequest* request,
    ::generate::v2::SetStatusReadyResponse* response
) {
    using namespace blitz::flag;
    BZ_INFO("Rank<{}> set status ready", blitz_stub->get_rank());
    /// \invariant "param_status"
    Invariant(blitz_stub->status_invariant(), Param);
    // disallowed backdoor for BlitzScale router
    Assertion((not Rdma), Param && Rdma);
    // noexcept
    blitz_stub->set_status_ready();
    return ::grpc::Status::OK;
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::Relay(
    ::grpc::ServerContext* context,
    const ::generate::v2::RelayRequest* request,
    ::generate::v2::RelayResponse* response
) {
    BZ_INFO(
        "Rank<{}> relay... not head? {}",
        blitz_stub->get_rank(),
        request->relax_not_head()
    );
    try {
        auto [batch_id, seq_num] = blitz_stub->relay_zag_task(
            request->rank(),
            request->relax_not_head()
        );
        if (batch_id != -1) {
            response->set_batch_id((uint64_t)batch_id);
            response->set_seq_num(seq_num);
            BZ_INFO(
                "Rank<{}> relay Some(Batch[{}] <: zigzag#{})",
                blitz_stub->get_rank(),
                batch_id,
                seq_num
            );
        } else {
            BZ_INFO("Rank<{}> relay None", rank);
        }
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::MigratePartial(
    ::grpc::ServerContext* context,
    const ::generate::v2::MigratePartialRequest* request,
    ::generate::v2::MigrateResponse* response
) {
    assert((request->src(blitz_stub->get_tp_rank()) == rank));
    blitz::rank_t peer = request->dst(blitz_stub->get_tp_rank());
    BZ_INFO(
        "Rank<{}> MigratePartial Batch[{}] {} half -~> Rank<{}>",
        rank,
        request->batch().id(),
        (request->fst_or_snd() == ::generate::v2::PARTIAL_CASE_FIRST ? "fst"
                                                                     : "snd"),
        peer
    );
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        auto start = std::chrono::steady_clock::now();

        const auto& pb_batch = request->batch();
        cached_batches_lck.lock();
        auto iter = cached_batches.find(pb_batch.id());
        cached_batches_lck.unlock();
        if (iter == cached_batches.end()) {
            std::string error_message = ::fmt::format(
                "Batch[{}] not found in server cache!",
                pb_batch.id()
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::OUT_OF_RANGE,
                error_message
            );
        }
        auto& batch_ref = *(iter->second);
        switch (request->fst_or_snd()) {
            case generate::v2::PARTIAL_CASE_FIRST: {
                auto& mm = blitz_stub->get_migration_manager();
                // assertions inside
                mm.send_kv_cache(
                    batch_ref,
                    blitz::fst_half,
                    peer,
                    request->num_layer()
                );
                break;
            }
            case generate::v2::PARTIAL_CASE_SECOND: {
                auto& mm = blitz_stub->get_migration_manager();
                // assertions inside
                mm.send_kv_cache(
                    batch_ref,
                    blitz::snd_half,
                    peer,
                    request->num_layer()
                );
                break;
            }
                return ::grpc::Status(
                    ::grpc::StatusCode::INVALID_ARGUMENT,
                    "PartialMigrate fst_or_snd not set!"
                );
        }
        // clear batch kvcache
        batch_ref.cache_manager->free(batch_ref.page_table.indices);
        batch_ref.cache_manager->free_block_meta(batch_ref.pinned_memory_idx);

        cached_batches_lck.lock();
        cached_batches.erase(iter);
        cached_batches_lck.unlock();
        auto stop = std::chrono::steady_clock::now();
        BZ_INFO(
            "Rank<{}> MigratePartial Batch[{}] {} half -~> Rank<{}> elapsed {}ms",
            rank,
            request->batch().id(),
            (request->fst_or_snd() == ::generate::v2::PARTIAL_CASE_FIRST
                 ? "fst"
                 : "snd"),
            peer,
            std::chrono::duration<double, std::milli>(stop - start).count()
        );
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::ImmigratePartial(
    ::grpc::ServerContext* context,
    const ::generate::v2::ImmigratePartialRequest* request,
    ::generate::v2::ImmigrateResponse* response
) {
    assert((request->dst(blitz_stub->get_tp_rank()) == rank));
    blitz::rank_t peer = request->src(blitz_stub->get_tp_rank());
    BZ_INFO(
        "Rank<{}> ImigratePartial Batch[{}] {} half <~- Rank<{}>...",
        rank,
        request->batch().id(),
        (request->fst_or_snd() == ::generate::v2::PARTIAL_CASE_FIRST ? "fst"
                                                                     : "snd"),
        peer
    );
    try {
        CUDA_CHECK(cudaSetDevice(cuda_device));
        switch (request->fst_or_snd()) {
            case generate::v2::PARTIAL_CASE_FIRST: {
                // create new batch
                const auto& pb_batch = request->batch();
                typename decltype(cached_batches)::iterator iter;
                using B = blitz::Batch<DType, IdType>;
                bool is_new_batch_id;
                auto& cache_manager = blitz_stub->get_cache_manager();
                cached_batches_lck.lock();
                /// \note use prefill Batch constructor
                ///       to ensure num_prompt_blocks consistent
                std::tie(iter, is_new_batch_id) = cached_batches.try_emplace(
                    pb_batch.id(),
                    std::make_unique<B>(
                        blitz::batch::prefill,
                        pb_batch,
                        &cache_manager,
                        hf_tokenizer
                    )
                );
                cached_batches_lck.unlock();
                assert(is_new_batch_id);
                snd_wait_fst_half_layers.notify_one(pb_batch.id());

                auto& mm = blitz_stub->get_migration_manager();
                B& batch_mref = *(iter->second);
                mm.recv_kv_cache(
                    batch_mref,
                    blitz::fst_half,
                    peer,
                    request->num_layer()
                );
                /// \bug HACK
                batch_mref.num_tokens = 0;
                batch_mref.ragged_indptr.clear();
                auto pb_cached_batch = batch_mref.to_pb();
                *response->mutable_batch() = std::move(pb_cached_batch);
                break;
            }
            case generate::v2::PARTIAL_CASE_SECOND: {
                /// \bug no sync mechanism for fst_half pred snd_half
                // get batch
                const auto& pb_batch = request->batch();
                auto batch_id = snd_wait_fst_half_layers.wait();
                assert((batch_id == pb_batch.id()));
                cached_batches_lck.lock();
                auto iter = cached_batches.find(pb_batch.id());
                cached_batches_lck.unlock();
                if (iter == cached_batches.end()) {
                    std::string error_message = ::fmt::format(
                        "Batch[{}] not found in server cache! Maybe add Sync for snd_half >>= fst_half",
                        pb_batch.id()
                    );
                    BZ_ERROR(error_message);
                    return ::grpc::Status(
                        ::grpc::StatusCode::OUT_OF_RANGE,
                        error_message
                    );
                }
                /**
                 * \todo add more Assertions!
                 */
                auto& mm = blitz_stub->get_migration_manager();
                auto& batch_mref = *(iter->second);
                mm.recv_kv_cache(
                    batch_mref,
                    blitz::snd_half,
                    peer,
                    request->num_layer()
                );

                auto pb_cached_batch = batch_mref.to_pb();
                *response->mutable_batch() = std::move(pb_cached_batch);
                break;
            }
                return ::grpc::Status(
                    ::grpc::StatusCode::INVALID_ARGUMENT,
                    "PartialImmigrate fst_or_snd not set!"
                );
        }
        BZ_INFO(
            "Rank<{}> ImigratePartial Batch[{}] {} half <~- Rank<{}> done.",
            rank,
            request->batch().id(),
            (request->fst_or_snd() == ::generate::v2::PARTIAL_CASE_FIRST
                 ? "fst"
                 : "snd"),
            peer
        );
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::NvlBroadcast(
    ::grpc::ServerContext* context,
    const ::generate::v2::BroadcastRequest* request,
    ::generate::v2::BroadcastResponse* response
) {
    Invariant(blitz_stub->status_invariant(), blitz::flag::Param);
    try {
        using rank_t = blitz::rank_t;
        BZ_INFO("Rank<{}> Nvl Broadcasting...", rank);
        const rank_t tp_size = blitz_stub->get_tp_size();
        const rank_t tp_rank = blitz_stub->get_tp_rank();
        assert(request->src_ranks_size() == tp_size);
        std::vector<rank_t> ranks {request->src_ranks(blitz_stub->get_tp_rank())
        };
        for (auto it = request->dst_ranks().begin() + tp_rank;
             it < request->dst_ranks().end();
             it += tp_size) {
            ranks.push_back(*it);
        }
        if (auto it = std::find(ranks.begin(), ranks.end(), rank);
            it == ranks.end()) {
            std::string error_message = ::fmt::format(
                "Rank<{}> Nvl Broadcasting w/ invalid arguments: src_ranks=<{}>; dst_ranks=<{}>!",
                rank,
                ::fmt::join(request->src_ranks(), ","),
                ::fmt::join(request->dst_ranks(), ",")
            );
            BZ_ERROR(error_message);
            return ::grpc::Status(
                ::grpc::StatusCode::INVALID_ARGUMENT,
                error_message
            );
        }
        blitz_stub->nccl_broadcast(std::move(ranks));
        blitz_stub->set_status_ready();
        Postcondition((blitz_stub->is_ready_weak()), false);
        return ::grpc::Status::OK;
    }
    HANDLE_GRPC_ERROR
}

template<typename DType, typename IdType>
::grpc::Status TextGenerationServiceImpl<DType, IdType>::TanzBroadcast(
    ::grpc::ServerContext* context,
    const ::generate::v2::BroadcastRequest* request,
    ::generate::v2::BroadcastResponse* response
) {
    /// \invariant "param_status"
    Invariant(blitz_stub->status_invariant(), blitz::flag::Param);
    try {
        using rank_t = blitz::rank_t;
        /**
         *  \brief validate arguments
         *  
         *  \param src_ranks {x, x1}, a list of SendingDecode, each is the src of Tanz chain
         *  \param dst_ranks {0, 1, 4, 5} ranks w/in one machine, join the Tanz chain
         *
         *  \pre  size dst_ranks == size src_ranks * (#GPU / #RNIC)
         *  \remark "subscription_ratio" := (#GPU / #RNIC)
         *  \note   if n == 2 then dst_ranks is viewed as {{0, 1}, {4, 5}}  
         */
        BZ_INFO("Rank<{}> Tanz Broadcasting...", rank);
        const rank_t tp_size = blitz_stub->get_tp_size();
        const rank_t tp_rank = blitz_stub->get_tp_rank();
        std::vector<rank_t> src_ranks;
        std::vector<rank_t> dst_ranks;
        for (auto it = request->src_ranks().begin() + tp_rank;
             it < request->src_ranks().end();
             it += tp_size) {
            src_ranks.push_back(*it);
        }
        for (auto it = request->dst_ranks().begin() + tp_rank;
             it < request->dst_ranks().end();
             it += tp_size) {
            dst_ranks.push_back(*it);
        }
        const size_t rt = dst_ranks.size() / src_ranks.size();
        /// \pre every rank has the same view of src/dst ranks
        for (size_t i = 0; i < src_ranks.size(); ++i) {
            /// \brief is source rank, initiate RDMA
            if (src_ranks[i] == rank) {
                std::vector<rank_t> ranks_in_chain = {rank};
                ranks_in_chain.insert(
                    ranks_in_chain.end(),
                    dst_ranks.cbegin() + i * rt,
                    dst_ranks.cbegin() + (i + 1) * rt
                );
                blitz_stub->get_status_mutex();
                blitz_stub->rdma_cast_params_in_chain_w_notify(
                    ranks_in_chain,
                    0,
                    blitz::tanz,
                    src_ranks.size(),
                    i
                );
                bool error = not blitz_stub->join_params_thread();
                /// \post "param_status"
                if (error) {
                    std::string error_message = ::fmt::format(
                        "Rank<{}> (Tanz, src) RDMA chain casting error!",
                        rank
                    );
                    BZ_ERROR(error_message);
                    return ::grpc::Status(
                        grpc::StatusCode::INTERNAL,
                        error_message
                    );
                }
                return ::grpc::Status::OK;
            }
        }
        for (size_t i = 0; i < dst_ranks.size(); ++i) {
            /// \brief is target rank, join RDMA & initiate NVL
            if (dst_ranks[i] == rank) {
                blitz_stub->get_status_mutex();
                size_t j = i / rt;
                std::vector<rank_t> ranks_in_rdma_chain = {src_ranks[j]};
                ranks_in_rdma_chain.insert(
                    ranks_in_rdma_chain.end(),
                    dst_ranks.cbegin() + j * rt,
                    dst_ranks.cbegin() + (j + 1) * rt
                );
                /// \pre paramater guard acquired
                blitz_stub->rdma_cast_params_in_chain_w_notify(
                    ranks_in_rdma_chain,
                    i - j * rt + 1,
                    blitz::tanz,
                    src_ranks.size(),
                    j
                );
                std::vector<rank_t> ranks_in_tanz_chain;
                for (auto it = dst_ranks.cbegin() + i % rt;
                     it < dst_ranks.cend();
                     it += rt) {
                    ranks_in_tanz_chain.push_back(*it);
                }
                /// \note Sync operation
                blitz_stub->tanz_broadcast(std::move(ranks_in_tanz_chain), rt);
                bool error = not blitz_stub->join_params_thread();
                /// \post parameter guard released
                if (error) {
                    std::string error_message = ::fmt::format(
                        "Rank<{}> (Tanz, dst) RDMA chain casting error!",
                        rank
                    );
                    BZ_ERROR(error_message);
                    return ::grpc::Status(
                        grpc::StatusCode::INTERNAL,
                        error_message
                    );
                }
                /// \pre parameter guard released
                blitz_stub->set_status_ready();
                return ::grpc::Status::OK;
            }
        }
        std::string error_message = ::fmt::format(
            "Rank<{}> invalid Tanz argument: src=<{}>, dst=<{}>",
            rank,
            ::fmt::join(request->src_ranks(), ","),
            ::fmt::join(request->src_ranks(), ",")
        );
        BZ_ERROR(error_message);
        return ::grpc::Status(
            ::grpc::StatusCode::INVALID_ARGUMENT,
            error_message
        );
    }
    HANDLE_GRPC_ERROR
}