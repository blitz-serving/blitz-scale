#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/service_type.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "blitz/stub.h"
#include "include/spinlock.hpp"
#include "include/tokenizer.hpp"
#include "generate.grpc.pb.h"
#include "generate.pb.h"
#include "model/hyper.h"

template<typename DType, typename IdType>
using Batch_t = std::unique_ptr<blitz::Batch<DType, IdType>>;

template<typename DType, typename IdType>
class TextGenerationServiceImpl final: public ::generate::v2::TextGenerationService::Service {
  private:
    std::unique_ptr<blitz::Stub<DType, IdType>> blitz_stub;
    // serialize concurrent ZagPrefill RPC
    // put member here to avoid cacheline contention
    std::atomic_uint32_t next_zag_seq;
    // make it atomic
    std::map<uint64_t, Batch_t<DType, IdType>> cached_batches;
    Spinlock cached_batches_lck;
    blitz::HuggingfaceTokenizer hf_tokenizer;
    // serialize concurrent batch creation when immigrant

    // cuda device
    const int cuda_device;

    // Rank
    const int rank;

    // Add the server_urls for the service discovery
    // Currently we only support one server
    const std::vector<std::string> server_urls;

    /// \brief helper functions
    /// \brief spin sync 2 events
    OneshotSemaphore<uint64_t> snd_wait_fst_half_layers; 

  public:
    // Constructor
    TextGenerationServiceImpl(
        const int world_size,
        const int rank,
        const int device,
        const char* ib_hca_name,
        std::unique_ptr<blitz::Stub<DType, IdType>> stub,
        blitz::model::GptHyperParam& hyper_param,
        blitz::HuggingfaceTokenizer&& tokenizer,
        std::vector<std::string>&& _server_urls
    );

    // Model Info
    virtual ::grpc::Status Info(
        ::grpc::ServerContext* context,
        const ::generate::v2::InfoRequest* request,
        ::generate::v2::InfoResponse* response
    ) override;

    // Service discovery
    virtual ::grpc::Status ServiceDiscovery(
        ::grpc::ServerContext* context,
        const ::generate::v2::ServiceDiscoveryRequest* request,
        ::generate::v2::ServiceDiscoveryResponse* response
    ) override;

    // Empties batch cache
    virtual ::grpc::Status ClearCache(
        ::grpc::ServerContext* context,
        const ::generate::v2::ClearCacheRequest* request,
        ::generate::v2::ClearCacheResponse* response
    ) override;

    // Remove requests from a cached batch
    virtual ::grpc::Status FilterBatch(
        ::grpc::ServerContext* context,
        const ::generate::v2::FilterBatchRequest* request,
        ::generate::v2::FilterBatchResponse* response
    ) override;

    // Warmup the model and compute max cache size
    virtual ::grpc::Status Warmup(
        ::grpc::ServerContext* context,
        const ::generate::v2::WarmupRequest* request,
        ::generate::v2::WarmupResponse* response
    ) override;

    // Prefill batch and decode first token
    virtual ::grpc::Status Prefill(
        ::grpc::ServerContext* context,
        const ::generate::v2::PrefillRequest* request,
        ::generate::v2::PrefillResponse* response
    ) override;

    // Decode token for a list of prefilled batches
    virtual ::grpc::Status Decode(
        ::grpc::ServerContext* context,
        const ::generate::v2::DecodeRequest* request,
        ::generate::v2::DecodeResponse* response
    ) override;

    // Health check
    virtual ::grpc::Status Health(
        ::grpc::ServerContext* context,
        const ::generate::v2::HealthRequest* request,
        ::generate::v2::HealthResponse* response
    ) override;

    //
    virtual ::grpc::Status PrefillV2(
        ::grpc::ServerContext* context,
        const ::generate::v2::PrefillV2Request* request,
        ::generate::v2::PrefillV2Response* response
    ) override;

    //
    virtual ::grpc::Status DecodeV2(
        ::grpc::ServerContext* context,
        const ::generate::v2::DecodeV2Request* request,
        ::generate::v2::DecodeV2Response* response
    ) override;

    //
    virtual ::grpc::Status ZagPrefill(
        ::grpc::ServerContext* context,
        const ::generate::v2::ZagPrefillRequest* request,
        ::generate::v2::PrefillV2Response* response
    ) override;

    virtual ::grpc::Status Migrate(
        ::grpc::ServerContext* context,
        const ::generate::v2::MigrateRequest* request,
        ::generate::v2::MigrateResponse* response
    ) override;

    virtual ::grpc::Status Immigrate(
        ::grpc::ServerContext* context,
        const ::generate::v2::ImmigrateRequest* request,
        ::generate::v2::ImmigrateResponse* response
    ) override;

    // RPC indicating parameter reception done, used for Router as callbacks
    virtual ::grpc::Status WaitRdmaDone(
        ::grpc::ServerContext* context,
        const ::generate::v2::WaitRdmaDoneRequest* request,
        ::generate::v2::WaitRdmaDoneResponse* response
    ) override;

    // Send parameters to dst rank
    virtual ::grpc::Status SendParams(
        ::grpc::ServerContext* context,
        const ::generate::v2::SendParamsRequest* request,
        ::generate::v2::SendParamsResponse* response
    ) override;

    // Receive parameters from src rank
    virtual ::grpc::Status RecvParams(
        ::grpc::ServerContext* context,
        const ::generate::v2::RecvParamsRequest* request,
        ::generate::v2::RecvParamsResponse* response
    ) override;

    virtual ::grpc::Status LoadParams(
        ::grpc::ServerContext* context,
        const ::generate::v2::LoadParamsRequest* request,
        ::generate::v2::LoadParamsResponse* response
    ) override;

    virtual ::grpc::Status ResetStatus(
        ::grpc::ServerContext* context,
        const ::generate::v2::ResetStatusRequest* request,
        ::generate::v2::ResetStatusResponse* response
    ) override;

    virtual ::grpc::Status SetStatusReady(
        ::grpc::ServerContext* context,
        const ::generate::v2::SetStatusReadyRequest* request,
        ::generate::v2::SetStatusReadyResponse* response
    ) override;
    
    virtual ::grpc::Status Relay(
        ::grpc::ServerContext* context,
        const ::generate::v2::RelayRequest* request,
        ::generate::v2::RelayResponse* response
    ) override;

    virtual ::grpc::Status MigratePartial(
        ::grpc::ServerContext* context,
        const ::generate::v2::MigratePartialRequest* request,
        ::generate::v2::MigrateResponse* response
    ) override;

    virtual ::grpc::Status ImmigratePartial(
        ::grpc::ServerContext* context,
        const ::generate::v2::ImmigratePartialRequest* request,
        ::generate::v2::ImmigrateResponse* response
    ) override;

    virtual ::grpc::Status NvlBroadcast(
        ::grpc::ServerContext* context,
        const ::generate::v2::BroadcastRequest* request,
        ::generate::v2::BroadcastResponse* response
    ) override;

    virtual ::grpc::Status RdmaBroadcast(
        ::grpc::ServerContext* context,
        const ::generate::v2::BroadcastRequest* request,
        ::generate::v2::BroadcastResponse* response
    ) override;

    virtual ::grpc::Status TanzBroadcast(
        ::grpc::ServerContext* context,
        const ::generate::v2::BroadcastRequest* request,
        ::generate::v2::BroadcastResponse* response
    ) override;
};

template class TextGenerationServiceImpl<half, int32_t>;
