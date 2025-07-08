// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
//
// Modifications by Blitz-serving:
//   - Replaced the stub function to backend, added more interface
use std::cmp::min;

use grpc_metadata::InjectTelemetryContext;
use pb::generate::v2::*;
use text_generation_service_client::TextGenerationServiceClient;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

use crate::error::{ClientError, Result};

pub fn encode_prefill_case(cases: Vec<PrefillCase>) -> u32 {
    let mut encoding = 0;
    cases
        .into_iter()
        .for_each(|case| encoding = encoding | case as u32);
    encoding
}

pub fn is_case_set(encoding: u32, case: PrefillCase) -> bool {
    encoding & case as u32 != 0
}

/// Blitz gRPC client
#[derive(Debug, Clone)]
pub struct Stub {
    stub: TextGenerationServiceClient<Channel>,
}

impl Stub {
    /// Returns a client connected to the given url
    pub async fn connect(uri: String) -> Result<Self> {
        if uri.starts_with("unix://") {
            Self::connect_uds(uri).await
        } else if uri.starts_with("http://") {
            Self::connect_tcp(uri).await
        } else {
            Err(ClientError::InvalidUri(uri))
        }
    }

    /// Returns a client connected to the given unix socket
    async fn connect_uds(uri: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn({
                let path = uri.strip_prefix("unix://").unwrap().to_string();
                move |_: Uri| tokio::net::UnixStream::connect(path.clone())
            }))
            .await
            .map_err(|err| {
                eprintln!("[rust client] connect fail: {:?}", err);
                err
            })?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    async fn connect_tcp(addr: String) -> Result<Self> {
        let channel = Channel::from_shared(addr)?.connect().await?;
        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.stub.service_discovery(request).await?;
        let uris = response.into_inner().urls.into_iter().collect();
        Ok(uris)
    }

    /// Get model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let request = tonic::Request::new(InfoRequest {}).inject_context();
        let response = self.stub.info(request).await?.into_inner();
        Ok(response)
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    /// Clear the past generations cache
    #[instrument(skip_all)]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest { id: batch_id }).inject_context();
        self.stub.clear_cache(request).await?;
        Ok(())
    }

    /// Filter a cached batch
    #[instrument(skip_all)]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<CachedBatch> {
        let request = tonic::Request::new(FilterBatchRequest {
            batch_id,
            request_ids,
        })
        .inject_context();
        let filtered_batch = self.stub.filter_batch(request).await?.into_inner();
        Ok(filtered_batch.batch)
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip_all)]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
    ) -> Result<Option<u32>> {
        let mut n_tokens = 0;
        let mut requests = Vec::new();
        // Create requests
        while n_tokens < max_prefill_tokens {
            let truncate = min(max_input_length, max_prefill_tokens - n_tokens);
            requests.push(Request {
                id: 0,
                // We truncate the input on the server side to be sure that it has the correct size
                inputs: "_test ".to_string().repeat(max_input_length as usize),
                truncate: Some(truncate),
                // Set sampling parameters to also take these ops into account in the max memory
                parameters: Some(NextTokenChooserParameters {
                    temperature: 0.9,
                    top_k: 10,
                    top_p: 0.9,
                    typical_p: 0.9,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 1.2,
                    watermark: true,
                }),
                stopping_parameters: StoppingCriteriaParameters {
                    max_new_tokens: max_total_tokens - truncate,
                    stop_sequences: vec![],
                    ignore_eos_token: true,
                },
                prefill_logprobs: false,
                top_n_tokens: 0,
                input_tokens: vec![0],
            });
            n_tokens += max_input_length;
        }

        let batch = Batch {
            id: 0,
            size: requests.len() as u32,
            requests,
            max_tokens: 0,
        };

        let request = tonic::Request::new(WarmupRequest {
            batch: Some(batch),
            max_input_length: Some(max_input_length),
            max_prefill_tokens: Some(max_prefill_tokens),
            max_total_tokens: Some(max_total_tokens),
        })
        .inject_context();
        let response = self.stub.warmup(request).await?.into_inner();
        Ok(response.max_supported_total_tokens)
    }
}

impl Stub {
    #[instrument(skip_all)]
    pub async fn prefill_v2(&mut self, request: PrefillV2Request) -> Result<PrefillV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.stub.prefill_v2(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn zag_prefill(&mut self, request: ZagPrefillRequest) -> Result<PrefillV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.stub.zag_prefill(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn decode_v2(&mut self, request: DecodeV2Request) -> Result<DecodeV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.stub.decode_v2(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn send_params(&mut self, dst: i32) -> Result<SendParamsResponse> {
        let request = tonic::Request::new(SendParamsRequest { dst }).inject_context();
        let response = self.stub.send_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn recv_params(&mut self, src: i32) -> Result<RecvParamsResponse> {
        let request = tonic::Request::new(RecvParamsRequest { src }).inject_context();
        let response = self.stub.recv_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn reset_status(&mut self) -> Result<ResetStatusResponse> {
        let request = tonic::Request::new(ResetStatusRequest {}).inject_context();
        let response = self.stub.reset_status(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn set_status_ready(&mut self) -> Result<SetStatusReadyResponse> {
        let request = tonic::Request::new(SetStatusReadyRequest {}).inject_context();
        let response = self.stub.set_status_ready(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn load_params_from_host_memory(
        &mut self,
        model_name: String,
    ) -> Result<LoadParamsResponse> {
        let request = tonic::Request::new(LoadParamsRequest {
            load_case: LoadParamCase::LoadFromHostMem as i32,
            model_name,
            model_path: None,
        })
        .inject_context();
        let response = self.stub.load_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn load_param_from_disk(
        &mut self,
        model_name: String,
        model_path: String,
    ) -> Result<LoadParamsResponse> {
        let request = tonic::Request::new(LoadParamsRequest {
            load_case: LoadParamCase::LoadFromDisk as i32,
            model_name,
            model_path: Some(model_path),
        })
        .inject_context();
        let response = self.stub.load_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn load_params(
        &mut self,
        case: LoadParamCase,
        model_name: String,
        model_path: Option<String>,
    ) -> Result<LoadParamsResponse> {
        let request = tonic::Request::new(LoadParamsRequest {
            load_case: case as i32,
            model_name,
            model_path,
        })
        .inject_context();
        let response = self.stub.load_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn wait_rdma_done(&mut self) -> Result<WaitRdmaDoneResponse> {
        let request = tonic::Request::new(WaitRdmaDoneRequest {}).inject_context();
        let response = self.stub.wait_rdma_done(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn migrate(
        &mut self,
        batch: Batch,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<MigrateResponse> {
        Ok(self
            .stub
            .migrate(
                tonic::Request::new(MigrateRequest {
                    batch,
                    src: src_ranks,
                    dst: dst_ranks,
                })
                .inject_context(),
            )
            .await?
            .into_inner())
    }

    #[instrument(skip_all)]
    pub async fn immigrate(
        &mut self,
        batch: Batch,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<ImmigrateResponse> {
        Ok(self
            .stub
            .immigrate(
                tonic::Request::new(ImmigrateRequest {
                    batch,
                    src: src_ranks,
                    dst: dst_ranks,
                })
                .inject_context(),
            )
            .await?
            .into_inner())
    }

    #[instrument(skip_all)]
    pub async fn relay(
        &mut self,
        rank: i32,
        relax_not_head: bool,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let request = tonic::Request::new(RelayRequest {
            rank,
            relax_not_head,
        })
        .inject_context();
        let response = self.stub.relay(request).await?.into_inner();
        Ok((response.batch_id, response.seq_num))
    }

    #[instrument(skip_all)]
    pub async fn migrate_fst(
        &mut self,
        batch: Batch,
        fst_layer: u32,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<MigrateResponse> {
        Ok(self
            .stub
            .migrate_partial(
                tonic::Request::new(MigratePartialRequest {
                    batch,
                    fst_or_snd: PartialCase::First.into(),
                    num_layer: fst_layer,
                    src: src_ranks,
                    dst: dst_ranks,
                })
                .inject_context(),
            )
            .await?
            .into_inner())
    }

    #[instrument(skip_all)]
    pub async fn immigrate_fst(
        &mut self,
        batch: Batch,
        fst_layer: u32,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<ImmigrateResponse> {
        Ok(self
            .stub
            .immigrate_partial(
                tonic::Request::new(ImmigratePartialRequest {
                    batch,
                    fst_or_snd: PartialCase::First.into(),
                    num_layer: fst_layer,
                    src: src_ranks,
                    dst: dst_ranks,
                })
                .inject_context(),
            )
            .await?
            .into_inner())
    }

    #[instrument(skip_all)]
    pub async fn migrate_snd(
        &mut self,
        batch: Batch,
        snd_layer: u32,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<MigrateResponse> {
        Ok({
            self.stub
                .migrate_partial(
                    tonic::Request::new(MigratePartialRequest {
                        batch,
                        fst_or_snd: PartialCase::Second.into(),
                        num_layer: snd_layer,
                        src: src_ranks,
                        dst: dst_ranks,
                    })
                    .inject_context(),
                )
                .await?
                .into_inner()
        })
    }

    #[instrument(skip_all)]
    pub async fn immigrate_snd(
        &mut self,
        batch: Batch,
        snd_layer: u32,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) -> Result<ImmigrateResponse> {
        Ok(self
            .stub
            .immigrate_partial(
                tonic::Request::new(ImmigratePartialRequest {
                    batch,
                    fst_or_snd: PartialCase::Second.into(),
                    num_layer: snd_layer,
                    src: src_ranks,
                    dst: dst_ranks,
                })
                .inject_context(),
            )
            .await?
            .into_inner())
    }

    #[instrument(skip_all)]
    pub async fn nvl_broadcast(
        &mut self,
        src_ranks: &Vec<i32>,
        dst_ranks: &Vec<i32>,
    ) -> Result<BroadcastResponse> {
        let request = tonic::Request::new(BroadcastRequest {
            src_ranks: src_ranks.clone(),
            dst_ranks: dst_ranks.clone(),
        })
        .inject_context();
        Ok(self.stub.nvl_broadcast(request).await?.into_inner())
    }

    #[instrument(skip_all)]
    pub async fn rdma_broadcast(
        &mut self,
        src_ranks: &Vec<i32>,
        dst_ranks: &Vec<i32>,
    ) -> Result<BroadcastResponse> {
        let request = tonic::Request::new(BroadcastRequest {
            src_ranks: src_ranks.clone(),
            dst_ranks: dst_ranks.clone(),
        })
        .inject_context();
        Ok(self.stub.rdma_broadcast(request).await?.into_inner())
    }

    #[instrument(skip_all)]
    pub async fn tanz_broadcast(
        &mut self,
        src_ranks: &Vec<i32>,
        dst_ranks: &Vec<i32>,
    ) -> Result<BroadcastResponse> {
        let request = tonic::Request::new(BroadcastRequest {
            src_ranks: src_ranks.clone(),
            dst_ranks: dst_ranks.clone(),
        })
        .inject_context();
        Ok(self.stub.tanz_broadcast(request).await?.into_inner())
    }
}
