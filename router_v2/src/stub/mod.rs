// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// (C) 2022-present Hugging Face Inc. -- Apache-2.0.
//
// Modifications by Blitz-serving:
//   - Replaced the stub function to backend, added more interface
//   - Split into engine/resource/admin sub-modules

mod engine;
mod resource;
mod admin;

use std::cmp::min;

use pb::generate::v2::*;
use engine_service_client::EngineServiceClient;
use resource_service_client::ResourceServiceClient;
use admin_service_client::AdminServiceClient;
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

/// Blitz gRPC client -- wraps 3 service clients sharing one channel
#[derive(Debug, Clone)]
pub struct Stub {
    pub(crate) engine: EngineServiceClient<Channel>,
    pub(crate) resource: ResourceServiceClient<Channel>,
    pub(crate) admin: AdminServiceClient<Channel>,
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
            engine: EngineServiceClient::new(channel.clone()),
            resource: ResourceServiceClient::new(channel.clone()),
            admin: AdminServiceClient::new(channel),
        })
    }

    async fn connect_tcp(addr: String) -> Result<Self> {
        let channel = Channel::from_shared(addr)?.connect().await?;
        Ok(Self {
            engine: EngineServiceClient::new(channel.clone()),
            resource: ResourceServiceClient::new(channel.clone()),
            admin: AdminServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.resource.service_discovery(request).await?;
        let uris = response.into_inner().urls.into_iter().collect();
        Ok(uris)
    }

    /// Get model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let request = tonic::Request::new(InfoRequest {}).inject_context();
        let response = self.engine.info(request).await?.into_inner();
        Ok(response)
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.admin.health(request).await?.into_inner();
        Ok(response)
    }

    /// Clear the past generations cache
    #[instrument(skip_all)]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest { id: batch_id }).inject_context();
        self.engine.clear_cache(request).await?;
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
        let filtered_batch = self.engine.filter_batch(request).await?.into_inner();
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
        let response = self.engine.warmup(request).await?.into_inner();
        Ok(response.max_supported_total_tokens)
    }
}

use grpc_metadata::InjectTelemetryContext;
