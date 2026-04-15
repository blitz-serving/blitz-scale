use grpc_metadata::InjectTelemetryContext;
use pb::generate::v2::*;
use tracing::instrument;

use crate::error::Result;
use super::Stub;

impl Stub {
    // --- ResourceService methods ---

    #[instrument(skip_all)]
    pub async fn send_params(&mut self, dst: i32) -> Result<SendParamsResponse> {
        let request = tonic::Request::new(SendParamsRequest { dst }).inject_context();
        let response = self.resource.send_params(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn recv_params(&mut self, src: i32) -> Result<RecvParamsResponse> {
        let request = tonic::Request::new(RecvParamsRequest { src }).inject_context();
        let response = self.resource.recv_params(request).await?.into_inner();
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
        let response = self.resource.load_params(request).await?.into_inner();
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
        let response = self.resource.load_params(request).await?.into_inner();
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
        let response = self.resource.load_params(request).await?.into_inner();
        Ok(response)
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
        Ok(self.resource.nvl_broadcast(request).await?.into_inner())
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
        Ok(self.resource.rdma_broadcast(request).await?.into_inner())
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
        Ok(self.resource.tanz_broadcast(request).await?.into_inner())
    }
}
