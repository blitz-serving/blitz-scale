use grpc_metadata::InjectTelemetryContext;
use pb::generate::v2::*;
use tracing::instrument;

use crate::error::Result;
use super::Stub;

impl Stub {
    // --- AdminService methods ---

    #[instrument(skip_all)]
    pub async fn reset_status(&mut self) -> Result<ResetStatusResponse> {
        let request = tonic::Request::new(ResetStatusRequest {}).inject_context();
        let response = self.admin.reset_status(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn set_status_ready(&mut self) -> Result<SetStatusReadyResponse> {
        let request = tonic::Request::new(SetStatusReadyRequest {}).inject_context();
        let response = self.admin.set_status_ready(request).await?.into_inner();
        Ok(response)
    }
}
