use grpc_metadata::InjectTelemetryContext;
use pb::generate::v2::*;
use tracing::instrument;

use crate::error::Result;
use super::Stub;

impl Stub {
    // --- EngineService methods ---

    #[instrument(skip_all)]
    pub async fn prefill_v2(&mut self, request: PrefillV2Request) -> Result<PrefillV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.engine.prefill_v2(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn zag_prefill(&mut self, request: ZagPrefillRequest) -> Result<PrefillV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.engine.zag_prefill(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn decode_v2(&mut self, request: DecodeV2Request) -> Result<DecodeV2Response> {
        let request = tonic::Request::new(request).inject_context();
        let response = self.engine.decode_v2(request).await?.into_inner();
        Ok(response)
    }

    #[instrument(skip_all)]
    pub async fn wait_rdma_done(&mut self) -> Result<WaitRdmaDoneResponse> {
        let request = tonic::Request::new(WaitRdmaDoneRequest {}).inject_context();
        let response = self.engine.wait_rdma_done(request).await?.into_inner();
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
            .engine
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
            .engine
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
        let response = self.engine.relay(request).await?.into_inner();
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
            .engine
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
            .engine
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
            self.engine
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
            .engine
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
}
