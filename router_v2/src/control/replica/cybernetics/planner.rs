use std::{
    cmp::{max, min},
    collections::VecDeque,
    sync::{atomic::Ordering, Arc},
    time,
};

use tokio::{sync::RwLock, task::yield_now};
use tracing::instrument;

use crate::{
    ControllerArgs, DisaggregationController, ReplicaState, KV_BLOCK_SIZE, SYSTEM_METRIC,
    ZIGZAG_ACTIVE_CNT,
};

impl DisaggregationController {
    #[instrument(skip_all)]
    pub(crate) async fn replica_state_moniter_loop(self: Arc<Self>) {
        let ControllerArgs {
            tokens_prefilled_per_sec,
            tokens_transferred_per_sec,
            max_blocks_per_replica,
            prefill_lower_bound,
            prefill_upper_bound,
            decode_lower_bound,
            decode_upper_bound,
            migration_lower_bound,
            migration_upper_bound,
            scale_down_threshold_millis,
            max_prefill_num,
            max_decode_num,
            min_prefill_num,
            min_decode_num,
            ..
        } = self.disaggregation_controller_args;

        let mut overprovision_timestamp: Vec<Option<time::Instant>> =
            vec![None; self.all_stubs.len()];
        // only used for impl ServerlessLLM
        let mut null_replica_state: Vec<ReplicaState> =
            vec![ReplicaState::Inactive; self.all_stubs.len()];
        // sending state marker only for Planner
        let replica_rdma_state: Arc<RwLock<Vec<ReplicaState>>> = Arc::new(RwLock::new(vec![
                ReplicaState::Inactive;
                self.all_stubs.len()
            ]));
        // sending state marker only for Planner
        let replica_nvl_states: Arc<RwLock<Vec<ReplicaState>>> = Arc::new(RwLock::new(vec![
                ReplicaState::Inactive;
                self.all_stubs.len()
            ]));

        let model_name = self.steersman.lock().await.get_managed_model_name();
        // \invariant `Skip`
        loop {
            let mut inactive_replica_indices = Vec::<usize>::default();
            // `Normal`:= eligible state transition
            // `u32`:= make best schedule choice based on used KV$ blocks
            let mut normal_prefill_replica_indices = Vec::<(u32, usize)>::default();
            let mut normal_decode_replica_indices = Vec::<(u32, usize)>::default();
            // `Shutting` := forced state transition => `Null`
            // `u32`:= make best schedule choice based on used KV$ blocks
            let mut shutting_prefill_replica_indices = Vec::<(u32, usize)>::default();
            let mut shutting_decode_replica_indices = Vec::<(u32, usize)>::default();
            // only used for Live::zigzag
            let mut num_zigzag_prefill_replica = 0;
            let mut old_prefill_replica_indices = Vec::<usize>::default();
            // \sum{P_token} + \sum{D_token}
            let mut current_waiting_decode_blocks = 0u32;
            // \sum{Unbatch_token}
            let current_waiting_prefill_tokens = self.batching_queue.waiting_prefill_tokens().await;

            let ((num_prefill_replica, num_decode_replica), current_zigzag_prefill_tokens) = self
                .view_replica_state(
                    &mut inactive_replica_indices,
                    &mut normal_prefill_replica_indices,
                    &mut normal_decode_replica_indices,
                    &mut shutting_prefill_replica_indices,
                    &mut shutting_decode_replica_indices,
                    &mut current_waiting_decode_blocks,
                    &mut num_zigzag_prefill_replica,
                    &mut old_prefill_replica_indices,
                    &mut null_replica_state,
                )
                .await;

            metrics::gauge!("blitz_prefill_replica", num_prefill_replica as f64);
            metrics::gauge!("blitz_decode_replica", num_decode_replica as f64);
            let tokens_waiting_migration =
                SYSTEM_METRIC.token_in_queue.load(Ordering::SeqCst) as u32;
            let tokens_consumed_per_sec = num_decode_replica * tokens_transferred_per_sec;

            let (mut prefill_scale_plan, mut decode_scale_plan) = self.generate_scale_target(
                num_prefill_replica,
                num_decode_replica,
                &normal_prefill_replica_indices,
                &normal_decode_replica_indices,
                current_waiting_decode_blocks,
                current_waiting_prefill_tokens + current_zigzag_prefill_tokens,
                tokens_waiting_migration.max(1),
                tokens_consumed_per_sec.max(1),
                max_blocks_per_replica,
                tokens_prefilled_per_sec,
                (decode_upper_bound, decode_lower_bound),
                (prefill_upper_bound, prefill_lower_bound),
                (migration_upper_bound, migration_lower_bound),
            );

            // config file take effect
            if prefill_scale_plan + num_prefill_replica as i32 > max_prefill_num as i32 {
                prefill_scale_plan = max_prefill_num as i32 - num_prefill_replica as i32;
                // tracing::warn!("Prefill scale plan exceeds max_prefill_num");
            }
            if (prefill_scale_plan + num_prefill_replica as i32) < min_prefill_num as i32 {
                prefill_scale_plan = min_prefill_num as i32 - num_prefill_replica as i32;
                // tracing::warn!("Prefill scale plan exceeds min_prefill_num");
            }

            if decode_scale_plan + num_decode_replica as i32 > max_decode_num as i32 {
                decode_scale_plan = max_decode_num as i32 - num_decode_replica as i32;
                // tracing::warn!("Decode scale plan exceeds max_decode_num");
            }
            if (decode_scale_plan + num_decode_replica as i32) < min_decode_num as i32 {
                decode_scale_plan = min_decode_num as i32 - num_decode_replica as i32;
                // tracing::warn!("Decode scale plan exceeds min_decode_num");
            }

            if cfg!(feature = "impl_blitz") {
                normal_prefill_replica_indices.sort_by_key(|&(k, _)| k);
                normal_decode_replica_indices.sort_by_key(|&(k, _)| k);
                shutting_prefill_replica_indices.sort_by_key(|&(k, _)| k);
                shutting_decode_replica_indices.sort_by_key(|&(k, _)| k);
                // increasing order
                let sorted_normal_prefill_replica_indices =
                    VecDeque::from(normal_prefill_replica_indices);
                let sorted_normal_decode_replica_indices =
                    VecDeque::from(normal_decode_replica_indices);
                let sorted_shutting_prefill_replica_indices =
                    VecDeque::from(shutting_prefill_replica_indices);
                let sorted_shutting_decode_replica_indices =
                    VecDeque::from(shutting_decode_replica_indices);

                // num_zigzag_prefill_replica := len({NewPrefill})
                // relay_prefill_replica_indices := {OldPrefill}
                let has_live_or_loading_scaling_prefill =
                    num_prefill_replica - sorted_normal_prefill_replica_indices.len() as u32 > 0;
                if has_live_or_loading_scaling_prefill {
                    // no ZIGZAG_ACTIVE_CNT -> loading
                    // otherwise, exists OldPrefill
                    assert!(
                        !old_prefill_replica_indices.is_empty()
                            || ZIGZAG_ACTIVE_CNT.load(Ordering::Acquire) == 0,
                        "There must be some OldPrefill during live zigzag scaling!"
                    );
                    // invariant: only Planner able to inc ZigzagActiveCnt
                    assert!(
                        num_zigzag_prefill_replica != 0
                            || ZIGZAG_ACTIVE_CNT.load(Ordering::Acquire) == 0
                    );
                }
                self.execute_scale_plan_blitz(
                    inactive_replica_indices,
                    sorted_normal_prefill_replica_indices,
                    sorted_normal_decode_replica_indices,
                    sorted_shutting_prefill_replica_indices,
                    sorted_shutting_decode_replica_indices,
                    prefill_scale_plan,
                    decode_scale_plan,
                    &mut old_prefill_replica_indices,
                    &model_name.clone(),
                    &mut overprovision_timestamp,
                    &mut null_replica_state,
                    &replica_rdma_state,
                    &replica_nvl_states,
                )
                .await;
            } else {
                if cfg!(feature = "mutate") {
                    // todo!("Add loading from disk in exec blitz");
                    let sorted_normal_prefill_replica_indices =
                        VecDeque::from(normal_prefill_replica_indices);
                    let sorted_normal_decode_replica_indices =
                        VecDeque::from(normal_decode_replica_indices);
                    let sorted_shutting_prefill_replica_indices =
                        VecDeque::from(shutting_prefill_replica_indices);
                    let sorted_shutting_decode_replica_indices =
                        VecDeque::from(shutting_decode_replica_indices);
                    self.execute_scale_plan_blitz(
                        inactive_replica_indices,
                        sorted_normal_prefill_replica_indices,
                        sorted_normal_decode_replica_indices,
                        sorted_shutting_prefill_replica_indices,
                        sorted_shutting_decode_replica_indices,
                        prefill_scale_plan,
                        decode_scale_plan,
                        &mut old_prefill_replica_indices,
                        &model_name.clone(),
                        &mut overprovision_timestamp,
                        &mut null_replica_state,
                        &replica_rdma_state,
                        &replica_nvl_states,
                    )
                    .await;
                } else {
                    self.execute_scale_plan_serverless(
                        inactive_replica_indices,
                        normal_prefill_replica_indices,
                        normal_decode_replica_indices,
                        shutting_prefill_replica_indices,
                        shutting_decode_replica_indices,
                        prefill_scale_plan,
                        decode_scale_plan,
                        &mut overprovision_timestamp,
                        &mut null_replica_state,
                        &replica_rdma_state,
                        &replica_nvl_states,
                    )
                    .await;
                }
            }

            yield_now().await;
        }
    }

    /// Scan cluster state
    ///
    /// `inactive_replica`: Inactive
    /// `normal_prefill_replica`: Prefill
    /// `normal_decode_replica`: Decode
    /// `shutting_prefill_replica`: ShuttingPrefill
    /// `shutting_decode_replica`: ShuttingDecode
    /// `old_prefill_replica`: OldPrefill
    ///
    /// `null_replica`: unused; only used for ServerlessLLM
    async fn view_replica_state(
        &self,
        inactive_replica_indices: &mut Vec<usize>,
        normal_prefill_replica_indices: &mut Vec<(u32, usize)>,
        normal_decode_replica_indices: &mut Vec<(u32, usize)>,
        shutting_prefill_replica_indices: &mut Vec<(u32, usize)>,
        shutting_decode_replica_indices: &mut Vec<(u32, usize)>,
        current_waiting_decode_blocks: &mut u32,
        num_zigzag_prefill_replica: &mut usize,
        old_prefill_replica_indices: &mut Vec<usize>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) -> ((u32, u32), u32) {
        let mut num_prefill_replica = 0;
        let mut num_decode_replica = 0;
        let mut num_live_waiting_tokens = 0;
        for (&replica_index, replica_metric) in self.replica_metrics.iter() {
            let st = replica_metric.state.read().await.clone();
            match st {
                ReplicaState::NvlinkSending
                | ReplicaState::RdmaSending
                | ReplicaState::RdmaLoading
                | ReplicaState::RdmaCasting
                | ReplicaState::NvlCasting
                | ReplicaState::TanzCasting => {
                    panic!(
                        "Replica<{}> in {:?}, which is planner's marker state!",
                        replica_index, st
                    );
                }
                ReplicaState::Inactive => {
                    inactive_replica_indices.push(replica_index);
                    if cfg!(feature = "impl_sllm") {
                        null_replica_state[replica_index] = ReplicaState::Inactive;
                        // assert_eq!(null_replica_state[index], ReplicaState::Inactive);
                    }
                }
                ReplicaState::ShuttingNull => {}
                // monotonicity of used_blocks
                ReplicaState::LoadingPrefill => {
                    *current_waiting_decode_blocks += replica_metric.get_used_blocks() as u32;
                    num_prefill_replica += 1;
                }
                // monotonicity of used_blocks
                ReplicaState::LoadingDecode
                | ReplicaState::MutatingToDecode
                | ReplicaState::AusDecode => {
                    *current_waiting_decode_blocks += replica_metric.get_used_blocks() as u32;
                    num_decode_replica += 1;
                }
                ReplicaState::Prefill | ReplicaState::AusPrefill => {
                    let num_blocks = replica_metric.get_used_blocks() as u32;
                    assert!(
                        replica_metric.get_partial_migration_cnt() != 0
                            || replica_metric.is_model_loaded()
                    );
                    normal_prefill_replica_indices.push((num_blocks, replica_index));
                    *current_waiting_decode_blocks += num_blocks;
                    num_prefill_replica += 1;
                }
                ReplicaState::Decode => {
                    let num_blocks = replica_metric.get_used_blocks() as u32;
                    normal_decode_replica_indices.push((num_blocks, replica_index));
                    *current_waiting_decode_blocks += num_blocks;
                    num_decode_replica += 1;
                }
                ReplicaState::ShuttingPrefill => {
                    shutting_prefill_replica_indices
                        .push((replica_metric.get_used_blocks() as u32, replica_index));
                }
                ReplicaState::ShuttingDecode => {
                    shutting_decode_replica_indices
                        .push((replica_metric.get_used_blocks() as u32, replica_index));
                }
                ReplicaState::NewPrefill | ReplicaState::RefractoryPrefill => {
                    *num_zigzag_prefill_replica += 1;
                    num_prefill_replica += 1;
                    num_live_waiting_tokens += replica_metric.get_used_blocks() * KV_BLOCK_SIZE;
                }
                ReplicaState::OldPrefill => {
                    old_prefill_replica_indices.push(replica_index);
                    num_prefill_replica += 1;
                }
            }
        }
        (
            (num_prefill_replica, num_decode_replica),
            num_live_waiting_tokens,
        )
    }

    ///
    /// # Args
    ///
    /// * current_waiting_prefill_blocks = \sum{Unbatch_tokens}
    /// * current_waiting_decode_blocks = \sum{P} + \sum{D}
    fn generate_scale_target(
        &self,
        num_prefill_replica: u32,
        num_decode_replica: u32,
        normal_prefill_replica_indices: &Vec<(u32, usize)>,
        normal_decode_replica_indices: &Vec<(u32, usize)>,
        current_waiting_decode_blocks: u32,
        current_waiting_prefill_tokens: u32,
        migrating_tokens: u32,
        max_tokens_consumed_per_sec: u32,
        max_blocks_per_replica: u32,
        max_tokens_prefilled_per_replica: u32,
        decode_bounds: (f32, f32),    // (upper bound, lower bound)
        prefill_bounds: (f32, f32),   // (upper bound, lower bound)
        migration_bounds: (f32, f32), // (upper bound, lower bound)
    ) -> (i32, i32) {
        assert!(num_prefill_replica > 0);
        assert!(num_decode_replica > 0);

        // ---------- Eq.1 Decode Memory ----------
        // decide based on used blocks
        let (decode_high, decode_low) = decode_bounds;
        assert!(decode_low > 0.);
        let min_num_decode_mem =
            (current_waiting_decode_blocks as f32 / decode_high / max_blocks_per_replica as f32)
                .ceil() as u32;
        let max_num_decode_mem =
            (current_waiting_decode_blocks as f32 / decode_low / max_blocks_per_replica as f32)
                .ceil() as u32;

        // ---------- Eq.2 KV$ Bandwidth ----------
        // \bug FIXME: hardcoded
        let prefill_tokens: u32 = normal_prefill_replica_indices
            .iter()
            .map(|(block, _)| block * KV_BLOCK_SIZE)
            .sum();
        // \bug FIXME: Maybe repeated calculation
        // \note only Decode's perspectives
        let (migration_high, migration_low) = migration_bounds;
        assert!(migration_low > 0.);
        let min_num_decode_kv =
            (prefill_tokens as f32 / migration_high / max_tokens_consumed_per_sec as f32).ceil()
                as u32;
        let max_num_decode_kv =
            (prefill_tokens as f32 / migration_low / max_tokens_consumed_per_sec as f32).ceil()
                as u32;

        // temporary solution for decode
        let min_num_decode_replica = max(min_num_decode_mem, min_num_decode_kv);
        let max_num_decode_replica = max(
            max_num_decode_mem.max(min_num_decode_replica),
            max_num_decode_kv.max(min_num_decode_replica),
        );

        // ---------- Eq.2 KV$ Bandwidth ----------
        let min_num_prefill_kv =
            (prefill_tokens as f32 / migration_high / max_tokens_consumed_per_sec as f32).ceil()
                as u32;
        let max_num_prefill_kv =
            (prefill_tokens as f32 / migration_high / max_tokens_consumed_per_sec as f32).ceil()
                as u32;

        // ---------- Eq.3 prefill thpt. ----------
        let (prefill_high, prefill_low) = prefill_bounds;
        let min_num_prefill_thpt = (current_waiting_prefill_tokens as f32
            / prefill_high
            / max_tokens_prefilled_per_replica as f32)
            .ceil() as u32;
        let max_num_prefill_thpt = (current_waiting_prefill_tokens as f32
            / prefill_low
            / max_tokens_prefilled_per_replica as f32)
            .ceil() as u32;

        // temporary solution for decode
        let min_num_prefill_replica = max(min_num_prefill_thpt, min_num_prefill_kv);
        let max_num_prefill_replica = max(
            max_num_prefill_thpt.max(min_num_prefill_replica),
            max_num_prefill_kv.max(min_num_prefill_replica),
        );

        // final solution
        let crt_prefill_replica_num = (if min_num_prefill_replica > num_prefill_replica {
            (min_num_prefill_replica + max_num_prefill_replica + 1) / 2 - num_prefill_replica
        } else if max_num_prefill_replica < num_prefill_replica {
            max_num_prefill_replica - num_prefill_replica
        } else {
            0
        } as i32)
            .max(-1)
            .max(-(num_prefill_replica as i32) + 1);
        let crt_decode_replica_num = (if min_num_decode_replica > num_decode_replica {
            (min_num_decode_replica + max_num_decode_replica + 1) / 2 - num_decode_replica
        } else if max_num_decode_replica < num_decode_replica {
            max_num_decode_replica - num_decode_replica
        } else {
            0
        } as i32)
            .max(-1)
            .max(-(num_decode_replica as i32) + 1);

        // \post crt_prefill_replica_num + num_prefill_replica > 0
        // \post crt_decode_replica_num + num_decode_replica > 0
        if crt_prefill_replica_num != 0 || crt_decode_replica_num != 0 {
            tracing::info!(
                "Control signal: P{:+} D{:+} | Prefill thpt : [{},{}]; KV$ : [{},{}] | Decode Mem : [{},{}]; KV$ : [{},{}]",
                crt_prefill_replica_num,
                crt_decode_replica_num,
                min_num_prefill_thpt,
                max_num_prefill_thpt,
                min_num_prefill_kv,
                max_num_prefill_kv,
                min_num_decode_mem,
                max_num_decode_mem,
                min_num_decode_kv,
                max_num_decode_kv
            );
        }
        return (crt_prefill_replica_num, crt_decode_replica_num);
    }
}
