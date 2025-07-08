mod exec_blitz;
mod exec_serverless;
mod planner;

use std::{
    collections::{HashMap, VecDeque},
    ops::Index,
    sync::{atomic::Ordering, Arc},
    time,
};

use futures::future::join_all;
use tokio::{spawn, sync::RwLock, time::sleep};

use crate::ReplicaState;

use super::{DisaggregationController, ReplicaMetric, D_MTX_NOTIFY, SCARCE_DECODE};

#[derive(Debug)]
pub(crate) enum ScalePlan {
    ScaleUp,
    ScaleDown,
    Stay,
}

/// \invariant not_skip
impl DisaggregationController {
    /// #2 Prefill => MutatingToDecode
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices
    async fn mutate_prefill_to_decode(
        &self,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
    ) {
        while *crct_num_prefill_replica < 0
            && *crct_num_decode_replica > 0
            && normal_prefill_replica_indices.len() > 1
        {
            let prefill_replica_index = normal_prefill_replica_indices.pop_front().unwrap().1;
            *self.replica_metrics[&prefill_replica_index]
                .state
                .write()
                .await = ReplicaState::MutatingToDecode;
            tracing::info!(
                "Replica<{}> Prefill => MutatingToDecode",
                prefill_replica_index
            );
            *crct_num_prefill_replica += 1;
            *crct_num_decode_replica -= 1;
        }
    }

    /// Prefill => ShuttingPrefill, put overprovisioning timestamp
    ///
    /// remark: shut down one, since Planner will scan the cluster next turn sooner
    ///
    /// invariant: valid_replica_indices
    /// invariant: valid_overpvn_ts (write)
    async fn shut_down_prefill_one(
        &self,
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) {
        if normal_prefill_replica_indices.len() > 1 {
            let prefill_replica_index = normal_prefill_replica_indices.pop_front().unwrap().1;
            *self.replica_metrics[&prefill_replica_index]
                .state
                .write()
                .await = ReplicaState::ShuttingPrefill;
            overprovision_timestamp[prefill_replica_index] = Some(time::Instant::now());
            null_replica_state[prefill_replica_index] = ReplicaState::Prefill;
            tracing::info!("Replica<{}> => ShuttingPrefill", prefill_replica_index);
        }
    }

    /// Decode => ShuttingDecode, put overprovisioning timestamp
    ///
    /// remark: shut down one, since Planner will scan the cluster next turn sooner
    ///
    /// invariant: valid_replica_indices
    /// invariant: valid_overpvn_ts (write)
    async fn shut_down_decode_one(
        &self,
        normal_decode_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) {
        if normal_decode_replica_indices.len() > 1 {
            let decode_replica_index = normal_decode_replica_indices.pop().unwrap();
            *self.replica_metrics[&decode_replica_index]
                .state
                .write()
                .await = ReplicaState::ShuttingDecode;
            overprovision_timestamp[decode_replica_index] = Some(time::Instant::now());
            null_replica_state[decode_replica_index] = ReplicaState::Decode;
            tracing::info!("Replica<{}> => ShuttingDecode", decode_replica_index);
        }
    }

    /// #3 Use nvlink to scale Prefill/Decode instance
    ///
    /// src_ranks: Prefill/Decode => SendingPrefill/Decode
    /// dst_ranks: Inactive => NvlCastingToPrefill/Decode
    ///
    /// invariant: updated_crct_num
    /// invariant: total_correctness
    async fn nvl_scale_up_p_and_d(
        self: &Arc<Self>,
        replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
        inactive_replica_indices: &mut Vec<usize>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        nvlink_chains: HashMap<usize, Vec<usize>>,
    ) {
        let tp_size = self.disaggregation_controller_args.tensor_parallel_size;
        assert_eq!(tp_size.count_ones(), 1);
        macro_rules! rank_to_replica {
            ($a:expr) => {
                $a as usize & (!(tp_size - 1))
            };
        }
        let mut crct_num_replica = *crct_num_prefill_replica + *crct_num_decode_replica;
        let mut choose_d_or_p = *crct_num_decode_replica > 0;
        for (src_rank, mut dst_replica_indices) in nvlink_chains.into_iter() {
            if crct_num_replica == 0 {
                break;
            }
            let src_replica_index = rank_to_replica!(src_rank);
            let _ = src_rank;
            let src_ranks: Vec<i32> = self.replica_to_ranks[&src_replica_index].clone();
            // find ALL needed dst ranks within one machine
            let dst_ranks: Vec<i32> = if (crct_num_replica as usize) < dst_replica_indices.len() {
                dst_replica_indices.truncate(crct_num_replica as usize);
                dst_replica_indices
                    .iter()
                    .map(|dst_rank| self.replica_to_ranks[&rank_to_replica!(*dst_rank)].clone())
                    .flat_map(|rank| rank.into_iter())
                    .collect()
            } else {
                dst_replica_indices
                    .iter()
                    .map(|dst_rank| self.replica_to_ranks[&rank_to_replica!(*dst_rank)].clone())
                    .flat_map(|rank| rank.into_iter())
                    .collect()
            };
            crct_num_replica -= (dst_ranks.len() / tp_size) as i32;
            tracing::info!("NvlCasting ++ {:?} +>+>+>+ {:?}", src_ranks, dst_ranks,);

            // TODO: extend nvl broadcast chain
            // pre: dst_ranks.len() <= crct_num_replica * tp_size
            if !dst_ranks.is_empty() {
                let mut src_stubs = self.replica_to_stubs[&src_replica_index].clone();
                for dst_rank in dst_replica_indices {
                    let dst_replica_index = rank_to_replica!(dst_rank);
                    let _ = dst_rank;
                    assert!(inactive_replica_indices.contains(&dst_replica_index));
                    let mut stubs = self.replica_to_stubs[&dst_replica_index].clone();
                    let replica_metric = self.replica_metrics[&dst_replica_index].clone();
                    let replica_nvl_state0 = replica_nvl_states.clone();
                    match (*crct_num_decode_replica > 0, choose_d_or_p) {
                        (true, true) => {
                            // scale a Decode
                            let src_rank0 = src_ranks.clone();
                            let dst_rank0 = self.replica_to_ranks[&dst_replica_index].clone();
                            let dst_ranks1 = dst_ranks.clone();
                            let steersman = self.steersman.clone();
                            // 1. mark Planner and Steersman state
                            replica_nvl_states.write().await[dst_replica_index] =
                                ReplicaState::NvlCasting;
                            {
                                let mut sman_lck = steersman.lock().await;
                                dst_rank0.iter().for_each(|rank| {
                                    sman_lck.wait_replica_nvlink(*rank as usize, 1);
                                });
                            }
                            // 1. Planner and Steersman marked
                            spawn(async move {
                                // 2. marking Worker state
                                tracing::info!(
                                    "Replica<{}> (nvl) => LoadingDecode",
                                    dst_replica_index
                                );
                                *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                                let _ = join_all(
                                    stubs
                                        .iter_mut()
                                        .map(|stub| stub.nvl_broadcast(&src_rank0, &dst_ranks1)),
                                )
                                .await
                                .into_iter()
                                .collect::<Result<Vec<_>, _>>()
                                .unwrap();
                                replica_metric.set_model_loaded(true);
                                tracing::info!("Replica<{}> => Decode", dst_replica_index);
                                *replica_metric.state.write().await = ReplicaState::AusDecode;
                                // 2. Worker marked
                                // 3. revert Steersman & Planner state
                                {
                                    let mut sman_lck = steersman.lock().await;
                                    src_rank0.iter().for_each(|rank| {
                                        sman_lck.post_replica_nvlink(*rank as usize)
                                    });
                                    dst_rank0.iter().for_each(|rank| {
                                        sman_lck.post_replica_nvlink(*rank as usize);
                                        sman_lck.record_model_loaded(*rank as usize);
                                    });
                                }
                                replica_nvl_state0.write().await[dst_replica_index] =
                                    ReplicaState::Inactive;
                                // 3. Steersman & Planner reverted
                            });
                            *crct_num_decode_replica -= 1;
                            inactive_replica_indices
                                .retain(|&replica_index| replica_index != dst_replica_index);
                            choose_d_or_p = !(*crct_num_prefill_replica > 0);
                            continue;
                        }
                        (true, false) => {}
                        (false, _) => {
                            choose_d_or_p = false;
                        }
                    }
                    match (*crct_num_prefill_replica > 0, choose_d_or_p) {
                        (true, false) => {
                            // scale a Prefill
                            let src_rank0 = src_ranks.clone();
                            let dst_rank0 = self.replica_to_ranks[&dst_replica_index].clone();
                            let dst_ranks1 = dst_ranks.clone();
                            let steersman = self.steersman.clone();
                            // 1. mark Planner and Steersman state
                            replica_nvl_states.write().await[dst_replica_index] =
                                ReplicaState::NvlCasting;
                            {
                                let mut sman_lck = steersman.lock().await;
                                dst_rank0.iter().for_each(|rank| {
                                    sman_lck.wait_replica_nvlink(*rank as usize, 1)
                                });
                            }
                            // 1. Planner and Steersman marked
                            spawn(async move {
                                // 2. marking Worker state
                                tracing::info!(
                                    "Replica<{}> (nvl) => LoadingPrefill",
                                    dst_replica_index
                                );
                                *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                                let _ = join_all(
                                    stubs
                                        .iter_mut()
                                        .map(|stub| stub.nvl_broadcast(&src_rank0, &dst_ranks1)),
                                )
                                .await
                                .into_iter()
                                .collect::<Result<Vec<_>, _>>()
                                .unwrap();
                                replica_metric.set_model_loaded(true);
                                tracing::info!("Replica<{}> => Prefill", dst_replica_index);
                                *replica_metric.state.write().await = ReplicaState::Prefill;
                                // 2. Worker marked
                                // 3. revert Steersman & Planner state
                                {
                                    let mut sman_lck = steersman.lock().await;
                                    src_rank0.iter().for_each(|rank| {
                                        sman_lck.post_replica_nvlink(*rank as usize)
                                    });
                                    dst_rank0.iter().for_each(|rank| {
                                        sman_lck.post_replica_nvlink(*rank as usize);
                                        sman_lck.record_model_loaded(*rank as usize);
                                    });
                                }
                                replica_nvl_state0.write().await[dst_replica_index] =
                                    ReplicaState::Inactive;
                                // 3. Steersman & Planner reverted
                            });
                            *crct_num_prefill_replica -= 1;
                            inactive_replica_indices
                                .retain(|&replica_index| replica_index != dst_replica_index);
                            choose_d_or_p = *crct_num_decode_replica > 0;
                            continue;
                        }
                        (true, true) => {}
                        (false, _) => {
                            choose_d_or_p = true;
                        }
                    }
                }
                // send src a nvl_broadcast RPC
                // let mut src_stub = self.all_stubs[src_rank].clone();
                let replica_metric = self.replica_metrics[&src_replica_index].clone();
                let replica_nvl_state0 = replica_nvl_states.clone();
                let st = replica_metric.state.read().await.clone();
                match st {
                    ReplicaState::Decode
                    | ReplicaState::Prefill
                    | ReplicaState::OldPrefill
                    | ReplicaState::ShuttingDecode
                    | ReplicaState::ShuttingPrefill
                    | ReplicaState::ShuttingNull
                    | ReplicaState::RefractoryPrefill
                    | ReplicaState::AusDecode
                    | ReplicaState::AusPrefill => {
                        // 1. mark Planner & Steersman state
                        tracing::info!("Replica<{}> => NvlinkSending", src_replica_index);
                        replica_nvl_states.write().await[src_replica_index] =
                            ReplicaState::NvlinkSending;
                        {
                            let mut sman_lck = self.steersman.lock().await;
                            src_ranks.iter().for_each(|rank| {
                                sman_lck.wait_replica_nvlink(
                                    *rank as usize,
                                    dst_ranks.len() / src_ranks.len(),
                                )
                            });
                        }
                        // 1. marked
                        spawn(async move {
                            // 2. mark Worker state
                            let _ =
                                join_all(src_stubs.iter_mut().map(|src_stub| {
                                    src_stub.nvl_broadcast(&src_ranks, &dst_ranks)
                                }))
                                .await
                                .into_iter()
                                .collect::<Result<Vec<_>, _>>()
                                .unwrap();
                            // 2. marked
                            // 3. revert Steersman & Planner state
                            // steersman skipped, since is reverted by dst ranks
                            replica_nvl_state0.write().await[src_replica_index] =
                                ReplicaState::Inactive;
                            // 3. reverted
                        });
                    }
                    _ => {
                        panic!(
                            "Refine transition system! Replica<{}> in state: {:?}",
                            src_replica_index, st
                        );
                    }
                }
            } else {
                tracing::warn!("Nvl casting dst ranks empty!");
                break;
            }
        }
    }

    /// #1 ShuttingDecode => AusDecode => Decode
    ///
    /// invariant: updated_crct_num
    /// invariant: total_correctness
    async fn shutting_decode_to_decode(
        self: &Arc<Self>,
        crct_num_decode_replica: &mut i32,
        shutting_decode_replica_indices: &mut VecDeque<(u32, usize)>,
        normal_decode_replica_indices: &mut VecDeque<(u32, usize)>,
    ) {
        if *crct_num_decode_replica > 0 {
            while shutting_decode_replica_indices.len() > 0 && *crct_num_decode_replica > 0 {
                let (blks, replica_index) = shutting_decode_replica_indices.pop_front().unwrap();
                tracing::info!("Replica<{}> ShuttingDecode => AusDecode", replica_index);
                *self.replica_metrics[&replica_index].state.write().await = ReplicaState::AusDecode;
                *crct_num_decode_replica -= 1;
                normal_decode_replica_indices.push_back((blks, replica_index));
            }
        }
    }

    /// #1 ShuttingPrefill => Prefill
    ///
    /// invariant: updated_crct_num
    /// invariant: total_correctness
    async fn shutting_prefill_to_prefill(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        shutting_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
    ) {
        if *crct_num_prefill_replica > 0 {
            while shutting_prefill_replica_indices.len() > 0 && *crct_num_prefill_replica > 0 {
                let (blks, replica_index) = shutting_prefill_replica_indices.pop_front().unwrap();
                tracing::info!("Replica<{}> ShuttingPrefill => Prefill", replica_index);
                *self.replica_metrics[&replica_index].state.write().await = ReplicaState::Prefill;
                *crct_num_prefill_replica -= 1;
                normal_prefill_replica_indices.push_back((blks, replica_index));
            }
        }
    }

    /// reset parameter status for consistency
    ///
    /// remark: reset server && router inner worker && steersman
    async fn deactivate_null(&self, replica_metric: &Arc<ReplicaMetric>, replica_index: usize) {
        // invalidate loaded parameter on server for consistency
        let mut stubs = self.replica_to_stubs[&replica_index].clone();
        for stub in stubs.iter_mut() {
            stub.reset_status().await.unwrap();
        }
        // invalidate router state
        *replica_metric.state.write().await = ReplicaState::Inactive;
        replica_metric.set_model_loaded(false);
        let mut sman_lck = self.steersman.lock().await;
        (replica_index..replica_index + stubs.len())
            .for_each(|rank| sman_lck.record_model_unloaded(rank));
    }

    /// ShuttingNull => Inactive
    ///
    /// invariant: valid_overpvn_ts (clear)
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices (null)
    async fn x_shutted_null_prefill(
        &self,
        crct_num_prefill_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) {
        while *crct_num_prefill_replica < 0 && shutted_null_replica_indices.len() > 0 {
            let replica_index = shutted_null_replica_indices.pop().unwrap();
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(
                *replica_metric.state.read().await,
                ReplicaState::ShuttingNull
            );
            tracing::info!("Replica<{}> : ShuttingNull => Inactive", replica_index);
            self.deactivate_null(&replica_metric, replica_index).await;
            overprovision_timestamp[replica_index] = None;

            *crct_num_prefill_replica += 1;
        }
    }

    /// ShuttingNull => Inactive
    ///
    /// invariant: valid_overpvn_ts (clear)
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices (null)
    async fn x_shutted_null_decode(
        &self,
        crct_num_decode_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) {
        while *crct_num_decode_replica < 0 && shutted_null_replica_indices.len() > 0 {
            let replica_index = shutted_null_replica_indices.pop().unwrap();
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(
                *replica_metric.state.read().await,
                ReplicaState::ShuttingNull
            );
            tracing::info!("Replica<{}> : ShuttingNull => Inactive", replica_index);
            self.deactivate_null(&replica_metric, replica_index).await;
            overprovision_timestamp[replica_index] = None;
            *crct_num_decode_replica += 1;
        }
    }

    /// #2 ShuttingNull => Prefill/Decode
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices (null)
    async fn shutted_null_to_p_and_d(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) {
        while shutted_null_replica_indices.len() > 0 {
            match (*crct_num_prefill_replica > 0, *crct_num_decode_replica > 0) {
                (true, true) => {
                    let replica_index = shutted_null_replica_indices.pop().unwrap();
                    tracing::info!("Replica<{}> ShuttingNull => Decode", replica_index);
                    self.shutted_null_to_decode_inner(replica_index).await;
                    *crct_num_decode_replica -= 1;
                    overprovision_timestamp[replica_index] = None;
                    if let Some(replica_index) = shutted_null_replica_indices.pop() {
                        tracing::info!("Replica<{}> ShuttingNull => Prefill", replica_index);
                        *self.replica_metrics[&replica_index].state.write().await =
                            ReplicaState::Prefill;
                        *crct_num_prefill_replica -= 1;
                        overprovision_timestamp[replica_index] = None;
                    }
                }
                (false, true) => {
                    self.shutted_null_to_decode(
                        crct_num_decode_replica,
                        shutted_null_replica_indices,
                        overprovision_timestamp,
                    )
                    .await;
                    return;
                }
                (true, false) => {
                    self.shutted_null_to_prefill(
                        crct_num_prefill_replica,
                        shutted_null_replica_indices,
                        overprovision_timestamp,
                    )
                    .await;
                    return;
                }
                (false, false) => {
                    return;
                }
            }
        }
    }

    async fn shutted_null_to_p_and_d_wo_mutation(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) {
        let null_replica_size = shutted_null_replica_indices.len();
        let mut unused_shutted_null = Vec::new();
        for _ in 0..null_replica_size {
            // while shutted_null_replica_indices.len() > 0 {
            // for null_replica_index in shutted_null_replica_indices.iter() {
            match (*crct_num_prefill_replica > 0, *crct_num_decode_replica > 0) {
                (true, true) => {
                    let replica_index = shutted_null_replica_indices.pop().unwrap();
                    match null_replica_state[replica_index] {
                        ReplicaState::Prefill => {
                            tracing::info!("Replica<{}> ShuttingNull => Prefill", replica_index);
                            *self.replica_metrics[&replica_index].state.write().await =
                                ReplicaState::Prefill;
                            *crct_num_prefill_replica -= 1;
                        }
                        ReplicaState::Decode => {
                            tracing::info!("Replica<{}> ShuttingNull => Decode", replica_index);
                            self.shutted_null_to_decode_inner(replica_index).await;
                            *crct_num_decode_replica -= 1;
                        }
                        _ => {
                            panic!("ServerlessLLM scaling policy failed!");
                        }
                    }
                }
                (false, true) => {
                    let replica_index = shutted_null_replica_indices.pop().unwrap();
                    if null_replica_state[replica_index] == ReplicaState::Decode {
                        tracing::info!("Replica<{}> ShuttingNull => Decode", replica_index);
                        self.shutted_null_to_decode_inner(replica_index).await;
                        *crct_num_decode_replica -= 1;
                    } else {
                        unused_shutted_null.push(replica_index);
                    }
                }
                (true, false) => {
                    let replica_index = shutted_null_replica_indices.pop().unwrap();
                    if null_replica_state[replica_index] == ReplicaState::Prefill {
                        tracing::info!("Replica<{}> ShuttingNull => Prefill", replica_index);
                        *self.replica_metrics[&replica_index].state.write().await =
                            ReplicaState::Prefill;
                        *crct_num_prefill_replica -= 1;
                    } else {
                        unused_shutted_null.push(replica_index);
                    }
                }
                (false, false) => {
                    return;
                }
            }
        }
        unused_shutted_null
            .iter()
            .for_each(|&index| shutted_null_replica_indices.push(index));
    }

    /// ShuttingNull => Inactive
    ///
    /// invariant: valid_replica_indices (null)
    /// invariant: valid_overpvn_ts
    async fn check_null_and_shut_one(
        self: &Arc<Self>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        #[allow(non_snake_case)] noExpiredDecode: &mut i32,
        null_replica_state: &mut Vec<ReplicaState>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
        replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
        can_sleep: bool,
    ) {
        let shutted_null_replica_indices = self
            .collect_null_replica_indices(noExpiredDecode, overprovision_timestamp)
            .await;
        let current_timestamp = time::Instant::now();
        let mut max_live_index: Option<usize> = None;
        let mut max_time_living = std::time::Duration::default();
        for &replica_index in shutted_null_replica_indices.iter() {
            let no_dep = replica_rdma_states.read().await[replica_index] == ReplicaState::Inactive
                && replica_nvl_states.read().await[replica_index] == ReplicaState::Inactive;
            let d =
                current_timestamp.duration_since(overprovision_timestamp[replica_index].unwrap());
            if no_dep && d > max_time_living {
                max_live_index = Some(replica_index);
                max_time_living = d;
            }
        }
        if let Some(index) = max_live_index {
            let &threshold_millis = &self
                .disaggregation_controller_args
                .scale_down_threshold_millis;
            let max_live_duration =
                current_timestamp.duration_since(overprovision_timestamp[index].unwrap());
            if max_live_duration.as_millis() > threshold_millis as u128 {
                tracing::info!(
                    "Replica<{}> : (time-to-live) ShuttingNull => Inactive",
                    index
                );
                self.deactivate_null(&self.replica_metrics[&index], index)
                    .await;
                overprovision_timestamp[index] = None;
                null_replica_state[index] = ReplicaState::Inactive;
            }
        } else if can_sleep {
            sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    /// #2 ShuttingNull => Prefill
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices (null)
    async fn shutted_null_to_prefill(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) {
        if *crct_num_prefill_replica > 0 {
            while shutted_null_replica_indices.len() > 0 && *crct_num_prefill_replica > 0 {
                let replica_index = shutted_null_replica_indices.pop().unwrap();
                tracing::info!("Replica<{}> ShuttingNull => Prefill", replica_index);
                *self.replica_metrics[&replica_index].state.write().await = ReplicaState::Prefill;
                *crct_num_prefill_replica -= 1;
                overprovision_timestamp[replica_index] = None;
            }
        }
    }

    /// #2 ShuttingNull => Decode
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices (null)
    async fn shutted_null_to_decode(
        self: &Arc<Self>,
        crct_num_decode_replica: &mut i32,
        shutted_null_replica_indices: &mut Vec<usize>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) {
        if *crct_num_decode_replica > 0 {
            while shutted_null_replica_indices.len() > 0 && *crct_num_decode_replica > 0 {
                let replica_index = shutted_null_replica_indices.pop().unwrap();
                tracing::debug!("Replica<{}> ShuttingNull => Decode", replica_index);
                self.shutted_null_to_decode_inner(replica_index).await;
                *crct_num_decode_replica -= 1;
                overprovision_timestamp[replica_index] = None;
            }
        }
    }

    /// impl for dst_mtx consistency
    async fn shutted_null_to_decode_inner(self: &Arc<Self>, replica_index: usize) {
        match D_MTX_NOTIFY
            .get()
            .unwrap()
            .get(replica_index)
            .unwrap()
            .swap(false, Ordering::AcqRel)
        {
            true => {
                /*
                   \pre ShuttingNull not yet get lock
                   \pre ShuttingNull must read FALSE
                */
                *self.replica_metrics[&replica_index].state.write().await = ReplicaState::Decode;
                // \post ShuttingNull shall not get lock
            }
            false => {
                // \pre ShuttingNull eventually get lock
                *self.replica_metrics[&replica_index].state.write().await = ReplicaState::AusDecode;
            }
        }
    }

    /// Mark all possible ShuttingNull
    ///
    /// conditionally mark decode replica w/o using any blocks
    ///
    /// invariant: updated_crct_num
    async fn collect_null_replica_indices(
        self: &Arc<Self>,
        crct_num_decode_replica: &mut i32,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
    ) -> Vec<usize> {
        let mut null_replica_indices = Vec::<usize>::new();
        let mut idle_decode =
            *crct_num_decode_replica < 0 && !SCARCE_DECODE.load(Ordering::Acquire);
        for (&replica_index, replica_metric) in self.replica_metrics.iter() {
            let st = replica_metric.state.read().await.clone();
            match st {
                ReplicaState::Decode => {
                    if idle_decode && replica_metric.get_used_blocks() == 0 {
                        if let Ok(_mtx) = replica_metric.dst_mutex.clone().try_lock_owned() {
                            if replica_metric.get_used_blocks() > 0 {
                                continue;
                            }
                            tracing::info!(
                                "Replica<{}> (expiring) Decode => ShuttingNull",
                                replica_index
                            );
                            D_MTX_NOTIFY
                                .get()
                                .unwrap()
                                .index(replica_index)
                                .store(true, Ordering::Release);
                            *replica_metric.state.write().await = ReplicaState::ShuttingNull;
                            overprovision_timestamp[replica_index] = Some(time::Instant::now());
                            null_replica_indices.push(replica_index);
                            if *crct_num_decode_replica < 0 {
                                *crct_num_decode_replica += 1;
                                idle_decode = *crct_num_decode_replica < 0
                                    && !SCARCE_DECODE.load(Ordering::Acquire);
                            } else {
                                idle_decode = false;
                            }
                        }
                    }
                }
                ReplicaState::ShuttingNull => {
                    if overprovision_timestamp[replica_index].is_none() {
                        overprovision_timestamp[replica_index] = Some(time::Instant::now());
                    }
                    null_replica_indices.push(replica_index);
                }
                _ => {}
            }
        }
        null_replica_indices
    }
}
