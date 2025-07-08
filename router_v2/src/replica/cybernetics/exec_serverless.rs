use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time;

use tokio::{spawn, sync::RwLock, task::yield_now};

use crate::{DisaggregationController, ReplicaState};

use pb::generate::v2::*;

use super::ScalePlan;

/// Implementation of ServerlessLLM
impl DisaggregationController {
    pub(crate) async fn execute_scale_plan_serverless(
        self: &Arc<Self>,
        mut inactive_replica_indices: Vec<usize>,
        mut normal_prefill_replica_indices: Vec<(u32, usize)>,
        mut normal_decode_replica_indices: Vec<(u32, usize)>,
        _shutting_prefill_replica_indices: Vec<(u32, usize)>,
        _shutting_decode_replica_indices: Vec<(u32, usize)>,
        crt_num_prefill_replica: i32,
        crt_num_decode_replica: i32,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
        replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) {
        // assign scale plan: up down or stay
        let mut prefill_scale_plan = ScalePlan::Stay;
        let mut decode_scale_plan = ScalePlan::Stay;
        if crt_num_prefill_replica > 0 {
            prefill_scale_plan = ScalePlan::ScaleUp;
        } else if crt_num_prefill_replica < 0 {
            prefill_scale_plan = ScalePlan::ScaleDown;
        }
        if crt_num_decode_replica > 0 {
            decode_scale_plan = ScalePlan::ScaleUp;
        } else if crt_num_decode_replica < 0 {
            decode_scale_plan = ScalePlan::ScaleDown;
        }

        let num_crt_prefill_replica = crt_num_prefill_replica.abs();
        let num_crt_decode_replica = crt_num_decode_replica.abs();
        #[allow(non_snake_case)]
        let mut noExpiredDecode = 1;
        // let need_decode_replica_num = 0;

        match (prefill_scale_plan, decode_scale_plan) {
            (ScalePlan::ScaleUp, ScalePlan::ScaleUp)
            | (ScalePlan::Stay, ScalePlan::ScaleUp)
            | (ScalePlan::ScaleUp, ScalePlan::Stay) => {
                let mut add_num_prefill = num_crt_prefill_replica;
                let mut add_num_decode = num_crt_decode_replica;
                if add_num_prefill + add_num_decode > inactive_replica_indices.len() as i32 {
                    add_num_prefill = inactive_replica_indices.len() as i32 - add_num_decode;
                }

                add_num_prefill = add_num_prefill.max(0);
                add_num_decode = add_num_decode.max(0);

                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;

                if shutted_null_replica_indices.len() > 0 {
                    tracing::info!(
                        "[1]Shutted Null Replica indices: {:?}",
                        shutted_null_replica_indices
                    );
                }

                self.shutted_null_to_p_and_d_wo_mutation(
                    &mut add_num_prefill,
                    &mut add_num_decode,
                    &mut shutted_null_replica_indices,
                    null_replica_state,
                )
                .await;

                if shutted_null_replica_indices.len() > 0 {
                    tracing::info!(
                        "[2]Shutted Null Replica indices: {:?}",
                        shutted_null_replica_indices
                    );
                }

                self.decode_scale_up_n_serverless(add_num_decode, &mut inactive_replica_indices)
                    .await;

                self.prefill_scale_up_n_serverless(add_num_prefill, &mut inactive_replica_indices)
                    .await;

                if shutted_null_replica_indices.len() > 0 {
                    tracing::info!(
                        "[3]Shutted Null Replica indices: {:?}",
                        shutted_null_replica_indices
                    );
                }

                self.check_null_and_shut_one(
                    overprovision_timestamp,
                    &mut noExpiredDecode,
                    null_replica_state,
                    replica_rdma_states,
                    replica_nvl_states,
                    false,
                )
                .await;

                if shutted_null_replica_indices.len() > 0 {
                    tracing::info!(
                        "[4]Shutted Null Replica indices: {:?}",
                        shutted_null_replica_indices
                    );
                }
            }
            (ScalePlan::ScaleDown, ScalePlan::ScaleDown)
            | (ScalePlan::Stay, ScalePlan::ScaleDown)
            | (ScalePlan::ScaleDown, ScalePlan::Stay) => {
                // if there's no enough inactive replicas, assign prefill first
                for _ in 0..num_crt_prefill_replica {
                    self.scale_down_prefill_one_sllm(
                        &mut normal_prefill_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }

                for _ in 0..num_crt_decode_replica {
                    self.scale_down_decode_one_sllm(
                        &mut normal_decode_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }

                self.check_null_and_shut_one(
                    overprovision_timestamp,
                    &mut noExpiredDecode,
                    null_replica_state,
                    replica_rdma_states,
                    replica_nvl_states,
                    false,
                )
                .await;
            }
            (ScalePlan::ScaleDown, ScalePlan::ScaleUp) => {
                // Scale down prefill replicas
                for _ in 0..num_crt_prefill_replica {
                    self.scale_down_prefill_one_sllm(
                        &mut normal_prefill_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
                // Try to scale up decode replicas
                self.decode_scale_up_n_serverless(
                    num_crt_decode_replica,
                    &mut inactive_replica_indices,
                )
                .await;

                self.check_null_and_shut_one(
                    overprovision_timestamp,
                    &mut noExpiredDecode,
                    null_replica_state,
                    replica_rdma_states,
                    replica_nvl_states,
                    false,
                )
                .await;
            }
            (ScalePlan::ScaleUp, ScalePlan::ScaleDown) => {
                // prefill scale up, decode scale down
                for _ in 0..num_crt_decode_replica {
                    self.scale_down_decode_one_sllm(
                        &mut normal_decode_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
                self.prefill_scale_up_n_serverless(
                    num_crt_prefill_replica,
                    &mut inactive_replica_indices,
                )
                .await;

                self.check_null_and_shut_one(
                    overprovision_timestamp,
                    &mut noExpiredDecode,
                    null_replica_state,
                    replica_rdma_states,
                    replica_nvl_states,
                    false,
                )
                .await;
            }
            (ScalePlan::Stay, ScalePlan::Stay) => {
                self.check_null_and_shut_one(
                    overprovision_timestamp,
                    &mut noExpiredDecode,
                    null_replica_state,
                    replica_rdma_states,
                    replica_nvl_states,
                    true,
                )
                .await;
            }
        }
    }

    pub(super) async fn decode_scale_up_n_serverless(
        self: &Arc<Self>,
        scale_up_num: i32,
        inactive_replica_indices: &mut Vec<usize>,
    ) {
        assert!(scale_up_num >= 0);

        let model_name = self.steersman.lock().await.get_managed_model_name().clone();
        let old = inactive_replica_indices.len();

        let mut cache_available_replica_indices: Vec<usize>;
        let mut cache_unavailable_replica_indices: Vec<usize>;

        if cfg!(feature = "cache_replace") {
            let mut cache_available_replica_indices_set = HashSet::<usize>::new();
            let mut cache_unavailable_replica_indices_set = HashSet::<usize>::new();
            let sman_lck = self.steersman.lock().await;
            std::mem::take(inactive_replica_indices)
                .into_iter()
                .for_each(|replica_index| {
                    if sman_lck.machine_has_cache(replica_index, model_name.as_str()) {
                        cache_available_replica_indices_set.insert(replica_index);
                    } else {
                        cache_unavailable_replica_indices_set.insert(replica_index);
                    }
                });

            cache_available_replica_indices =
                cache_available_replica_indices_set.into_iter().collect();
            cache_unavailable_replica_indices =
                cache_unavailable_replica_indices_set.into_iter().collect();
        } else if cfg!(feature = "cache_all_miss") {
            cache_available_replica_indices = vec![];
            cache_unavailable_replica_indices = std::mem::take(inactive_replica_indices)
        } else if cfg!(feature = "cache_all_hit") {
            cache_unavailable_replica_indices = vec![];
            cache_available_replica_indices = std::mem::take(inactive_replica_indices)
        } else if cfg!(feature = "motiv") {
            cache_unavailable_replica_indices = vec![];
            cache_available_replica_indices = std::mem::take(inactive_replica_indices)
        } else {
            unreachable!("When ServerlessLLM is enabled, one of {{\"cache_replace\", \"cache_all_hit\", \"cache_all_miss\"}} must be enabled");
        }

        let new = cache_unavailable_replica_indices.len() + cache_available_replica_indices.len();
        assert_eq!(old, new);

        for _ in 0..scale_up_num {
            if cfg!(feature = "cache_replace") {
                if cache_available_replica_indices.len() > 0 {
                    let replica_index = cache_available_replica_indices.pop().unwrap();
                    let replica_metric = self.replica_metrics[&replica_index].clone();
                    let steersman0 = self.steersman.clone();
                    let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                    let orchest = self.clone();
                    let model_name0 = model_name.clone();
                    spawn(async move {
                        tracing::info!("Replica<{}> => LoadingDecode", replica_index);
                        *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                        let start = std::time::Instant::now();
                        orchest
                            .load_param_w_case_inner(
                                dst_ranks.clone(),
                                LoadParamCase::LoadFromHostMem,
                                &model_name0,
                            )
                            .await;
                        tracing::info!("Replica<{}> => Decode", replica_index);
                        replica_metric.set_model_loaded(true);
                        *replica_metric.state.write().await = ReplicaState::MutatingToDecode;
                        tracing::info!(
                            "Replica<{}> (host cache) Decode; elapse: {:?}",
                            replica_index,
                            start.elapsed()
                        );
                        let mut sman_lck = steersman0.lock().await;
                        dst_ranks.iter().for_each(|rank| {
                            sman_lck.record_model_loaded(*rank as usize);
                        });
                    });
                } else if cache_unavailable_replica_indices.len() > 0 {
                    let replica_index = cache_unavailable_replica_indices.pop().unwrap();
                    let replica_metric = self.replica_metrics[&replica_index].clone();
                    let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                    let steersman0 = self.steersman.clone();
                    tracing::info!("Decode scale up replica [{}] through SSD", replica_index);
                    let orchest = self.clone();
                    let model_name0 = model_name.clone();
                    spawn(async move {
                        tracing::info!("Replica<{}> (host cache) => LoadingDecode", replica_index);
                        *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                        let start = std::time::Instant::now();
                        orchest
                            .load_param_w_case_inner(
                                dst_ranks.clone(),
                                LoadParamCase::LoadFromHostMem,
                                &model_name0,
                            )
                            .await;
                        tracing::info!("Replica<{}> => Decode", replica_index);
                        replica_metric.set_model_loaded(true);
                        *replica_metric.state.write().await = ReplicaState::MutatingToDecode;
                        tracing::info!(
                            "Replica<{}> (host cache) Decode; elapse: {:?}",
                            replica_index,
                            start.elapsed()
                        );
                        let mut sman_lck = steersman0.lock().await;
                        dst_ranks.iter().for_each(|rank| {
                            sman_lck.record_model_loaded(*rank as usize);
                        });
                    });
                } else {
                    break;
                }
            } else if cfg!(feature = "cache_all_miss") {
                if cache_unavailable_replica_indices.is_empty() {
                    break;
                }
                let replica_index = cache_unavailable_replica_indices.pop().unwrap();
                let replica_metric = self.replica_metrics[&replica_index].clone();
                {}
                let steersman0 = self.steersman.clone();
                let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                let controller = self.clone();
                let model_name0 = model_name.clone();
                spawn(async move {
                    tracing::info!("Replica<{}> (ssd) => LoadingDecode", replica_index);
                    *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                    let start = std::time::Instant::now();
                    controller
                        .load_param_w_case_inner(
                            dst_ranks.clone(),
                            LoadParamCase::LoadFromDisk,
                            &model_name0,
                        )
                        .await;
                    tracing::info!("Replica<{}> => Decode", replica_index);
                    replica_metric.set_model_loaded(true);
                    *replica_metric.state.write().await = ReplicaState::MutatingToDecode;
                    tracing::info!(
                        "Replica<{}> (ssd) Decode; elapse: {:?}",
                        replica_index,
                        start.elapsed()
                    );
                    let mut sman_lck = steersman0.lock().await;
                    dst_ranks.iter().for_each(|rank| {
                        sman_lck.record_model_loaded(*rank as usize);
                    });
                });
            } else if cfg!(feature = "cache_all_hit") {
                if cache_available_replica_indices.is_empty() {
                    break;
                }
                let steersman0 = self.steersman.clone();
                let replica_index = cache_unavailable_replica_indices.pop().unwrap();
                let replica_metric = self.replica_metrics[&replica_index].clone();
                let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                let controller = self.clone();
                let model_name0 = model_name.clone();
                spawn(async move {
                    tracing::info!("Replica<{}> (host cache) => LoadingDecode", replica_index);
                    *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                    let start = std::time::Instant::now();
                    controller
                        .load_param_w_case_inner(
                            dst_ranks.clone(),
                            LoadParamCase::LoadFromDisk,
                            &model_name0,
                        )
                        .await;
                    tracing::info!("Replica<{}> => Decode", replica_index);
                    replica_metric.set_model_loaded(true);
                    *replica_metric.state.write().await = ReplicaState::MutatingToDecode;
                    tracing::info!(
                        "Replica<{}> (ssd) Decode; elapse: {:?}",
                        replica_index,
                        start.elapsed()
                    );
                    let mut sman_lck = steersman0.lock().await;
                    dst_ranks.iter().for_each(|rank| {
                        sman_lck.record_model_loaded(*rank as usize);
                    });
                });
            } else if cfg!(feature = "motiv") {
                if cache_available_replica_indices.is_empty() {
                    break;
                }
                let steersman0 = self.steersman.clone();
                let replica_index = cache_unavailable_replica_indices.pop().unwrap();
                let replica_metric = self.replica_metrics[&replica_index].clone();
                let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                let controller = self.clone();
                spawn(async move {
                    tracing::info!("Replica<{}> (host cache) => LoadingDecode", replica_index);
                    *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                    let start = std::time::Instant::now();
                    controller.mock_transfer_params(dst_ranks.clone()).await;
                    tracing::info!("Replica<{}> => Decode", replica_index);
                    replica_metric.set_model_loaded(true);
                    *replica_metric.state.write().await = ReplicaState::MutatingToDecode;
                    tracing::info!(
                        "Replica<{}> (ssd) Decode; elapse: {:?}",
                        replica_index,
                        start.elapsed()
                    );
                    let mut sman_lck = steersman0.lock().await;
                    dst_ranks.iter().for_each(|rank| {
                        sman_lck.record_model_loaded(*rank as usize);
                    });
                });
            } else {
                unreachable!("Should be one of 'enable_cache', 'mock_load' or 'mock_cache_hit'");
            }

            yield_now().await;
        }

        cache_available_replica_indices
            .into_iter()
            .chain(cache_unavailable_replica_indices.into_iter())
            .for_each(|replica_index| inactive_replica_indices.push(replica_index));
    }

    pub(super) async fn prefill_scale_up_n_serverless(
        self: &Arc<Self>,
        scale_up_num: i32,
        inactive_replica_indices: &mut Vec<usize>,
    ) {
        assert!(scale_up_num >= 0);
        let model_name = self.steersman.lock().await.get_managed_model_name();

        let old = inactive_replica_indices.len();

        let mut cache_available_replica_indices: Vec<usize>;
        let mut cache_unavailable_replica_indices: Vec<usize>;

        if cfg!(feature = "cache_replace") {
            let mut cache_available_replica_indices_set = HashSet::<usize>::new();
            let mut cache_unavailable_replica_indices_set = HashSet::<usize>::new();
            let sman_lck = self.steersman.lock().await;
            std::mem::take(inactive_replica_indices)
                .into_iter()
                .for_each(|replica_index| {
                    if sman_lck.machine_has_cache(replica_index, &model_name) {
                        cache_available_replica_indices_set.insert(replica_index);
                    } else {
                        cache_unavailable_replica_indices_set.insert(replica_index);
                    }
                });
            drop(sman_lck);
            cache_available_replica_indices =
                cache_available_replica_indices_set.into_iter().collect();
            cache_unavailable_replica_indices =
                cache_unavailable_replica_indices_set.into_iter().collect();
        } else if cfg!(feature = "cache_all_miss") {
            cache_available_replica_indices = vec![];
            cache_unavailable_replica_indices = std::mem::take(inactive_replica_indices)
        } else if cfg!(feature = "cache_all_hit") {
            cache_unavailable_replica_indices = vec![];
            cache_available_replica_indices = std::mem::take(inactive_replica_indices)
        } else if cfg!(feature = "motiv") {
            cache_unavailable_replica_indices = vec![];
            cache_available_replica_indices = std::mem::take(inactive_replica_indices)
        } else {
            unreachable!("When ServerlessLLM is enabled, one of 'enable_cache', 'mock_load' or 'mock_cache_hit' should also be enabled");
        }

        let new = cache_unavailable_replica_indices.len() + cache_available_replica_indices.len();
        assert_eq!(old, new);

        for _ in 0..scale_up_num {
            if cache_available_replica_indices.len() > 0 {
                if cfg!(feature = "cache_all_miss") {
                    assert!(false, "Mock Load cannot have cache_available_instance");
                }
                let replica_index = cache_available_replica_indices.pop().unwrap();
                let replica_metric = self.replica_metrics[&replica_index].clone();
                let steersman0 = self.steersman.clone();
                tracing::info!("Replica<{}> (host cache) => LoadingPrefill", replica_index);
                // 1. marked
                let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                let orchest = self.clone();
                let model_name0 = model_name.clone();
                spawn(async move {
                    *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                    let start = std::time::Instant::now();
                    if cfg!(feature = "motiv") {
                        orchest.mock_transfer_params(dst_ranks.clone()).await;
                    } else {
                        orchest
                            .load_param_w_case_inner(
                                dst_ranks.clone(),
                                LoadParamCase::LoadFromHostMem,
                                &model_name0,
                            )
                            .await;
                    }
                    tracing::info!("Replica<{}> => Prefill", replica_index);
                    replica_metric.set_model_loaded(true);
                    *replica_metric.state.write().await = ReplicaState::Prefill;
                    tracing::info!(
                        "Replica<{}> (host cache) Prefill; elapse: {:?}",
                        replica_index,
                        start.elapsed()
                    );
                    let mut sman_lck = steersman0.lock().await;
                    dst_ranks.iter().for_each(|rank| {
                        sman_lck.record_model_loaded(*rank as usize);
                    });
                });
            } else if cache_unavailable_replica_indices.len() > 0 {
                if cfg!(feature = "cache_all_hit") {
                    assert!(
                        false,
                        "Mock Cache Hit cannot have cache_unavailabel instance"
                    );
                }
                let replica_index = cache_unavailable_replica_indices.pop().unwrap();
                let replica_metric = self.replica_metrics[&replica_index].clone();
                tracing::info!("Replica<{}> (ssd) => LoadingPrefill", replica_index);
                // 1. marked
                tracing::info!("Prefill scale up replica [{}] through SSD", replica_index);
                let dst_ranks = self.replica_to_ranks[&replica_index].clone();
                let orchest = self.clone();
                let model_name0 = model_name.clone();
                let steersman0 = self.steersman.clone();
                spawn(async move {
                    tracing::info!("Replica<{}> => Prefill", replica_index);
                    *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                    let start = std::time::Instant::now();
                    orchest
                        .load_param_w_case_inner(
                            dst_ranks.clone(),
                            LoadParamCase::LoadFromDisk,
                            &model_name0,
                        )
                        .await;
                    replica_metric.set_model_loaded(true);
                    *replica_metric.state.write().await = ReplicaState::Prefill;
                    tracing::info!(
                        "Replica<{}> (ssd) Prefill; elapse: {:?}",
                        replica_index,
                        start.elapsed()
                    );

                    let mut sman_lck = steersman0.lock().await;
                    dst_ranks.iter().for_each(|rank| {
                        sman_lck.record_model_loaded(*rank as usize);
                    });
                });
            } else {
                break;
            }

            yield_now().await;
        }

        cache_available_replica_indices
            .into_iter()
            .chain(cache_unavailable_replica_indices.into_iter())
            .for_each(|replica_index| inactive_replica_indices.push(replica_index));
    }

    /// Prefill => ShuttingPrefill impl 4 S-LLM
    ///
    /// invariant: valid_replica_indices (shut down)
    /// invariant: valid_overpvn_ts
    async fn scale_down_prefill_one_sllm(
        &self,
        normal_prefill_replica_indices: &mut Vec<(u32, usize)>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) {
        if normal_prefill_replica_indices.len() == 0 {
            return;
        }
        let prefill_replica_index = normal_prefill_replica_indices.remove(0).1;
        tracing::info!("Replica<{}> => ShuttingPrefill", prefill_replica_index);
        let replica = self.replica_metrics[&prefill_replica_index].as_ref();
        *replica.state.write().await = ReplicaState::ShuttingPrefill;
        overprovision_timestamp[prefill_replica_index] = Some(time::Instant::now());
        null_replica_state[prefill_replica_index] = ReplicaState::Prefill
    }

    /// Decode => ShuttingDecode impl 4 S-LLM
    ///
    /// invariant: valid_replica_indices (shut down)
    /// invariant: valid_overpvn_ts
    async fn scale_down_decode_one_sllm(
        &self,
        normal_decode_replica_indices: &mut Vec<(u32, usize)>,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
    ) {
        if normal_decode_replica_indices.len() == 0 {
            return;
        }
        let decode_replica_index = normal_decode_replica_indices.pop().unwrap().1;
        tracing::info!("Replica<{}> => ShuttingDecode", decode_replica_index);
        let replica = self.replica_metrics[&decode_replica_index].as_ref();
        *replica.state.write().await = ReplicaState::ShuttingDecode;
        overprovision_timestamp[decode_replica_index] = Some(time::Instant::now());
        null_replica_state[decode_replica_index] = ReplicaState::Decode;
    }
}
