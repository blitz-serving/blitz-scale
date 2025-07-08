use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicI32;
use std::sync::{atomic::Ordering, Arc};
use std::time;

use futures::future::join_all;
use tokio::{
    spawn,
    sync::RwLock,
    task::{yield_now, JoinHandle},
};

use super::ScalePlan;
use crate::{
    DisaggregationController, ReplicaState, CTRL_LOOP_CNT, FLOW_WATCHER, PLANNER_BUSY,
    RELAY_DEACTIVE_CNT, SCARCE_DECODE, ZIGZAG_ACTIVE_CNT,
};

impl DisaggregationController {
    pub(crate) async fn execute_scale_plan_blitz(
        self: &Arc<Self>,
        mut inactive_replica_indices: Vec<usize>,
        mut normal_prefill_replica_indices: VecDeque<(u32, usize)>,
        mut normal_decode_replica_indices: VecDeque<(u32, usize)>,
        mut shutting_prefill_replica_indices: VecDeque<(u32, usize)>,
        mut shutting_decode_replica_indices: VecDeque<(u32, usize)>,
        crct_num_prefill_replica: i32,
        crct_num_decode_replica: i32,
        _old_prefill_replica_indices: &mut Vec<usize>,
        model_name: &str,
        overprovision_timestamp: &mut Vec<Option<time::Instant>>,
        null_replica_state: &mut Vec<ReplicaState>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
        replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) {
        CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
        let mut crct_num_prefill_replica = crct_num_prefill_replica;
        let mut crct_num_decode_replica = crct_num_decode_replica;
        #[allow(non_snake_case)]
        let mut noExpiredDecode = 1;

        let mut prefill_scale_plan = ScalePlan::Stay;
        let mut decode_scale_plan = ScalePlan::Stay;
        if crct_num_prefill_replica > 0 {
            prefill_scale_plan = ScalePlan::ScaleUp;
        } else if crct_num_prefill_replica < 0 {
            prefill_scale_plan = ScalePlan::ScaleDown;
        }
        if crct_num_decode_replica > 0 {
            decode_scale_plan = ScalePlan::ScaleUp;
        } else if crct_num_decode_replica < 0 {
            decode_scale_plan = ScalePlan::ScaleDown;
        }

        // Scaling execution priority
        //
        // 1. ShuttingPrefill/Decode -> Prefill/Decode
        // 2. CollectNull -> ShuttingNull -> Prefill/Decode
        // 3. Longest NvlCasting Chain -> ShuttingNull -> Prefill/Decode
        // 4. Widest TanzCasting Matrix -> Prefill/Decode
        // 5. Longest RdmaCasting Chain -> Prefill/Decode
        match (prefill_scale_plan, decode_scale_plan) {
            (ScalePlan::ScaleUp, ScalePlan::ScaleUp) => {
                self.shutting_decode_to_decode(
                    &mut crct_num_decode_replica,
                    &mut shutting_decode_replica_indices,
                    &mut normal_decode_replica_indices,
                )
                .await;
                self.shutting_prefill_to_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutting_prefill_replica_indices,
                    &mut normal_prefill_replica_indices,
                )
                .await;
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;
                self.shutted_null_to_p_and_d(
                    &mut crct_num_prefill_replica,
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if cfg!(feature = "impl_nvl") {
                    if crct_num_prefill_replica + crct_num_decode_replica > 0 {
                        // get MAX nvlink casting domain
                        // Vec::<NvlChainEdge<src?, dst>>
                        let max_nvlink_chain_edges = self
                            .steersman
                            .lock()
                            .await
                            .assign_nvlink_chains(&model_name, &mut inactive_replica_indices);
                        let mut nvlink_chains = HashMap::new();
                        for (src_replica_index, dst_replica_index) in max_nvlink_chain_edges {
                            if let Some(k) = src_replica_index {
                                nvlink_chains
                                    .entry(k)
                                    .or_insert_with(Vec::new)
                                    .push(dst_replica_index);
                            }
                        }
                        // no need to manually filter, just check
                        // invariant: Nvl src & dst are not occupied
                        let nvl_mrk_lck = replica_nvl_states.read().await;
                        nvlink_chains
                            .keys()
                            .chain(
                                nvlink_chains
                                    .values()
                                    .flat_map(|replica_indices| replica_indices.iter()),
                            )
                            .for_each(|replica_index| {
                                assert_eq!(
                                    nvl_mrk_lck[*replica_index],
                                    ReplicaState::Inactive,
                                    "Rank<{}> reported availale by Sman, but marked as {:?}",
                                    replica_index,
                                    nvl_mrk_lck[*replica_index]
                                )
                            });
                        drop(nvl_mrk_lck);
                        // operate on each chain, currently one NvlChain/Machine
                        self.nvl_scale_up_p_and_d(
                            replica_nvl_states,
                            &mut inactive_replica_indices,
                            &mut crct_num_prefill_replica,
                            &mut crct_num_decode_replica,
                            nvlink_chains,
                        )
                        .await;
                    }
                }
                // \invariant "coro_yield"
                if crct_num_prefill_replica + crct_num_decode_replica > 0
                    && inactive_replica_indices.len() > 0
                {
                    // choose ranks to avoid bandwidth interference
                    let mut max_rdma_sending_replica_indices = self
                        .span_max_rdma_sending_replicas(
                            &normal_prefill_replica_indices,
                            &normal_decode_replica_indices,
                            shutting_decode_replica_indices,
                            shutted_null_replica_indices,
                            replica_rdma_states,
                            replica_nvl_states,
                        )
                        .await;
                    // todo:    mutation and detect scaling order
                    //
                    // pre:     updated_crct_num
                    // def:     need_to_scale := crct_num_prefill + crt_num_decode > 0
                    // post:    need_to_scale
                    if cfg!(feature = "test_live")
                        && crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                    {
                        if max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            self.live_p2p_scale_up_n_prefill(
                                &mut crct_num_prefill_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut normal_prefill_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await;
                        }
                        yield_now().await;
                        return;
                    }
                    if cfg!(feature = "impl_tanz") && max_rdma_sending_replica_indices.len() > 0 {
                        let handle = self
                            .tanz_scale_up_n_null(
                                &mut crct_num_prefill_replica,
                                &mut crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                &mut normal_prefill_replica_indices,
                                replica_rdma_states,
                                replica_nvl_states,
                            )
                            .await;
                        if let Some(handle) = handle {
                            PLANNER_BUSY.store(true, Ordering::Release);
                            handle.await.unwrap();
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            PLANNER_BUSY.store(false, Ordering::Release);
                        } else {
                            yield_now().await;
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                    }
                    if crct_num_decode_replica + crct_num_prefill_replica > 0
                        && cfg!(feature = "impl_fast")
                    {
                        if cfg!(feature = "live") {
                            let ((num_scaled_prefill, num_scaled_decode), handle) = self
                                .rdma_bcast_scale_w_live(
                                    &mut crct_num_prefill_replica,
                                    &mut crct_num_decode_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    &mut normal_prefill_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            if (num_scaled_prefill + num_scaled_decode) > 0 {
                                PLANNER_BUSY.store(true, Ordering::Release);
                                handle.await.unwrap();
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                                PLANNER_BUSY.store(false, Ordering::Release);
                            } else {
                                yield_now().await;
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            }
                            return;
                        }
                        // pre: need_to_scale
                        let num_scaled_up_prefill = if crct_num_prefill_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            // \def "eligible_src_replica" := max_sending_replica_indices.len() > 0
                            // \pre  "need_to_scale"
                            // \pre  "eligible_src_replica"
                            // \pre  "valid_src_dst_replica_indices"
                            let (nscaled, _) = self
                                .rdma_bcast_scale_n_prefill(
                                    model_name,
                                    crct_num_prefill_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            nscaled
                        } else {
                            0
                        };
                        let num_scaled_up_decode = if crct_num_decode_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            // \pre  "need_to_scale"
                            // \pre  "eligible_src_replica"
                            // \pre  "valid_src_dst_replica_indices"
                            let (nscaled, _) = self
                                .rdma_bcast_scale_n_decode(
                                    model_name,
                                    crct_num_decode_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            nscaled
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            (crct_num_prefill_replica - num_scaled_up_prefill)
                                + (crct_num_decode_replica - num_scaled_up_decode)
                                > 0,
                            Ordering::Release,
                        );
                    } else if cfg!(feature = "impl_rdma") {
                        let num_scaled_up_prefill = if crct_num_prefill_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                        {
                            // pre: need_to_scale
                            // pre: eligible_src_replica
                            // pre: valid_src_dst_replica_indices
                            self.rdma_p2p_scale_up_n_prefill(
                                model_name,
                                crct_num_prefill_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await
                        } else {
                            0
                        };
                        let num_scaled_up_decode = if crct_num_decode_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            self.rdma_p2p_scale_up_n_decode(
                                model_name,
                                crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            (crct_num_prefill_replica - num_scaled_up_prefill)
                                + (crct_num_decode_replica - num_scaled_up_decode)
                                > 0,
                            Ordering::Release,
                        );
                    } else if cfg!(feature = "impl_sllm") {
                        if crct_num_decode_replica > 0 {
                            self.decode_scale_up_n_serverless(
                                crct_num_decode_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                        if crct_num_prefill_replica > 0 {
                            self.prefill_scale_up_n_serverless(
                                crct_num_prefill_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                    }
                } else {
                    SCARCE_DECODE.store(false, Ordering::Release);
                }
            }
            (ScalePlan::ScaleUp, ScalePlan::Stay) => {
                self.shutting_prefill_to_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutting_prefill_replica_indices,
                    &mut normal_prefill_replica_indices,
                )
                .await;
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;
                // \post likely scarce decode, don't clear that flag

                self.shutted_null_to_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if cfg!(feature = "impl_nvl") {
                    assert!((crct_num_decode_replica <= 0));
                    if crct_num_prefill_replica > 0 {
                        let max_nvlink_chain_edges = self
                            .steersman
                            .lock()
                            .await
                            .assign_nvlink_chains(&model_name, &mut inactive_replica_indices);
                        let mut nvlink_chains = HashMap::new();
                        for (src_replica_index, dst_replica_index) in max_nvlink_chain_edges {
                            if let Some(k) = src_replica_index {
                                nvlink_chains
                                    .entry(k)
                                    .or_insert_with(Vec::new)
                                    .push(dst_replica_index);
                            }
                        }
                        // no need to manually filter, just check
                        // invariant: Nvl src & dst are not occupied
                        let nvl_mrk_lck = replica_nvl_states.read().await;
                        nvlink_chains
                            .keys()
                            .chain(
                                nvlink_chains
                                    .values()
                                    .flat_map(|replica_indices| replica_indices.iter()),
                            )
                            .for_each(|replica_index| {
                                assert_eq!(
                                    nvl_mrk_lck[*replica_index],
                                    ReplicaState::Inactive,
                                    "Rank<{}> reported availale by Sman, but marked as {:?}",
                                    replica_index,
                                    nvl_mrk_lck[*replica_index]
                                )
                            });
                        drop(nvl_mrk_lck);
                        // operate on each chain, currently one NvlChain/Machine
                        self.nvl_scale_up_p_and_d(
                            replica_nvl_states,
                            &mut inactive_replica_indices,
                            &mut crct_num_prefill_replica,
                            &mut crct_num_decode_replica,
                            nvlink_chains,
                        )
                        .await;
                    }
                }
                // \invariant : "coro_yield"
                assert_eq!(crct_num_decode_replica, 0);
                if crct_num_prefill_replica > 0 && inactive_replica_indices.len() > 0 {
                    // choose ranks to avoid bandwidth interference
                    let mut max_rdma_sending_replica_indices = self
                        .span_max_rdma_sending_replicas(
                            &normal_prefill_replica_indices,
                            &normal_decode_replica_indices,
                            shutting_decode_replica_indices,
                            shutted_null_replica_indices,
                            replica_rdma_states,
                            replica_nvl_states,
                        )
                        .await;
                    // \post: need_to_scale
                    // \post: valid_replica_indices
                    if cfg!(feature = "test_live")
                        && crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                    {
                        if max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            self.live_p2p_scale_up_n_prefill(
                                &mut crct_num_prefill_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut normal_prefill_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await;
                        }
                        yield_now().await;
                        return;
                    }
                    if cfg!(feature = "impl_tanz") && max_rdma_sending_replica_indices.len() > 0 {
                        let handle = self
                            .tanz_scale_up_n_null(
                                &mut crct_num_prefill_replica,
                                &mut crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                &mut normal_prefill_replica_indices,
                                replica_rdma_states,
                                replica_nvl_states,
                            )
                            .await;
                        // \post correct signal updated
                        // \note "fallthrough"
                        if let Some(handle) = handle {
                            PLANNER_BUSY.store(true, Ordering::Release);
                            handle.await.unwrap();
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            PLANNER_BUSY.store(false, Ordering::Release);
                        } else {
                            yield_now().await;
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                    }
                    if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_fast")
                    {
                        if cfg!(feature = "live") {
                            let ((num_scaled_prefill, num_scaled_decode), handle) = self
                                .rdma_bcast_scale_w_live(
                                    &mut crct_num_prefill_replica,
                                    &mut crct_num_decode_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    &mut normal_prefill_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            if (num_scaled_prefill + num_scaled_decode) > 0 {
                                PLANNER_BUSY.store(true, Ordering::Release);
                                handle.await.unwrap();
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                                PLANNER_BUSY.store(false, Ordering::Release);
                            } else {
                                yield_now().await;
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            }
                            return;
                        }
                        // pre: valid_src_replica_indices
                        // pre: valid_dst_replica_indices
                        // pre: need_to_scale
                        self.rdma_bcast_scale_n_prefill(
                            model_name,
                            crct_num_prefill_replica,
                            &mut max_rdma_sending_replica_indices,
                            &mut inactive_replica_indices,
                            replica_rdma_states,
                        )
                        .await;
                    } else if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_rdma")
                    {
                        // \pre : "valid_src_replica_indices"
                        // \pre : "valid_dst_replica_indices"
                        // \pre : "need_to_scale"
                        self.rdma_p2p_scale_up_n_prefill(
                            model_name,
                            crct_num_prefill_replica,
                            &mut max_rdma_sending_replica_indices,
                            &mut inactive_replica_indices,
                            replica_rdma_states,
                        )
                        .await;
                    } else if cfg!(feature = "impl_sllm") {
                        if crct_num_decode_replica > 0 {
                            self.decode_scale_up_n_serverless(
                                crct_num_decode_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                        if crct_num_prefill_replica > 0 {
                            self.prefill_scale_up_n_serverless(
                                crct_num_prefill_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                    }
                }
            }
            (ScalePlan::Stay, ScalePlan::ScaleUp) => {
                self.shutting_decode_to_decode(
                    &mut crct_num_decode_replica,
                    &mut shutting_decode_replica_indices,
                    &mut normal_decode_replica_indices,
                )
                .await;
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;
                self.shutted_null_to_decode(
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if cfg!(feature = "impl_nvl") {
                    assert!((crct_num_prefill_replica <= 0));
                    if crct_num_decode_replica > 0 {
                        let max_nvlink_chain_edges = self
                            .steersman
                            .lock()
                            .await
                            .assign_nvlink_chains(&model_name, &mut inactive_replica_indices);
                        let mut nvlink_chains = HashMap::new();
                        for (src_replica_index, dst_replica_index) in max_nvlink_chain_edges {
                            if let Some(k) = src_replica_index {
                                nvlink_chains
                                    .entry(k)
                                    .or_insert_with(Vec::new)
                                    .push(dst_replica_index);
                            }
                        }
                        // no need to manually filter, just check
                        // invariant: Nvl src & dst are not occupied
                        let nvl_mrk_lck = replica_nvl_states.read().await;
                        nvlink_chains
                            .keys()
                            .chain(
                                nvlink_chains
                                    .values()
                                    .flat_map(|replica_indices| replica_indices.iter()),
                            )
                            .for_each(|replica_index| {
                                assert_eq!(
                                    nvl_mrk_lck[*replica_index],
                                    ReplicaState::Inactive,
                                    "Rank<{}> reported availale by Sman, but marked as {:?}",
                                    replica_index,
                                    nvl_mrk_lck[*replica_index]
                                )
                            });
                        drop(nvl_mrk_lck);
                        // operate on each chain, currently one NvlChain/Machine
                        self.nvl_scale_up_p_and_d(
                            replica_nvl_states,
                            &mut inactive_replica_indices,
                            &mut crct_num_prefill_replica,
                            &mut crct_num_decode_replica,
                            nvlink_chains,
                        )
                        .await;
                    }
                }
                // \invariant : "coro_yield"
                assert_eq!(crct_num_prefill_replica, 0);
                if crct_num_decode_replica > 0 && inactive_replica_indices.len() > 0 {
                    // choose ranks to avoid bandwidth interference
                    let mut max_rdma_sending_replica_indices = self
                        .span_max_rdma_sending_replicas(
                            &normal_prefill_replica_indices,
                            &normal_decode_replica_indices,
                            shutting_decode_replica_indices,
                            shutted_null_replica_indices,
                            replica_rdma_states,
                            replica_nvl_states,
                        )
                        .await;
                    if cfg!(feature = "impl_tanz") && max_rdma_sending_replica_indices.len() > 0 {
                        let handle = self
                            .tanz_scale_up_n_null(
                                &mut crct_num_prefill_replica,
                                &mut crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                &mut normal_prefill_replica_indices,
                                replica_rdma_states,
                                replica_nvl_states,
                            )
                            .await;
                        SCARCE_DECODE.store(crct_num_decode_replica > 0, Ordering::Release);
                        if let Some(handle) = handle {
                            PLANNER_BUSY.store(true, Ordering::Release);
                            handle.await.unwrap();
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            PLANNER_BUSY.store(false, Ordering::Release);
                        } else {
                            yield_now().await;
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                    }
                    if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_fast")
                    {
                        let num_scaled_up_decode = if crct_num_decode_replica > 0 {
                            self.rdma_p2p_scale_up_n_decode(
                                model_name,
                                crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            crct_num_decode_replica - num_scaled_up_decode > 0,
                            Ordering::Release,
                        );
                    } else if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_rdma")
                    {
                        let num_scaled_up_decode = if crct_num_prefill_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                        {
                            let (nscaled, _) = self
                                .rdma_bcast_scale_n_decode(
                                    model_name,
                                    crct_num_prefill_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            nscaled
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            crct_num_decode_replica - num_scaled_up_decode > 0,
                            Ordering::Release,
                        );
                    } else if cfg!(feature = "impl_sllm") {
                        if crct_num_decode_replica > 0 {
                            self.decode_scale_up_n_serverless(
                                crct_num_decode_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                        if crct_num_prefill_replica > 0 {
                            self.prefill_scale_up_n_serverless(
                                crct_num_prefill_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                    }
                }
            }
            (ScalePlan::ScaleUp, ScalePlan::ScaleDown) => {
                self.shutting_prefill_to_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutting_prefill_replica_indices,
                    &mut normal_prefill_replica_indices,
                )
                .await;
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(
                        &mut crct_num_decode_replica,
                        overprovision_timestamp,
                    )
                    .await;
                self.shutted_null_to_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                self.x_shutted_null_decode(
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if cfg!(feature = "impl_nvl") {
                    assert!((crct_num_decode_replica <= 0));
                    if crct_num_prefill_replica > 0 {
                        let max_nvlink_chain_edges = self
                            .steersman
                            .lock()
                            .await
                            .assign_nvlink_chains(&model_name, &mut inactive_replica_indices);
                        let mut nvlink_chains = HashMap::new();
                        for (src_replica_index, dst_replica_index) in max_nvlink_chain_edges {
                            if let Some(k) = src_replica_index {
                                nvlink_chains
                                    .entry(k)
                                    .or_insert_with(Vec::new)
                                    .push(dst_replica_index);
                            }
                        }
                        // no need to manually filter, just check
                        // invariant: Nvl src & dst are not occupied
                        let nvl_mrk_lck = replica_nvl_states.read().await;
                        nvlink_chains
                            .keys()
                            .chain(
                                nvlink_chains
                                    .values()
                                    .flat_map(|replica_indices| replica_indices.iter()),
                            )
                            .for_each(|replica_index| {
                                assert_eq!(
                                    nvl_mrk_lck[*replica_index],
                                    ReplicaState::Inactive,
                                    "Rank<{}> reported availale by Sman, but marked as {:?}",
                                    replica_index,
                                    nvl_mrk_lck[*replica_index]
                                )
                            });
                        drop(nvl_mrk_lck);
                        // operate on each chain, currently one NvlChain/Machine
                        self.nvl_scale_up_p_and_d(
                            replica_nvl_states,
                            &mut inactive_replica_indices,
                            &mut crct_num_prefill_replica,
                            &mut crct_num_decode_replica,
                            nvlink_chains,
                        )
                        .await;
                    }
                }
                // \pre :: Skip
                assert!(crct_num_decode_replica <= 0);
                let mut max_rdma_sending_replica_indices: Vec<usize>;
                if crct_num_prefill_replica > 0 && inactive_replica_indices.len() > 0 {
                    // choose ranks
                    max_rdma_sending_replica_indices = self
                        .span_max_rdma_sending_replicas(
                            &normal_prefill_replica_indices,
                            &normal_decode_replica_indices,
                            shutting_decode_replica_indices,
                            shutted_null_replica_indices,
                            replica_rdma_states,
                            replica_nvl_states,
                        )
                        .await;
                    if cfg!(feature = "test_live")
                        && crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                    {
                        if max_rdma_sending_replica_indices.len() > 0
                            && inactive_replica_indices.len() > 0
                        {
                            self.live_p2p_scale_up_n_prefill(
                                &mut crct_num_prefill_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut normal_prefill_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await;
                        }
                        yield_now().await;
                        return;
                    }
                    if cfg!(feature = "impl_tanz") && max_rdma_sending_replica_indices.len() > 0 {
                        let handle = self
                            .tanz_scale_up_n_null(
                                &mut crct_num_prefill_replica,
                                &mut crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                &mut normal_prefill_replica_indices,
                                replica_rdma_states,
                                replica_nvl_states,
                            )
                            .await;
                        if let Some(handle) = handle {
                            PLANNER_BUSY.store(true, Ordering::Release);
                            handle.await.unwrap();
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            PLANNER_BUSY.store(false, Ordering::Release);
                        } else {
                            yield_now().await;
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                        // \post correct signal updated
                        // \note "fallthrough"
                    }
                    if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_fast")
                    {
                        if cfg!(feature = "live") {
                            let ((num_scaled_prefill, num_scaled_decode), handle) = self
                                .rdma_bcast_scale_w_live(
                                    &mut crct_num_prefill_replica,
                                    &mut crct_num_decode_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    &mut normal_prefill_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            if (num_scaled_prefill + num_scaled_decode) > 0 {
                                PLANNER_BUSY.store(true, Ordering::Release);
                                handle.await.unwrap();
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                                PLANNER_BUSY.store(false, Ordering::Release);
                            } else {
                                yield_now().await;
                                CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            }
                            return;
                        }
                        // pre: valid_src_replica_indices
                        // pre: valid_dst_replica_indices
                        // pre: need_to_scale
                        self.rdma_bcast_scale_n_prefill(
                            model_name,
                            crct_num_prefill_replica,
                            &mut max_rdma_sending_replica_indices,
                            &mut inactive_replica_indices,
                            replica_rdma_states,
                        )
                        .await;
                    } else if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_rdma")
                    {
                        // \pre : "valid_src_replica_indices"
                        // \pre : "valid_dst_replica_indices"
                        // \pre : "need_to_scale"
                        self.rdma_p2p_scale_up_n_prefill(
                            model_name,
                            crct_num_prefill_replica,
                            &mut max_rdma_sending_replica_indices,
                            &mut inactive_replica_indices,
                            replica_rdma_states,
                        )
                        .await;
                    } else if cfg!(feature = "impl_sllm") {
                        if crct_num_decode_replica > 0 {
                            self.decode_scale_up_n_serverless(
                                crct_num_decode_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                        if crct_num_prefill_replica > 0 {
                            self.prefill_scale_up_n_serverless(
                                crct_num_prefill_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                    }
                } else {
                    max_rdma_sending_replica_indices = normal_decode_replica_indices
                        .iter()
                        .map(|&(_, replica_index)| replica_index)
                        .collect();
                }
                let mut normal_decode_replica_indices = normal_decode_replica_indices
                    .into_iter()
                    .map(|(_, replica_index)| replica_index)
                    .filter(|replica_index| {
                        max_rdma_sending_replica_indices.contains(replica_index)
                    })
                    .collect();
                self.shut_down_decode_one(
                    &mut normal_decode_replica_indices,
                    overprovision_timestamp,
                    null_replica_state,
                )
                .await;
            }
            (ScalePlan::ScaleDown, ScalePlan::ScaleUp) => {
                self.shutting_decode_to_decode(
                    &mut crct_num_decode_replica,
                    &mut shutting_decode_replica_indices,
                    &mut normal_decode_replica_indices,
                )
                .await;
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;
                self.shutted_null_to_decode(
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                self.x_shutted_null_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if cfg!(feature = "impl_nvl") {
                    assert!((crct_num_prefill_replica <= 0));
                    if crct_num_decode_replica > 0 {
                        let max_nvlink_chain_edges = self
                            .steersman
                            .lock()
                            .await
                            .assign_nvlink_chains(&model_name, &mut inactive_replica_indices);
                        let mut nvlink_chains = HashMap::new();
                        for (src_replica_index, dst_replica_index) in max_nvlink_chain_edges {
                            if let Some(k) = src_replica_index {
                                nvlink_chains
                                    .entry(k)
                                    .or_insert_with(Vec::new)
                                    .push(dst_replica_index);
                            }
                        }
                        // no need to manually filter, just check
                        // invariant: Nvl src & dst are not occupied
                        let nvl_mrk_lck = replica_nvl_states.read().await;
                        nvlink_chains
                            .keys()
                            .chain(
                                nvlink_chains
                                    .values()
                                    .flat_map(|replica_indices| replica_indices.iter()),
                            )
                            .for_each(|replica_index| {
                                assert_eq!(
                                    nvl_mrk_lck[*replica_index],
                                    ReplicaState::Inactive,
                                    "Rank<{}> reported availale by Sman, but marked as {:?}",
                                    replica_index,
                                    nvl_mrk_lck[*replica_index]
                                )
                            });
                        drop(nvl_mrk_lck);
                        // operate on each chain, currently one NvlChain/Machine
                        self.nvl_scale_up_p_and_d(
                            replica_nvl_states,
                            &mut inactive_replica_indices,
                            &mut crct_num_prefill_replica,
                            &mut crct_num_decode_replica,
                            nvlink_chains,
                        )
                        .await;
                    }
                }
                if crct_num_prefill_replica < 0 {
                    self.mutate_prefill_to_decode(
                        &mut crct_num_prefill_replica,
                        &mut crct_num_decode_replica,
                        &mut normal_prefill_replica_indices,
                    )
                    .await;
                }
                assert!(crct_num_prefill_replica <= 0);
                if crct_num_decode_replica > 0 && inactive_replica_indices.len() > 0 {
                    // choose ranks
                    let mut max_rdma_sending_replica_indices = self
                        .span_max_rdma_sending_replicas(
                            &normal_prefill_replica_indices,
                            &normal_decode_replica_indices,
                            shutting_decode_replica_indices,
                            shutted_null_replica_indices,
                            replica_rdma_states,
                            replica_nvl_states,
                        )
                        .await;
                    if cfg!(feature = "impl_tanz") && max_rdma_sending_replica_indices.len() > 0 {
                        let handle = self
                            .tanz_scale_up_n_null(
                                &mut crct_num_prefill_replica,
                                &mut crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                &mut normal_prefill_replica_indices,
                                replica_rdma_states,
                                replica_nvl_states,
                            )
                            .await;
                        if let Some(handle) = handle {
                            PLANNER_BUSY.store(true, Ordering::Release);
                            handle.await.unwrap();
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                            PLANNER_BUSY.store(false, Ordering::Release);
                        } else {
                            yield_now().await;
                            CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
                        }
                        return;
                    }
                    if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_fast")
                    {
                        let num_scaled_up_decode = if crct_num_decode_replica > 0 {
                            self.rdma_p2p_scale_up_n_decode(
                                model_name,
                                crct_num_decode_replica,
                                &mut max_rdma_sending_replica_indices,
                                &mut inactive_replica_indices,
                                replica_rdma_states,
                            )
                            .await
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            crct_num_decode_replica - num_scaled_up_decode > 0,
                            Ordering::Release,
                        );
                    } else if crct_num_prefill_replica > 0
                        && max_rdma_sending_replica_indices.len() > 0
                        && inactive_replica_indices.len() > 0
                        && cfg!(feature = "impl_rdma")
                    {
                        let num_scaled_up_decode = if crct_num_prefill_replica > 0
                            && max_rdma_sending_replica_indices.len() > 0
                        {
                            let (nscaled, _) = self
                                .rdma_bcast_scale_n_decode(
                                    model_name,
                                    crct_num_prefill_replica,
                                    &mut max_rdma_sending_replica_indices,
                                    &mut inactive_replica_indices,
                                    replica_rdma_states,
                                )
                                .await;
                            nscaled
                        } else {
                            0
                        };
                        SCARCE_DECODE.store(
                            crct_num_decode_replica - num_scaled_up_decode > 0,
                            Ordering::Release,
                        );
                    } else if cfg!(feature = "impl_sllm") {
                        if crct_num_decode_replica > 0 {
                            self.decode_scale_up_n_serverless(
                                crct_num_decode_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                        if crct_num_prefill_replica > 0 {
                            self.prefill_scale_up_n_serverless(
                                crct_num_prefill_replica,
                                &mut inactive_replica_indices,
                            )
                            .await;
                        }
                    }
                }
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
            (ScalePlan::Stay, ScalePlan::ScaleDown) => {
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(
                        &mut crct_num_decode_replica,
                        overprovision_timestamp,
                    )
                    .await;
                self.x_shutted_null_decode(
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                assert_eq!(crct_num_prefill_replica, 0);
                if crct_num_decode_replica < 0 {
                    let mut normal_decode_replica_indices = normal_decode_replica_indices
                        .into_iter()
                        .map(|(_, replica_index)| replica_index)
                        .collect();
                    self.shut_down_decode_one(
                        &mut normal_decode_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
            }
            (ScalePlan::ScaleDown, ScalePlan::Stay) => {
                assert_eq!(crct_num_decode_replica, 0);
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(&mut noExpiredDecode, overprovision_timestamp)
                    .await;
                self.x_shutted_null_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                assert_eq!(crct_num_decode_replica, 0);
                if crct_num_prefill_replica < 0 {
                    self.shut_down_prefill_one(
                        &mut normal_prefill_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
                SCARCE_DECODE.store(false, Ordering::Release);
            }
            (ScalePlan::ScaleDown, ScalePlan::ScaleDown) => {
                let mut shutted_null_replica_indices = self
                    .collect_null_replica_indices(
                        &mut crct_num_decode_replica,
                        overprovision_timestamp,
                    )
                    .await;
                self.x_shutted_null_decode(
                    &mut crct_num_decode_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                self.x_shutted_null_prefill(
                    &mut crct_num_prefill_replica,
                    &mut shutted_null_replica_indices,
                    overprovision_timestamp,
                )
                .await;
                if crct_num_decode_replica < 0 {
                    let mut normal_decode_replica_indices = normal_decode_replica_indices
                        .into_iter()
                        .map(|(_, replica_index)| replica_index)
                        .collect();
                    self.shut_down_decode_one(
                        &mut normal_decode_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
                if crct_num_prefill_replica < 0 {
                    self.shut_down_prefill_one(
                        &mut normal_prefill_replica_indices,
                        overprovision_timestamp,
                        null_replica_state,
                    )
                    .await;
                }
                SCARCE_DECODE.store(false, Ordering::Release);
            }
        };
        CTRL_LOOP_CNT.fetch_add(1, Ordering::AcqRel);
        yield_now().await;
    }

    async fn span_max_rdma_sending_replicas(
        self: &Arc<Self>,
        normal_prefill_replica_indices: &VecDeque<(u32, usize)>,
        normal_decode_replica_indices: &VecDeque<(u32, usize)>,
        shutting_decode_replica_indices: VecDeque<(u32, usize)>,
        shutted_null_replica_indices: Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
        _replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> Vec<usize> {
        let rdma_mrk_lck = replica_rdma_states.read().await;
        let rdma_sending_replicas = if cfg!(feature = "unidirect") {
            Vec::from_iter(
                normal_prefill_replica_indices
                    .iter()
                    .map(|&(_, replica_index)| replica_index)
                    .chain(
                        normal_decode_replica_indices
                            .iter()
                            .map(|&(_, replica_index)| replica_index),
                    )
                    .chain(
                        shutting_decode_replica_indices
                            .into_iter()
                            .map(|(_, replica_index)| replica_index),
                    )
                    .chain(shutted_null_replica_indices.into_iter())
                    .filter(|replica_index| rdma_mrk_lck[*replica_index] == ReplicaState::Inactive),
            )
        } else {
            Vec::from_iter(
                normal_decode_replica_indices
                    .iter()
                    .map(|&(_, replica_index)| replica_index)
                    .chain(
                        shutting_decode_replica_indices
                            .into_iter()
                            .map(|(_, replica_index)| replica_index),
                    )
                    .chain(shutted_null_replica_indices.into_iter())
                    .filter(|replica_index| rdma_mrk_lck[*replica_index] == ReplicaState::Inactive),
            )
        };
        rdma_sending_replicas
    }

    /// Filter out 1 of 2 (possible) SendingDecodes on the same RNIC
    ///
    /// pre: need_to_scale
    #[cfg(feature = "buddy_nic")]
    async fn filter_buddy_nic_src_rank(
        self: &Arc<Self>,
        src_replica_indices: &mut VecDeque<(u32, usize)>,
    ) {
        // roll out buddy nic
        let mut prev_nic_num = None;
        let mut new_src_replica_indices = VecDeque::new();
        src_replica_indices
            .make_contiguous()
            .sort_by_key(|&(_, index)| index);
        for &(blocks, replica_index) in src_replica_indices.iter() {
            let same_nic = prev_nic_num == Some(replica_index >> 1);
            prev_nic_num = Some(replica_index >> 1);
            // use same nic -> same flow direction -> continue
            if same_nic
                && *self.replica_metrics[&(replica_index ^ 1)]
                    .state
                    .read()
                    .await
                    == ReplicaState::RdmaSending
            {
                continue;
            }
            new_src_replica_indices.push_back((blocks, replica_index));
        }
        swap(src_replica_indices, &mut new_src_replica_indices);
    }

    /// Filter out 1 of 2 (possible) Inactives on the same RNIC
    ///
    /// pre: need_to_scale
    #[cfg(feature = "buddy_nic")]
    async fn filter_buddy_nic_dst_rank(self: &Arc<Self>, dst_replica_indices: &mut Vec<usize>) {
        // roll out buddy nic
        let mut prev_nic_num = None;
        let mut new_dst_replica_indices = Vec::new();
        dst_replica_indices.sort();
        for &replica_index in dst_replica_indices.iter() {
            let same_nic = prev_nic_num == Some(replica_index >> 1);
            prev_nic_num = Some(replica_index >> 1);
            // use same nic -> same flow direction -> continue
            if same_nic
                && *self.replica_metrics[&(replica_index ^ 1)]
                    .state
                    .read()
                    .await
                    == ReplicaState::Inactive
            {
                continue;
            }
            new_dst_replica_indices.push(replica_index);
        }
        swap(dst_replica_indices, &mut new_dst_replica_indices);
    }

    /// pre: need_to_scale
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices
    async fn live_p2p_scale_up_n_prefill(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        src_replica_indices: &mut Vec<usize>, // SendingDecode
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
        inactive_replica_indices: &mut Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) {
        // eligibility
        assert!(*crct_num_prefill_replica > 0);
        assert!(src_replica_indices.len() > 0);
        assert!(inactive_replica_indices.len() > 0);

        let num_max_new_prefill_replica = min(
            src_replica_indices
                .len()
                .min(inactive_replica_indices.len()),
            normal_prefill_replica_indices.len(),
        );
        let num_add_new_prefill_replica = min(
            num_max_new_prefill_replica as i32,
            *crct_num_prefill_replica,
        ) as usize;

        for _ in 0..num_add_new_prefill_replica {
            let src_replica_index = src_replica_indices.pop().unwrap();
            let new_replica_index = inactive_replica_indices.pop().unwrap();
            let (_, old_replica_index) = normal_prefill_replica_indices.pop_front().unwrap();
            // SendingDecode ++ NewPrefill
            let orchest = self.clone();
            let dst_replica_metric = self.replica_metrics[&new_replica_index].clone();
            // 1. mark Planner state
            tracing::info!("Replica<{}> (zigzag) => SendingDecode", src_replica_index);
            {
                let mut rdma_mrk_lck = replica_rdma_states.write().await;
                rdma_mrk_lck[src_replica_index] = ReplicaState::RdmaSending;
                rdma_mrk_lck[new_replica_index] = ReplicaState::RdmaLoading;
            }
            let replica_rdma_state0 = replica_rdma_states.clone();
            // 1. marked
            spawn(async move {
                orchest
                    .rdma_p2p_n_inner(
                        orchest.replica_to_ranks[&src_replica_index].clone(),
                        orchest.replica_to_ranks[&new_replica_index].clone(),
                    )
                    .await;
                // 2. mark Worker loaded
                tracing::info!("Replica<{}> : (live) Parameter loaded!", new_replica_index);
                dst_replica_metric.set_model_loaded(true);
                // 2. marked
                while *orchest.replica_metrics[&new_replica_index]
                    .state
                    .read()
                    .await
                    == ReplicaState::NewPrefill
                {
                    yield_now().await;
                }
                assert!(
                    *orchest.replica_metrics[&new_replica_index]
                        .state
                        .read()
                        .await
                        == ReplicaState::RefractoryPrefill
                        || *orchest.replica_metrics[&new_replica_index]
                            .state
                            .read()
                            .await
                            == ReplicaState::Prefill
                );
                // 3. revert Steersman & Planner state
                {
                    let mut sman_lck = orchest.steersman.lock().await;
                    (new_replica_index
                        ..new_replica_index
                            + orchest.disaggregation_controller_args.tensor_parallel_size)
                        .for_each(|rank| sman_lck.record_model_loaded(rank));
                }
                let mut rdma_mrk_lck = replica_rdma_state0.write().await;
                rdma_mrk_lck[src_replica_index] = ReplicaState::Inactive;
                rdma_mrk_lck[new_replica_index] = ReplicaState::Inactive;
                // 3. reverted
            });
            tracing::info!("Replica<{}> (zigzag) => NewPrefill", new_replica_index);
            *self.replica_metrics[&new_replica_index].state.write().await =
                ReplicaState::NewPrefill;
            tracing::info!("Replica<{}> (zigzag) => OldPrefill", old_replica_index);
            *self.replica_metrics[&old_replica_index].state.write().await =
                ReplicaState::OldPrefill;
            ZIGZAG_ACTIVE_CNT.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// pre: need_to_scale
    ///
    /// invariant: updated_crct_num
    /// invariant: valid_replica_indices
    async fn tanz_scale_up_n_null(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        src_replica_indices: &mut Vec<usize>,
        inactive_replica_indices: &mut Vec<usize>,
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
        replica_nvl_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> Option<JoinHandle<()>> {
        // pre: "valid_replica_indices"
        // <len(column), list(ranks)>
        let mut dst_ranks_in_tanz_chains = self
            .steersman
            .lock()
            .await
            .assign_tanz_chains(inactive_replica_indices);

        // inccreasing order
        dst_ranks_in_tanz_chains
            .make_contiguous()
            .sort_by(|a, b| a.0.cmp(&b.0));

        // preparation
        let mut crct_num_replica = *crct_num_prefill_replica + *crct_num_decode_replica;
        let max_one_mchn_rank_num = dst_ranks_in_tanz_chains
            .back()
            .unwrap_or(&(0, Vec::new()))
            .0;

        // max rows of Tanz matrix (a.k.a widest lane)
        let max_lane_num = src_replica_indices
            .len()
            .min(crct_num_replica as usize)
            .min(max_one_mchn_rank_num / self.disaggregation_controller_args.tensor_parallel_size);
        let mut src_replica_indices0 = src_replica_indices.clone();
        let mut dst_replica_indices0 = Vec::<usize>::new();
        src_replica_indices0.truncate(max_lane_num);

        // assign src and dsts
        for (mchn_rank_num, dst_replica_indices) in dst_ranks_in_tanz_chains.iter_mut() {
            if crct_num_replica <= 0 {
                break;
            }
            if *mchn_rank_num / self.disaggregation_controller_args.tensor_parallel_size
                >= max_lane_num
            {
                // can handle this case
                dst_replica_indices.truncate(max_lane_num);
                dst_replica_indices0.extend(dst_replica_indices.iter());
                crct_num_replica -= max_lane_num as i32;
            }
        }

        if max_lane_num > 1 && src_replica_indices0.len() > 0 && dst_replica_indices0.len() > 0 {
            // 1. mark Planner & Steersman state
            let mut rdma_mrk_lck = replica_rdma_states.write().await;
            for src_replica_index in src_replica_indices0.iter() {
                rdma_mrk_lck[*src_replica_index] = ReplicaState::RdmaSending;
            }
            let mut nvl_mrk_lck = replica_nvl_states.write().await;
            let mut sman_lck = self.steersman.lock().await;
            for dst_replica_index in dst_replica_indices0.iter() {
                rdma_mrk_lck[*dst_replica_index] = ReplicaState::TanzCasting;
                nvl_mrk_lck[*dst_replica_index] = ReplicaState::TanzCasting;
                (*dst_replica_index
                    ..*dst_replica_index
                        + self.disaggregation_controller_args.tensor_parallel_size)
                    .for_each(|rank| sman_lck.wait_replica_nvlink(rank, 1));
            }
            // 1. marked
            drop(sman_lck);
            drop(nvl_mrk_lck);
            drop(rdma_mrk_lck);
            // 2. entail Worker state
            assert_eq!(dst_replica_indices0.len() % src_replica_indices0.len(), 0);
            let handles = self
                .walzer_tanzen_n(
                    crct_num_prefill_replica,
                    crct_num_decode_replica,
                    &src_replica_indices0,
                    &dst_replica_indices0,
                )
                .await;
            if cfg!(feature = "live") {
                // trigger OldPrefill
                // len(dst_replicas) >= len(src_replicas)
                let num_add_old_prefill = src_replica_indices0
                    .len()
                    .min(normal_prefill_replica_indices.len());
                for _ in 0..num_add_old_prefill {
                    let (_, old_replica_index) =
                        normal_prefill_replica_indices.pop_front().unwrap();
                    tracing::info!("Replica<{}> (zigzag) => OldPrefill", old_replica_index);
                    *self.replica_metrics[&old_replica_index].state.write().await =
                        ReplicaState::OldPrefill;
                }
                // NOTE: in case not enough OldPrefill is scaled
                for _ in num_add_old_prefill..src_replica_indices0.len() {
                    RELAY_DEACTIVE_CNT.fetch_sub(1, Ordering::SeqCst);
                }
            }
            if cfg!(feature = "mutate")
                && *crct_num_prefill_replica < 0
                && *crct_num_decode_replica > 0
            {
                let num_mutate_p2d = crct_num_prefill_replica
                    .abs()
                    .min(*crct_num_decode_replica)
                    .min(normal_prefill_replica_indices.len() as i32)
                    as usize;
                for _ in 0..num_mutate_p2d {
                    let (_, mutating_prefill_replica_index) =
                        normal_prefill_replica_indices.pop_back().unwrap();
                    *self.replica_metrics[&mutating_prefill_replica_index]
                        .state
                        .write()
                        .await = ReplicaState::MutatingToDecode;
                    tracing::info!(
                        "Replica<{}> => MutatingToDecode",
                        mutating_prefill_replica_index
                    );
                }
            }
            // 3. revert Planner & Steersman mark
            let replica_rdma_state0 = replica_rdma_states.clone();
            let replica_nvl_state0 = replica_nvl_states.clone();
            let orchest = self.clone();
            let steersman = self.steersman.clone();
            let src_replica_indices1 = src_replica_indices0.clone();
            let dst_replica_indices1 = dst_replica_indices0.clone();
            let handle = spawn(async move {
                join_all(handles).await;
                let mut rdma_mrk_lck = replica_rdma_state0.write().await;
                for replica_index in src_replica_indices1 {
                    rdma_mrk_lck[replica_index] = ReplicaState::Inactive;
                }
                let mut nvl_mrk_lck = replica_nvl_state0.write().await;
                let mut sman_lck = steersman.lock().await;
                for replica_index in dst_replica_indices1 {
                    rdma_mrk_lck[replica_index] = ReplicaState::Inactive;
                    nvl_mrk_lck[replica_index] = ReplicaState::Inactive;
                    (replica_index
                        ..replica_index
                            + orchest.disaggregation_controller_args.tensor_parallel_size)
                        .for_each(|rank| sman_lck.post_replica_nvlink(rank));
                    while *orchest.replica_metrics[&replica_index].state.read().await
                        == ReplicaState::NewPrefill
                    {
                        yield_now().await;
                    }
                    assert!(
                        *orchest.replica_metrics[&replica_index].state.read().await
                            == ReplicaState::RefractoryPrefill
                            || *orchest.replica_metrics[&replica_index].state.read().await
                                == ReplicaState::Prefill
                    );
                    (replica_index
                        ..replica_index
                            + orchest.disaggregation_controller_args.tensor_parallel_size)
                        .for_each(|rank| sman_lck.record_model_loaded(rank));
                }
            });
            let _ = src_replica_indices.split_off(max_lane_num);
            inactive_replica_indices.retain(|x| !dst_replica_indices0.contains(x));
            *crct_num_prefill_replica -=
                (dst_replica_indices0.len() as i32 - *crct_num_decode_replica).max(0);
            *crct_num_decode_replica -=
                (*crct_num_decode_replica).min(dst_replica_indices0.len() as i32);
            Some(handle)
        } else if src_replica_indices0.len() > 0 && dst_replica_indices0.len() > 0 {
            assert_eq!(
                max_lane_num, 1,
                "Tanz fallback path triggered! num_lane={}",
                max_lane_num
            );
            tracing::warn!(
                "Degenerated Tanz matrix: max_lane_num={}; src_ranks={:?}, dst_ranks={:?}!",
                max_lane_num,
                src_replica_indices0,
                dst_replica_indices0
            );
            // NOTE: dst_ranks are truncated by 1, therefore scatter scaling onto different machines
            if *crct_num_prefill_replica > *crct_num_decode_replica {
                let (_, handle) = self
                    .rdma_bcast_scale_n_prefill(
                        "",
                        dst_replica_indices0.len() as i32,
                        &mut src_replica_indices0,
                        &mut dst_replica_indices0,
                        replica_rdma_states,
                    )
                    .await;
                Some(handle)
            } else {
                let (_, handle) = self
                    .rdma_bcast_scale_n_decode(
                        "",
                        dst_replica_indices0.len() as i32,
                        &mut src_replica_indices0,
                        &mut dst_replica_indices0,
                        replica_rdma_states,
                    )
                    .await;
                Some(handle)
            }
        } else {
            tracing::warn!("Run out of GPUs!");
            None
        }
    }

    /// P2P scale up n Decode instance 4 "impl_rdma"
    ///
    /// "scale" in fn name -> mark Planner & Steersman & Worker state!
    ///
    /// src: Prefill/Decode => SendingPrefill/Decode => Prefill/Decode
    /// dst: Inactive => LoadingDecode => Decode
    async fn rdma_p2p_scale_up_n_decode(
        self: &Arc<Self>,
        model_name: &str,
        n: i32,
        src_replica_indices: &mut Vec<usize>,
        dst_replica_indices: &mut Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> i32 {
        let mut m = 0;
        for _ in 0..n {
            match (src_replica_indices.pop(), dst_replica_indices.pop()) {
                (Some(src_replica_index), Some(dst_replica_index)) => {
                    let dst_replica_metric = self.replica_metrics[&dst_replica_index].clone();
                    // 1. mark Planner state
                    let mut rdma_mrk_lck = replica_rdma_states.write().await;
                    rdma_mrk_lck[src_replica_index] = ReplicaState::RdmaSending;
                    rdma_mrk_lck[dst_replica_index] = ReplicaState::RdmaLoading;
                    drop(rdma_mrk_lck);
                    // 1. marked
                    // RDMA bw monitor
                    let mut flow_watcher = FLOW_WATCHER.lock().await;
                    let param_size =
                        self.steersman.lock().await.get_model_param_size(model_name) as usize;
                    flow_watcher.append_param(src_replica_index, param_size);
                    flow_watcher.recv_param(dst_replica_index, param_size);
                    // 2. mark Worker
                    tracing::info!(
                        "Replica<{}> (rdma::p2p) => LoadingDecode",
                        dst_replica_index
                    );
                    *dst_replica_metric.state.write().await = ReplicaState::LoadingDecode;
                    let orchest = self.clone();
                    let replica_rdma_state0 = replica_rdma_states.clone();
                    spawn(async move {
                        orchest
                            .rdma_p2p_n_inner(
                                orchest.replica_to_ranks[&src_replica_index].clone(),
                                orchest.replica_to_ranks[&dst_replica_index].clone(),
                            )
                            .await;
                        tracing::info!(
                            "Replica<{}> (rdma::p2p) LoadingDecode => AusDecode",
                            dst_replica_index
                        );
                        *dst_replica_metric.state.write().await = ReplicaState::AusDecode;
                        dst_replica_metric.set_model_loaded(true);
                        // 2. marked
                        // 3. revert Steersman & Planner
                        {
                            let mut sman_lck = orchest.steersman.lock().await;
                            (dst_replica_index
                                ..dst_replica_index
                                    + orchest.disaggregation_controller_args.tensor_parallel_size)
                                .for_each(|rank| sman_lck.record_model_loaded(rank));
                        }
                        let mut rdma_mrk_lck = replica_rdma_state0.write().await;
                        rdma_mrk_lck[src_replica_index] = ReplicaState::Inactive;
                        rdma_mrk_lck[dst_replica_index] = ReplicaState::Inactive;
                        // 3. reverted
                    });
                    m += 1;
                }
                _ => {
                    break;
                }
            }
        }
        m
    }

    /// P2P scale up n prefill instance 4 "impl_rdma"
    ///
    /// src: Prefill/Decode => SendingPrefill/Decode => Prefill/Decode
    /// dst: Inactive => LoadingPrefill => Prefill
    ///
    /// pre:  valid_replica_indices
    /// post: valid_replica_indices
    async fn rdma_p2p_scale_up_n_prefill(
        self: &Arc<Self>,
        model_name: &str,
        n: i32,
        src_replica_indices: &mut Vec<usize>,
        dst_replica_indices: &mut Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> i32 {
        let mut m = 0;
        for _ in 0..n {
            match (src_replica_indices.pop(), dst_replica_indices.pop()) {
                (Some(src_replica_index), Some(dst_replica_index)) => {
                    let dst_replica_metric = self.replica_metrics[&dst_replica_index].clone();
                    // 1. mark Planner state
                    let mut rdma_mrk_lck = replica_rdma_states.write().await;
                    rdma_mrk_lck[src_replica_index] = ReplicaState::RdmaSending;
                    rdma_mrk_lck[dst_replica_index] = ReplicaState::RdmaLoading;
                    drop(rdma_mrk_lck);
                    // 1. marked
                    // RDMA bw monitor
                    let mut flow_watcher = FLOW_WATCHER.lock().await;
                    let param_size =
                        self.steersman.lock().await.get_model_param_size(model_name) as usize;
                    flow_watcher.append_param(src_replica_index, param_size);
                    flow_watcher.recv_param(dst_replica_index, param_size);
                    // 2. mark Worker
                    tracing::info!(
                        "Replica<{}> (rdma::p2p) => LoadingPrefill",
                        dst_replica_index
                    );
                    *dst_replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                    let orchest = self.clone();
                    let replica_rdma_state0 = replica_rdma_states.clone();
                    spawn(async move {
                        orchest
                            .rdma_p2p_n_inner(
                                orchest.replica_to_ranks[&src_replica_index].clone(),
                                orchest.replica_to_ranks[&dst_replica_index].clone(),
                            )
                            .await;
                        tracing::info!(
                            "Replica<{}> (rdma::p2p) LoadingPrefill => Prefill",
                            dst_replica_index
                        );
                        *dst_replica_metric.state.write().await = ReplicaState::Prefill;
                        dst_replica_metric.set_model_loaded(true);
                        // 2. marked
                        // 3. revert Steersman & Planner
                        {
                            let mut sman_lck = orchest.steersman.lock().await;
                            (dst_replica_index
                                ..dst_replica_index
                                    + orchest.disaggregation_controller_args.tensor_parallel_size)
                                .for_each(|rank| sman_lck.record_model_loaded(rank));
                        }
                        let mut rdma_mrk_lck = replica_rdma_state0.write().await;
                        rdma_mrk_lck[src_replica_index] = ReplicaState::Inactive;
                        rdma_mrk_lck[dst_replica_index] = ReplicaState::Inactive;
                        // 3. reverted
                    });
                    m += 1;
                }
                _ => {
                    break;
                }
            }
        }
        m
    }

    /// MULTICAST scale up prefill && decode instanc 4 "impl_fast"
    ///
    /// NOTE: turn chain end as NewPrefill
    ///
    /// src: Decode => SendingDecode => Decode
    /// dst: Inactive => RdmaCastingDecode | LoadingDecode => Decode
    /// dst: Inactive => RdmaCastingPrefill | NewPrefill => Prefill
    ///
    /// return: (scaled_prefill, scaled_decode), handle
    async fn rdma_bcast_scale_w_live(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        src_replica_indices: &mut Vec<usize>,
        dst_replica_indices: &mut Vec<usize>,
        normal_prefill_replica_indices: &mut VecDeque<(u32, usize)>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> ((i32, i32), JoinHandle<()>) {
        assert!(src_replica_indices.len() > 0);
        assert!(dst_replica_indices.len() > 0);
        // 1. measure # chains
        //    min(src_replica_indices, crct_num_prefill_replica)
        let num_chains = min(
            min(src_replica_indices.len(), dst_replica_indices.len()),
            (*crct_num_prefill_replica).max(0) as usize,
        );
        let mut dst_bcast_chains = vec![(Vec::<usize>::new(), Vec::<i32>::new()); num_chains];
        let mut dst_chain_it = dst_bcast_chains.iter_mut();
        let n1 = (*crct_num_prefill_replica).max(0);
        let n2 = (*crct_num_decode_replica).max(0);
        for _ in 0..(n1 + n2) {
            match dst_replica_indices.pop() {
                Some(replica_index) => {
                    let dst_chain: &mut (Vec<usize>, Vec<i32>);
                    if let Some(dst_chain0) = dst_chain_it.next() {
                        dst_chain = dst_chain0
                    } else {
                        dst_chain_it = dst_bcast_chains.iter_mut();
                        dst_chain = dst_chain_it.next().unwrap();
                    }
                    let (dst_replica_indices_in_chain, dst_ranks_in_chain) = dst_chain;
                    dst_replica_indices_in_chain.push(replica_index);
                    dst_ranks_in_chain.extend(self.replica_to_ranks[&replica_index].iter());
                    assert_eq!(
                        *self.replica_metrics[&replica_index].state.read().await,
                        ReplicaState::Inactive
                    );
                }
                None => {
                    break;
                }
            }
        }
        // 2. assemble chains
        let mut wrk_hndl_list: Vec<JoinHandle<()>> = Vec::new();
        let mut m1 = 0;
        let mut m2 = 0;
        for dst_chain in dst_bcast_chains {
            let src_replica_index = src_replica_indices.pop().unwrap();
            let dst_replica_indices2 = dst_chain.0;
            let src_ranks: Vec<i32> = self.replica_to_ranks[&src_replica_index].clone();
            let dst_ranks: Vec<i32> = dst_chain.1;
            // a. assign src replica
            {
                let replica_index = src_replica_index;
                let replica_metric = self.replica_metrics[&replica_index].clone();
                assert_eq!(*replica_metric.state.read().await, ReplicaState::Decode);
                assert_eq!(
                    replica_rdma_states.read().await[replica_index],
                    ReplicaState::Inactive
                );
                let src_ranks0 = src_ranks.clone();
                let dst_ranks0 = dst_ranks.clone();
                let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
                let replica_rdma_state0 = replica_rdma_states.clone();
                tracing::info!("Replica<{}> (rdma::chain) => SendingDecode", replica_index);
                // 1. mark Planner state
                replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaSending;
                // 1. marked
                wrk_hndl_list.push(spawn(async move {
                    // 2. skip Worker state
                    join_all(
                        replica_stubs
                            .iter_mut()
                            .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                    join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    // 2. skipped
                    // 3. revert Planner state
                    tracing::info!("Replica<{}> (rdma::chain::done) => Decode", replica_index);
                    replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                    // 3. reverted
                }));
            }
            // b. assign dst replica
            let mut is_chain_end = true;
            for replica_index in dst_replica_indices2.into_iter().rev() {
                let replica_metric = self.replica_metrics[&replica_index].clone();
                assert_eq!(*replica_metric.state.read().await, ReplicaState::Inactive);
                assert_eq!(
                    replica_rdma_states.read().await[replica_index],
                    ReplicaState::Inactive
                );
                let src_ranks0 = src_ranks.clone();
                let dst_ranks0 = dst_ranks.clone();
                let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
                let replica_rdma_state0 = replica_rdma_states.clone();
                let steersman = self.steersman.clone();
                let is_chain_end0 = is_chain_end;
                is_chain_end = false;
                // 1. mark Planner state
                tracing::info!(
                    "Replica<{}> (rdma::chain::marker) => RdmaCastingDecode",
                    replica_index
                );
                replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaCasting;
                // 1. marked
                if cfg!(feature = "live") && is_chain_end0 && m1 < n1 {
                    wrk_hndl_list.push(spawn(async move {
                        ZIGZAG_ACTIVE_CNT.fetch_add(1, Ordering::AcqRel);
                        tracing::info!("Replica<{}> (zigzag) => NewPrefill", replica_index);
                        *replica_metric.state.write().await = ReplicaState::NewPrefill;
                        // 2. mark Worker state
                        join_all(
                            replica_stubs
                                .iter_mut()
                                .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                        )
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                        join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                            .await
                            .into_iter()
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap();
                        // don't change worker state directly
                        replica_metric.set_model_loaded(true);
                        // 2. Worker marked
                        // 3. revert Steersman Planner state
                        let mut sman_lck = steersman.lock().await;
                        (replica_index..replica_index + replica_stubs.len())
                            .for_each(|rank| sman_lck.record_model_loaded(rank));
                        drop(sman_lck);
                        replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                        // 3. reverted
                    }));
                    m1 += 1;
                } else if m2 < n2 {
                    wrk_hndl_list.push(spawn(async move {
                        tracing::info!("Replica<{}> (rdma::chian) => LoadingDecode", replica_index);
                        *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                        // 2. mark Worker state
                        join_all(
                            replica_stubs
                                .iter_mut()
                                .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                        )
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                        join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                            .await
                            .into_iter()
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap();
                        tracing::info!("Replica<{}> (rdma::chain) => Decode", replica_index);
                        *replica_metric.state.write().await = ReplicaState::AusDecode;
                        replica_metric.set_model_loaded(true);
                        // 2. Worker marked
                        // 3. revert Steersman Planner state
                        let mut sman_lck = steersman.lock().await;
                        (replica_index..replica_index + replica_stubs.len())
                            .for_each(|rank| sman_lck.record_model_loaded(rank));
                        drop(sman_lck);
                        replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                        // 3. reverted
                    }));
                    m2 += 1;
                } else if m1 < n1 {
                    wrk_hndl_list.push(spawn(async move {
                        tracing::info!(
                            "Replica<{}> (rdma::chian) => LoadingPrefill",
                            replica_index
                        );
                        *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                        // 2. mark Worker state
                        join_all(
                            replica_stubs
                                .iter_mut()
                                .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                        )
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                        join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                            .await
                            .into_iter()
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap();
                        tracing::info!("Replica<{}> (rdma::chain) => Prefill", replica_index);
                        *replica_metric.state.write().await = ReplicaState::Prefill;
                        replica_metric.set_model_loaded(true);
                        // 2. Worker marked
                        // 3. revert Steersman Planner state
                        let mut sman_lck = steersman.lock().await;
                        (replica_index..replica_index + replica_stubs.len())
                            .for_each(|rank| sman_lck.record_model_loaded(rank));
                        drop(sman_lck);
                        replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                        // 3. reverted
                    }));
                    m1 += 1;
                }
            }
        }
        let handle = spawn(async move {
            join_all(wrk_hndl_list).await;
        });
        // 3. launch OldPrefill
        if cfg!(feature = "live") && m1 > 0 {
            let num_add_new_prefill = min(m1 as usize, num_chains);
            let num_add_old_prefill =
                min(num_add_new_prefill, normal_prefill_replica_indices.len());
            for _ in 0..num_add_old_prefill {
                let (_, old_replica_index) = normal_prefill_replica_indices.pop_front().unwrap();
                tracing::info!("Replica<{}> (zigzag) => OldPrefill", old_replica_index);
                *self.replica_metrics[&old_replica_index].state.write().await =
                    ReplicaState::OldPrefill;
            }
            // NOTE: in case not enough OldPrefill is scaled
            for _ in num_add_old_prefill..num_add_new_prefill {
                RELAY_DEACTIVE_CNT.fetch_sub(1, Ordering::SeqCst);
            }
        }
        ((m1, m2), handle)
    }

    /// MULTICAST scale up n decode instance 4 "impl_fast"
    ///
    /// src: Decode => SendingDecode => Decode
    /// dst: Inactive => RdmaCastingDecode | LoadingDecode => Decode
    async fn rdma_bcast_scale_n_decode(
        self: &Arc<Self>,
        _model_name: &str,
        n: i32,
        src_replica_indices: &mut Vec<usize>,
        dst_replica_indices: &mut Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> (i32, JoinHandle<()>) {
        assert!(cfg!(feature = "fast"));
        assert!(src_replica_indices.len() > 0);
        assert!(dst_replica_indices.len() > 0);
        let src_replica_index = src_replica_indices.pop().unwrap();
        let mut dst_replica_indices2 = Vec::<usize>::new();
        let mut dst_ranks_in_chain = Vec::<i32>::new();
        let mut m = 0;
        while m < n {
            match dst_replica_indices.pop() {
                Some(replica_index) => {
                    dst_ranks_in_chain.extend(self.replica_to_ranks[&replica_index].iter());
                    dst_replica_indices2.push(replica_index);
                    assert_eq!(
                        *self.replica_metrics[&replica_index].state.read().await,
                        ReplicaState::Inactive
                    );
                    m += 1;
                }
                None => {
                    break;
                }
            }
        }
        let src_ranks: Vec<i32> = self.replica_to_ranks[&src_replica_index].clone();
        let dst_ranks: Vec<i32> = dst_ranks_in_chain;
        let mut wrk_hndl_list: Vec<JoinHandle<()>> = Vec::new();
        {
            // src replica
            let replica_index = src_replica_index;
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(*replica_metric.state.read().await, ReplicaState::Decode);
            assert_eq!(
                replica_rdma_states.read().await[replica_index],
                ReplicaState::Inactive
            );
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
            let replica_rdma_state0 = replica_rdma_states.clone();
            // 1. mark Planner state
            tracing::info!("Replica<{}> (rdma::chain) => SendingDecode", replica_index);
            replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaSending;
            // 1. marked
            wrk_hndl_list.push(spawn(async move {
                // 2. skip Worker state
                join_all(
                    replica_stubs
                        .iter_mut()
                        .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
                join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                // 2. skipped
                // 3. revert Planner state
                tracing::info!("Replica<{}> (rdma::chain) => Decode", replica_index);
                replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                // 3. reverted
            }));
        }
        for replica_index in dst_replica_indices2.into_iter() {
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(*replica_metric.state.read().await, ReplicaState::Inactive);
            assert_eq!(
                replica_rdma_states.read().await[replica_index],
                ReplicaState::Inactive
            );
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
            let replica_rdma_state0 = replica_rdma_states.clone();
            let steersman = self.steersman.clone();
            // 1. mark Planner state
            tracing::info!(
                "Replica<{}> (rdma::chain::marker) => RdmaCastingDecode",
                replica_index
            );
            replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaCasting;
            // 1. marked
            wrk_hndl_list.push(spawn(async move {
                // 2. mark Worker state
                tracing::info!("Replica<{}> (rdma::chian) => LoadingDecode", replica_index);
                *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                join_all(
                    replica_stubs
                        .iter_mut()
                        .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
                join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                tracing::info!("Replica<{}> (rdma::chain) => Decode", replica_index);
                *replica_metric.state.write().await = ReplicaState::AusDecode;
                replica_metric.set_model_loaded(true);
                // 2. Worker marked
                // 3. revert Steersman Planner state
                let mut sman_lck = steersman.lock().await;
                (replica_index..replica_index + replica_stubs.len())
                    .for_each(|rank| sman_lck.record_model_loaded(rank));
                drop(sman_lck);
                replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                // 3. reverted
            }));
        }
        let tp_size = &self.disaggregation_controller_args.tensor_parallel_size;
        assert_eq!(m as usize, dst_ranks.len() / tp_size);
        let handle = spawn(async move {
            join_all(wrk_hndl_list).await;
        });
        (m, handle)
    }

    /// MULTICAST scale up n decode instance 4 "impl_fast"
    ///
    /// src: Decode => SendingDecode => Decode
    /// dst: Inactive => RdmaCastingPrefill => Prefill
    async fn rdma_bcast_scale_n_prefill(
        self: &Arc<Self>,
        _model_name: &str,
        n: i32,
        src_replica_indices: &mut Vec<usize>,
        dst_replica_indices: &mut Vec<usize>,
        replica_rdma_states: &Arc<RwLock<Vec<ReplicaState>>>,
    ) -> (i32, JoinHandle<()>) {
        assert!(cfg!(feature = "fast"));
        assert!(src_replica_indices.len() > 0);
        assert!(dst_replica_indices.len() > 0);
        let src_replica_index = src_replica_indices.pop().unwrap();
        let mut dst_replica_indices2: Vec<usize> = Vec::new();
        let mut dst_ranks_in_chain: Vec<i32> = Vec::new();
        let mut m = 0;
        while m < n {
            match dst_replica_indices.pop() {
                Some(replica_index) => {
                    dst_replica_indices2.push(replica_index);
                    dst_ranks_in_chain.extend(self.replica_to_ranks[&replica_index].iter());
                    let st = self.replica_metrics[&replica_index].state.read().await;
                    assert_eq!(
                        *st,
                        ReplicaState::Inactive,
                        "Replica<{}> expect to be inactive, but is {:?}",
                        replica_index,
                        st
                    );
                    m += 1;
                }
                None => {
                    break;
                }
            }
        }
        let src_ranks: Vec<i32> = self.replica_to_ranks[&src_replica_index].clone();
        let dst_ranks: Vec<i32> = dst_ranks_in_chain;
        let mut wrk_hndl_list: Vec<JoinHandle<()>> = Vec::new();
        {
            // src replica
            let replica_index = src_replica_index;
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(*replica_metric.state.read().await, ReplicaState::Decode);
            assert_eq!(
                replica_rdma_states.read().await[replica_index],
                ReplicaState::Inactive
            );
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
            let replica_rdma_state0 = replica_rdma_states.clone();
            // 1. mark Planner state
            tracing::info!("Replica<{}> (rdma::chain) => SendingDecode", replica_index);
            replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaSending;
            // 1. marked
            wrk_hndl_list.push(spawn(async move {
                // 2. skip Worker state
                join_all(
                    replica_stubs
                        .iter_mut()
                        .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
                join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                // 2. skipped
                // 3. revert Planner state
                tracing::info!("Replica<{}> (rdma::chain::marker) => Decode", replica_index);
                replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                // 3. reverted
            }));
        }
        for replica_index in dst_replica_indices2.into_iter() {
            let replica_metric = self.replica_metrics[&replica_index].clone();
            assert_eq!(*replica_metric.state.read().await, ReplicaState::Inactive);
            assert_eq!(
                replica_rdma_states.read().await[replica_index],
                ReplicaState::Inactive
            );
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            let mut replica_stubs = self.replica_to_stubs[&replica_index].clone();
            let replica_rdma_state0 = replica_rdma_states.clone();
            let steersman = self.steersman.clone();
            // 1. mark Planner state
            tracing::info!(
                "Replica<{}> (rdma::chain::marker) => RdmaCastingPrefill",
                replica_index
            );
            replica_rdma_states.write().await[replica_index] = ReplicaState::RdmaCasting;
            // 1. marked
            wrk_hndl_list.push(spawn(async move {
                // 2. mark Worker state
                tracing::info!("Replica<{}> (rdma::chain) => LoadingPrefill", replica_index);
                *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                join_all(
                    replica_stubs
                        .iter_mut()
                        .map(|stub| stub.rdma_broadcast(&src_ranks0, &dst_ranks0)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
                join_all(replica_stubs.iter_mut().map(|stub| stub.wait_rdma_done()))
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                tracing::info!("Replica<{}> (rdma::chain) => Prefill", replica_index);
                *replica_metric.state.write().await = ReplicaState::Prefill;
                replica_metric.set_model_loaded(true);
                // 2. Worker marked
                // 3. revert Steersman Planner state
                let mut sman_lck = steersman.lock().await;
                (replica_index..replica_index + replica_stubs.len())
                    .for_each(|rank| sman_lck.record_model_loaded(rank));
                drop(sman_lck);
                replica_rdma_state0.write().await[replica_index] = ReplicaState::Inactive;
                // 3. reverted
            }));
        }
        let tp_size = &self.disaggregation_controller_args.tensor_parallel_size;
        assert_eq!(m as usize, dst_ranks.len() / tp_size);
        let handle = spawn(async move {
            join_all(wrk_hndl_list).await;
        });
        (m, handle)
    }

    /// Inner-most worker function, only change worker state & mark
    ///
    /// post: updated_loading
    async fn walzer_tanzen_n(
        self: &Arc<Self>,
        crct_num_prefill_replica: &mut i32,
        crct_num_decode_replica: &mut i32,
        src_replica_indices: &Vec<usize>,
        dst_replica_indices: &Vec<usize>,
    ) -> Vec<JoinHandle<()>> {
        let mut handles: Vec<JoinHandle<()>> = Vec::new();
        let ncol = dst_replica_indices.len() / src_replica_indices.len();
        let src_ranks: Vec<i32> = src_replica_indices
            .iter()
            .map(|replica_index| self.replica_to_ranks[replica_index].clone())
            .flat_map(|ranks| ranks.into_iter())
            .collect();
        let dst_ranks: Vec<i32> = dst_replica_indices
            .iter()
            .map(|replica_index| self.replica_to_ranks[replica_index].clone())
            .flat_map(|ranks| ranks.into_iter())
            .collect();
        for replica_index in src_replica_indices.iter() {
            let mut stubs = self.replica_to_stubs[replica_index].clone();
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            handles.push(spawn(async move {
                join_all(
                    stubs
                        .iter_mut()
                        .map(|stub| stub.tanz_broadcast(&src_ranks0, &dst_ranks0)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            }));
        }
        for &replica_index in dst_replica_indices.iter() {
            let replica_metric = self.replica_metrics[&replica_index].clone();
            let mut stubs = self.replica_to_stubs[&replica_index].clone();
            let src_ranks0 = src_ranks.clone();
            let dst_ranks0 = dst_ranks.clone();
            if cfg!(feature = "live") && replica_index % ncol == 0 {
                ZIGZAG_ACTIVE_CNT.fetch_add(1, Ordering::AcqRel);
                tracing::info!("Replica<{}> (zigzag) => NewPrefill", replica_index);
                *self.replica_metrics[&replica_index].state.write().await =
                    ReplicaState::NewPrefill;
                *crct_num_prefill_replica -= 1;
                handles.push(spawn(async move {
                    join_all(
                        stubs
                            .iter_mut()
                            .map(|stub| stub.tanz_broadcast(&src_ranks0, &dst_ranks0)),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                    // FIXME: defer state transition for Zigzag
                    replica_metric.set_model_loaded(true);
                }));
            } else if *crct_num_decode_replica > 0 {
                tracing::info!("Replica<{}> (tanzen) => LoadingDecode", replica_index);
                *replica_metric.state.write().await = ReplicaState::LoadingDecode;
                *crct_num_decode_replica -= 1;
                handles.push(spawn(async move {
                    join_all(
                        stubs
                            .iter_mut()
                            .map(|stub| stub.tanz_broadcast(&src_ranks0, &dst_ranks0)),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                    // FIXME: defer state transition for Zigzag
                    replica_metric.set_model_loaded(true);
                    tracing::info!("Replica<{}> (tanzen) => Decode", replica_index);
                    *replica_metric.state.write().await = ReplicaState::AusDecode;
                }));
            } else {
                tracing::info!("Replica<{}> (tanzen) => LoadingPrefill", replica_index);
                *replica_metric.state.write().await = ReplicaState::LoadingPrefill;
                *crct_num_prefill_replica -= 1;
                handles.push(spawn(async move {
                    join_all(
                        stubs
                            .iter_mut()
                            .map(|stub| stub.tanz_broadcast(&src_ranks0, &dst_ranks0)),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                    // FIXME: defer state transition for Zigzag
                    replica_metric.set_model_loaded(true);
                    tracing::info!("Replica<{}> (tanzen) => Prefill", replica_index);
                    *replica_metric.state.write().await = ReplicaState::Prefill;
                }));
            }
        }
        handles
    }
}
