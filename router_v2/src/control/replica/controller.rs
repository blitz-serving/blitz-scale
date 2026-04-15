use core::panic;
use futures::{future::join_all, join};
use nohash_hasher::IntMap;
use pb::generate::v2::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicI64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    spawn,
    sync::{Mutex, OwnedMutexGuard, RwLock},
    task::yield_now,
    time::sleep,
};
use tracing::instrument;

use super::globals::*;
use super::helpers::*;
use super::metrics::*;
use super::migration::MigrationQueue;
use super::relay_queue::*;
use super::steersman::Steersman;
use super::stubs_filter_batch_with_metric;
use super::{config::*, Model};
use crate::{
    encode_prefill_case,
    error::ClientError,
    datapath::infer::{filter_send_generations, filter_send_generations_on_prefill_done},
    datapath::queue::{Entry, Queue},
    Stub,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ControllerArgs {
    pub tokens_prefilled_per_sec: u32,
    pub tokens_transferred_per_sec: u32,

    pub max_blocks_per_replica: u32,
    pub max_prefill_num: u32,
    pub max_decode_num: u32,
    pub min_prefill_num: u32,
    pub min_decode_num: u32,

    pub prefill_lower_bound: f32,
    pub prefill_upper_bound: f32,
    pub decode_lower_bound: f32,
    pub decode_upper_bound: f32,
    pub migration_lower_bound: f32,
    pub migration_upper_bound: f32,

    pub scale_down_threshold_millis: u64,

    pub num_hidden_layers: u32,
    pub num_gpus_per_node: usize,

    pub mock_load_millis: u64,
    pub mock_transfer_millis: u64,
    pub tensor_parallel_size: usize,
}

#[derive(Clone)]
pub struct DisaggregationController {
    /// Only used when `manually_scale` is enabled.
    pub(crate) command_channel_txs: Vec<async_channel::Sender<ReplicaCommand>>,

    pub(crate) disaggregation_controller_args: ControllerArgs,
    pub(crate) all_stubs: Vec<Stub>,
    pub(crate) replica_metrics: BTreeMap<usize, Arc<ReplicaMetric>>,
    pub(crate) batching_queue: Queue,
    pub(crate) steersman: Arc<Mutex<Steersman>>,
    pub(crate) replica_to_stubs: HashMap<usize, Vec<Stub>>,
    pub(crate) replica_to_ranks: HashMap<usize, Vec<i32>>,
}

impl DisaggregationController {
    #[instrument(skip_all)]
    pub(crate) async fn report_system_metrics(self: Arc<Self>, interval_millis: u64) {
        loop {
            let prefill_tokens = SYSTEM_METRIC.prefill_tokens.swap(0, Ordering::SeqCst);
            let decode_tokens = SYSTEM_METRIC.decode_tokens.swap(0, Ordering::SeqCst);
            tracing::info!(
                "System Metrics: prefill_tokens: {}, decode_tokens: {}",
                prefill_tokens,
                decode_tokens,
            );

            // report replica metrics
            let tp_size = self.disaggregation_controller_args.tensor_parallel_size;
            assert_eq!(self.all_stubs.len(), self.replica_metrics.len() * tp_size);
            for (&index, replica_metric) in self.replica_metrics.iter() {
                let state = replica_metric.state.read().await.clone();
                let used_blocks = replica_metric.get_used_blocks();
                let model_loaded = replica_metric.is_model_loaded();
                let loop_cnt = SYSTEM_METRIC.loop_counts[index].swap(0, Ordering::AcqRel);
                tracing::info!(
                    "Replica<{}>::(state) {:?}, used_blocks: {}, model_loaded: {}, loop_cnt: {}",
                    index,
                    state,
                    used_blocks,
                    model_loaded,
                    loop_cnt,
                );
                assert_eq!(self.all_stubs.len(), self.replica_metrics.len() * tp_size);
            }
            tokio::time::sleep(Duration::from_millis(interval_millis)).await;
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn report_cur_waiting_prefill_tokens(self: Arc<Self>, interval_ms: u64) {
        tokio::time::sleep(Duration::from_millis(interval_ms / 2)).await;
        loop {
            let cur_waiting_prefill_token = self.batching_queue.waiting_prefill_tokens().await;
            if cur_waiting_prefill_token != 0 {
                // tracing::info!("Cur waiting prefill tokens: {}", cur_waiting_prefill_token);
            }
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn report_network_flow(self: Arc<Self>, interval_ms: u64) {
        tokio::time::sleep(Duration::from_millis(interval_ms / 2)).await;

        loop {
            let (flow_out, flow_in) = {
                let mut g = FLOW_WATCHER.lock().await;
                let flows = g.get_all();
                g.clear();
                flows
            };
            let mut replica_flow_map: HashMap<usize, (usize, usize)> = HashMap::new();
            // 0-7 insert (0,0)
            for i in 0..8 {
                replica_flow_map.insert(i, (0, 0));
            }
            for (replica_index, flow) in flow_in {
                if let Some((in_flow, _)) = replica_flow_map.get_mut(&replica_index) {
                    *in_flow = flow;
                } else {
                    replica_flow_map.insert(replica_index, (flow, 0));
                }
            }
            for (replica_index, flow) in flow_out {
                if let Some((_, out_flow)) = replica_flow_map.get_mut(&replica_index) {
                    *out_flow = flow;
                } else {
                    replica_flow_map.insert(replica_index, (0, flow));
                }
            }
            // for 0 - 7
            tracing::info!(
                "Waiting migration token num: {}",
                SYSTEM_METRIC.token_in_queue.load(Ordering::Acquire)
            );
            if cfg!(not(feature = "impl_sllm")) {
                let ctrl_loop_cnt = CTRL_LOOP_CNT.load(Ordering::Acquire);
                assert!(
                    PLANNER_BUSY.load(Ordering::Acquire) || ctrl_loop_cnt > 0,
                    "Control loop stuck!"
                );
                CTRL_LOOP_CNT.store(0, Ordering::Release);
            }

            for (replica_index, (flow_in, flow_out)) in replica_flow_map {
                if (flow_in + flow_out) > 0 {
                    tracing::info!(
                        "Replica[{}]: Flow in: {}MB, Flow out: {}MB",
                        replica_index,
                        flow_in,
                        flow_out
                    );
                }
            }
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn auto_scale_event_loop(self: Arc<Self>) {
        if cfg!(feature = "manually_scale") {
            panic!("Manually scale is enabled. Auto scale is disabled");
        } else {
            self.replica_state_moniter_loop().await;
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn reject_kv_cache_event_loop(self: Arc<Self>) {
        let mut locked_decode_map = HashMap::new();
        loop {
            let mut decode_replicas = Vec::new();
            for (replica_index, replica_metric) in &self.replica_metrics {
                match *replica_metric.state.read().await {
                    ReplicaState::Decode | ReplicaState::RdmaSending => {
                        decode_replicas.push((replica_metric.get_used_blocks(), *replica_index));
                    }
                    _ => {}
                }
            }

            let num_decode_replica = decode_replicas.len() as u32;
            let total_used_decode_blocks: u32 = decode_replicas
                .iter()
                .map(|(used_blocks, _)| *used_blocks)
                .sum();

            let mut should_lock = HashSet::new();
            for (used_blocks, decode_replica_index) in decode_replicas {
                if used_blocks * num_decode_replica > total_used_decode_blocks * 2
                    && used_blocks * 2 > self.disaggregation_controller_args.max_blocks_per_replica
                {
                    should_lock.insert(decode_replica_index);
                }
            }

            locked_decode_map
                .keys()
                .cloned()
                .collect::<HashSet<_>>()
                .difference(&should_lock)
                .for_each(|decode_replica_index| {
                    assert!(locked_decode_map.remove(decode_replica_index).is_some());
                    tracing::info!("Decode replica [{}] released", decode_replica_index);
                });

            for decode_replica_index in
                should_lock.difference(&locked_decode_map.keys().cloned().collect::<HashSet<_>>())
            {
                if locked_decode_map.is_empty() {
                    locked_decode_map.insert(
                        *decode_replica_index,
                        self.replica_metrics[decode_replica_index]
                            .dst_mutex
                            .clone()
                            .lock_owned()
                            .await,
                    );
                    tracing::info!("Decode replica [{}] locked", decode_replica_index);
                } else {
                    if let Ok(guard) = self.replica_metrics[decode_replica_index]
                        .dst_mutex
                        .clone()
                        .try_lock_owned()
                    {
                        panic!("Unintended task!");
                        locked_decode_map.insert(*decode_replica_index, guard);
                        tracing::info!("Decode replica [{}] locked", decode_replica_index);
                    }
                }
            }

            yield_now().await;
        }
    }

    #[instrument(skip_all)]
    #[cfg(feature = "manually_scale")]
    pub(crate) async fn trigger_scale_up(
        self: &Arc<Self>,
        src_replica: Vec<usize>,
        dst_replica: Vec<usize>,
        scale_to: String,
    ) {
        if cfg!(not(feature = "manually_scale")) {
            panic!("Manually scale is disabled");
        }

        if cfg!(feature = "sched_naive") {
            tracing::info!("sched_naive trigger scale up");
            let controller = self.clone();
            let dst_replica_index = *dst_replica.iter().min().unwrap();
            // spawn(async move { controller.transfer_params(src_replica, dst_replica).await });

            controller.transfer_params(src_replica, dst_replica).await;
            tracing::info!("sched_naive params transfer done");
            if scale_to.contains("Prefill") {
                self.command_channel_txs[dst_replica_index]
                    .send(ReplicaCommand::TransToPrefill)
                    .await
                    .unwrap();
            } else {
                self.command_channel_txs[dst_replica_index]
                    .send(ReplicaCommand::TransToDecode)
                    .await
                    .unwrap();
            }
        } else {
            let controller = self.clone();
            let src_replica_index = *src_replica.iter().min().unwrap();
            let dst_replica_index = *dst_replica.iter().min().unwrap();
            spawn({
                let dst_replica = dst_replica.clone();
                let src_replica = src_replica.clone();
                async move {
                    controller.transfer_params(src_replica, dst_replica).await;
                }
            });

            let res = join!(
                self.command_channel_txs[src_replica_index].send(ReplicaCommand::TransToOldPrefill),
                self.command_channel_txs[dst_replica_index].send(ReplicaCommand::TransToNewPrefill)
            );
            assert!(matches!(res, (Ok(()), Ok(()))));
        }
    }

    #[instrument(skip_all)]
    pub(crate) async fn trigger_scale_down(&self, replica_stub_indices: Vec<usize>) {
        if cfg!(not(feature = "manually_scale")) {
            panic!("Manually scale is disabled");
        }
        self.command_channel_txs[*replica_stub_indices.iter().min().unwrap()]
            .send(ReplicaCommand::Deactivate)
            .await
            .unwrap();
    }

    #[instrument(skip_all)]
    pub(crate) async fn trigger_mutate_to_decode(&self, replica_stub_indices: Vec<usize>) {
        if cfg!(not(feature = "manually_scale")) {
            panic!("Manually scale is disabled");
        }
        self.command_channel_txs[*replica_stub_indices.iter().min().unwrap()]
            .send(ReplicaCommand::TransToDecode)
            .await
            .unwrap();
    }

    /// Simulate the process of loading params from SSD / Cache / NVLink
    #[instrument(skip_all)]
    #[cfg(feature = "mock_transfer")]
    pub(crate) async fn mock_load_params(self: &Arc<Self>, dst_replica: Vec<usize>) {
        let mock_load_millis = self.disaggregation_controller_args.mock_load_millis;
        assert_ne!(mock_load_millis, 0);
        let mut dst_stubs = dst_replica
            .iter()
            .map(|&index| self.all_stubs[index].clone())
            .collect::<Vec<_>>();
        let dst_replica_metric = self.replica_metrics[dst_replica.iter().min().unwrap()].clone();
        tracing::info!(
            "mock_load_params on {:?} for {}ms",
            dst_replica,
            mock_load_millis
        );
        sleep(Duration::from_millis(mock_load_millis)).await;
        join_all(dst_stubs.iter_mut().map(Stub::set_status_ready)).await;
        if cfg!(feature = "sched_naive") && !cfg!(feature = "optimize_op_loading") {
            sleep(Duration::from_millis(100)).await;
        }
        dst_replica_metric.set_model_loaded(true);
    }

    #[instrument(skip_all)]
    pub(crate) async fn load_param_w_case_inner(
        self: &Arc<Self>,
        dst_ranks: Vec<i32>,
        case: generate::v2::LoadParamCase,
        model_name: &str,
    ) {
        let mut dst_stubs = dst_ranks
            .iter()
            .map(|rank| self.all_stubs[*rank as usize].clone())
            .collect::<Vec<_>>();
        let s_time = Instant::now();
        let model_path = self.steersman.lock().await.get_model_path(model_name);
        join_all(dst_stubs.iter_mut().map(|stub| {
            stub.load_params(case, String::from(model_name), Some(model_path.clone()))
        }))
        .await;
        let e_time = Instant::now();
        tracing::info!(
            "load_params case {:?} on {:?} takes {:?}ms",
            case,
            dst_ranks,
            (e_time - s_time).as_millis()
        );
    }

    #[instrument(skip_all)]
    pub(crate) async fn mock_transfer_params(self: &Arc<Self>, dst_ranks: Vec<i32>) {
        let mock_load_millis = self.disaggregation_controller_args.mock_load_millis;
        assert_ne!(mock_load_millis, 0);
        let mut dst_stubs = dst_ranks
            .iter()
            .map(|&index| self.all_stubs[index as usize].clone())
            .collect::<Vec<_>>();
        tracing::info!(
            "mock_load_params on {:?} for {}ms",
            dst_ranks,
            mock_load_millis
        );
        sleep(Duration::from_millis(mock_load_millis)).await;
        join_all(dst_stubs.iter_mut().map(Stub::set_status_ready)).await;
    }

    /// Rdma P2P transfer parameter
    ///
    /// no "scale" in fn name -> only possible (un)mark worker state
    /// "inner" in fn name -> do not mark any state
    pub(crate) async fn rdma_p2p_n_inner(
        self: &Arc<Self>,
        src_ranks: Vec<i32>,
        dst_ranks: Vec<i32>,
    ) {
        tracing::info!("Rdma p2p: {:?} ~+> {:?}", src_ranks, dst_ranks);
        let mut pair_stubs = src_ranks
            .into_iter()
            .zip(dst_ranks.into_iter())
            .map(|(src_rank, dst_rank)| {
                (
                    (src_rank, self.all_stubs[src_rank as usize].clone()),
                    (dst_rank, self.all_stubs[dst_rank as usize].clone()),
                )
            })
            .collect::<Vec<_>>();

        join_all(pair_stubs.iter_mut().map(
            |((src_rank, src_stub), (dst_rank, dst_stub))| async move {
                let send_recv_res = join!(
                    src_stub.send_params(*dst_rank),
                    dst_stub.recv_params(*src_rank)
                );
                assert!(matches!(send_recv_res, (Ok(_), Ok(_))));
                let wait_res = join!(src_stub.wait_rdma_done(), dst_stub.wait_rdma_done());
                assert!(matches!(wait_res, (Ok(_), Ok(_))));
            },
        ))
        .await;
    }
}

pub fn start_disaggregation_event_loop(
    queue: Queue,
    all_stubs: Vec<Stub>,
    config: DisaggregationConfig,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    model: Model,
    disaggregation_controller_args: ControllerArgs,
) -> Arc<DisaggregationController> {
    let DisaggregationConfig {
        init_states,
        replicas,
        machines,
    } = config;

    let (command_channel_txs, command_channel_rxs): (Vec<_>, Vec<_>) = (0..all_stubs.len())
        .map(|_| async_channel::unbounded::<ReplicaCommand>())
        .unzip();

    let mut replica_metrics = BTreeMap::default();

    let all_ranks = replicas
        .iter()
        .flat_map(|ranks| ranks.iter())
        .collect::<Vec<_>>();
    assert_eq!(init_states.len(), all_ranks.len());
    assert_eq!(all_ranks.len(), machines.len());

    // Create mutexes with initial states. The mutexes are used to synchronize the P-to-D kv cache migration
    let (init_guards, dst_mutexes): (Vec<_>, Vec<_>) = replicas
        .iter()
        .map(|ranks| {
            let replica_index = ranks[0];
            let init_state = &init_states[replica_index];
            assert!(ranks.iter().all(|rank| &init_states[*rank] == init_state));
            let mutex = Arc::new(Mutex::new(()));
            if !matches!(init_state, ReplicaState::Decode) {
                // Non-decode replica should not recv blocks from decode replica.
                (Some(mutex.clone().try_lock_owned().unwrap()), mutex)
            } else {
                // Decode replica should receive blocks from Prefill replica.
                // We leave the lock released
                (None, mutex)
            }
        })
        .unzip();

    let mut steersman = Steersman::new(HOST_CACHE_TIME_TO_LIVE_IN_MS);
    steersman.init_model_config(model);

    for ranks in replicas.iter() {
        let replica_index = *ranks.iter().min().unwrap();
        let machine = machines[replica_index];
        tracing::info!(
            "Replica<{}>::<{:?}> is on machine {}",
            replica_index,
            ranks,
            machine
        );
        ranks
            .iter()
            .for_each(|rank| steersman.register_replica(*rank, machine));
    }
    assert_eq!(replicas.len(), dst_mutexes.len());
    let metrics = replicas
        .iter()
        .zip(dst_mutexes.clone().into_iter())
        .map(|(ranks, dst_mutex)| {
            let replica_index = *ranks.iter().min().unwrap();
            let init_state = init_states[replica_index].clone();
            match init_state {
                ReplicaState::Inactive => {}
                _ => {
                    ranks
                        .iter()
                        .for_each(|rank| steersman.record_model_loaded(*rank));
                }
            };
            Arc::new(ReplicaMetric::new(
                !matches!(init_state, ReplicaState::Inactive),
                replica_index,
                RwLock::const_new(init_state),
                dst_mutex,
            ))
        })
        .collect::<Vec<_>>();

    let migration_queue = MigrationQueue::spawn_migration_queue(
        all_stubs.clone(),
        replicas.clone(),
        metrics.clone(),
        dst_mutexes.clone(),
    );

    let steersman = Arc::new(Mutex::new(steersman));
    let mut replica_stubs_map: HashMap<usize, Vec<Stub>> = HashMap::new();
    let mut replica_ranks_map: HashMap<usize, Vec<i32>> = HashMap::new();

    for (((ranks, metric), dst_mutex), init_guard) in replicas
        .into_iter()
        .zip(metrics.into_iter())
        .zip(dst_mutexes.into_iter())
        .zip(init_guards.into_iter())
    {
        let replica_index = *ranks.iter().min().unwrap();
        let command_channel_rx = command_channel_rxs[replica_index].clone();
        replica_stubs_map.insert(
            replica_index,
            ranks.iter().map(|&rank| all_stubs[rank].clone()).collect(),
        );
        replica_ranks_map.insert(
            replica_index,
            ranks.iter().map(|rank| *rank as i32).collect(),
        );

        // FIXME: add warmups
        let mut replica_stubs = replica_stubs_map[&replica_index].clone();
        spawn(async move {
            join_all(
                replica_stubs
                    .iter_mut()
                    .map(|stub| stub.warmup(2048, 4096, 4096)),
            )
            .await
            .into_iter()
            .collect::<Result<VecDeque<_>, ClientError>>()
            .unwrap()
            .pop_front()
            .unwrap()
        });

        spawn(super::state_machine::disaggregation_event_loop_inner(
            ranks.clone(),
            metric.clone(),
            dst_mutex,
            init_guard,
            all_stubs.clone(),
            command_channel_rx,
            queue.clone(),
            migration_queue.clone(),
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            disaggregation_controller_args.num_hidden_layers,
            disaggregation_controller_args.tensor_parallel_size,
            steersman.clone(),
        ));

        replica_metrics.insert(replica_index, metric);
    }

    D_MTX_NOTIFY.get_or_init(|| {
        (0..all_stubs.len())
            .map(|_| AtomicBool::new(false))
            .collect::<Vec<_>>()
    });
    let controller = DisaggregationController {
        command_channel_txs,
        disaggregation_controller_args,
        all_stubs,
        replica_metrics,
        batching_queue: queue,
        steersman: steersman.clone(),
        replica_to_stubs: replica_stubs_map,
        replica_to_ranks: replica_ranks_map,
    };
    let controller = Arc::new(controller);
    spawn(controller.clone().report_system_metrics(1000));
    spawn(controller.clone().report_network_flow(1000));
    spawn(controller.clone().report_cur_waiting_prefill_tokens(100));
    if cfg!(feature = "decode_load_balance") && !cfg!(feature = "manually_scale") {
        todo!();
        spawn(controller.clone().reject_kv_cache_event_loop());
    }
    if cfg!(feature = "manually_scale") {
        return controller;
    } else {
        spawn(controller.clone().auto_scale_event_loop());
        return controller;
    }
}
