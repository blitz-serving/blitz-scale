use core::panic;
use futures::{future::join_all, join, FutureExt};
use nohash_hasher::IntMap;
use pb::generate::{self, v2::*};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    mem::take,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering},
        Arc, LazyLock, OnceLock,
    },
    time::{Duration, Instant},
};
use tokio::{
    spawn,
    sync::{oneshot, Mutex, OwnedMutexGuard, RwLock},
    task::{yield_now, JoinHandle},
    time::sleep,
};
use tracing::instrument;

use super::metrics::*;
use super::relay_queue::*;
use super::steersman::Steersman;
use super::stubs_filter_batch_with_metric;
use super::{config::*, Model};
use crate::{
    encode_prefill_case,
    error::ClientError,
    infer::{filter_send_generations, filter_send_generations_on_prefill_done},
    queue::{Entry, Queue},
    Stub,
};

pub static MAX_BLOCKS_PER_REPLICA: OnceLock<u32> = OnceLock::new();
static INIT_ZIGZAG_REQ_LIMIT: isize = 4;
// decrease request limit by 1 per 100ms
static DEC_ZIGZAG_REQ_LIMIT_RT: u128 = 200;

const HOST_CACHE_TIME_TO_LIVE_IN_MS: u128 = 30000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicaCommand {
    /// Inactive to NewPrefill
    TransToNewPrefill,
    /// Prefill to OldPrefill
    TransToOldPrefill,
    /// Prefill to Decode
    TransToDecode,
    /// Prefill/Decode to Inactive
    Deactivate,
    /// Inactive to Prefill. Only legal when `sched_naive` is enabled.
    TransToPrefill,
}

#[derive(Debug)]
pub(crate) struct MigratingBatch {
    pub(crate) batch: Batch,
    pub(crate) tokens: Tokens,
    pub(crate) entries: IntMap<u64, Entry>,
}

pub(crate) struct MigratingPartialBatch {
    pub(crate) batch: Batch,
    pub(crate) tokens: Option<Tokens>,
    pub(crate) entries: Option<IntMap<u64, Entry>>,
    pub(crate) layer: u32,
    pub(crate) fst2snd_tx: Option<oneshot::Sender<(usize, JoinHandle<OwnedMutexGuard<()>>)>>,
}

#[derive(Debug)]
pub(crate) struct MigratedBatch {
    pub(crate) batch: CachedBatch,
    pub(crate) tokens: Tokens,
    entries: IntMap<u64, Entry>,
}

impl Into<MigratedBatch> for MigratingBatch {
    fn into(self) -> MigratedBatch {
        MigratedBatch {
            batch: batch_to_cached_batch(self.batch),
            tokens: self.tokens,
            entries: self.entries,
        }
    }
}

fn batch_to_cached_batch(batch: Batch) -> CachedBatch {
    let Batch {
        id,
        requests,
        size,
        max_tokens,
    } = batch;
    let request_ids = requests.iter().map(|r| r.id).collect::<Vec<_>>();
    // let all_tokens = requests
    //     .iter()
    //     .map(|r| CacheTokens {
    //         ids: r.input_tokens.clone(),
    //     })
    //     .collect::<Vec<_>>();
    CachedBatch {
        id,
        request_ids,
        size,
        max_tokens,
        // all_tokens,
    }
}

pub(crate) fn restore_cached_batch(
    filtered_cached_batch: CachedBatch,
    origin_batch: Batch,
) -> Batch {
    let Batch { requests, .. } = origin_batch;
    let CachedBatch {
        id,
        request_ids,
        size,
        max_tokens,
        // all_tokens: _,
    } = filtered_cached_batch;
    let request_ids = request_ids.into_iter().collect::<HashSet<_>>();
    let requests = requests
        .into_iter()
        .filter(|request| request_ids.contains(&request.id))
        .collect::<Vec<_>>();
    Batch {
        id,
        requests,
        size,
        max_tokens,
    }
}

#[derive(Clone)]
pub(crate) struct DisaggregationController {
    /// Only used when `manually_scale` is enabled.
    command_channel_txs: Vec<async_channel::Sender<ReplicaCommand>>,

    pub(crate) disaggregation_controller_args: ControllerArgs,
    pub(crate) all_stubs: Vec<Stub>,
    pub(crate) replica_metrics: BTreeMap<usize, Arc<ReplicaMetric>>,
    pub(crate) batching_queue: Queue,
    pub(crate) steersman: Arc<Mutex<Steersman>>,
    pub(crate) replica_to_stubs: HashMap<usize, Vec<Stub>>,
    pub(crate) replica_to_ranks: HashMap<usize, Vec<i32>>,
}

pub(crate) static KV_BLOCK_SIZE: u32 = 64;
pub(crate) static SYSTEM_METRIC: LazyLock<Arc<SystemMetric>> =
    LazyLock::new(|| Arc::new(SystemMetric::new()));

// TODO: should as disaggregation_event_loop param in the future
static RELAY_ASYNC_QUEUE: LazyLock<Arc<AsyncQueue>> = LazyLock::new(|| Arc::new(AsyncQueue::new()));
pub(crate) static FLOW_WATCHER: LazyLock<Arc<Mutex<Flow>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Flow::new())));
pub(crate) static PLANNER_BUSY: AtomicBool = AtomicBool::new(false);
pub(crate) static ZIGZAG_ACTIVE_CNT: AtomicU64 = AtomicU64::new(0);
pub(crate) static RELAY_DEACTIVE_CNT: AtomicI64 = AtomicI64::new(0);

/// someone can't get decode mutex -> disable expriring decode
pub(crate) static SCARCE_DECODE: AtomicBool = AtomicBool::new(false);
/// \sync patch up to pass decode mutex guard
pub(crate) static D_MTX_NOTIFY: OnceLock<Vec<AtomicBool>> = OnceLock::new();
/// \debug alert to locate Cybernetics improgressiveness
pub(crate) static CTRL_LOOP_CNT: AtomicU64 = AtomicU64::new(0);

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

pub(crate) fn start_disaggregation_event_loop(
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

        spawn(disaggregation_event_loop_inner(
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

async fn disaggregation_event_loop_inner(
    replica_stub_indices: Vec<usize>,
    replica_metric: Arc<ReplicaMetric>,
    dst_mutex: Arc<Mutex<()>>,
    init_guard: Option<OwnedMutexGuard<()>>,
    all_stubs: Vec<Stub>,
    command_channel_rx: async_channel::Receiver<ReplicaCommand>,
    batching_queue: Queue,
    migration_queue: MigrationQueue,
    _waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    _max_waiting_tokens: usize,
    num_hidden_layers: u32,
    tensor_parallel_size: usize,
    steersman: Arc<Mutex<Steersman>>,
) {
    // ident of this replica
    let replica_index = *replica_stub_indices.iter().min().unwrap();
    let mut replica_stubs = replica_stub_indices
        .iter()
        .map(|&rank| all_stubs[rank].clone())
        .collect::<Vec<_>>();
    // netwwork guard for decoding replica
    let mut guard = init_guard;
    // decoding replica -> prefilling replica
    let mut pending_lock_acquirer = Option::<JoinHandle<OwnedMutexGuard<()>>>::None;

    let mut global_entries = IntMap::default();
    let mut global_batches = Vec::new();

    // This is used as a barrier to prevent the replica from being shut down before all migrations are finished.
    // Migrations include both P-to-P migration and P-to-D migration.
    let unfinished_migration_count = Arc::new(AtomicI64::new(0));

    let mut zigzag_req_limit = INIT_ZIGZAG_REQ_LIMIT;
    let mut zigzag_timestamp: Option<Instant> = None;
    // sequence number to mark zigzag#
    let mut next_zigzag_seq_num: u32 = 0;
    // for refractory concurrent
    let head_zag_seq_num = Arc::new(AtomicU32::new(0));
    let pending_zag_prefills = Arc::new(Mutex::new(VecDeque::<
        JoinHandle<(PrefillV2Response, Batch, IntMap<u64, Entry>, u32)>,
    >::new()));
    let mut refractory_head_task: Option<JoinHandle<()>> = None;

    'NextState: loop {
        let tmp_ro_state = if cfg!(feature = "manually_scale") {
            if let Ok(command) = command_channel_rx.try_recv() {
                let mut guard = replica_metric.state.write().await;
                let transferred_state = match (&*guard, command) {
                    (ReplicaState::Inactive, ReplicaCommand::TransToNewPrefill) => {
                        ReplicaState::NewPrefill
                    }
                    (ReplicaState::Prefill, ReplicaCommand::TransToOldPrefill) => {
                        ReplicaState::OldPrefill
                    }
                    (ReplicaState::Inactive, ReplicaCommand::TransToPrefill) => {
                        ReplicaState::Prefill
                    }
                    (ReplicaState::Inactive, ReplicaCommand::TransToDecode) => {
                        ReplicaState::MutatingToDecode
                    }
                    (ReplicaState::Prefill, ReplicaCommand::Deactivate) => {
                        ReplicaState::ShuttingPrefill
                    }
                    (ReplicaState::Decode, ReplicaCommand::Deactivate) => {
                        ReplicaState::ShuttingDecode
                    }
                    (ReplicaState::Prefill, ReplicaCommand::TransToDecode) => {
                        ReplicaState::MutatingToDecode
                    }
                    (state, command) => panic!(
                        "Rank<{}> Invalid state transfer {:?} {:?}",
                        replica_index, state, command
                    ),
                };
                *guard = transferred_state.clone();
                transferred_state
            } else {
                replica_metric.state.read().await.clone()
            }
        } else {
            replica_metric.state.read().await.clone()
        };
        SYSTEM_METRIC.loop_counts[replica_index].fetch_add(1, Ordering::AcqRel);

        match tmp_ro_state {
            ReplicaState::Inactive | ReplicaState::LoadingPrefill | ReplicaState::LoadingDecode => {
                assert!(guard.is_some());
                yield_now().await;
            }
            ReplicaState::ShuttingNull => {
                // has guard, to mutating to decode
                let need_guard = D_MTX_NOTIFY
                    .get()
                    .unwrap()
                    .get(replica_index)
                    .unwrap()
                    .swap(false, Ordering::AcqRel);
                if need_guard && guard.is_none() {
                    guard = Some(dst_mutex.clone().lock_owned().await);
                }
                // post: guard.is_some() -> invariant
                yield_now().await;
            }
            ReplicaState::Decode => {
                assert!(
                    guard.is_none(),
                    "Rank<{}> dst mutex assert failed",
                    replica_index
                );
                while let Some(migrated_batch) = migration_queue.try_consume(replica_index) {
                    let all_token_num: u32 = migrated_batch
                        .entries
                        .iter()
                        .map(|e| e.1.request.input_length)
                        .sum();
                    let MigratedBatch {
                        batch,
                        tokens,
                        entries,
                    } = migrated_batch;
                    let mut flow_watcher = FLOW_WATCHER.lock().await;
                    flow_watcher.recv_token(replica_index, all_token_num as usize);
                    global_entries.extend(entries);
                    global_batches.push((batch, tokens));
                }

                if !global_batches.is_empty() {
                    // Prefill finished and do decode immediately without receiving from the channel.
                    let (batches, last_iter_tokens): (Vec<_>, Vec<_>) =
                        take(&mut global_batches).into_iter().unzip();
                    let batch_size = batches
                        .iter()
                        .map(|b: &CachedBatch| b.size as usize)
                        .sum::<usize>();
                    SYSTEM_METRIC
                        .decode_tokens
                        .fetch_add(batch_size, Ordering::AcqRel);
                    // Prefill finished and do decode immediately without receiving from the channel.
                    let request = DecodeV2Request {
                        batches,
                        last_iter_tokens,
                    };
                    let response = join_all(
                        replica_stubs
                            .iter_mut()
                            .map(|stub| stub.decode_v2(request.clone())),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<VecDeque<_>, ClientError>>()
                    .unwrap()
                    .pop_front()
                    .unwrap();
                    let DecodeV2Response {
                        generations, batch, ..
                    } = response;
                    filter_send_generations(&generations, &mut global_entries);
                    if let Some((batch, _)) = stubs_filter_batch_with_metric(
                        &mut replica_stubs,
                        replica_metric.clone(),
                        batch,
                        generations,
                        &mut global_entries,
                    )
                    .await
                    {
                        global_batches.push((batch, Tokens::default()));
                    }
                } else {
                    yield_now().await;
                }
            }
            ReplicaState::Prefill => {
                assert!(guard.is_some());
                if *MAX_BLOCKS_PER_REPLICA.get().unwrap() < replica_metric.get_used_blocks() {
                    yield_now().await;
                    continue;
                }

                if let Some((mut new_entries, new_batch, _)) = batching_queue
                    .next_batch(
                        None,
                        max_batch_prefill_tokens,
                        max_batch_total_tokens,
                        Some(
                            *MAX_BLOCKS_PER_REPLICA.get().unwrap()
                                - replica_metric.get_used_blocks(),
                        ),
                    )
                    .await
                {
                    replica_metric
                        .add_used_blocks(new_batch.max_tokens / replica_metric.block_size);
                    let mut token_sum: u32 = 0;
                    for elem in new_entries.iter() {
                        token_sum += elem.1.request.input_length;
                    }
                    SYSTEM_METRIC
                        .prefill_tokens
                        .fetch_add(token_sum.try_into().unwrap(), Ordering::AcqRel);
                    let request = PrefillV2Request {
                        batch: new_batch.clone(),
                        forward_case: encode_prefill_case(vec![PrefillCase::Normal]),
                        ..Default::default()
                    };
                    let PrefillV2Response {
                        generations,
                        batch: cached_batch,
                        ..
                    } = join_all(
                        replica_stubs
                            .iter_mut()
                            .map(|stub| stub.prefill_v2(request.clone())),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<VecDeque<_>, ClientError>>()
                    .unwrap()
                    .pop_front()
                    .unwrap();

                    filter_send_generations_on_prefill_done(&generations, &mut new_entries);
                    filter_send_generations(&generations, &mut new_entries);
                    if let Some((filtered_cached_batch, tokens)) = stubs_filter_batch_with_metric(
                        &mut replica_stubs,
                        replica_metric.clone(),
                        cached_batch,
                        generations,
                        &new_entries,
                    )
                    .await
                    {
                        let migrating_batch = MigratingBatch {
                            batch: restore_cached_batch(filtered_cached_batch, new_batch),
                            tokens,
                            entries: new_entries,
                        };
                        let mut flow_watcher = FLOW_WATCHER.lock().await;
                        let all_token_num: u32 = migrating_batch
                            .entries
                            .iter()
                            .map(|e| e.1.request.input_length)
                            .sum();
                        flow_watcher.append_token(replica_index, all_token_num as usize);
                        migration_queue
                            .append(
                                migrating_batch,
                                replica_index,
                                unfinished_migration_count.clone(),
                            )
                            .await;
                    }
                }
            }
            ReplicaState::NewPrefill => {
                if cfg!(feature = "impl_blitz") {
                    assert!(guard.is_some());
                    if zigzag_timestamp.is_none() {
                        zigzag_timestamp = Some(Instant::now());
                    }

                    // Set the model_ready flag so that NewPrefill stop to acquire
                    // new requests from batching queue, and => Refractory mode
                    let mut model_loaded = replica_metric.is_model_loaded();
                    tracing::debug!("Replica<{}> Model loaded? {}", replica_index, model_loaded);

                    if model_loaded {
                        // NOTE: wait another request to make sure server is complied w/ refractory mode
                        if pending_zag_prefills.lock().await.len() > 0 {
                            let prefill_v2_response: PrefillV2Response;
                            let new_batch: Batch;
                            let new_entries: IntMap<u64, Entry>;

                            (prefill_v2_response, new_batch, new_entries, ..) =
                                pending_zag_prefills
                                    .lock()
                                    .await
                                    .pop_front()
                                    .unwrap()
                                    .await
                                    .unwrap();
                            head_zag_seq_num.fetch_add(1, Ordering::AcqRel);

                            tracing::info!(
                                "Replica<{}> (New => Refrac) get Batch[{}]; pending zigzag prefill queue len is {}",
                                replica_index,
                                new_batch.id,
                                pending_zag_prefills.lock().await.len()
                            );
                            assert!(prefill_v2_response
                                .pp_info
                                .as_ref()
                                .expect("PipeParaInfo must be set")
                                .start_layer
                                .is_none());

                            post_prefill_pre_decode(
                                &replica_metric,
                                &migration_queue,
                                replica_index,
                                &mut replica_stubs,
                                &unfinished_migration_count,
                                prefill_v2_response.generations,
                                prefill_v2_response.batch,
                                new_batch,
                                new_entries,
                            )
                            .await;
                        }

                        let replica_stubs1 = replica_stub_indices
                            .iter()
                            .map(|&index| all_stubs[index].clone())
                            .collect::<Vec<_>>();
                        let replica_metric1 = Arc::clone(&replica_metric);

                        let pending_zag_prefills1 = Arc::clone(&pending_zag_prefills);
                        let head_zag_seq_num1 = Arc::clone(&head_zag_seq_num);

                        let migration_queue1 = migration_queue.clone();
                        let unfinished_migration_count1 = Arc::clone(&unfinished_migration_count);

                        refractory_head_task = Some(create_refract_head_task(
                            replica_index,
                            replica_stubs1,
                            replica_metric1,
                            pending_zag_prefills1,
                            head_zag_seq_num1,
                            migration_queue1,
                            unfinished_migration_count1,
                        ));
                        // transition point#0
                        tracing::info!(
                            "Replica<{}> NewPrefill => RefractoryPrefill",
                            replica_index
                        );
                        // reset zigzag inner state
                        next_zigzag_seq_num = 0;
                        zigzag_req_limit = INIT_ZIGZAG_REQ_LIMIT;
                        *replica_metric.state.write().await = ReplicaState::RefractoryPrefill;
                        yield_now().await;
                        continue 'NextState;
                    }

                    if *MAX_BLOCKS_PER_REPLICA.get().unwrap() < replica_metric.get_used_blocks() {
                        yield_now().await;
                        continue;
                    }

                    if !model_loaded {
                        if let Some((relay_rank, sender)) = RELAY_ASYNC_QUEUE.try_recv() {
                            tracing::info!(
                                "Replica<{}> Before get server's relay response",
                                replica_index
                            );
                            assert_eq!(relay_rank.len(), replica_stubs.len());
                            let response = join_all(
                                replica_stubs
                                    .iter_mut()
                                    .zip(relay_rank.clone())
                                    .into_iter()
                                    .map(|(stub, rank)| stub.relay(rank as i32, false)),
                            )
                            .await
                            .into_iter()
                            .collect::<Result<VecDeque<_>, ClientError>>()
                            .unwrap()
                            .pop_front()
                            .unwrap();
                            tracing::info!("Replica<{}> get Old Prefill's Sender", replica_index);

                            if let (Some(batch_id), Some(seq_num)) = response {
                                tracing::info!(
                                    "Relay response: Batch[{}] zag#{}",
                                    batch_id,
                                    seq_num
                                );
                                let mut prefill_v2_response: PrefillV2Response;
                                let mut new_batch: Batch;
                                let mut new_entries: IntMap<u64, Entry>;

                                loop {
                                    (prefill_v2_response, new_batch, new_entries, ..) =
                                        pending_zag_prefills
                                            .lock()
                                            .await
                                            .pop_front()
                                            .unwrap()
                                            .await
                                            .unwrap();
                                    head_zag_seq_num.fetch_add(1, Ordering::AcqRel);
                                    tracing::info!(
                                        "Replica<{}> (new) pending zigzag prefill queue len is {}",
                                        replica_index,
                                        pending_zag_prefills.lock().await.len()
                                    );

                                    tracing::info!("Prefill response: {}", new_batch.id);
                                    if prefill_v2_response
                                        .pp_info
                                        .as_ref()
                                        .expect("PipeParaInfo must be set")
                                        .start_layer
                                        .is_none()
                                    {
                                        post_prefill_pre_decode(
                                            &replica_metric,
                                            &migration_queue,
                                            replica_index,
                                            &mut replica_stubs,
                                            &unfinished_migration_count,
                                            prefill_v2_response.generations,
                                            prefill_v2_response.batch,
                                            new_batch,
                                            new_entries,
                                        )
                                        .await;
                                        //
                                        model_loaded = true;
                                    } else {
                                        break;
                                    }
                                }
                                // post: relay::resp == zag::resp
                                assert_eq!(batch_id, new_batch.id);
                                // corner case: Relay a not start batch
                                let start_layer =
                                    &prefill_v2_response.pp_info.as_ref().unwrap().start_layer;
                                match start_layer {
                                    Some(pipe_para_info::StartLayer::TfmLayer(num_layer)) => {
                                        // pass to background task to migrate kv cache.
                                        // add partial migration to background migration task
                                        replica_metric.add_partial_migration_cnt();
                                        tracing::info!(
                                            "Batch[{}]::({:?}) add migration fst task from [{}]",
                                            new_batch.id,
                                            prefill_v2_response
                                                .pp_info
                                                .as_ref()
                                                .unwrap()
                                                .num_layer_per_rank,
                                            replica_stub_indices.iter().min().unwrap(),
                                        );
                                        assert_eq!(
                                            *num_layer,
                                            prefill_v2_response
                                                .pp_info
                                                .as_ref()
                                                .unwrap()
                                                .num_layer_per_rank[0]
                                        );
                                        migration_queue
                                            .append_partial_fst(
                                                new_batch.clone(),
                                                *num_layer,
                                                replica_index,
                                                unfinished_migration_count.clone(),
                                            )
                                            .await;
                                        tracing::warn!(
                                            "Append partial first done, batch[{}]",
                                            new_batch.id
                                        );
                                        // pass to OldPrefill task
                                        let prefill_request = PrefillV2Request {
                                            batch: new_batch,
                                            forward_case: encode_prefill_case(vec![
                                                PrefillCase::NaivePp,
                                                PrefillCase::Immigrate,
                                            ]),
                                            pp_info: prefill_v2_response.pp_info,
                                            pipe_peer: replica_stub_indices
                                                .iter()
                                                .map(|&index| index as i32)
                                                .collect::<Vec<_>>(),
                                        };
                                        let partial_prefill_request = PartialPrefill {
                                            request: Some(prefill_request),
                                            zag_request: None,
                                            entries: new_entries,
                                        };
                                        sender.send(Some(partial_prefill_request)).unwrap();
                                    }
                                    Some(pipe_para_info::StartLayer::EmbeddingLayer(1)) => {
                                        // pass to OldPrefill task
                                        let id = new_batch.id;
                                        let prefill_request = PrefillV2Request {
                                            batch: new_batch,
                                            forward_case: encode_prefill_case(vec![
                                                PrefillCase::Normal,
                                            ]),
                                            pp_info: prefill_v2_response.pp_info,
                                            pipe_peer: replica_stub_indices
                                                .iter()
                                                .map(|&index| index as i32)
                                                .collect::<Vec<_>>(),
                                        };
                                        let partial_prefill_request = PartialPrefill {
                                            request: Some(prefill_request),
                                            zag_request: None,
                                            entries: new_entries,
                                        };
                                        tracing::info!("Batch[{}] relay start layer", id);
                                        sender.send(Some(partial_prefill_request)).unwrap();
                                        join_all(
                                            replica_stubs
                                                .iter_mut()
                                                .map(|stub| stub.clear_cache(Some(id))),
                                        )
                                        .await;
                                        // TODO: sub used blocks
                                    }
                                    _ => {
                                        panic!(
                                            "Invalid relay Batch[{}] start_layer: {:?}",
                                            new_batch.id, start_layer
                                        );
                                    }
                                }
                            } else {
                                sender.send(None).unwrap();
                                model_loaded = replica_metric.is_model_loaded();
                            }

                            if model_loaded {
                                let replica_stubs1 = replica_stub_indices
                                    .iter()
                                    .map(|&index| all_stubs[index].clone())
                                    .collect::<Vec<_>>();
                                let replica_metric1 = Arc::clone(&replica_metric);

                                let pending_zag_prefills1 = Arc::clone(&pending_zag_prefills);
                                let head_zag_seq_num1 = Arc::clone(&head_zag_seq_num);

                                let migration_queue1 = migration_queue.clone();
                                let unfinished_migration_count1 =
                                    Arc::clone(&unfinished_migration_count);

                                // NOTE: wait another request to make sure server is complied w/ refractory mode
                                if pending_zag_prefills.lock().await.len() > 0 {
                                    let prefill_v2_response: PrefillV2Response;
                                    let new_batch: Batch;
                                    let new_entries: IntMap<u64, Entry>;

                                    (prefill_v2_response, new_batch, new_entries, ..) =
                                        pending_zag_prefills
                                            .lock()
                                            .await
                                            .pop_front()
                                            .unwrap()
                                            .await
                                            .unwrap();
                                    head_zag_seq_num.fetch_add(1, Ordering::AcqRel);

                                    tracing::info!(
                                        "Replica<{}> (New => Refrac) get Batch[{}]; pending zigzag prefill queue len is {}",
                                        replica_index,
                                        new_batch.id,
                                        pending_zag_prefills.lock().await.len()
                                    );
                                    assert!(prefill_v2_response
                                        .pp_info
                                        .as_ref()
                                        .expect("PipeParaInfo must be set")
                                        .start_layer
                                        .is_none());

                                    post_prefill_pre_decode(
                                        &replica_metric,
                                        &migration_queue,
                                        replica_index,
                                        &mut replica_stubs,
                                        &unfinished_migration_count,
                                        prefill_v2_response.generations,
                                        prefill_v2_response.batch,
                                        new_batch,
                                        new_entries,
                                    )
                                    .await;
                                }

                                refractory_head_task = Some(create_refract_head_task(
                                    replica_index,
                                    replica_stubs1,
                                    replica_metric1,
                                    pending_zag_prefills1,
                                    head_zag_seq_num1,
                                    migration_queue1,
                                    unfinished_migration_count1,
                                ));
                                tracing::info!(
                                    "Replica<{}> NewPrefill => RefractoryPrefill",
                                    replica_index
                                );
                                next_zigzag_seq_num = 0;
                                zigzag_req_limit = INIT_ZIGZAG_REQ_LIMIT;
                                *replica_metric.state.write().await =
                                    ReplicaState::RefractoryPrefill;
                                yield_now().await;
                                continue 'NextState;
                            }
                        }

                        if pending_zag_prefills.lock().await.len() >= (zigzag_req_limit as usize) {
                            yield_now().await;
                            continue;
                        } else {
                            zigzag_req_limit = INIT_ZIGZAG_REQ_LIMIT
                                - (Instant::now()
                                    .duration_since(zigzag_timestamp.unwrap())
                                    .as_millis()
                                    / DEC_ZIGZAG_REQ_LIMIT_RT)
                                    as isize;
                            zigzag_req_limit = zigzag_req_limit.max(2);
                        }
                        if let Some((new_entries, new_batch, _)) = batching_queue
                            .next_batch(
                                None,
                                max_batch_prefill_tokens,
                                max_batch_total_tokens,
                                Some(
                                    *MAX_BLOCKS_PER_REPLICA.get().unwrap()
                                        - replica_metric.get_used_blocks(),
                                ),
                            )
                            .await
                        {
                            tracing::info!(
                                "Requests {:?} are zag",
                                new_batch
                                    .requests
                                    .iter()
                                    .map(|request| request.id)
                                    .collect::<Vec<_>>()
                            );
                            replica_metric
                                .add_used_blocks(new_batch.max_tokens / replica_metric.block_size);
                            let zag_seq_num = next_zigzag_seq_num;
                            next_zigzag_seq_num += 1;
                            let replica_metric = replica_metric.clone();
                            let mut replica_stubs = replica_stubs.clone();
                            let response_future = spawn(async move {
                                tracing::info!(
                                    "Batch [{}] assigned to replica {}",
                                    new_batch.id,
                                    replica_index
                                );

                                let token_sum = new_entries
                                    .iter()
                                    .map(|ele| ele.1.request.input_length)
                                    .sum::<u32>()
                                    as usize;

                                SYSTEM_METRIC
                                    .prefill_tokens
                                    .fetch_add(token_sum, Ordering::SeqCst);

                                let prefill_request = ZagPrefillRequest {
                                    batch: new_batch.clone(),
                                    forward_case: encode_prefill_case(vec![PrefillCase::NaivePp]),
                                    pp_info: Some(PipeParaInfo {
                                        start_layer: Some(
                                            pipe_para_info::StartLayer::EmbeddingLayer(1),
                                        ),
                                        ..Default::default()
                                    }),
                                    zag_layers: 1,
                                    zag_seq_num,
                                };

                                let response = join_all(
                                    replica_stubs
                                        .iter_mut()
                                        .map(|stub| stub.zag_prefill(prefill_request.clone())),
                                )
                                .await
                                .into_iter()
                                .collect::<Result<VecDeque<_>, ClientError>>()
                                .expect(
                                    format!("Used blocks{}", replica_metric.get_used_blocks())
                                        .as_str(),
                                )
                                .pop_front()
                                .unwrap();
                                (response, new_batch, new_entries, next_zigzag_seq_num)
                            });
                            pending_zag_prefills.lock().await.push_back(response_future);
                        }
                    } else {
                        yield_now().await;
                        continue;
                    }
                }
            }
            ReplicaState::RefractoryPrefill => {
                if cfg!(feature = "impl_blitz") {
                    assert!(guard.is_some());

                    // `await` is a RR barrier
                    if let Some((relay_rank, sender)) = RELAY_ASYNC_QUEUE.try_recv() {
                        let response = join_all(
                            replica_stubs
                                .iter_mut()
                                .zip(relay_rank.clone())
                                .into_iter()
                                .map(|(stub, rank)| stub.relay(rank as i32, true)),
                        )
                        .await
                        .into_iter()
                        .collect::<Result<VecDeque<_>, ClientError>>()
                        .unwrap()
                        .pop_front()
                        .unwrap();

                        if let (Some(batch_id), Some(seq_num)) = response {
                            // the head of queue can't be interrupted
                            // return value is second task
                            tracing::info!("Relay response: {} in refractory phase", batch_id);

                            let handle = {
                                let mut pending_guard = pending_zag_prefills.lock().await;
                                let head_seq_num = head_zag_seq_num.load(Ordering::Acquire);
                                let handle =
                                    pending_guard.remove((seq_num - head_seq_num) as usize);
                                tracing::info!(
                                    "Replica<{}> (refract) pending zigzag prefill queue len is {}",
                                    replica_index,
                                    pending_guard.len(),
                                );
                                // OK; pending zigzag queue guard ensures head_zag_seq_num
                                // BEFORE-OR-AFTER atomicity
                                head_zag_seq_num.store(head_seq_num + 1, Ordering::Release);
                                handle
                            };
                            let (prefill_v2_response, new_batch, new_entries, ..) =
                                handle.unwrap().await.unwrap();

                            // tracing::info!("Get prefill response: {}", new_batch.id);
                            assert_eq!(batch_id, new_batch.id);

                            tracing::info!(
                                "Relay response Batch[{}] pp info: {:?}",
                                new_batch.id,
                                prefill_v2_response.pp_info.as_ref().unwrap().start_layer
                            );
                            let start_layer =
                                &prefill_v2_response.pp_info.as_ref().unwrap().start_layer;
                            match start_layer {
                                Some(pipe_para_info::StartLayer::TfmLayer(num_layer)) => {
                                    // pass to background task to migrate kv cache.
                                    // add partial migration to background migration task
                                    replica_metric.add_partial_migration_cnt();
                                    tracing::info!(
                                        "Batch[{}]::({:?}) add migration fst task from [{}]",
                                        new_batch.id,
                                        prefill_v2_response
                                            .pp_info
                                            .as_ref()
                                            .unwrap()
                                            .num_layer_per_rank,
                                        replica_stub_indices.iter().min().unwrap(),
                                    );
                                    assert_eq!(
                                        *num_layer,
                                        prefill_v2_response
                                            .pp_info
                                            .as_ref()
                                            .unwrap()
                                            .num_layer_per_rank[0]
                                    );
                                    migration_queue
                                        .append_partial_fst(
                                            new_batch.clone(),
                                            *num_layer,
                                            replica_index,
                                            unfinished_migration_count.clone(),
                                        )
                                        .await;
                                    // pass to OldPrefill task
                                    let prefill_request = PrefillV2Request {
                                        batch: new_batch,
                                        forward_case: encode_prefill_case(vec![
                                            PrefillCase::NaivePp,
                                            PrefillCase::Immigrate,
                                        ]),
                                        pp_info: prefill_v2_response.pp_info,
                                        pipe_peer: replica_stub_indices
                                            .iter()
                                            .map(|&index| index as i32)
                                            .collect::<Vec<_>>(),
                                    };
                                    let partial_prefill_request = PartialPrefill {
                                        request: Some(prefill_request),
                                        zag_request: None,
                                        entries: new_entries,
                                    };
                                    sender.send(Some(partial_prefill_request)).unwrap();
                                }
                                Some(pipe_para_info::StartLayer::EmbeddingLayer(1)) => {
                                    // pass to OldPrefill task
                                    let id = new_batch.id;
                                    let prefill_request = PrefillV2Request {
                                        batch: new_batch,
                                        forward_case: encode_prefill_case(vec![
                                            PrefillCase::Normal,
                                        ]),
                                        pp_info: prefill_v2_response.pp_info,
                                        pipe_peer: replica_stub_indices
                                            .iter()
                                            .map(|&index| index as i32)
                                            .collect::<Vec<_>>(),
                                    };
                                    let partial_prefill_request = PartialPrefill {
                                        request: Some(prefill_request),
                                        zag_request: None,
                                        entries: new_entries,
                                    };
                                    tracing::info!("Batch[{}] relay start layer", id);
                                    sender.send(Some(partial_prefill_request)).unwrap();
                                    join_all(
                                        replica_stubs
                                            .iter_mut()
                                            .map(|stub| stub.clear_cache(Some(id))),
                                    )
                                    .await;
                                    // TODO: sub used blocks
                                }
                                _ => {
                                    panic!(
                                        "Invalid relay Batch[{}] start_layer: {:?}",
                                        new_batch.id, start_layer
                                    );
                                }
                            }
                        } else {
                            let len = pending_zag_prefills.lock().await.len();
                            if len > 1 {
                                tracing::error!(
                                    "Rank<{}> pending zag prefill len is {}",
                                    replica_index,
                                    len
                                );
                                assert!(pending_zag_prefills.lock().await.len() <= 1);
                            }
                            ZIGZAG_ACTIVE_CNT.fetch_sub(1, Ordering::SeqCst);
                            RELAY_DEACTIVE_CNT.fetch_add(1, Ordering::SeqCst);
                            sender.send(None).unwrap();
                            // await the (possible) last zag request
                            let _ = refractory_head_task
                                .unwrap()
                                .await
                                .expect("Fail to join refractory_head_task!");
                            refractory_head_task = None;
                            zigzag_timestamp = None;
                            // all zag request is processed
                            tracing::info!(
                                "Replica<{}> Refractory => Normal Prefill",
                                replica_index
                            );
                            *replica_metric.state.write().await = ReplicaState::Prefill;
                            yield_now().await;
                            continue 'NextState;
                        }
                    }
                    yield_now().await;
                }
            }
            ReplicaState::OldPrefill => {
                assert!(guard.is_some());
                // let relay_rank = replica_index as u32;
                tracing::info!("Replica<{}> before get relay answer", replica_index);
                if let Some(PartialPrefill {
                    request,
                    mut entries,
                    ..
                }) = RELAY_ASYNC_QUEUE.append(replica_stub_indices.clone()).await
                {
                    let new_batch = request.as_ref().unwrap().batch.clone();
                    tracing::info!(
                        "Replica<{}> (old) recv Batch[{}]",
                        replica_index,
                        new_batch.id
                    );

                    let PrefillV2Response {
                        generations,
                        batch: cached_batch,
                        ..
                    } = join_all(
                        replica_stubs
                            .iter_mut()
                            .map(|stub| stub.prefill_v2(request.clone().unwrap())),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<VecDeque<_>, ClientError>>()
                    .unwrap()
                    .pop_front()
                    .unwrap();
                    replica_metric
                        .add_used_blocks(new_batch.max_tokens / replica_metric.block_size);

                    // [debug] :: corner case: generation length == 1
                    let input_entries_len = entries.len();
                    filter_send_generations_on_prefill_done(&generations, &mut entries);
                    filter_send_generations(&generations, &mut entries);
                    assert_eq!(
                        input_entries_len,
                        entries.len(),
                        "Generation length == 1 for Batch[{}]",
                        new_batch.id
                    );

                    let fst_start_layer = request
                        .as_ref()
                        .unwrap()
                        .pp_info
                        .as_ref()
                        .unwrap()
                        .start_layer
                        .as_ref();
                    match fst_start_layer {
                        Some(pipe_para_info::StartLayer::TfmLayer(num_layer)) => {
                            replica_metric.add_partial_migration_cnt();
                            if let Some((filtered_cached_batch, tokens)) =
                                stubs_filter_batch_with_metric(
                                    &mut replica_stubs,
                                    replica_metric.clone(),
                                    cached_batch,
                                    generations,
                                    &entries,
                                )
                                .await
                            {
                                tracing::info!(
                                    "Batch[{}] from old prefill to decode",
                                    new_batch.id
                                );
                                let migrating_batch = MigratingBatch {
                                    batch: restore_cached_batch(filtered_cached_batch, new_batch),
                                    tokens,
                                    entries,
                                };
                                let all_token_num: u32 = migrating_batch
                                    .entries
                                    .iter()
                                    .map(|e| e.1.request.input_length)
                                    .sum();
                                FLOW_WATCHER
                                    .lock()
                                    .await
                                    .append_token(replica_index, all_token_num as usize);
                                migration_queue
                                    .append_partial_snd(
                                        migrating_batch,
                                        *num_layer,
                                        &replica_stub_indices,
                                        unfinished_migration_count.clone(),
                                    )
                                    .await;
                            }
                        }
                        Some(pipe_para_info::StartLayer::EmbeddingLayer(1)) => {
                            // todo
                            let mut token_sum: u32 = 0;
                            for ele in entries.iter() {
                                token_sum += ele.1.request.input_length;
                            }
                            SYSTEM_METRIC
                                .prefill_tokens
                                .fetch_add(token_sum.try_into().unwrap(), Ordering::AcqRel);
                            if let Some((filtered_cached_batch, tokens)) =
                                stubs_filter_batch_with_metric(
                                    &mut replica_stubs,
                                    replica_metric.clone(),
                                    cached_batch,
                                    generations,
                                    &entries,
                                )
                                .await
                            {
                                let migrating_batch = MigratingBatch {
                                    batch: restore_cached_batch(filtered_cached_batch, new_batch),
                                    tokens,
                                    entries,
                                };
                                let mut flow_watcher = FLOW_WATCHER.lock().await;
                                let all_token_num: u32 = migrating_batch
                                    .entries
                                    .iter()
                                    .map(|e| e.1.request.input_length)
                                    .sum();
                                flow_watcher.append_token(replica_index, all_token_num as usize);
                                migration_queue
                                    .append(
                                        migrating_batch,
                                        replica_index,
                                        unfinished_migration_count.clone(),
                                    )
                                    .await;
                            }
                        }
                        _ => {
                            panic!(
                                "Invalid relay Batch[{}] start_layer: {:?}",
                                new_batch.id, fst_start_layer
                            );
                        }
                    }
                } else {
                    // NOTE: a None doesn't mean *stopped*, just *skipped*
                    // Remark: RefractoryPrefill => Prefill will shutdown an OldPrefill by adding RelayDeactiveCnt
                    let mut current = RELAY_DEACTIVE_CNT.load(Ordering::Acquire);
                    while current > 0 {
                        // possible take action
                        match RELAY_DEACTIVE_CNT.compare_exchange(
                            current,
                            current - 1,
                            Ordering::SeqCst,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => {
                                // no other thread concurrently modify this counter
                                tracing::info!("Replica<{}> OldPrefill => Prefill", replica_index);
                                *replica_metric.state.write().await = ReplicaState::Prefill;
                                break;
                            }
                            Err(new) => current = new,
                        }
                    }
                    yield_now().await;
                }
            }
            ReplicaState::MutatingToDecode => {
                assert!(
                    guard.is_some(),
                    "replica_index: {} MutatingToDecode guard is None",
                    replica_index
                );
                let migrated_batches = migration_queue.flush(replica_index).await;
                for migrated_batch in migrated_batches {
                    let MigratedBatch {
                        batch,
                        tokens,
                        entries,
                    } = migrated_batch;
                    global_entries.extend(entries);
                    global_batches.push((batch, tokens));
                }
                tracing::info!("Replica<{}> MutatingToDecode => Decode", replica_index);
                *replica_metric.state.write().await = ReplicaState::Decode;
                drop(take(&mut guard));
                continue 'NextState;
            }
            ReplicaState::AusPrefill => {
                if guard.is_none() {
                    guard = Some(dst_mutex.clone().lock_owned().await);
                }
                tracing::info!("Replica<{}> => Prefill", replica_index);
                *replica_metric.state.write().await = ReplicaState::Prefill;
                continue 'NextState;
            }
            ReplicaState::AusDecode => {
                if pending_lock_acquirer.is_some() {
                    guard = Some(take(&mut pending_lock_acquirer).unwrap().await.unwrap());
                }
                *replica_metric.state.write().await = ReplicaState::Decode;
                drop(take(&mut guard));
                continue 'NextState;
            }
            ReplicaState::ShuttingDecode => {
                if global_batches.is_empty() && guard.is_some() {
                    assert!(pending_lock_acquirer.is_none());
                    assert!(global_entries.is_empty());
                    *replica_metric.state.write().await = ReplicaState::ShuttingNull;
                    tracing::info!(
                        "Replica<{}> : ShuttingDecode => ShuttingNull",
                        replica_index
                    );
                    continue 'NextState;
                }

                /*
                   \note \ne current `task` eventually acquires lock
                   \maybe falsely wrong arguement?
                */
                if guard.is_none() {
                    match &mut pending_lock_acquirer {
                        Some(join_handle) => {
                            if join_handle.is_finished() {
                                guard =
                                    Some(take(&mut pending_lock_acquirer).unwrap().await.unwrap());
                                yield_now().await;
                            }
                        }
                        None => {
                            pending_lock_acquirer = Some(spawn(dst_mutex.clone().lock_owned()));
                        }
                    }
                } else {
                    assert!(pending_lock_acquirer.is_none());
                }

                while let Some(migrated_batch) = migration_queue.try_consume(replica_index) {
                    let all_token_num: u32 = migrated_batch
                        .entries
                        .iter()
                        .map(|e| e.1.request.input_length)
                        .sum();
                    let MigratedBatch {
                        batch,
                        tokens,
                        entries,
                    } = migrated_batch;
                    let mut flow_watcher = FLOW_WATCHER.lock().await;
                    flow_watcher.recv_token(replica_index, all_token_num as usize);
                    global_entries.extend(entries);
                    global_batches.push((batch, tokens));
                }

                if !global_batches.is_empty() {
                    // Prefill finished and do decode immediately without receiving from the channel.
                    let (batches, last_iter_tokens): (Vec<_>, Vec<_>) =
                        take(&mut global_batches).into_iter().unzip();
                    // Prefill finished and do decode immediately without receiving from the channel.
                    let request = DecodeV2Request {
                        batches,
                        last_iter_tokens,
                    };
                    let response = join_all(
                        replica_stubs
                            .iter_mut()
                            .map(|stub| stub.decode_v2(request.clone())),
                    )
                    .await
                    .into_iter()
                    .collect::<Result<VecDeque<_>, ClientError>>()
                    .unwrap()
                    .pop_front()
                    .unwrap();
                    let DecodeV2Response {
                        generations, batch, ..
                    } = response;
                    filter_send_generations(&generations, &mut global_entries);
                    if let Some((batch, _)) = stubs_filter_batch_with_metric(
                        &mut replica_stubs,
                        replica_metric.clone(),
                        batch,
                        generations,
                        &mut global_entries,
                    )
                    .await
                    {
                        global_batches.push((batch, Tokens::default()));
                    }
                } else {
                    yield_now().await;
                }
            }
            ReplicaState::ShuttingPrefill => {
                assert!(guard.is_some());
                if unfinished_migration_count.load(Ordering::Acquire) == 0 {
                    assert!(global_batches.is_empty());
                    assert!(global_entries.is_empty());
                    *replica_metric.state.write().await = ReplicaState::ShuttingNull;
                    tracing::info!(
                        "Replica<{}> : ShuttingPrefill => ShuttingNull",
                        replica_index
                    );
                    continue 'NextState;
                }
                yield_now().await;
            }
            ReplicaState::NvlinkSending
            | ReplicaState::RdmaSending
            | ReplicaState::RdmaLoading
            | ReplicaState::RdmaCasting
            | ReplicaState::NvlCasting
            | ReplicaState::TanzCasting => {
                panic!(
                    "Replica<{}> in {:?}, which is planner's marker state!",
                    replica_index, tmp_ro_state
                );
            }
        }
    }
}

fn create_refract_head_task(
    replica_index: usize,
    mut replica_stubs: Vec<Stub>,
    replica_metric: Arc<ReplicaMetric>,
    pending_zag_prefills: Arc<
        Mutex<VecDeque<JoinHandle<(PrefillV2Response, Batch, IntMap<u64, Entry>, u32)>>>,
    >,
    head_zag_seq_num: Arc<AtomicU32>,
    migration_queue: MigrationQueue,
    unfinished_migration_count: Arc<AtomicI64>,
) -> JoinHandle<()> {
    spawn(async move {
        let mut prefill_v2_response: PrefillV2Response;
        let mut new_batch: Batch;
        let mut new_entries: IntMap<u64, Entry>;
        'Finish: loop {
            let handle = {
                let mut pending_guard = pending_zag_prefills.lock().await;
                if pending_guard.len() == 0 {
                    tracing::info!(
                        "Replica<{}> refractory zag prefill queue len is 0",
                        replica_index
                    );
                    head_zag_seq_num.store(0, Ordering::Release);
                    break 'Finish;
                }
                head_zag_seq_num.fetch_add(1, Ordering::AcqRel);
                let handle = pending_guard.pop_front();
                tracing::info!(
                    "Replica<{}> (head) pending zigzag prefill queue len is {}",
                    replica_index,
                    pending_guard.len()
                );
                handle
            };
            (prefill_v2_response, new_batch, new_entries, ..) = handle.unwrap().await.unwrap();
            tracing::info!("Prefill response: {} in head task", new_batch.id);
            assert!(prefill_v2_response
                .pp_info
                .as_ref()
                .expect("PipeParaInfo must be set")
                .start_layer
                .is_none());
            post_prefill_pre_decode(
                &replica_metric,
                &migration_queue,
                replica_index,
                &mut replica_stubs,
                &unfinished_migration_count,
                prefill_v2_response.generations,
                prefill_v2_response.batch,
                new_batch,
                new_entries,
            )
            .await;
        }
        tracing::info!("Replica<{}> refractory task break", replica_index);
    })
}

async fn post_prefill_pre_decode(
    replica_metric: &Arc<ReplicaMetric>,
    migration_queue: &MigrationQueue,
    replica_index: usize,
    replica_stubs: &mut Vec<Stub>,
    unfinished_migration_count: &Arc<AtomicI64>,
    generations: Vec<Generation>,
    cached_batch: Option<CachedBatch>,
    new_batch: Batch,
    mut new_entries: HashMap<
        u64,
        Entry,
        std::hash::BuildHasherDefault<nohash_hasher::NoHashHasher<u64>>,
    >,
) {
    // If start_layer is not set, it means the prefill is done.
    filter_send_generations_on_prefill_done(&generations, &mut new_entries);
    filter_send_generations(&generations, &mut new_entries);
    if let Some((filtered_cached_batch, tokens)) = stubs_filter_batch_with_metric(
        replica_stubs,
        replica_metric.clone(),
        cached_batch,
        generations,
        &new_entries,
    )
    .await
    {
        let migrating_batch = MigratingBatch {
            batch: restore_cached_batch(filtered_cached_batch, new_batch),
            tokens,
            entries: new_entries,
        };
        let mut flow_watcher = FLOW_WATCHER.lock().await;
        let all_token_num: u32 = migrating_batch
            .entries
            .iter()
            .map(|e| e.1.request.input_length)
            .sum();
        flow_watcher.append_token(replica_index, all_token_num as usize);
        migration_queue
            .append(
                migrating_batch,
                replica_index,
                unfinished_migration_count.clone(),
            )
            .await;
    }
}

/// P2D migration queue.
#[derive(Debug, Clone)]
pub(crate) struct MigrationQueue {
    producer_txs: Arc<HashMap<usize, async_channel::Sender<MigrationCommand>>>,
    consumer_rxs: Arc<HashMap<usize, async_channel::Receiver<MigratedBatch>>>,
    /// New Prefill puts a channel, Old Prefill gets Decode replica index
    all_partial_batches:
        Arc<Mutex<HashMap<usize, oneshot::Receiver<(usize, JoinHandle<OwnedMutexGuard<()>>)>>>>,
    /// New Prefill mark partial batch as eligible
    eligible_partial_batches: Arc<RwLock<HashSet<u64>>>,
}

pub(crate) enum MigrationCommand {
    Produce {
        unfinished_migration_count: Arc<AtomicI64>,
        migrating_batch: MigratingBatch,
    },
    ProducePartial {
        unfinished_migration_count: Arc<AtomicI64>,
        migrating_batch: MigratingPartialBatch,
    },
    Flush {
        response_tx: oneshot::Sender<Vec<MigratedBatch>>,
    },
}

impl MigrationQueue {
    #[instrument(skip_all)]
    pub fn spawn_migration_queue(
        all_stubs: Vec<Stub>,
        all_replica_stub_indices: Vec<Vec<usize>>,
        metrics: Vec<Arc<ReplicaMetric>>,
        dst_mutexes: Vec<Arc<Mutex<()>>>,
    ) -> Self {
        let mut producer_txs = HashMap::new();
        let mut consumer_rxs = HashMap::new();
        let all_partial_batches = Arc::new(Mutex::new(HashMap::new()));
        let eligible_partial_batches = Arc::new(RwLock::new(HashSet::new()));

        let (consumer_txs_vec, consumer_rxs_vec): (Vec<_>, Vec<_>) = all_replica_stub_indices
            .iter()
            .map(|_| async_channel::unbounded::<MigratedBatch>())
            .unzip();

        for ((ranks, src_replica_metric), consumer_rx) in all_replica_stub_indices
            .iter()
            .cloned()
            .zip(metrics.iter().cloned())
            .zip(consumer_rxs_vec.iter().cloned())
        {
            let replica_index = *ranks.iter().min().unwrap();
            let (producer_tx, producer_rx) = async_channel::unbounded::<MigrationCommand>();

            producer_txs.insert(replica_index, producer_tx);
            consumer_rxs.insert(replica_index, consumer_rx);

            let consumer_txs = consumer_txs_vec.clone();
            let partial_batches = all_partial_batches.clone();
            let eligible_partial_batches = eligible_partial_batches.clone();
            let all_stubs = all_stubs.clone();
            let replica_metrics = metrics.clone();
            let replicas = all_replica_stub_indices.clone();
            let dst_mutexes = dst_mutexes.clone();
            let replica_stubs = ranks
                .iter()
                .map(|&rank| all_stubs[rank].clone())
                .collect::<Vec<_>>();

            spawn(async move {
                let mut waiting_full_batches = VecDeque::<(MigratingBatch, Arc<AtomicI64>)>::new();
                let mut waiting_partial_batches =
                    BTreeMap::<u64, (MigratingPartialBatch, Arc<AtomicI64>)>::new();
                let mut doing_full_migration = Option::<oneshot::Receiver<()>>::None;
                let mut doing_partial_migration = Option::<oneshot::Receiver<()>>::None;
                let tokio_task_id = format!("(watcher#{})", replica_index);

                loop {
                    while let Ok(element) = producer_rx.try_recv() {
                        match element {
                            MigrationCommand::Produce {
                                migrating_batch,
                                unfinished_migration_count,
                            } => {
                                // metrics::blitz_waiting_full_batches
                                waiting_full_batches
                                    .push_back((migrating_batch, unfinished_migration_count));
                            }
                            MigrationCommand::ProducePartial {
                                unfinished_migration_count,
                                migrating_batch,
                            } => {
                                // metrics::blitz_waiting_partial_batches
                                waiting_partial_batches.insert(
                                    migrating_batch.batch.id,
                                    (migrating_batch, unfinished_migration_count),
                                );
                            }
                            MigrationCommand::Flush { response_tx } => {
                                // metrics::blitz_waiting_full_batches
                                let mut migrated_batches = Vec::<MigratedBatch>::new();
                                take(&mut waiting_full_batches).into_iter().for_each(
                                    |(migrating_batch, unfinished_count)| {
                                        SYSTEM_METRIC.token_in_queue.fetch_sub(
                                            migrating_batch
                                                .entries
                                                .iter()
                                                .map(|entry| entry.1.request.input_length as usize)
                                                .sum(),
                                            Ordering::AcqRel,
                                        );
                                        migrated_batches.push(migrating_batch.into());
                                        unfinished_count.fetch_sub(1, Ordering::AcqRel);
                                    },
                                );
                                let batch_ids = migrated_batches
                                    .iter()
                                    .map(|b| b.batch.id)
                                    .collect::<Vec<_>>();
                                tracing::info!(
                                    "Batches {:?} flushed on replica {}",
                                    batch_ids,
                                    replica_index
                                );
                                response_tx.send(migrated_batches).unwrap();
                            }
                        }
                    }

                    // prioritise zag prefill to normal prefill
                    match &mut doing_partial_migration {
                        Some(recver) => match recver.try_recv() {
                            Ok(_) => doing_partial_migration = None,
                            Err(oneshot::error::TryRecvError::Empty) => {}
                            Err(oneshot::error::TryRecvError::Closed) => panic!("sender dropped"),
                        },
                        None => {}
                    };

                    if doing_partial_migration.is_none() && !waiting_partial_batches.is_empty() {
                        let (&batch_id, (mb, _)) = waiting_partial_batches.iter().next().unwrap();
                        // get partial migration task
                        match &mb.fst2snd_tx {
                            Some(_) => {
                                // Initiator ::
                                // Select an available destination replica whose lock is not held by other tasks.
                                'hold_mtx: loop {
                                    for (i, (dst_mutex, dst_replica_metric)) in
                                        dst_mutexes.iter().zip(replica_metrics.iter()).enumerate()
                                    {
                                        let src_replica_index = replica_index;
                                        let dst_replica_index = *replicas[i].iter().min().unwrap();
                                        // [initiator] :: check dst replica free blocks
                                        let num_migrating_blocks =
                                            mb.batch.max_tokens / dst_replica_metric.block_size;
                                        // [debug]: debug flag for block count overflow
                                        if let Ok(g) = dst_mutex.clone().try_lock_owned() {
                                            if *dst_replica_metric.state.read().await
                                                == ReplicaState::ShuttingNull
                                            {
                                                continue;
                                            }
                                            // assert: check again after acquiring the lock.
                                            let dst_replica_used_blocks =
                                                dst_replica_metric.get_used_blocks();
                                            if dst_replica_used_blocks + num_migrating_blocks
                                                > *MAX_BLOCKS_PER_REPLICA.get().unwrap()
                                            {
                                                drop(g);
                                                tracing::warn!("Replica<{}> sending to Replica<{}> Batch [{}] Block Overflow", src_replica_index, dst_replica_index, batch_id);
                                                yield_now().await;
                                                continue;
                                            } else {
                                                let (
                                                    _,
                                                    (
                                                        migrating_partial_batch,
                                                        unfinished_migration_count,
                                                    ),
                                                ) = waiting_partial_batches.pop_first().unwrap();
                                                tracing::warn!(
                                                    "Lock decoding Replica<{}> guard for Batch[{}]!",
                                                    dst_replica_index, batch_id
                                                );
                                                assert_ne!(src_replica_index, dst_replica_index, "Replica<{}> <-> Replica<{}> partial migrating batch[{}]", src_replica_index, dst_replica_index, migrating_partial_batch.batch.id);
                                                // commit: send partial kv cache
                                                // metrics::blitz_used_blocks
                                                dst_replica_metric
                                                    .add_used_blocks(num_migrating_blocks);
                                                let src_stubs = replica_stubs.clone();
                                                let dst_stubs = replicas[i]
                                                    .iter()
                                                    .map(|&rank| all_stubs[rank].clone())
                                                    .collect::<Vec<_>>();
                                                let src_stub_indices = ranks.clone();
                                                let dst_stub_indices = replicas[i].clone();
                                                let (tx, rx) = oneshot::channel();
                                                doing_partial_migration = Some(rx);
                                                let src_replica_metric = src_replica_metric.clone();
                                                let MigratingPartialBatch {
                                                    batch,
                                                    layer,
                                                    fst2snd_tx,
                                                    ..
                                                } = migrating_partial_batch;
                                                tracing::info!(
                                                    "Issue Batch[{}] partial migrate fst...",
                                                    batch_id
                                                );
                                                let handle = spawn(async move {
                                                    tracing::info!(
                                                        "Replica<{}> -~> Replica<{}>: migrating Batch[{}] fst partial",
                                                        src_replica_index,
                                                        dst_replica_index,
                                                        batch.id,
                                                    );

                                                    let batch_id = batch.id;
                                                    let handle = Self::trans_partial_kv_cache(
                                                        batch,
                                                        Some(layer),
                                                        None,
                                                        src_stub_indices,
                                                        src_stubs,
                                                        dst_stub_indices,
                                                        dst_stubs,
                                                    );
                                                    // fst partial batch
                                                    let _ = handle.await.unwrap().pop().unwrap();

                                                    src_replica_metric.sub_partial_migration_cnt();
                                                    unfinished_migration_count
                                                        .fetch_sub(1, Ordering::AcqRel);
                                                    src_replica_metric
                                                        .sub_used_blocks(num_migrating_blocks);
                                                    // [initiator]
                                                    // don't send MigratedBatch to consumer; follower's duty
                                                    tx.send(()).unwrap();
                                                    tracing::warn!("Move decoding Replica<{}> guard for Batch[{}]...", dst_replica_index, batch_id);
                                                    g
                                                });
                                                fst2snd_tx.unwrap().send((i, handle)).unwrap();
                                                eligible_partial_batches
                                                    .write()
                                                    .await
                                                    .insert(batch_id);
                                                break 'hold_mtx;
                                            }
                                        }
                                        // \pre not break -> not hold any lock
                                    }
                                    SCARCE_DECODE.store(true, Ordering::Release);
                                    yield_now().await;
                                }
                            }
                            None => {
                                let elgb_rlck = eligible_partial_batches.read().await;
                                if let Some(&batch_id) = waiting_partial_batches
                                    .keys()
                                    .find(|&&batch_id| elgb_rlck.contains(&batch_id))
                                {
                                    drop(elgb_rlck);
                                    eligible_partial_batches.write().await.remove(&batch_id);
                                    // [follower] :: get initiator's previous choice
                                    let (migrating_partial_batch, unfinished_migration_count) =
                                        waiting_partial_batches.remove(&batch_id).unwrap();
                                    let snd_rx: oneshot::Receiver<(
                                        usize,
                                        JoinHandle<OwnedMutexGuard<()>>,
                                    )> = partial_batches
                                        .lock()
                                        .await
                                        .remove(&(migrating_partial_batch.batch.id as usize))
                                        .unwrap();

                                    tracing::debug!(
                                        "Batch[{}] wait partial migrate snd...",
                                        migrating_partial_batch.batch.id
                                    );
                                    // [follower] :: check dst replica free blocks
                                    let (i, fst_handle) = snd_rx.await.unwrap();
                                    tracing::debug!(
                                        "Issue Batch[{}] partial migrate snd...",
                                        migrating_partial_batch.batch.id
                                    );
                                    let num_migrating_blocks =
                                        migrating_partial_batch.batch.max_tokens
                                            / replica_metrics[i].block_size;
                                    let consumer_tx = consumer_txs[i].clone();
                                    let src_replica_index = replica_index;
                                    let dst_replica_index = *replicas[i].iter().min().unwrap();
                                    let src_stubs = replica_stubs.clone();
                                    let dst_stubs = replicas[i]
                                        .iter()
                                        .map(|&index| all_stubs[index].clone())
                                        .collect::<Vec<_>>();
                                    let src_stub_indices = ranks.clone();
                                    let dst_stub_indices = replicas[i].clone();
                                    let (tx, rx) = oneshot::channel();
                                    doing_partial_migration = Some(rx);
                                    let src_replica_metric = src_replica_metric.clone();
                                    spawn(async move {
                                        let MigratingPartialBatch {
                                            batch,
                                            tokens,
                                            entries,
                                            layer,
                                            ..
                                        } = migrating_partial_batch;

                                        tracing::info!(
                                            "Replica<{}> -~> Replica<{}>: migrating snd partial batch[{}]",
                                            src_replica_index,
                                            dst_replica_index,
                                            batch.id,
                                        );

                                        let batch = Self::trans_partial_kv_cache(
                                            batch,
                                            None,
                                            Some(layer),
                                            src_stub_indices,
                                            src_stubs,
                                            dst_stub_indices,
                                            dst_stubs,
                                        )
                                        .await
                                        .unwrap()
                                        .pop()
                                        .unwrap();

                                        let batch_id = batch.id;
                                        // snd partial batch
                                        let migrated_batch = MigratedBatch {
                                            batch,
                                            tokens: tokens.unwrap(),
                                            entries: entries.unwrap(),
                                        };
                                        src_replica_metric.sub_partial_migration_cnt();
                                        unfinished_migration_count.fetch_sub(1, Ordering::AcqRel);
                                        src_replica_metric.sub_used_blocks(num_migrating_blocks);
                                        let _ = fst_handle.await.unwrap();
                                        tracing::debug!(
                                            "Drop decoding Replica<{}> guard for Batch[{}].",
                                            i,
                                            batch_id
                                        );
                                        consumer_tx.send(migrated_batch).await.unwrap();
                                        tx.send(()).unwrap();
                                    });
                                    // [debug] :: possible error state: follower issue 2 migrations onto the same QP
                                    continue;
                                }
                            }
                        }
                        yield_now().await;
                    }

                    // only ReplicaState::Refractory is possible to do both zag migration & p2d migration
                    match &mut doing_full_migration {
                        Some(recver) => match recver.try_recv() {
                            Ok(_) => doing_full_migration = None,
                            Err(oneshot::error::TryRecvError::Empty) => {}
                            Err(oneshot::error::TryRecvError::Closed) => panic!("sender dropped"),
                        },
                        None => {}
                    };

                    if doing_full_migration.is_none() && !waiting_full_batches.is_empty() {
                        tracing::debug!(
                            "{}Batch[{}] wait full migration...",
                            tokio_task_id,
                            &waiting_full_batches.front().unwrap().0.batch.id
                        );
                        // Select an available destination replica whose lock is not held by other tasks.
                        for (i, (dst_mutex, dst_replica_metric)) in
                            dst_mutexes.iter().zip(replica_metrics.iter()).enumerate()
                        {
                            let src_index = replica_index;
                            let dst_index = *replicas[i].iter().min().unwrap();

                            let num_migrating_blocks =
                                waiting_full_batches.front().unwrap().0.batch.max_tokens
                                    / dst_replica_metric.block_size;

                            // Check if the destination replica has enough space to accept the migrating batch.
                            if let Ok(g) = dst_mutex.clone().try_lock_owned() {
                                if *dst_replica_metric.state.read().await
                                    == ReplicaState::ShuttingNull
                                {
                                    continue;
                                }
                                // Check again after acquiring the lock.
                                let dst_replica_used_blocks = dst_replica_metric.get_used_blocks();
                                if dst_replica_used_blocks + num_migrating_blocks
                                    > *MAX_BLOCKS_PER_REPLICA.get().unwrap()
                                {
                                    drop(g);
                                    tracing::warn!("Replica<{}> sending to Replica<{}> Batch [{}] Block Overflow", src_index, dst_index, waiting_full_batches.front().unwrap().0.batch.id);
                                    yield_now().await;
                                    continue;
                                } else {
                                    let (migrating_batch, unfinished_migration_count) =
                                        waiting_full_batches.pop_front().unwrap();
                                    let consumer_tx = consumer_txs[i].clone();
                                    tracing::debug!(
                                        "Lock decoding Replica<{}> guard for Batch[{}]!",
                                        i,
                                        migrating_batch.batch.id
                                    );

                                    let src_index = replica_index;
                                    let dst_index = *replicas[i].iter().min().unwrap();
                                    assert_ne!(
                                        src_index, dst_index,
                                        "Replica<{}> <-> Replica<{}> migrating batch[{}]",
                                        src_index, dst_index, migrating_batch.batch.id
                                    );
                                    dst_replica_metric.add_used_blocks(num_migrating_blocks);
                                    let src_stubs = replica_stubs.clone();
                                    let dst_stubs = replicas[i]
                                        .iter()
                                        .map(|&index| all_stubs[index].clone())
                                        .collect::<Vec<_>>();
                                    let src_stub_indices = ranks.clone();
                                    let dst_stub_indices = replicas[i].clone();
                                    let (tx, rx) = oneshot::channel();
                                    doing_full_migration = Some(rx);
                                    let src_replica_metric = src_replica_metric.clone();

                                    spawn(async move {
                                        let MigratingBatch {
                                            batch,
                                            tokens,
                                            entries,
                                        } = migrating_batch;

                                        let src = *src_stub_indices.iter().min().unwrap();
                                        let dst = *dst_stub_indices.iter().min().unwrap();
                                        assert_eq!(replica_index, src);
                                        tracing::info!(
                                            "Migrate Batch[{}] <{:?}> -~> <{:?}>",
                                            batch.id,
                                            src_stub_indices,
                                            dst_stub_indices,
                                        );

                                        let batch = Self::trans_kv_cache(
                                            batch,
                                            src_stub_indices,
                                            src_stubs,
                                            dst_stub_indices,
                                            dst_stubs,
                                        )
                                        .await
                                        .unwrap()
                                        .pop()
                                        .unwrap();

                                        tracing::info!(
                                            "Migrate Batch Done[{}] <{}> -~> <{}>",
                                            batch.id,
                                            src,
                                            dst,
                                        );

                                        let batch_id = batch.id;
                                        let migrated_batch = MigratedBatch {
                                            batch,
                                            tokens,
                                            entries,
                                        };

                                        unfinished_migration_count.fetch_sub(1, Ordering::AcqRel);
                                        src_replica_metric.sub_used_blocks(num_migrating_blocks);
                                        consumer_tx.send(migrated_batch).await.unwrap();
                                        drop(g);
                                        tracing::debug!(
                                            "Drop decoding Replica<{}> guard for Batch[{}].",
                                            i,
                                            batch_id
                                        );
                                        tx.send(()).unwrap();
                                    });
                                    break;
                                }
                            }
                        }
                        // \pre not break -> not hold any lock
                        SCARCE_DECODE.store(true, Ordering::Release);
                    }
                    yield_now().await;
                }
            });
        }

        Self {
            producer_txs: Arc::new(producer_txs),
            consumer_rxs: Arc::new(consumer_rxs),
            all_partial_batches,
            eligible_partial_batches,
        }
    }

    pub(crate) async fn append(
        &self,
        migrating_batch: MigratingBatch,
        prefill_replica_index: usize,
        prefill_replica_unfinished_migration_count: Arc<AtomicI64>,
    ) {
        // metrics::blitz
        let producer_tx = self.producer_txs.get(&prefill_replica_index).unwrap();
        prefill_replica_unfinished_migration_count.fetch_add(1, Ordering::AcqRel);
        SYSTEM_METRIC.token_in_queue.fetch_add(
            migrating_batch
                .entries
                .iter()
                .map(|entry| entry.1.request.input_length as usize)
                .sum(),
            Ordering::AcqRel,
        );
        producer_tx
            .send(MigrationCommand::Produce {
                unfinished_migration_count: prefill_replica_unfinished_migration_count,
                migrating_batch,
            })
            .await
            .unwrap();
    }

    pub(crate) async fn append_partial_fst(
        &self,
        batch: Batch,
        num_layer: u32,
        prefill_replica_index: usize,
        prefill_replica_unfinished_migration_count: Arc<AtomicI64>,
    ) {
        // metrics::blitz
        let producer_tx = self.producer_txs.get(&prefill_replica_index).unwrap();
        prefill_replica_unfinished_migration_count.fetch_add(1, Ordering::AcqRel);
        SYSTEM_METRIC.token_in_queue.fetch_add(
            batch
                .requests
                .iter()
                .map(|r| r.input_tokens.len() as usize)
                .sum(),
            Ordering::AcqRel,
        );
        let (fst_tx, snd_rx) = oneshot::channel::<(usize, JoinHandle<OwnedMutexGuard<()>>)>();
        let mut m = self.all_partial_batches.lock().await;
        m.insert(batch.id as usize, snd_rx);
        drop(m);
        producer_tx
            .send(MigrationCommand::ProducePartial {
                unfinished_migration_count: prefill_replica_unfinished_migration_count,
                migrating_batch: MigratingPartialBatch {
                    batch,
                    tokens: None,
                    entries: None,
                    layer: num_layer,
                    fst2snd_tx: Some(fst_tx),
                },
            })
            .await
            .unwrap();
    }

    pub(crate) async fn append_partial_snd(
        &self,
        migrating_batch: MigratingBatch,
        num_layer: u32,
        prefill_replica_stub_indices: &Vec<usize>,
        prefill_replica_unfinished_migration_count: Arc<AtomicI64>,
    ) {
        // metrics::blitz
        let src_replica_index = *prefill_replica_stub_indices.iter().min().unwrap();
        let producer_tx = self.producer_txs.get(&src_replica_index).unwrap();
        prefill_replica_unfinished_migration_count.fetch_add(1, Ordering::AcqRel);
        producer_tx
            .send(MigrationCommand::ProducePartial {
                unfinished_migration_count: prefill_replica_unfinished_migration_count,
                migrating_batch: MigratingPartialBatch {
                    batch: migrating_batch.batch,
                    tokens: Some(migrating_batch.tokens),
                    entries: Some(migrating_batch.entries),
                    layer: num_layer,
                    fst2snd_tx: None,
                },
            })
            .await
            .unwrap();
    }

    pub(crate) async fn flush(&self, mutating_replica_index: usize) -> Vec<MigratedBatch> {
        // metrics::blitz
        let producer_tx = self.producer_txs.get(&mutating_replica_index).unwrap();
        let (tx, rx) = oneshot::channel();
        producer_tx
            .send(MigrationCommand::Flush { response_tx: tx })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    pub(crate) fn try_consume(&self, decode_replica_index: usize) -> Option<MigratedBatch> {
        let ret = self
            .consumer_rxs
            .get(&decode_replica_index)
            .unwrap()
            .try_recv()
            .ok();
        if ret.is_some() {
            let migrated_batch = ret.as_ref().unwrap();
            SYSTEM_METRIC.token_in_queue.fetch_sub(
                migrated_batch
                    .entries
                    .iter()
                    .map(|entry| entry.1.request.input_length as usize)
                    .sum(),
                Ordering::AcqRel,
            );
        }
        ret
    }
}

impl MigrationQueue {
    async fn trans_kv_cache(
        batch: Batch,
        src_rank_indices: Vec<usize>,
        src_stubs: Vec<Stub>,
        dst_rank_indices: Vec<usize>,
        dst_stubs: Vec<Stub>,
    ) -> Result<Vec<CachedBatch>, ClientError> {
        metrics::increment_counter!("blitz_kv_cache_times");
        let src_ranks = src_rank_indices
            .iter()
            .map(|i| *i as i32)
            .collect::<Vec<_>>();
        let dst_ranks = dst_rank_indices
            .iter()
            .map(|i| *i as i32)
            .collect::<Vec<_>>();
        join_all(
            src_stubs
                .clone()
                .into_iter()
                .map(|mut stub| {
                    let batch = batch.clone();
                    let src_ranks0 = src_ranks.clone();
                    let dst_ranks0 = dst_ranks.clone();
                    async move {
                        stub.migrate(batch.clone(), src_ranks0, dst_ranks0)
                            .await
                            .map(|res| res.batch)
                    }
                    .boxed()
                })
                .chain(dst_stubs.into_iter().map(|mut stub| {
                    let batch = batch.clone();
                    let src_ranks0 = src_ranks.clone();
                    let dst_ranks0 = dst_ranks.clone();
                    async move {
                        stub.immigrate(batch.clone(), src_ranks0, dst_ranks0)
                            .await
                            .map(|res| res.batch)
                    }
                    .boxed()
                })),
        )
        .await
        .into_iter()
        .collect()
    }

    async fn trans_partial_kv_cache(
        batch: Batch,
        fst_layer: Option<u32>,
        snd_layer: Option<u32>,
        src_stub_indices: Vec<usize>,
        src_stubs: Vec<Stub>,
        dst_stub_indices: Vec<usize>,
        dst_stubs: Vec<Stub>,
    ) -> Result<Vec<CachedBatch>, ClientError> {
        metrics::increment_counter!("blitz_partial_kv_cache_times");
        let src_ranks = src_stub_indices
            .iter()
            .map(|i| *i as i32)
            .collect::<Vec<_>>();
        let dst_ranks = dst_stub_indices
            .iter()
            .map(|i| *i as i32)
            .collect::<Vec<_>>();
        join_all(
            src_stubs
                .clone()
                .into_iter()
                .map(|mut stub| {
                    let batch = batch.clone();
                    let src_ranks0 = src_ranks.clone();
                    let dst_ranks0 = dst_ranks.clone();
                    match (fst_layer, snd_layer) {
                        (Some(fst_layer), None) => {
                            async move {
                                stub.migrate_fst(
                                    batch,
                                    fst_layer,
                                    src_ranks0,
                                    dst_ranks0,
                                )
                                .await
                                .map(|res| res.batch)
                            }
                            .boxed()
                        }
                        (None, Some(snd_layer)) => {
                            async move {
                                stub.migrate_snd(
                                    batch,
                                    snd_layer,
                                    src_ranks0,
                                    dst_ranks0,
                                )
                                .await
                                .map(|res| res.batch)
                            }
                            .boxed()
                        }
                        _ => {
                            panic!("trans_partial_kv_cache fst_layer & snd_layer not mutually exclusive!");
                        }
                    }
                })
                .chain(dst_stubs.into_iter().map(|mut stub| {
                    let batch = batch.clone();
                    let src_ranks0 = src_ranks.clone();
                    let dst_ranks0 = dst_ranks.clone();
                    match (fst_layer, snd_layer) {
                        (Some(fst_layer), None) => {
                            async move {
                                stub.immigrate_fst(
                                    batch,
                                    fst_layer,
                                    src_ranks0,
                                    dst_ranks0,
                                )
                                .await
                                .map(|res| res.batch)
                            }
                            .boxed()
                        }
                        (None, Some(snd_layer)) => {
                            async move {
                                stub.immigrate_snd(
                                    batch,
                                    snd_layer,
                                    src_ranks0,
                                    dst_ranks0,
                                )
                                .await
                                .map(|res| res.batch)
                            }
                            .boxed()
                        }
                        _ => {
                            panic!("trans_partial_kv_cache fst_layer & snd_layer not mutually exclusive!");
                        }
                    }
                })),
        )
        .await
        .into_iter()
        .collect()
    }
}
