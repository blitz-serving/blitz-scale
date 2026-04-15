use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    mem::take,
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc,
    },
};

use futures::{future::join_all, FutureExt};
use pb::generate::v2::*;
use tokio::{
    spawn,
    sync::{oneshot, Mutex, OwnedMutexGuard, RwLock},
    task::{yield_now, JoinHandle},
};

use super::globals::*;
use super::helpers::*;
use super::metrics::*;
use crate::{
    error::ClientError,
    datapath::queue::Entry,
    Stub,
};

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
