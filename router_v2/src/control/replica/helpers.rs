use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicI64, AtomicU32, Ordering},
        Arc,
    },
};

use nohash_hasher::IntMap;
use pb::generate::v2::*;
use tokio::{
    spawn,
    sync::{oneshot, Mutex, OwnedMutexGuard},
    task::JoinHandle,
};

use super::globals::*;
use super::metrics::*;
use super::migration::MigrationQueue;
use super::stubs_filter_batch_with_metric;
use crate::{
    datapath::infer::{filter_send_generations, filter_send_generations_on_prefill_done},
    datapath::queue::Entry,
    Stub,
};

#[derive(Debug)]
pub struct MigratingBatch {
    pub(crate) batch: Batch,
    pub(crate) tokens: Tokens,
    pub(crate) entries: IntMap<u64, Entry>,
}

pub struct MigratingPartialBatch {
    pub(crate) batch: Batch,
    pub(crate) tokens: Option<Tokens>,
    pub(crate) entries: Option<IntMap<u64, Entry>>,
    pub(crate) layer: u32,
    pub(crate) fst2snd_tx: Option<oneshot::Sender<(usize, JoinHandle<OwnedMutexGuard<()>>)>>,
}

#[derive(Debug)]
pub struct MigratedBatch {
    pub(crate) batch: CachedBatch,
    pub(crate) tokens: Tokens,
    pub(crate) entries: IntMap<u64, Entry>,
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

pub fn batch_to_cached_batch(batch: Batch) -> CachedBatch {
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

pub fn restore_cached_batch(
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

pub fn create_refract_head_task(
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
    use std::collections::VecDeque;
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

pub async fn post_prefill_pre_decode(
    replica_metric: &Arc<ReplicaMetric>,
    migration_queue: &MigrationQueue,
    replica_index: usize,
    replica_stubs: &mut Vec<Stub>,
    unfinished_migration_count: &Arc<AtomicI64>,
    generations: Vec<Generation>,
    cached_batch: Option<CachedBatch>,
    new_batch: Batch,
    mut new_entries: std::collections::HashMap<
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
