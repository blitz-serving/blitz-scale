use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    mem::take,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
};

use crate::{error::Result, queue::Entry, Stub};

use futures::{future::join_all, join};
use nohash_hasher::IntMap;
use pb::generate::v2::*;
use tokio::{spawn, task::JoinHandle};

pub(crate) mod config;
mod cybernetics;
mod disaggregation;
mod metrics;
mod relay_queue;
mod steersman;

pub use disaggregation::*;
pub use metrics::*;
pub use steersman::*;

pub static MOCK_TRANSFER_MILLIS: OnceLock<u64> = OnceLock::new();
pub static MOCK_LOAD_MILLIS: OnceLock<u64> = OnceLock::new();

/// This function will spawn a background task to transfer model parameters
/// from old instance to new instance.
///
/// After all parameters are transferred, the `model_ready` flag will be set to true.

#[allow(unused)]
async fn stub_filter_batch(
    stub: &mut Stub,
    next_batch: Option<CachedBatch>,
    mut generations: Vec<Generation>,
    entries: &IntMap<u64, Entry>,
) -> Option<(CachedBatch, Tokens)> {
    let mut batch = next_batch?;
    // No need to filter
    if batch.request_ids.len() == entries.len()
        && batch.request_ids.iter().cloned().collect::<BTreeSet<_>>()
            == entries.keys().cloned().collect::<BTreeSet<_>>()
    {
        let mut tokens = Tokens {
            ids: Vec::with_capacity(batch.size as usize),
            logprobs: Vec::with_capacity(batch.size as usize),
            texts: Vec::with_capacity(batch.size as usize),
            is_special: Vec::with_capacity(batch.size as usize),
        };
        generations.into_iter().for_each(|mut generation| {
            tokens.ids.push(generation.tokens.ids[0]);
            tokens.texts.push(take(&mut generation.tokens.texts[0]));
            // tokens.logprobs.push(generation.tokens.logprobs[0]);
            // tokens.is_special.push(generation.tokens.is_special[0]);
        });
        return Some((batch, tokens));
    } else {
        let mut indexed_request_id = batch
            .request_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<HashMap<_, _>>();

        // Retain only requests that are still in entries
        batch.request_ids.retain(|id| entries.contains_key(id));
        if batch.request_ids.is_empty() {
            // All requests have been filtered out and next batch is now empty.
            // Clear it from the Python shards cache
            stub.clear_cache(Some(batch.id)).await.unwrap();
            None
        } else {
            let batch = stub
                .filter_batch(batch.id, batch.request_ids)
                .await
                .unwrap();
            let mut tokens = Tokens {
                ids: Vec::with_capacity(batch.size as usize),
                logprobs: Vec::with_capacity(batch.size as usize),
                texts: Vec::with_capacity(batch.size as usize),
                is_special: Vec::with_capacity(batch.size as usize),
            };

            batch.request_ids.iter().for_each(|request_id| {
                let mut generation =
                    take(&mut generations[indexed_request_id.remove(request_id).unwrap()]);
                tokens.ids.push(generation.tokens.ids[0]);
                tokens.texts.push(take(&mut generation.tokens.texts[0]));
                // tokens.logprobs.push(generation.tokens.logprobs[0]);
                // tokens.is_special.push(generation.tokens.is_special[0]);
            });
            return Some((batch, tokens));
        }
    }
}

async fn stubs_filter_batch(
    stubs: &mut Vec<Stub>,
    next_batch: Option<CachedBatch>,
    mut generations: Vec<Generation>,
    entries: &IntMap<u64, Entry>,
) -> Option<(CachedBatch, Tokens)> {
    let mut batch = next_batch?;
    // No need to filter
    if batch.request_ids.len() == entries.len()
        && batch.request_ids.iter().cloned().collect::<BTreeSet<_>>()
            == entries.keys().cloned().collect::<BTreeSet<_>>()
    {
        tracing::debug!(
            "No need to filter: request ids {:?} entries {:?}",
            batch.request_ids,
            entries.keys()
        );
        let mut tokens = Tokens {
            ids: Vec::with_capacity(batch.size as usize),
            logprobs: Vec::with_capacity(batch.size as usize),
            texts: Vec::with_capacity(batch.size as usize),
            is_special: Vec::with_capacity(batch.size as usize),
        };
        generations.into_iter().for_each(|mut generation| {
            tokens.ids.push(generation.tokens.ids[0]);
            tokens.texts.push(take(&mut generation.tokens.texts[0]));
            // tokens.logprobs.push(generation.tokens.logprobs[0]);
            // tokens.is_special.push(generation.tokens.is_special[0]);
        });
        return Some((batch, tokens));
    } else {
        let mut indexed_request_id = batch
            .request_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<HashMap<_, _>>();

        // Retain only requests that are still in entries
        batch.request_ids.retain(|id| entries.contains_key(id));
        if batch.request_ids.is_empty() {
            // All requests have been filtered out and next batch is now empty.
            // Clear it from the Python shards cache
            tracing::debug!("Clear cache of batch {}", batch.id);
            join_all(
                stubs
                    .iter_mut()
                    .map(|stub| stub.clear_cache(Some(batch.id))),
            )
            .await
            .into_iter()
            .collect::<Result<VecDeque<_>>>()
            .unwrap();
            None
        } else {
            // tracing::info!("Retained requests {:?}", batch.request_ids);
            let batch = join_all(
                stubs
                    .iter_mut()
                    .map(|instance| instance.filter_batch(batch.id, batch.request_ids.clone())),
            )
            .await
            .into_iter()
            .collect::<Result<VecDeque<_>>>()
            .unwrap()
            .pop_front()
            .unwrap();
            let mut tokens = Tokens {
                ids: Vec::with_capacity(batch.size as usize),
                logprobs: Vec::with_capacity(batch.size as usize),
                texts: Vec::with_capacity(batch.size as usize),
                is_special: Vec::with_capacity(batch.size as usize),
            };

            batch.request_ids.iter().for_each(|request_id| {
                let mut generation =
                    take(&mut generations[indexed_request_id.remove(request_id).unwrap()]);
                if generation.tokens.ids.len() > 0 {
                    tokens.ids.push(generation.tokens.ids[0]);
                }
                if generation.tokens.texts.len() > 0 {
                    tokens.texts.push(take(&mut generation.tokens.texts[0]));
                }
                // tokens.logprobs.push(generation.tokens.logprobs[0]);
                // tokens.is_special.push(generation.tokens.is_special[0]);
            });
            return Some((batch, tokens));
        }
    }
}

async fn stubs_filter_batch_with_metric(
    stubs: &mut Vec<Stub>,
    replica_metric: Arc<ReplicaMetric>,
    next_batch: Option<CachedBatch>,
    mut generations: Vec<Generation>,
    entries: &IntMap<u64, Entry>,
) -> Option<(CachedBatch, Tokens)> {
    let mut batch = next_batch?;
    // No need to filter
    if batch.request_ids.len() == entries.len()
        && batch.request_ids.iter().cloned().collect::<BTreeSet<_>>()
            == entries.keys().cloned().collect::<BTreeSet<_>>()
    {
        tracing::debug!(
            "No need to filter: request ids {:?} entries {:?}",
            batch.request_ids,
            entries.keys()
        );
        let mut tokens = Tokens {
            ids: Vec::with_capacity(batch.size as usize),
            logprobs: Vec::with_capacity(batch.size as usize),
            texts: Vec::with_capacity(batch.size as usize),
            is_special: Vec::with_capacity(batch.size as usize),
        };
        generations.into_iter().for_each(|mut generation| {
            if generation.tokens.ids.len() > 0 {
                tokens.ids.push(generation.tokens.ids[0]);
            }
            if generation.tokens.texts.len() > 0 {
                tokens.texts.push(take(&mut generation.tokens.texts[0]));
            }
            // tokens.logprobs.push(generation.tokens.logprobs[0]);
            // tokens.is_special.push(generation.tokens.is_special[0]);
        });
        return Some((batch, tokens));
    } else {
        let mut indexed_request_id = batch
            .request_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<HashMap<_, _>>();

        // Retain only requests that are still in entries
        batch.request_ids.retain(|id| entries.contains_key(id));
        if batch.request_ids.is_empty() {
            // All requests have been filtered out and next batch is now empty.
            // Clear it from the Python shards cache
            tracing::debug!("Clear cache of batch {}", batch.id);
            join_all(
                stubs
                    .iter_mut()
                    .map(|stub| stub.clear_cache(Some(batch.id))),
            )
            .await
            .into_iter()
            .collect::<Result<VecDeque<_>>>()
            .unwrap();
            let block_cnt = batch.max_tokens / replica_metric.block_size;
            replica_metric.sub_used_blocks(block_cnt);
            // tracing::info!("Sub used blocks: {}", block_cnt);
            None
        } else {
            // tracing::info!("Retained requests {:?}", batch.request_ids);
            let new_batch = join_all(
                stubs
                    .iter_mut()
                    .map(|instance| instance.filter_batch(batch.id, batch.request_ids.clone())),
            )
            .await
            .into_iter()
            .collect::<Result<VecDeque<_>>>()
            .unwrap()
            .pop_front()
            .unwrap();
            let block_cnt = (batch.max_tokens - new_batch.max_tokens) / replica_metric.block_size;
            replica_metric.sub_used_blocks(block_cnt);
            // tracing::info!("Sub used blocks: {}", block_cnt);
            let mut tokens = Tokens {
                ids: Vec::with_capacity(new_batch.size as usize),
                logprobs: Vec::with_capacity(new_batch.size as usize),
                texts: Vec::with_capacity(new_batch.size as usize),
                is_special: Vec::with_capacity(new_batch.size as usize),
            };

            new_batch.request_ids.iter().for_each(|request_id| {
                let mut generation =
                    take(&mut generations[indexed_request_id.remove(request_id).unwrap()]);
                tokens.ids.push(generation.tokens.ids[0]);
                tokens.texts.push(take(&mut generation.tokens.texts[0]));
                // tokens.logprobs.push(generation.tokens.logprobs[0]);
                // tokens.is_special.push(generation.tokens.is_special[0]);
            });
            return Some((new_batch, tokens));
        }
    }
}
