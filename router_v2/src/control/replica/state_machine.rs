use futures::future::join_all;
use nohash_hasher::IntMap;
use pb::generate::v2::*;
use std::{
    collections::VecDeque,
    mem::take,
    sync::{
        atomic::{AtomicI64, AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    spawn,
    sync::{Mutex, OwnedMutexGuard},
    task::{yield_now, JoinHandle},
    time::Instant,
};

use super::globals::*;
use super::helpers::*;
use super::metrics::*;
use super::migration::MigrationQueue;
use super::relay_queue::*;
use super::steersman::Steersman;
use super::stubs_filter_batch_with_metric;
use super::config::*;
use crate::{
    encode_prefill_case,
    error::ClientError,
    datapath::infer::{filter_send_generations, filter_send_generations_on_prefill_done},
    datapath::queue::{Entry, Queue},
    Stub,
};

pub(crate) async fn disaggregation_event_loop_inner(
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
