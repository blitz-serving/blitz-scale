use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    time,
};

use crate::KV_BLOCK_SIZE;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::instrument;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub(crate) enum ReplicaState {
    /// Inactive replica, w/ dst_mtx
    Inactive,
    /// Normal prefill replica
    Prefill,
    /// Zigzag::New, prefill rep. for fst half
    NewPrefill,
    /// Zigzag::Old, prefill rep. for snd half
    OldPrefill,
    /// Zigzag::New => Normal transient st., w/ model loaded
    RefractoryPrefill,
    /// Normal Decode replica.
    Decode,
    /// Prefill => Null transient st., waiting unfinished kV$ migration
    ShuttingPrefill,
    /// Decode => Null transient st., waiting unfinished decoding req.
    ShuttingDecode,
    /// Shutted down w/o unfinished job left, w/ dst_mtx
    ShuttingNull,
    /// Prefill => Decode emplace, w/o dst_mtx
    MutatingToDecode,
    /// eligible to Prefill, weakest premise
    AusPrefill,
    /// elgible to Decode, weakest premise
    AusDecode,
    /// Marker state for Worker, another coroutine eventually modify this state to Prefill
    LoadingPrefill,
    /// Marker state for Worker, another coroutine eventually modify this state to Decode
    LoadingDecode,
    /// Marker state for Planner
    RdmaSending,
    /// Marker state for Planner
    RdmaLoading,
    /// Marker state for Planner
    NvlinkSending,
    /// NVLink dst st., => Decode when transfer is done
    NvlCasting,
    /// RDMA BCast dst st., => Prefill when transfer is done
    RdmaCasting,
    /// Tanz BCast dst st., ranks join the Tänze
    TanzCasting,
}

macro_rules! match_trans {
    ($value:expr, $($pattern:pat => $result:expr),*) => {
        match $value {
            $($pattern => $result,)*
        }
    };
}

/// Logic of Transition System
///
/// aRb := a -> b
/// reflexitivity := aRa
// impl PartialOrd for ReplicaState {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         use ReplicaState::*;
//         use std::cmp::Ordering;

//         // reflexitivity
//         match match_trans!((self, other),
//             (Inactive, Inactive) | (Prefill, Prefill) | (Decode, Decode) => Some(Ordering::Equal),
//             (Inactive, LoadingDecode) | (Inactive, LoadingPrefill) | (Inactive, RdmaCastingPrefill {..}) | (Inactive, RdmaCastingDecode{..}) | (Inactive, NvlCastingDecode{..}) | (Inactive, NvlCastingPrefill{..}) | (Inactive, WalzerCastingNull{..}) => Some(Ordering::Less),
//             (Prefill, NewPrefill) | (Prefill, OldPrefill) | (Prefill, MutatingToDecode) | (Prefill, ShuttingPrefill) => Some(Ordering::Less),
//             (Decode, )
//         )
//     }
// }

#[derive(Debug)]
pub(crate) struct SystemMetric {
    pub(crate) prefill_tokens: AtomicUsize,
    pub(crate) decode_tokens: AtomicUsize,
    pub(crate) loop_counts: Vec<AtomicUsize>,
    pub(crate) token_in_queue: AtomicUsize,
}

impl SystemMetric {
    pub(crate) fn new() -> Self {
        Self {
            prefill_tokens: AtomicUsize::new(0),
            decode_tokens: AtomicUsize::new(0),
            loop_counts: (0..32).map(|_| AtomicUsize::new(0)).collect(),
            token_in_queue: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Flow {
    pub(crate) flow_out: HashMap<usize, AtomicUsize>,
    pub(crate) flow_in: HashMap<usize, AtomicUsize>,
}

impl Flow {
    pub(crate) fn new() -> Self {
        Self {
            flow_out: HashMap::new(),
            flow_in: HashMap::new(),
        }
    }

    pub(crate) fn append_token(&mut self, replica_index: usize, token_num: usize) {
        // tracing::info!("Migrating token num {}", token_num);
        self.flow_out
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add((token_num as f32 * 0.5) as usize, Ordering::AcqRel);
    }

    pub(crate) fn recv_token(&mut self, replica_index: usize, token_num: usize) {
        self.flow_in
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add((token_num as f32 * 0.5) as usize, Ordering::AcqRel);
    }

    pub(crate) fn append_param(&mut self, replica_index: usize, param_size_in_gb: usize) {
        self.flow_out
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(param_size_in_gb * 1024, Ordering::AcqRel);
    }

    pub(crate) fn recv_param(&mut self, replica_index: usize, param_size_in_gb: usize) {
        self.flow_in
            .entry(replica_index)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(param_size_in_gb * 1024, Ordering::AcqRel);
    }

    pub(crate) fn get_all(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut ret_flow_out = Vec::new();
        let mut ret_flow_in = Vec::new();
        for (&replica_index, flow) in self.flow_out.iter() {
            let flow = flow.load(Ordering::Acquire);
            ret_flow_out.append(&mut vec![(replica_index, flow)]);
        }
        for (&replica_index, flow) in self.flow_in.iter() {
            let flow = flow.load(Ordering::Acquire);
            ret_flow_in.append(&mut vec![(replica_index, flow)]);
        }
        (ret_flow_out, ret_flow_in)
    }

    pub(crate) fn clear(&mut self) {
        self.flow_out.clear();
        self.flow_in.clear();
    }
}

#[derive(Debug)]
pub(crate) struct ReplicaMetric {
    pub(crate) block_size: u32,
    used_blocks: AtomicU32,
    model_loaded: AtomicBool,

    #[allow(unused)]
    replica_index: usize,
    /// lock() <-> act as decode
    pub(crate) dst_mutex: Arc<Mutex<()>>,
    /// replica state for event loop
    pub(crate) state: RwLock<ReplicaState>,
    /// ongoing Zigzag partial layer migration tasks.
    pub(crate) flying_partial_migration_batches: Arc<AtomicI64>,
}

impl ReplicaMetric {
    pub(crate) fn new(
        model_loaded: bool,
        replica_index: usize,
        state: RwLock<ReplicaState>,
        dst_mutex: Arc<Mutex<()>>,
    ) -> Self {
        Self {
            block_size: KV_BLOCK_SIZE,
            used_blocks: AtomicU32::new(0),
            model_loaded: AtomicBool::new(model_loaded),
            replica_index,
            dst_mutex,
            state,
            flying_partial_migration_batches: Arc::new(AtomicI64::new(0)),
        }
    }

    pub(crate) fn set_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.store(used_blocks, Ordering::Release);
    }

    pub(crate) fn get_used_blocks(&self) -> u32 {
        self.used_blocks.load(Ordering::Acquire)
    }

    #[instrument(skip_all)]
    pub(crate) fn add_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.fetch_add(used_blocks, Ordering::AcqRel);
        // tracing::info!(
        //     "Add used blocks: {}",
        //     used_blocks,
        // );
    }

    #[instrument(skip_all)]
    pub(crate) fn sub_used_blocks(&self, used_blocks: u32) {
        self.used_blocks.fetch_sub(used_blocks, Ordering::AcqRel);
        // tracing::info!(
        //     "Sub used blocks: {}",
        //     used_blocks,
        // );
    }

    pub(crate) fn set_model_loaded(&self, model_loaded: bool) {
        self.model_loaded.store(model_loaded, Ordering::Release);
    }

    pub(crate) fn is_model_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::Acquire)
    }

    pub(crate) fn add_partial_migration_cnt(&self) {
        self.flying_partial_migration_batches
            .fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn sub_partial_migration_cnt(&self) {
        self.flying_partial_migration_batches
            .fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn get_partial_migration_cnt(&self) -> i64 {
        self.flying_partial_migration_batches
            .load(Ordering::Acquire)
    }
}
