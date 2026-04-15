use std::sync::{
    atomic::{AtomicBool, AtomicI64, AtomicU64},
    Arc, LazyLock, OnceLock,
};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use super::metrics::*;
use super::relay_queue::*;

pub static MAX_BLOCKS_PER_REPLICA: OnceLock<u32> = OnceLock::new();
pub static INIT_ZIGZAG_REQ_LIMIT: isize = 4;
// decrease request limit by 1 per 100ms
pub static DEC_ZIGZAG_REQ_LIMIT_RT: u128 = 200;

pub const HOST_CACHE_TIME_TO_LIVE_IN_MS: u128 = 30000;

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

pub static KV_BLOCK_SIZE: u32 = 64;
pub static SYSTEM_METRIC: LazyLock<Arc<SystemMetric>> =
    LazyLock::new(|| Arc::new(SystemMetric::new()));

// TODO: should as disaggregation_event_loop param in the future
pub static RELAY_ASYNC_QUEUE: LazyLock<Arc<AsyncQueue>> =
    LazyLock::new(|| Arc::new(AsyncQueue::new()));
pub static FLOW_WATCHER: LazyLock<Arc<Mutex<Flow>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Flow::new())));
pub static PLANNER_BUSY: AtomicBool = AtomicBool::new(false);
pub static ZIGZAG_ACTIVE_CNT: AtomicU64 = AtomicU64::new(0);
pub static RELAY_DEACTIVE_CNT: AtomicI64 = AtomicI64::new(0);

/// someone can't get decode mutex -> disable expriring decode
pub static SCARCE_DECODE: AtomicBool = AtomicBool::new(false);
/// \sync patch up to pass decode mutex guard
pub static D_MTX_NOTIFY: OnceLock<Vec<AtomicBool>> = OnceLock::new();
/// \debug alert to locate Cybernetics improgressiveness
pub static CTRL_LOOP_CNT: AtomicU64 = AtomicU64::new(0);
