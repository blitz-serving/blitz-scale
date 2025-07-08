use super::metrics::*;
use nohash_hasher::IntMap;
use pb::generate::v2::*;
use serde::{Deserialize, Serialize};

use crate::queue::Entry;

/// Partial layer prefill batch
#[derive(Debug)]
pub(super) struct PartialPrefill {
    pub(super) request: Option<PrefillV2Request>,
    pub(super) zag_request: Option<ZagPrefillRequest>,
    pub(super) entries: IntMap<u64, Entry>,
}

/// managed P-D disaggregation cluster config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DisaggregationConfig {
    pub(crate) init_states: Vec<ReplicaState>,
    pub(crate) replicas: Vec<Vec<usize>>,
    pub(crate) machines: Vec<usize>,
}

#[cfg(test)]
#[test]
fn test_dump_disaggregation_config() {
    let config = DisaggregationConfig {
        init_states: vec![ReplicaState::Inactive, ReplicaState::Prefill],
        replicas: vec![vec![0, 1], vec![2, 3]],
        machines: vec![0, 0, 1, 1],
    };
    let serialized = serde_json::to_string(&config).unwrap();
    println!("{}", serialized);
}
