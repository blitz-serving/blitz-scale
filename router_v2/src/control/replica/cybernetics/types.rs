//! Pure types and functions for cybernetics scaling decisions.
//!
//! This module isolates the decision logic from side-effecting execution,
//! enabling unit testing and future formal verification of the autoscaler.

use super::super::controller::ControllerArgs;
use super::super::metrics::ReplicaState;

/// A snapshot of the system state at a point in time.
///
/// Contains all the information the planner needs to make a scaling decision,
/// without any mutable references or async operations.
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// Per-replica: (replica_index, state, used_blocks, model_loaded)
    pub replicas: Vec<ReplicaSnapshot>,
    /// Current prefill token throughput (tokens/sec)
    pub prefill_token_rate: f32,
    /// Current decode token throughput (tokens/sec)
    pub decode_token_rate: f32,
    /// Number of tokens waiting in the migration queue
    pub tokens_in_queue: usize,
    /// Number of tokens waiting in the batching queue for prefill
    pub waiting_prefill_tokens: usize,
}

/// Snapshot of a single replica's state.
#[derive(Debug, Clone)]
pub struct ReplicaSnapshot {
    pub replica_index: usize,
    pub state: ReplicaState,
    pub used_blocks: u32,
    pub max_blocks: u32,
    pub model_loaded: bool,
}

/// The scaling decision produced by the planner.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleDecision {
    /// No action needed; system is balanced.
    NoOp,
    /// Scale up a new prefill replica.
    ScaleUpPrefill {
        /// Index of the inactive replica to activate.
        target_replica: usize,
    },
    /// Scale up a new decode replica.
    ScaleUpDecode {
        /// Index of the inactive replica to activate.
        target_replica: usize,
    },
    /// Scale down a replica.
    ScaleDown {
        /// Index of the replica to deactivate.
        target_replica: usize,
    },
    /// Mutate a prefill replica into a decode replica.
    Mutate {
        /// Index of the prefill replica to mutate.
        target_replica: usize,
    },
}

/// Pure function: given a system snapshot and controller configuration,
/// compute the optimal scaling decision.
///
/// This function has NO side effects and can be tested deterministically.
pub fn plan_scaling(snapshot: &SystemSnapshot, config: &ControllerArgs) -> ScaleDecision {
    let num_prefill = snapshot
        .replicas
        .iter()
        .filter(|r| matches!(r.state, ReplicaState::Prefill | ReplicaState::NewPrefill | ReplicaState::OldPrefill | ReplicaState::RefractoryPrefill))
        .count() as u32;

    let num_decode = snapshot
        .replicas
        .iter()
        .filter(|r| matches!(r.state, ReplicaState::Decode))
        .count() as u32;

    let num_inactive = snapshot
        .replicas
        .iter()
        .filter(|r| matches!(r.state, ReplicaState::Inactive | ReplicaState::ShuttingNull))
        .count();

    // Check if decode replicas are overloaded
    let avg_decode_usage: f32 = if num_decode > 0 {
        let total_usage: f32 = snapshot
            .replicas
            .iter()
            .filter(|r| matches!(r.state, ReplicaState::Decode))
            .map(|r| r.used_blocks as f32 / r.max_blocks as f32)
            .sum();
        total_usage / num_decode as f32
    } else {
        0.0
    };

    // Check if prefill is underprovisioned
    let prefill_pressure = if snapshot.waiting_prefill_tokens > 0 {
        snapshot.waiting_prefill_tokens as f32
            / (config.tokens_prefilled_per_sec as f32 * num_prefill.max(1) as f32)
    } else {
        0.0
    };

    // Decision logic
    if avg_decode_usage > config.decode_upper_bound && num_prefill > config.min_prefill_num {
        // Decode is overloaded and we have spare prefill capacity: mutate
        if let Some(candidate) = snapshot
            .replicas
            .iter()
            .find(|r| matches!(r.state, ReplicaState::Prefill))
        {
            return ScaleDecision::Mutate {
                target_replica: candidate.replica_index,
            };
        }
    }

    if avg_decode_usage > config.decode_upper_bound
        && num_inactive > 0
        && num_decode < config.max_decode_num
    {
        // Scale up a decode replica
        if let Some(candidate) = snapshot
            .replicas
            .iter()
            .find(|r| matches!(r.state, ReplicaState::Inactive | ReplicaState::ShuttingNull))
        {
            return ScaleDecision::ScaleUpDecode {
                target_replica: candidate.replica_index,
            };
        }
    }

    if prefill_pressure > config.prefill_upper_bound
        && num_inactive > 0
        && num_prefill < config.max_prefill_num
    {
        // Scale up a prefill replica
        if let Some(candidate) = snapshot
            .replicas
            .iter()
            .find(|r| matches!(r.state, ReplicaState::Inactive | ReplicaState::ShuttingNull))
        {
            return ScaleDecision::ScaleUpPrefill {
                target_replica: candidate.replica_index,
            };
        }
    }

    if avg_decode_usage < config.decode_lower_bound && num_decode > config.min_decode_num {
        // Scale down an underutilized decode replica
        if let Some(candidate) = snapshot
            .replicas
            .iter()
            .filter(|r| matches!(r.state, ReplicaState::Decode))
            .min_by_key(|r| r.used_blocks)
        {
            return ScaleDecision::ScaleDown {
                target_replica: candidate.replica_index,
            };
        }
    }

    ScaleDecision::NoOp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ControllerArgs {
        ControllerArgs {
            tokens_prefilled_per_sec: 13000,
            tokens_transferred_per_sec: 20000,
            max_blocks_per_replica: 8000,
            max_prefill_num: 4,
            max_decode_num: 4,
            min_prefill_num: 1,
            min_decode_num: 1,
            prefill_lower_bound: 0.15,
            prefill_upper_bound: 0.4,
            decode_lower_bound: 0.45,
            decode_upper_bound: 0.8,
            migration_lower_bound: 0.0,
            migration_upper_bound: 2.0,
            scale_down_threshold_millis: 1500,
            num_hidden_layers: 32,
            num_gpus_per_node: 32,
            mock_load_millis: 0,
            mock_transfer_millis: 0,
            tensor_parallel_size: 1,
        }
    }

    #[test]
    fn test_noop_when_balanced() {
        let snapshot = SystemSnapshot {
            replicas: vec![
                ReplicaSnapshot {
                    replica_index: 0,
                    state: ReplicaState::Prefill,
                    used_blocks: 1000,
                    max_blocks: 8000,
                    model_loaded: true,
                },
                ReplicaSnapshot {
                    replica_index: 1,
                    state: ReplicaState::Decode,
                    used_blocks: 4000,
                    max_blocks: 8000,
                    model_loaded: true,
                },
            ],
            prefill_token_rate: 5000.0,
            decode_token_rate: 3000.0,
            tokens_in_queue: 0,
            waiting_prefill_tokens: 0,
        };
        assert_eq!(plan_scaling(&snapshot, &make_config()), ScaleDecision::NoOp);
    }
}
