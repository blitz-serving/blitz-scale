//! State transition definitions for formal verification.
//!
//! This module defines the complete set of valid state transitions for
//! replica state machines. The `next_state` function is a pure function
//! (no side effects) that can be used for model checking and property
//! verification.

use super::metrics::ReplicaState;

/// Events that can trigger a state transition in the replica state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateEvent {
    /// A new batch has been received for prefilling.
    BatchReceived,
    /// A batch has completed its decode iterations.
    BatchCompleted,
    /// A KV cache migration (P-to-D) has finished.
    MigrationDone,
    /// The autoscaler requests a scale-up (add replica).
    ScaleUpRequested,
    /// The autoscaler requests a scale-down (remove replica).
    ScaleDownRequested,
    /// Model parameters have been fully loaded on this replica.
    ModelLoaded,
    /// A mutation (Prefill -> Decode role change) has been requested.
    MutationRequested,
    /// All in-flight requests on a shutting-down replica have drained.
    DrainComplete,
    /// All pending migrations on a shutting-down prefill replica have flushed.
    FlushComplete,
    /// The overprovision timer has expired (scale-down cooldown).
    OverprovisionExpired,
}

/// Pure function: computes the next state without side effects.
///
/// Returns `Some(next)` for valid transitions, `None` for invalid ones.
/// This function enumerates ALL valid transitions observed in the
/// `disaggregation_event_loop_inner` state machine.
pub fn next_state(current: ReplicaState, event: StateEvent) -> Option<ReplicaState> {
    match (current, event) {
        // --- Scale-up paths ---
        (ReplicaState::Inactive, StateEvent::ScaleUpRequested) => {
            Some(ReplicaState::LoadingPrefill)
        }
        (ReplicaState::LoadingPrefill, StateEvent::ModelLoaded) => Some(ReplicaState::Prefill),
        (ReplicaState::LoadingDecode, StateEvent::ModelLoaded) => Some(ReplicaState::Decode),

        // --- ZigZag (NewPrefill / OldPrefill / Refractory) ---
        (ReplicaState::Prefill, StateEvent::ScaleUpRequested) => Some(ReplicaState::OldPrefill),
        (ReplicaState::Inactive, StateEvent::BatchReceived) => Some(ReplicaState::NewPrefill),
        (ReplicaState::NewPrefill, StateEvent::ModelLoaded) => {
            Some(ReplicaState::RefractoryPrefill)
        }
        (ReplicaState::RefractoryPrefill, StateEvent::FlushComplete) => {
            Some(ReplicaState::Prefill)
        }
        (ReplicaState::OldPrefill, StateEvent::FlushComplete) => Some(ReplicaState::Prefill),

        // --- Mutation (Prefill -> Decode) ---
        (ReplicaState::Prefill, StateEvent::MutationRequested) => {
            Some(ReplicaState::MutatingToDecode)
        }
        (ReplicaState::MutatingToDecode, StateEvent::MigrationDone) => {
            Some(ReplicaState::Decode)
        }

        // --- Auspicious states (planner-driven) ---
        (ReplicaState::AusPrefill, StateEvent::ModelLoaded) => Some(ReplicaState::Prefill),
        (ReplicaState::AusDecode, StateEvent::ModelLoaded) => Some(ReplicaState::Decode),

        // --- Scale-down paths ---
        (ReplicaState::Prefill, StateEvent::ScaleDownRequested) => {
            Some(ReplicaState::ShuttingPrefill)
        }
        (ReplicaState::Decode, StateEvent::ScaleDownRequested) => {
            Some(ReplicaState::ShuttingDecode)
        }
        (ReplicaState::ShuttingPrefill, StateEvent::FlushComplete) => {
            Some(ReplicaState::ShuttingNull)
        }
        (ReplicaState::ShuttingDecode, StateEvent::DrainComplete) => {
            Some(ReplicaState::ShuttingNull)
        }
        (ReplicaState::ShuttingNull, StateEvent::ScaleUpRequested) => {
            Some(ReplicaState::LoadingPrefill)
        }

        // --- Planner marker states (transfer in progress) ---
        (ReplicaState::RdmaSending, StateEvent::MigrationDone) => Some(ReplicaState::Decode),
        (ReplicaState::RdmaLoading, StateEvent::ModelLoaded) => Some(ReplicaState::Prefill),
        (ReplicaState::NvlinkSending, StateEvent::MigrationDone) => Some(ReplicaState::Decode),
        (ReplicaState::NvlCasting, StateEvent::ModelLoaded) => Some(ReplicaState::Decode),
        (ReplicaState::RdmaCasting, StateEvent::ModelLoaded) => Some(ReplicaState::Prefill),
        (ReplicaState::TanzCasting, StateEvent::ModelLoaded) => Some(ReplicaState::Prefill),

        // All other combinations are invalid transitions
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inactive_to_loading_prefill() {
        assert_eq!(
            next_state(ReplicaState::Inactive, StateEvent::ScaleUpRequested),
            Some(ReplicaState::LoadingPrefill)
        );
    }

    #[test]
    fn test_invalid_transition_returns_none() {
        assert_eq!(
            next_state(ReplicaState::Decode, StateEvent::ModelLoaded),
            None
        );
    }

    #[test]
    fn test_prefill_to_shutting_prefill() {
        assert_eq!(
            next_state(ReplicaState::Prefill, StateEvent::ScaleDownRequested),
            Some(ReplicaState::ShuttingPrefill)
        );
    }

    #[test]
    fn test_mutating_to_decode() {
        assert_eq!(
            next_state(ReplicaState::MutatingToDecode, StateEvent::MigrationDone),
            Some(ReplicaState::Decode)
        );
    }
}
