use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;

use std::cmp::min;
use std::collections::VecDeque;
use std::time::Duration;

use nohash_hasher::{BuildNoHashHasher, IntMap};
use pb::generate::v2::*;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,

    pub prev_token_time: Option<Instant>,
    pub max_time_between_tokens: Duration,
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new(
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
        speculate: u32,
    ) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(queue_task(
            requires_padding,
            block_size,
            window_size,
            speculate,
            queue_receiver,
        ));

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    /// Get the number of waiting prefill tokens
    #[instrument(skip_all)]
    pub(crate) async fn waiting_prefill_tokens(&self) -> u32 {
        let (tx, rx) = oneshot::channel();
        self.queue_sender
            .send(QueueCommand::WaitingPrefillTokens(tx))
            .unwrap();
        rx.await.unwrap()
    }

    // Get the next batch
    #[instrument(skip_all)]
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        block_budget: Option<u32>,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                block_budget,
                response_sender,
                span: Span::current(),
            })
            .unwrap();
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.unwrap()
    }
}

// Background task responsible of the queue state
async fn queue_task(
    requires_padding: bool,
    block_size: u32,
    window_size: Option<u32>,
    speculate: u32,
    mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    let mut state = State::new(requires_padding, block_size, window_size, speculate);

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append(entry, span) => {
                span.in_scope(|| state.append(*entry));
                metrics::increment_gauge!("blitz_queue_size", 1.0);
            }
            QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                block_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch =
                    state.next_batch(min_size, prefill_token_budget, token_budget, block_budget);
                response_sender.send(next_batch).unwrap();
                metrics::gauge!("blitz_queue_size", state.entries.len() as f64);
            }),
            QueueCommand::WaitingPrefillTokens(sender) => {
                sender.send(state.waiting_prefill_tokens()).unwrap();
            }
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,

    /// Speculation amount
    speculate: u32,

    /// Waiting prefill tokens in the queue
    waiting_prefill_tokens: u32,
}

impl State {
    fn new(
        requires_padding: bool,
        block_size: u32,
        window_size: Option<u32>,
        speculate: u32,
    ) -> Self {
        Self {
            entries: VecDeque::with_capacity(256),

            next_batch_id: 0,
            requires_padding,
            block_size,
            window_size,
            speculate,
            waiting_prefill_tokens: 0,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Update waiting prefill tokens
        self.waiting_prefill_tokens += entry.request.input_length;

        // Push entry in the queue
        self.entries.push_back((entry.request.request_id, entry));
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        block_budget: Option<u32>,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        let mut prefill_tokens: u32 = 0;
        let mut max_tokens = 0;

        let num_all_enties = self.entries.len() as u32;
        let mut consumed_entires: u32 = 0;

        // Pop entries starting from the front of the queue
        while let Some((id, mut entry)) = self.entries.pop_front() {
            consumed_entires += 1;
            let entry_prefill_tokens;
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_closed() {
                metrics::increment_counter!("blitz_request_failure", "err" => "dropped");
                continue;
            }

            let max_request_tokens;
            if self.requires_padding {
                unimplemented!("Padding is unsupported now!");
            } else {
                // pad to block size
                entry_prefill_tokens = entry.request.input_length;
                prefill_tokens += entry_prefill_tokens;

                let max_new_tokens = match self.window_size {
                    None => entry.request.stopping_parameters.max_new_tokens,
                    Some(window_size) => min(
                        window_size.saturating_sub(entry.request.input_length),
                        entry.request.stopping_parameters.max_new_tokens,
                    ),
                };
                // pad to block size
                max_request_tokens =
                    (entry.request.input_length + max_new_tokens + self.block_size - 1)
                        / self.block_size
                        * self.block_size;
            }

            if let Some(block_budget) = block_budget {
                if max_request_tokens + max_tokens > block_budget * self.block_size {
                    self.entries.push_front((id, entry));
                    break;
                }
            }

            if prefill_tokens > prefill_token_budget
                || (max_request_tokens + self.speculate) > token_budget
            {
                // Entry is over budget
                // Add it back to the front
                self.entries.push_front((id, entry));
                break;
            }

            assert!(
                self.waiting_prefill_tokens >= entry_prefill_tokens,
                "{} {}",
                self.waiting_prefill_tokens,
                entry_prefill_tokens
            );
            self.waiting_prefill_tokens -= entry_prefill_tokens;
            max_tokens += max_request_tokens;

            tracing::debug!(
                "input length {} max new tokens {}",
                entry.request.input_length,
                entry.request.stopping_parameters.max_new_tokens
            );

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                inputs: entry.request.inputs.clone(),
                truncate: Some(entry.request.truncate),
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: entry.request.stopping_parameters.clone(),
                top_n_tokens: entry.request.top_n_tokens,
                input_tokens: entry.request.input_tokens.clone(),
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Empty batch
        if batch_requests.is_empty() {
            return None;
        }

        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            // Batch is too small
            if batch_requests.len() < min_size {
                // Add back entries to the queue in the correct order
                for r in batch_requests.into_iter().rev() {
                    let id = r.id;
                    let entry = batch_entries.remove(&id).unwrap();
                    self.entries.push_front((id, entry));
                }

                return None;
            }
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens,
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("blitz_batch_next_size", batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }

    fn waiting_prefill_tokens(&self) -> u32 {
        self.waiting_prefill_tokens
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        block_budget: Option<u32>,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
    WaitingPrefillTokens(oneshot::Sender<u32>),
}

#[cfg(test)]
mod tests {
    use super::*;

    use tracing::info_span;

    fn default_entry() -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, receiver_tx) = mpsc::unbounded_channel();

        let entry = Entry {
            request: ValidGenerateRequest {
                request_id: 0,
                inputs: "".to_string(),
                input_length: 0,
                truncate: 0,
                decoder_input_details: false,
                parameters: NextTokenChooserParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    typical_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                    watermark: false,
                },
                stopping_parameters: StoppingCriteriaParameters {
                    ignore_eos_token: false,
                    max_new_tokens: 1,
                    stop_sequences: vec![],
                },
                top_n_tokens: 0,
                input_tokens: vec![0],
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            prev_token_time: None,
            max_time_between_tokens: Duration::from_micros(0),
        };
        (entry, receiver_tx)
    }

    #[test]
    fn test_append() {
        let mut state = State::new(false, 1, None, 0);
        let (entry, _guard) = default_entry();

        assert_eq!(state.entries.len(), 0);

        state.append(entry);

        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_next_batch_empty() {
        let mut state = State::new(false, 1, None, 0);

        assert!(state.next_batch(None, 1, 1, None).is_none());
        assert!(state.next_batch(Some(1), 1, 1, None).is_none());
    }

    #[test]
    fn test_next_batch_min_size() {
        let mut state = State::new(false, 1, None, 0);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 2, 2, None).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        assert!(state.next_batch(Some(2), 2, 2, None).is_none());

        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 2);
    }

    #[test]
    fn test_next_batch_token_budget() {
        let mut state = State::new(false, 1, None, 0);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 1, 1, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        let (entries, batch, _) = state.next_batch(None, 3, 3, None).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);

        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 2);
    }

    #[tokio::test]
    async fn test_queue_append() {
        let queue = Queue::new(false, 1, None, 0);
        let (entry, _guard) = default_entry();
        queue.append(entry);
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = Queue::new(false, 1, None, 0);

        assert!(queue.next_batch(None, 1, 1, None).await.is_none());
        assert!(queue.next_batch(Some(1), 1, 1, None).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = Queue::new(false, 1, None, 0);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 2, 2, None).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        // Not enough requests pending
        assert!(queue.next_batch(Some(2), 2, 2, None).await.is_none());
        // Not enough token budget
        assert!(queue.next_batch(Some(1), 0, 0, None).await.is_none());
        // Ok
        let (entries2, batch2, _) = queue.next_batch(Some(1), 2, 2, None).await.unwrap();
        assert_eq!(entries2.len(), 1);
        assert!(entries2.contains_key(&2));
        assert!(entries2.get(&2).unwrap().batch_time.is_some());
        assert_eq!(batch2.id, 1);
        assert_eq!(batch2.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_budget() {
        let queue = Queue::new(false, 1, None, 0);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 1, 1, None).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        let (entries, batch, _) = queue.next_batch(None, 3, 3, None).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_speculate() {
        let queue = Queue::new(false, 1, None, 2);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        // Budget of 1 is not enough
        assert!(queue.next_batch(None, 1, 1, None).await.is_none());

        let (entries, batch, _) = queue.next_batch(None, 6, 6, None).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_dropped_receiver() {
        let queue = Queue::new(false, 1, None, 0);
        let (entry, _) = default_entry();
        queue.append(entry);

        assert!(queue.next_batch(None, 1, 1, None).await.is_none());
    }
}
