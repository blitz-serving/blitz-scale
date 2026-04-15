use super::config::*;
use tokio::sync::oneshot;

pub(super) struct AsyncQueueElement {
    pub(super) rank: Vec<usize>,
    pub(super) sender: oneshot::Sender<Option<PartialPrefill>>,
}

#[derive(Debug, Clone)]
pub(super) struct AsyncQueue {
    async_channel_sender: async_channel::Sender<AsyncQueueElement>,
    async_channel_receiver: async_channel::Receiver<AsyncQueueElement>,
}

impl AsyncQueue {
    pub(super) fn new() -> Self {
        let (async_channel_sender, async_channel_receiver) =
            async_channel::unbounded::<AsyncQueueElement>();

        Self {
            async_channel_sender,
            async_channel_receiver,
        }
    }

    pub(super) async fn append(&self, indices: Vec<usize>) -> Option<PartialPrefill> {
        // metrics::blitz
        let (sender, receiver) = oneshot::channel();
        self.async_channel_sender
            .send(AsyncQueueElement {
                rank: indices,
                sender,
            })
            .await
            .unwrap();
        receiver.await.unwrap()
    }

    pub(super) async fn recv(&self) -> (Vec<usize>, oneshot::Sender<Option<PartialPrefill>>) {
        // metrics::blitz
        let res = self.async_channel_receiver.recv().await.unwrap();
        (res.rank, res.sender)
    }

    pub(super) fn try_recv(&self) -> Option<(Vec<usize>, oneshot::Sender<Option<PartialPrefill>>)> {
        // metrics::blitz
        let ok = self.async_channel_receiver.try_recv().ok();
        if let Some(res) = ok {
            Some((res.rank, res.sender))
        } else {
            None
        }
    }
}
