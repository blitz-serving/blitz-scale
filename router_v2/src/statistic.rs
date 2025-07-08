use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::AsyncWriteExt;

static PREFILL_TOKENS: AtomicUsize = AtomicUsize::new(0);
static DECODE_TOKENS: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn increase_prefill_tokens(count: usize) {
    PREFILL_TOKENS.fetch_add(count, Ordering::SeqCst);
}

pub(crate) fn get_prefill_tokens() -> usize {
    PREFILL_TOKENS.load(Ordering::SeqCst)
}

#[allow(dead_code)]
pub(crate) fn increase_decode_tokens(count: usize) {
    DECODE_TOKENS.fetch_add(count, Ordering::SeqCst);
}

pub(crate) fn get_decode_tokens() -> usize {
    DECODE_TOKENS.load(Ordering::SeqCst)
}

pub async fn statistic(statistic_path: Option<String>) {
    if statistic_path.is_none() {
        tracing::warn!("Statistic path is not set");
        return;
    }
    let mut file = tokio::fs::File::create(statistic_path.unwrap())
        .await
        .unwrap();

    let mut round = 0;
    let mut previous_prefill_tokens = 0;
    let mut previous_decode_tokens = 0;

    const MS: u64 = 500;

    file.write_all("epoch,prefill_tokens,decode_tokens,time\n".as_bytes())
        .await
        .unwrap();
    loop {
        round += 1;
        tokio::time::sleep(Duration::from_millis(MS)).await;
        let prefill_tokens = get_prefill_tokens();
        let decode_tokens = get_decode_tokens();
        let content = format!(
            "{},{},{},{}\n",
            round,
            prefill_tokens - previous_prefill_tokens,
            decode_tokens - previous_decode_tokens,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        );
        previous_prefill_tokens = prefill_tokens;
        previous_decode_tokens = decode_tokens;
        file.write_all(content.as_bytes()).await.unwrap();
    }
}
