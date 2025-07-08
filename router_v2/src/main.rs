// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
//
// Modifications by Blitz-serving:
//   - Passed more args for disaggregation controller in blitzscale
use std::fs::File;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::Duration;

use axum::http::HeaderValue;
use clap::Parser;
use futures::future::join_all;
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use router_v2::error::ClientError;
use router_v2::{
    parse_deployment, server, statistic, ControllerArgs, Deployment, HubModelInfo, Model, Stub,
    MAX_BLOCKS_PER_REPLICA,
};
use thiserror::Error;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tokio::spawn;
use tower_http::cors::AllowOrigin;
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,
    #[clap(default_value = "1024", long, env)]
    max_input_length: usize,
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,
    #[clap(default_value = "1.2", long, env)]
    waiting_served_ratio: f32,
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(long)]
    client_config: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value_t = false, long)]
    use_tokenizer: bool,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
    #[clap(long, env)]
    ngrok: bool,
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,
    #[clap(long, env)]
    ngrok_edge: Option<String>,
    #[clap(long, env)]
    log_path: Option<String>,
    #[clap(long, env)]
    statistic_path: Option<String>,
    #[clap(default_value = "disaggregation", long, value_parser = parse_deployment)]
    deployment: Deployment,
    #[clap(required = true, long)]
    deployment_config_path: String,

    #[clap(long, default_value_t = 13000)]
    tokens_prefilled_per_sec: u32,
    #[clap(long, default_value_t = 20000)]
    tokens_transferred_per_sec: u32,
    #[clap(long, default_value_t = 8000)]
    max_blocks_per_replica: u32,

    #[clap(long, default_value_t = 8)]
    max_prefill_num: u32,
    #[clap(long, default_value_t = 8)]
    max_decode_num: u32,

    #[clap(long, default_value_t = 1)]
    min_prefill_num: u32,
    #[clap(long, default_value_t = 1)]
    min_decode_num: u32,

    #[clap(long, default_value_t = 0.15)]
    prefill_lower_bound: f32,
    #[clap(long, default_value_t = 0.4)]
    prefill_upper_bound: f32,
    #[clap(long, default_value_t = 0.45)]
    decode_lower_bound: f32,
    #[clap(long, default_value_t = 0.8)]
    decode_upper_bound: f32,

    #[clap(long, default_value_t = 0.0)]
    migration_lower_bound: f32,
    #[clap(long, default_value_t = 2.0)]
    migration_upper_bound: f32,

    #[clap(long, default_value_t = 1500)]
    scale_down_threshold_millis: u64,

    #[clap(long, default_value_t = 32)]
    num_hidden_layers: u32,

    #[clap(long, required = true)]
    model_name: String,

    #[clap(long, required = true)]
    model_path: String,

    #[clap(long, required = true)]
    parameter_size: f32,

    #[clap(long, default_value_t = 32)]
    num_gpus_per_node: usize,

    #[clap(long, default_value_t = 0)]
    mock_load_millis: u64,

    #[clap(long, default_value_t = 0)]
    mock_transfer_millis: u64,

    #[clap(long, default_value_t = 1)]
    tensor_parallel_size: usize,
}

fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        hostname,
        port,
        client_config,
        tokenizer_name,
        use_tokenizer,
        revision,
        validation_workers,
        json_output,
        otlp_endpoint,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        log_path,
        statistic_path,
        deployment,
        deployment_config_path,

        tokens_prefilled_per_sec,
        tokens_transferred_per_sec,

        max_blocks_per_replica,
        prefill_lower_bound,
        prefill_upper_bound,
        decode_lower_bound,
        decode_upper_bound,
        scale_down_threshold_millis,

        max_prefill_num,
        max_decode_num,
        min_prefill_num,
        min_decode_num,
        migration_lower_bound,
        migration_upper_bound,

        num_hidden_layers,
        num_gpus_per_node,
        mock_load_millis,
        mock_transfer_millis,
        tensor_parallel_size,

        model_name,
        model_path,
        parameter_size,
    } = args;

    // Validate args
    if max_input_length >= max_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_input_length` must be < `max_total_tokens`".to_string(),
        ));
    }
    if max_input_length as u32 > max_batch_prefill_tokens {
        return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be >= `max_input_length`. Given: {max_batch_prefill_tokens} and {max_input_length}")));
    }

    if validation_workers == 0 {
        return Err(RouterError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }

    if let Some(ref max_batch_total_tokens) = max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {max_batch_prefill_tokens} and {max_batch_total_tokens}")));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_batch_total_tokens}")));
        }
    }

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let cors_allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin
                .iter()
                .map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    // Parse Huggingface hub token
    let authorization_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();

    // Tokenizer instance
    // This will only be used to validate payloads
    let local_path = Path::new(&tokenizer_name);
    let local_model = local_path.exists() && local_path.is_dir();
    let tokenizer = if use_tokenizer {
        if local_model {
            // Load local tokenizer
            Tokenizer::from_file(local_path.join("tokenizer.json")).ok()
        } else {
            // Download and instantiate tokenizer
            // We need to download it outside of the Tokio runtime
            let params = FromPretrainedParameters {
                revision: revision.clone().unwrap_or("main".to_string()),
                auth_token: authorization_token.clone(),
                ..Default::default()
            };
            Tokenizer::from_pretrained(tokenizer_name.clone(), Some(params)).ok()
        }
    } else {
        None
    };

    let _guard = init_logging(otlp_endpoint, json_output, log_path);

    let server_future = async {
        if tokenizer.is_none() {
            tracing::warn!("Could not find a fast tokenizer implementation for {tokenizer_name}");
            tracing::warn!("Rust input length validation and truncation is disabled");
        }

        // Get Model info
        let model_info = match local_model {
            true => HubModelInfo {
                model_id: tokenizer_name.clone(),
                sha: None,
                pipeline_tag: None,
            },
            false => get_model_info(&tokenizer_name, revision, authorization_token)
                .await
                .unwrap_or_else(|| {
                    tracing::warn!("Could not retrieve model info from the Hugging Face hub.");
                    HubModelInfo {
                        model_id: tokenizer_name.to_string(),
                        sha: None,
                        pipeline_tag: None,
                    }
                }),
        };

        // if pipeline-tag == text-generation we default to return_full_text = true
        let compat_return_full_text = match &model_info.pipeline_tag {
            None => {
                tracing::warn!("no pipeline tag found for model {tokenizer_name}");
                false
            }
            Some(pipeline_tag) => pipeline_tag.as_str() == "text-generation",
        };

        // Read uris from client_config
        let mut buf = String::new();
        File::open(client_config)
            .unwrap()
            .read_to_string(&mut buf)
            .unwrap();

        let stubs = join_all(
            serde_json::from_str::<Vec<String>>(buf.as_str())
                .unwrap()
                .into_iter()
                .map(|uri| Stub::connect(uri)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<Stub>, ClientError>>()
        .map_err(|e| RouterError::Connection(e))?;

        // Get info from the shard
        tracing::warn!("The shard info is not set properly by the st-server");
        let shard_info = Default::default();

        let disaggregation_controller_args = ControllerArgs {
            tokens_prefilled_per_sec,
            tokens_transferred_per_sec,

            max_blocks_per_replica,
            prefill_lower_bound,
            prefill_upper_bound,
            decode_lower_bound,
            decode_upper_bound,
            scale_down_threshold_millis,

            max_prefill_num,
            max_decode_num,
            min_prefill_num,
            min_decode_num,
            migration_lower_bound,
            migration_upper_bound,

            num_hidden_layers,
            num_gpus_per_node,
            mock_transfer_millis,
            mock_load_millis,
            tensor_parallel_size,
        };

        assert!(
            *MAX_BLOCKS_PER_REPLICA.get_or_init(|| max_blocks_per_replica)
                == max_blocks_per_replica
        );

        let mut temp_vec = Vec::with_capacity(stubs.len());
        let map_fn = |index: usize, t: Option<u32>| {
            let max_supported_batch_total_tokens = match t {
                // Older models do not support automatic max-batch-total-tokens
                None => {
                    let max_batch_total_tokens = max_batch_total_tokens.unwrap_or(
                        16000.max((max_total_tokens as u32).max(max_batch_prefill_tokens)),
                    );
                    tracing::warn!("Model does not support automatic max batch total tokens");
                    max_batch_total_tokens
                }
                // Flash attention models return their max supported total tokens
                Some(max_supported_batch_total_tokens) => {
                    // Warn if user added his own max-batch-total-tokens as we will ignore it
                    if max_batch_total_tokens.is_some() {
                        tracing::warn!(
                            "`--max-batch-total-tokens` is deprecated for Flash \
                        Attention models."
                        );
                        tracing::warn!(
                            "Inferred max batch total tokens: {max_supported_batch_total_tokens}"
                        );
                    }
                    if max_total_tokens as u32 > max_supported_batch_total_tokens {
                        let err_msg = format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_supported_batch_total_tokens}");
                        return Err(RouterError::ArgumentValidation(err_msg));
                    }
                    max_supported_batch_total_tokens
                }
            };
            tracing::info!("Setting max batch total tokens to {max_supported_batch_total_tokens}");
            tracing::info!("Connected {}", index);
            temp_vec.push(max_supported_batch_total_tokens);
            Ok(())
        };

        let addr = match hostname.parse() {
            Ok(ip) => SocketAddr::new(ip, port),
            Err(_) => {
                tracing::warn!("Invalid hostname, defaulting to 0.0.0.0");
                SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port)
            }
        };

        // let max_supported_batch_total_tokens = temp_vec.into_iter().min().unwrap();
        let max_supported_batch_total_tokens = 16000;
        let model = Model {
            model_name,
            model_path,
            parameter_size,
        };

        let _ = spawn(statistic(statistic_path));

        // Run server
        server::run(
            model_info,
            shard_info,
            compat_return_full_text,
            max_concurrent_requests,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_supported_batch_total_tokens,
            max_waiting_tokens,
            stubs,
            deployment,
            deployment_config_path,
            tokenizer,
            validation_workers,
            addr,
            cors_allow_origin,
            ngrok,
            ngrok_authtoken,
            ngrok_edge,
            model,
            true,
            disaggregation_controller_args,
        )
        .await?;
        Ok(())
    };

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server_future)
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
fn init_logging(
    otlp_endpoint: Option<String>,
    json_output: bool,
    log_path: Option<String>,
) -> Option<WorkerGuard> {
    let mut layers = Vec::new();
    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let guard = match log_path {
        Some(path) => {
            let (non_blocking, guard) = NonBlocking::new(std::fs::File::create(path).unwrap());
            let fmt_layer = fmt_layer.with_ansi(false).with_writer(non_blocking);
            let fmt_layer = match json_output {
                true => fmt_layer.json().flatten_event(true).boxed(),
                false => fmt_layer.boxed(),
            };
            layers.push(fmt_layer);
            Some(guard)
        }
        None => {
            let fmt_layer = match json_output {
                true => fmt_layer.json().flatten_event(true).boxed(),
                false => fmt_layer.boxed(),
            };
            layers.push(fmt_layer);
            Option::None
        }
    };

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "blitz.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
    return guard;
}

/// get model info from the Huggingface Hub
pub async fn get_model_info(
    model_id: &str,
    revision: Option<String>,
    token: Option<String>,
) -> Option<HubModelInfo> {
    let revision = match revision {
        None => {
            tracing::warn!("`--revision` is not set");
            tracing::warn!("We strongly advise to set it to a known supported commit.");
            "main".to_string()
        }
        Some(revision) => revision,
    };

    let client = reqwest::Client::new();
    // Poor man's urlencode
    let revision = revision.replace('/', "%2F");
    let url = format!("https://huggingface.co/api/models/{model_id}/revision/{revision}");
    let mut builder = client.get(url).timeout(Duration::from_secs(5));
    if let Some(token) = token {
        builder = builder.bearer_auth(token);
    }

    let response = builder.send().await.ok()?;

    if response.status().is_success() {
        let hub_model_info: HubModelInfo =
            serde_json::from_str(&response.text().await.ok()?).ok()?;
        if let Some(sha) = &hub_model_info.sha {
            tracing::info!(
                "Serving revision {sha} of model {}",
                hub_model_info.model_id
            );
        }
        Some(hub_model_info)
    } else {
        None
    }
}

#[allow(unused)]
#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Unable to connect to the Python model shards: {0}")]
    Connection(ClientError),
    #[error("Unable to clear the Python model shards cache: {0}")]
    Cache(ClientError),
    #[error("Unable to get the Python model shards info: {0}")]
    Info(ClientError),
    #[error("Unable to warmup the Python model shards: {0}")]
    Warmup(ClientError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
    #[error("Axum webserver failed: {0}")]
    Axum(#[from] axum::BoxError),
}
