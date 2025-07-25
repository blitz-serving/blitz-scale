// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// This file is a **modified** version of
// text-generation-inference/src/token_stream.rs
// © 2022-present Hugging Face Inc. – Apache-2.0.
//
// Modifications by Blitz-serving:
//   - Add Manually scale interface
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::health::Health;
use crate::infer::{InferError, InferResponse, InferStreamResponse};
use crate::validation::ValidationError;
use crate::{
    BestOfSequence, CompatGenerateRequest, ControllerArgs, Deployment, Details, ErrorResponse,
    FinishReason, GenerateParameters, GenerateRequest, GenerateResponse, HubModelInfo, Infer, Info,
    Model, PrefillToken, StreamDetails, StreamResponse, Stub, Token, Validation,
};

use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::stream::StreamExt;
use futures::Stream;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use pb::generate::v2::InfoResponse;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::time::Instant;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info_span, instrument, Instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Generate tokens if `stream == false` or a stream of token if `stream == true`
#[utoipa::path(
post,
tag = "Blitz",
path = "/",
request_body = CompatGenerateRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = GenerateResponse),
("text/event-stream" = StreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip_all)]
async fn compat_generate(
    Extension(default_return_full_text): Extension<bool>,
    infer: Extension<Infer>,
    Json(mut req): Json<CompatGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // default return_full_text given the pipeline_tag
    if req.parameters.return_full_text.is_none() {
        req.parameters.return_full_text = Some(default_return_full_text)
    }

    // switch on stream
    if req.stream {
        Ok(generate_stream(infer, Json(req.into()))
            .await
            .into_response())
    } else {
        let (headers, Json(generation)) = generate(infer, Json(req.into())).await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(vec![generation])).into_response())
    }
}

/// Blitz endpoint info
#[utoipa::path(
get,
tag = "Blitz",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "Blitz",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "Blitz is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
#[instrument(skip_all)]
/// Health check method
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match health.check().await {
        true => Ok(()),
        false => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
                error_type: "healthcheck".to_string(),
            }),
        )),
    }
}

/// Generate tokens
#[utoipa::path(
post,
tag = "Blitz",
path = "/generate",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = GenerateResponse),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
// #[instrument(
// skip_all,
// fields(
// parameters = ? req.parameters,
// total_time,
// validation_time,
// queue_time,
// inference_time,
// time_per_token,
// seed,
// )
// )]
#[instrument(skip_all)]
async fn generate(
    infer: Extension<Infer>,
    Json(req): Json<GenerateRequest>,
) -> Result<(HeaderMap, Json<GenerateResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("blitz_request_count");

    tracing::debug!("Input: {}", req.inputs);

    let compute_characters = req.inputs.chars().count();
    let mut add_prompt = None;
    if req.parameters.return_full_text.unwrap_or(false) {
        add_prompt = Some(req.inputs.clone());
    }

    let details: bool = req.parameters.details || req.parameters.decoder_input_details;

    // Inference
    let (response, best_of_responses) = match req.parameters.best_of {
        Some(best_of) if best_of > 1 => {
            let (response, best_of_responses) = infer.generate_best_of(req, best_of).await?;
            (response, Some(best_of_responses))
        }
        _ => (infer.generate(req).await?, None),
    };

    let request_id = response.request_id;
    let first_token_time = response.first_token_time;
    let max_time_between_tokens = response.max_time_between_tokens;
    let max_time_between_tokens_except_first = response.max_time_between_tokens_except_first;
    let avg_time_between_tokens = response.avg_time_between_tokens;
    let p90_time_between_tokens = response.p90_time_between_tokens;
    let p95_time_between_tokens = response.p95_time_between_tokens;
    let p99_time_between_tokens = response.p99_time_between_tokens;
    let input_length = response.input_length;
    let output_length = response.output_length;

    // Token details
    let details = match details {
        true => {
            // convert best_of_responses
            let best_of_sequences = best_of_responses.map(|responses: Vec<InferResponse>| {
                responses
                    .into_iter()
                    .map(|response: InferResponse| {
                        // Add prompt if return_full_text
                        let mut output_text = response.generated_text.text;
                        if let Some(prompt) = &add_prompt {
                            output_text = prompt.clone() + &output_text;
                        }

                        BestOfSequence {
                            generated_text: output_text,
                            finish_reason: FinishReason::from(
                                response.generated_text.finish_reason,
                            ),
                            generated_tokens: response.generated_text.generated_tokens,
                            prefill: response.prefill,
                            tokens: response.tokens,
                            top_tokens: response.top_tokens,
                            seed: response.generated_text.seed,
                        }
                    })
                    .collect()
            });

            Some(Details {
                finish_reason: FinishReason::from(response.generated_text.finish_reason),
                generated_tokens: response.generated_text.generated_tokens,
                prefill: response.prefill,
                tokens: response.tokens,
                seed: response.generated_text.seed,
                best_of_sequences,
                top_tokens: response.top_tokens,
            })
        }
        false => None,
    };

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;
    let time_per_token = inference_time / response.generated_text.generated_tokens;

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));
    span.record("time_per_token", format!("{time_per_token:?}"));
    span.record("seed", format!("{:?}", response.generated_text.seed));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", request_id.to_string().parse().unwrap());
    headers.insert(
        "x-first-token-time",
        first_token_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-token",
        time_per_token.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-first-decode-token-time",
        response
            .first_decode_token_time
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );

    headers.insert(
        "x-max-time-between-tokens",
        max_time_between_tokens
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-max-time-between-tokens-except-first",
        max_time_between_tokens_except_first
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-avg-time-between-tokens",
        avg_time_between_tokens
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-p90-time-between-tokens",
        p90_time_between_tokens
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-p95-time-between-tokens",
        p95_time_between_tokens
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-p99-time-between-tokens",
        p99_time_between_tokens
            .as_millis()
            .to_string()
            .parse()
            .unwrap(),
    );
    headers.insert("x-input-length", input_length.to_string().parse().unwrap());
    headers.insert(
        "x-output-length",
        output_length.to_string().parse().unwrap(),
    );

    // Metrics
    metrics::increment_counter!("blitz_request_success");
    metrics::histogram!("blitz_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "blitz_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("blitz_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "blitz_request_inference_duration",
        inference_time.as_secs_f64()
    );
    metrics::histogram!(
        "blitz_request_mean_time_per_token_duration",
        time_per_token.as_secs_f64()
    );
    metrics::histogram!(
        "blitz_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    // Send response
    let mut output_text = response.generated_text.text;
    if let Some(prompt) = add_prompt {
        output_text = prompt + &output_text;
    }

    tracing::debug!("Output: {}", output_text);
    // tracing::info!("Success");

    let response = GenerateResponse {
        generated_text: output_text,
        details,
    };
    Ok((headers, Json(response)))
}

#[instrument(skip_all)]
#[cfg(feature = "manually_scale")]
async fn modify_cluster_state_manually(
    infer: Extension<Infer>,
    Json(req): Json<ModifyClusterStateRequest>,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    tracing::info!("ModifyClusterStateRequest: {:?}", req);
    match req {
        ModifyClusterStateRequest::TriggerPrefillUp {
            old_stub_indices,
            new_stub_indices,
        } => match infer
            .manually_trigger_scale_up(old_stub_indices, new_stub_indices, "Prefill".to_string())
            .await
        {
            Ok(_) => Ok(()),
            Err(error) => Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error,
                    error_type: StatusCode::BAD_REQUEST.to_string(),
                }),
            )),
        },
        ModifyClusterStateRequest::TriggerDecodeUp {
            old_stub_indices,
            new_stub_indices,
        } => match infer
            .manually_trigger_scale_up(old_stub_indices, new_stub_indices, "Decode".to_string())
            .await
        {
            Ok(_) => Ok(()),
            Err(error) => Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error,
                    error_type: StatusCode::BAD_REQUEST.to_string(),
                }),
            )),
        },
        ModifyClusterStateRequest::TriggerScaleDown { stub_indices } => {
            match infer.manually_trigger_scale_down(stub_indices).await {
                Ok(_) => Ok(()),
                Err(error) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error,
                        error_type: StatusCode::BAD_REQUEST.to_string(),
                    }),
                )),
            }
        }
        ModifyClusterStateRequest::TriggerMutation { stub_indices } => {
            match infer.manually_trigger_mutate_to_decode(stub_indices).await {
                Ok(_) => Ok(()),
                Err(error) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error,
                        error_type: StatusCode::BAD_REQUEST.to_string(),
                    }),
                )),
            }
        }
        ModifyClusterStateRequest::TriggerNoramalDown { stub_indices } => {
            match infer.manually_trigger_scale_down(stub_indices).await {
                Ok(_) => Ok(()),
                Err(error) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error,
                        error_type: StatusCode::BAD_REQUEST.to_string(),
                    }),
                )),
            }
        }
        ModifyClusterStateRequest::TriggerNormalUp {
            old_stub_indices,
            new_stub_indices,
        } => {
            match infer
                .manually_trigger_scale_up(old_stub_indices, new_stub_indices, "Normal".to_string())
                .await
            {
                Ok(_) => Ok(()),
                Err(error) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error,
                        error_type: StatusCode::BAD_REQUEST.to_string(),
                    }),
                )),
            }
        }
    }
}

/// Generate a stream of token using Server-Sent Events
#[utoipa::path(
post,
tag = "Blitz",
path = "/generate_stream",
request_body = GenerateRequest,
responses(
(status = 200, description = "Generated Text", body = StreamResponse,
content_type = "text/event-stream"),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"}),
content_type = "text/event-stream"),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"}),
content_type = "text/event-stream"),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"}),
content_type = "text/event-stream"),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"}),
content_type = "text/event-stream"),
)
)]
#[instrument(skip_all)]
// #[instrument(
// skip_all,
// fields(
// parameters = ? req.parameters,
// total_time,
// validation_time,
// queue_time,
// inference_time,
// time_per_token,
// seed,
// )
// )]
async fn generate_stream(
    Extension(infer): Extension<Infer>,
    Json(req): Json<GenerateRequest>,
) -> (
    HeaderMap,
    Sse<impl Stream<Item = Result<Event, Infallible>>>,
) {
    let span = tracing::Span::current();
    let start_time = Instant::now();
    metrics::increment_counter!("blitz_request_count");

    tracing::debug!("Input: {}", req.inputs);

    let compute_characters = req.inputs.chars().count();

    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", "gpu+optimized".parse().unwrap());
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());

    let stream = async_stream::stream! {
        // Inference
        let mut end_reached = false;
        let mut error = false;

        let mut add_prompt = None;
        if req.parameters.return_full_text.unwrap_or(false) {
            add_prompt = Some(req.inputs.clone());
        }
        let details = req.parameters.details;

        let best_of = req.parameters.best_of.unwrap_or(1);
        if best_of != 1 {
            let err = InferError::from(ValidationError::BestOfStream);
            metrics::increment_counter!("blitz_request_failure", "err" => "validation");
            tracing::error!("{err}");
            yield Ok(Event::from(err));
        } else if req.parameters.decoder_input_details {
            let err = InferError::from(ValidationError::PrefillDetailsStream);
            metrics::increment_counter!("blitz_request_failure", "err" => "validation");
            tracing::error!("{err}");
            yield Ok(Event::from(err));
        } else {
            match infer.generate_stream(req).instrument(info_span!(parent: &span, "async_stream")).await {
                // Keep permit as long as generate_stream lives
                Ok((_request_id, _permit, mut response_stream)) => {
                    // Server-Sent Event stream
                    while let Some(response) = response_stream.next().await {
                        match response {
                            Ok(response) => {
                                match response {
                                    // DIY prefill_done is ignored
                                    InferStreamResponse::PrefillDone => {}
                                    // Prefill is ignored
                                    InferStreamResponse::Prefill(_) => {}
                                    // Yield event for every new token
                                    InferStreamResponse::Intermediate{
                                        token,
                                        top_tokens,
                                    } => {
                                        tracing::debug!(parent: &span, "Token: {:?}", token);

                                        // StreamResponse
                                        let stream_token = StreamResponse {
                                            token,
                                            top_tokens,
                                            generated_text: None,
                                            details: None,
                                        };

                                        yield Ok(Event::default().json_data(stream_token).unwrap())
                                    }
                                    // Yield event for last token and compute timings
                                    InferStreamResponse::End {
                                        token,
                                        generated_text,
                                        start,
                                        queued,
                                        top_tokens,
                                        max_time_between_tokens: _,
                                    } => {
                                        // Token details
                                        let details = match details {
                                            true => Some(StreamDetails {
                                                finish_reason: FinishReason::from(generated_text.finish_reason),
                                                generated_tokens: generated_text.generated_tokens,
                                                seed: generated_text.seed,
                                            }),
                                            false => None,
                                        };

                                        // Timings
                                        let total_time = start_time.elapsed();
                                        let validation_time = queued - start_time;
                                        let queue_time = start - queued;
                                        let inference_time = Instant::now() - start;
                                        let time_per_token = inference_time / generated_text.generated_tokens;

                                        // Tracing metadata
                                        span.record("total_time", format!("{total_time:?}"));
                                        span.record("validation_time", format!("{validation_time:?}"));
                                        span.record("queue_time", format!("{queue_time:?}"));
                                        span.record("inference_time", format!("{inference_time:?}"));
                                        span.record("time_per_token", format!("{time_per_token:?}"));
                                        span.record("seed", format!("{:?}", generated_text.seed));

                                        // Metrics
                                        metrics::increment_counter!("blitz_request_success");
                                        metrics::histogram!("blitz_request_duration", total_time.as_secs_f64());
                                        metrics::histogram!("blitz_request_validation_duration", validation_time.as_secs_f64());
                                        metrics::histogram!("blitz_request_queue_duration", queue_time.as_secs_f64());
                                        metrics::histogram!("blitz_request_inference_duration", inference_time.as_secs_f64());
                                        metrics::histogram!("blitz_request_mean_time_per_token_duration", time_per_token.as_secs_f64());
                                        metrics::histogram!("blitz_request_generated_tokens", generated_text.generated_tokens as f64);

                                        // StreamResponse
                                        end_reached = true;

                                        let mut output_text = generated_text.text;
                                        if let Some(prompt) = add_prompt {
                                            output_text = prompt + &output_text;
                                        }

                                        tracing::debug!(parent: &span, "Output: {}", output_text);
                                        tracing::info!(parent: &span, "Success");

                                        let stream_token = StreamResponse {
                                            token,
                                            top_tokens,
                                            generated_text: Some(output_text),
                                            details
                                        };

                                        yield Ok(Event::default().json_data(stream_token).unwrap());
                                        break;
                                    }
                                }
                            }
                            // yield error
                            Err(err) => {
                                error = true;
                                yield Ok(Event::from(err));
                                break;
                            }
                        }
                    }
                },
                // yield error
                Err(err) => {
                    error = true;
                    yield Ok(Event::from(err));
                }
            }
            // Check if generation reached the end
            // Skip if we already sent an error
            if !end_reached && !error {
                let err = InferError::IncompleteGeneration;
                metrics::increment_counter!("blitz_request_failure", "err" => "incomplete");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            }
        }
    };

    (headers, Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
get,
tag = "Blitz",
path = "/metrics",
responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: InfoResponse,
    compat_return_full_text: bool,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    stubs: Vec<Stub>,
    deployment: Deployment,
    config_path: String,
    tokenizer: Option<Tokenizer>,
    validation_workers: usize,
    addr: SocketAddr,
    allow_origin: Option<AllowOrigin>,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    model: Model,
    manually_modify_state_enabled: bool,
    controller_args: ControllerArgs,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    health,
    get_model_info,
    compat_generate,
    generate,
    generate_stream,
    metrics,
    ),
    components(
    schemas(
    Info,
    CompatGenerateRequest,
    GenerateRequest,
    GenerateParameters,
    PrefillToken,
    Token,
    GenerateResponse,
    BestOfSequence,
    Details,
    FinishReason,
    StreamResponse,
    StreamDetails,
    ErrorResponse,
    )
    ),
    tags(
    (name = "Blitz", description = "Hugging Face Blitz API")
    ),
    info(
    title = "Blitz",
    license(
    name = "Apache 2.0",
    url = "https://www.apache.org/licenses/LICENSE-2.0"
    )
    )
    )]
    struct ApiDoc;

    // Create state
    let validation = Validation::new(
        validation_workers,
        tokenizer,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_length,
        max_total_tokens,
    );
    let generation_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new();

    let config_str = std::fs::read_to_string(config_path).expect("Failed to read config file");

    let infer = match deployment {
        Deployment::Disaggregation => {
            let config = serde_json::from_str(&config_str).expect("Failed to parse config file");
            Infer::create_disaggregation(
                stubs,
                config,
                validation,
                waiting_served_ratio,
                max_batch_prefill_tokens,
                max_batch_total_tokens,
                max_waiting_tokens,
                max_concurrent_requests,
                shard_info.requires_padding,
                shard_info.window_size,
                shard_info.speculate,
                generation_health,
                model,
                manually_modify_state_enabled,
                controller_args,
            )
        }
    };

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("blitz_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("blitz_request_generated_tokens"));
    let generated_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Input Length buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("blitz_request_max_new_tokens"));
    let max_new_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("blitz_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();
    // Speculated tokens buckets
    let skipped_matcher = Matcher::Full(String::from("blitz_request_skipped_tokens"));
    let skipped_buckets: Vec<f64> = (0..shard_info.speculate + 1).map(|x| x as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &generated_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap()
        .set_buckets_for_metric(skipped_matcher, &skipped_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    describe_metric();
    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_total_tokens,
        max_waiting_tokens,
        validation_workers,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    // Create router
    let mut app_box: Box<Router> = Box::<Router>::default();
    #[cfg(not(feature = "manually_scale"))]
    {
        *app_box = Router::new()
            .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
            // Base routes
            .route("/", post(compat_generate))
            .route("/info", get(get_model_info))
            .route("/generate", post(generate))
            .route("/generate_stream", post(generate_stream))
            // AWS Sagemaker route
            .route("/invocations", post(compat_generate))
            // Base Health route
            .route("/health", get(health))
            // Inference API health route
            .route("/", get(health))
            // AWS Sagemaker health route
            .route("/ping", get(health))
            // Prometheus metrics route
            .route("/metrics", get(metrics))
            .layer(Extension(info))
            .layer(Extension(health_ext.clone()))
            .layer(Extension(compat_return_full_text))
            .layer(Extension(infer))
            .layer(Extension(prom_handle.clone()))
            .layer(OtelAxumLayer::default())
            .layer(cors_layer);
    };
    #[cfg(feature = "manually_scale")]
    {
        *app_box = Router::new()
            .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
            // Base routes
            .route("/", post(compat_generate))
            .route("/info", get(get_model_info))
            .route("/generate", post(generate))
            .route("/generate_stream", post(generate_stream))
            .route("/modify_cluster_state", post(modify_cluster_state_manually))
            // AWS Sagemaker route
            .route("/invocations", post(compat_generate))
            // Base Health route
            .route("/health", get(health))
            // Inference API health route
            .route("/", get(health))
            // AWS Sagemaker health route
            .route("/ping", get(health))
            // Prometheus metrics route
            .route("/metrics", get(metrics))
            .layer(Extension(info))
            .layer(Extension(health_ext.clone()))
            .layer(Extension(compat_return_full_text))
            .layer(Extension(infer))
            .layer(Extension(prom_handle.clone()))
            .layer(OtelAxumLayer::default())
            .layer(cors_layer);
    };
    let app = *app_box;

    if ngrok {
        #[cfg(feature = "ngrok")]
        {
            use ngrok::config::TunnelBuilder;

            let _ = addr;

            let authtoken =
                ngrok_authtoken.expect("`ngrok-authtoken` must be set when using ngrok tunneling");

            let edge = ngrok_edge.expect("`ngrok-edge` must be set when using ngrok tunneling");

            let tunnel = ngrok::Session::builder()
                .authtoken(authtoken)
                .connect()
                .await
                .unwrap()
                .labeled_tunnel()
                .label("edge", edge);

            let listener = tunnel.listen().await.unwrap();

            // Run prom metrics and health locally too
            tokio::spawn(
                axum::Server::bind(&addr)
                    .serve(
                        Router::new()
                            .route("/health", get(health))
                            .route("/metrics", get(metrics))
                            .layer(Extension(health_ext))
                            .layer(Extension(prom_handle))
                            .into_make_service(),
                    )
                    //Wait until all requests are finished to shut down
                    .with_graceful_shutdown(shutdown_signal()),
            );

            // Run server
            axum::Server::builder(listener)
                .serve(app.into_make_service())
                //Wait until all requests are finished to shut down
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        #[cfg(not(feature = "ngrok"))]
        {
            let _ngrok_authtoken = ngrok_authtoken;
            let _ngrok_domain = ngrok_domain;
            let _ngrok_username = ngrok_username;
            let _ngrok_password = ngrok_password;

            panic!("`blitz-router` was compiled without the `ngrok` feature");
        }
    } else {
        // Run server
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            // Wait until all requests are finished to shut down
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }
    Ok(())
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
    opentelemetry::global::shutdown_tracer_provider();
}

impl From<i32> for FinishReason {
    fn from(finish_reason: i32) -> Self {
        let finish_reason = pb::generate::v2::FinishReason::try_from(finish_reason).unwrap();
        match finish_reason {
            pb::generate::v2::FinishReason::Length => FinishReason::Length,
            pb::generate::v2::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            pb::generate::v2::FinishReason::StopSequence => FinishReason::StopSequence,
        }
    }
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::GenerationError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::IncompleteGeneration => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}

fn describe_blitz_metric() {
    metrics::describe_gauge!("blitz_relay_queue_size", "Current size of the relay queue");
    metrics::describe_gauge!(
        "blitz_migration_queue_size",
        "Current size of the migration queue"
    );

    metrics::describe_gauge!("blitz_used_blocks", "Total blocks used");

    metrics::describe_counter!(
        "blitz_scale_up_times",
        "Times trigger scale up(manual & auto)"
    );

    metrics::describe_counter!(
        "blitz_nvbroadcast_times",
        "Times when scale using nvbroadcast"
    );
    metrics::describe_counter!(
        "blitz_NVLink_p2p_times",
        "Times when scale using NVLink p2p"
    );
    metrics::describe_counter!("blitz_RDMA_p2p_times", "Times when scale using RDMA p2p");

    metrics::describe_gauge!(
        "blitz_waiting_full_batches",
        "Num of batches in MigrationQueue.waiting_full_batches"
    );
    metrics::describe_gauge!(
        "blitz_waiting_partial_batches",
        "Num of batches in MigrationQueue.waiting_partial_batches"
    );

    // done
    metrics::describe_counter!(
        "trans_kv_cache_times",
        "Times trans full kvcache from src to dst"
    );
    metrics::describe_counter!(
        "trans_partial_kv_cache_times",
        "Times trans partial kvcache from src to dst"
    );
    metrics::describe_gauge!(
        "blitz_decode_replica",
        "Current Num of Decode instance(Including all kinds of decode, except shutting decode)"
    );
    metrics::describe_gauge!(
        "blitz_prefill_replica",
        "Current Num of Prefill instance(Including all kinds of prefill, except shutting prefill)"
    );
    metrics::describe_gauge!("blitz_shutting_replica", "Current Num of Shutting instance");
    // pub(crate) prefill_tokens: AtomicUsize,
    // pub(crate) decode_tokens: AtomicUsize,
    // pub(crate) loop_counts: Vec<AtomicUsize>,
    // pub(crate) token_in_queue: AtomicUsize,

    // pub(crate) block_size: u32,
    // used_blocks: AtomicU32,
    // model_loaded: AtomicBool,

    // #[allow(unused)]
    // replica_index: usize,
    // /// lock() <-> act as decode
    // pub(crate) dst_mutex: Arc<Mutex<()>>,
    // /// replica state for event loop
    // pub(crate) state: RwLock<ReplicaState>,
    // /// ongoing Zigzag partial layer migration tasks.
    // pub(crate) flying_partial_migration_batches: Arc<AtomicI64>,
}

fn describe_metric() {
    metrics::describe_counter!("blitz_request_count", "Total number of requests received");
    metrics::describe_counter!(
        "blitz_request_success",
        "Total number of successful requests"
    );
    metrics::describe_counter!(
        "blitz_request_failure",
        "Total number of failed requests, labeled by error type"
    );
    metrics::describe_gauge!("blitz_queue_size", "Current size of the request queue");
    metrics::describe_histogram!(
        "blitz_request_duration",
        "Total time taken to process a request (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_validation_duration",
        "Time spent on request validation (seconds)"
    );

    // previous done
    metrics::describe_histogram!(
        "blitz_request_queue_duration",
        "Time spent in the request queue (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_inference_duration",
        "Time spent on inference (seconds)"
    );
    metrics::describe_histogram!(
        "blitz_request_mean_time_per_token_duration",
        "Mean time per generated token (seconds)"
    );

    // after done
    metrics::describe_histogram!(
        "blitz_request_generated_tokens",
        "Number of tokens generated per request"
    );
    metrics::describe_histogram!(
        "blitz_request_input_length",
        "Input length (number of tokens) per request"
    );
    metrics::describe_histogram!(
        "blitz_request_max_new_tokens",
        "Max new tokens requested per request"
    );
    metrics::describe_histogram!("blitz_batch_next_size", "Batch size for next batch");
    metrics::describe_histogram!(
        "blitz_request_skipped_tokens",
        "Number of speculated/skipped tokens per request"
    );
    describe_blitz_metric();
}
