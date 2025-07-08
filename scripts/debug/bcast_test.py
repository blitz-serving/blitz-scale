#!/usr/bin/env python3

import grpc
import argparse
import sys
import os
import json
from typing import Dict, Callable, Any, List

import generate_pb2
import generate_pb2_grpc

TP_SIZE = [1]

def create_stub(server_address: str) -> generate_pb2_grpc.TextGenerationServiceStub:
    """Create a grpc stub for the TextGenerationService."""
    channel = grpc.insecure_channel(server_address)
    return generate_pb2_grpc.TextGenerationServiceStub(channel)

def input_int(prompt: str) -> int:
    """Get integer input from user."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

def input_ints(prompt: str) -> List[int]:
    """Get list of integers from user."""
    while True:
        try:
            return [int(x) for x in input(prompt).split()]
        except ValueError:
            print("Please enter valid integers separated by spaces.")

def input_bool(prompt: str) -> bool:
    """Get boolean input from user."""
    while True:
        response = input(prompt).lower()
        if response in ['y', 'yes', 'true', '1']:
            return True
        if response in ['n', 'no', 'false', '0']:
            return False
        print("Please enter 'y' or 'n'.")

def get_info_params() -> Dict[str, Any]:
    """Get parameters for Info request."""
    return {}

def get_health_params() -> Dict[str, Any]:
    """Get parameters for Health request."""
    return {}

def get_service_discovery_params() -> Dict[str, Any]:
    """Get parameters for ServiceDiscovery request."""
    return {}

def get_clear_cache_params() -> Dict[str, Any]:
    """Get parameters for ClearCache request."""
    params = {}
    if input_bool("Do you want to specify a batch ID? (y/n): "):
        params['batch_id'] = input_int("Enter batch ID: ")
    return params

def get_filter_batch_params() -> Dict[str, Any]:
    """Get parameters for FilterBatch request."""
    params = {}
    params['batch_id'] = input_int("Enter batch ID: ")
    params['request_ids'] = input_ints("Enter request IDs (space-separated): ")
    return params

def get_warmup_params() -> Dict[str, Any]:
    """Get parameters for Warmup request."""
    params = {}
    if input_bool("Do you want to specify max input length? (y/n): "):
        params['max_input_length'] = input_int("Enter max input length: ")
    if input_bool("Do you want to specify max prefill tokens? (y/n): "):
        params['max_prefill_tokens'] = input_int("Enter max prefill tokens: ")
    if input_bool("Do you want to specify max total tokens? (y/n): "):
        params['max_total_tokens'] = input_int("Enter max total tokens: ")
    return params

def get_prefill_params() -> Dict[str, Any]:
    """Get parameters for Prefill request."""
    params = {}
    params['batch_id'] = input_int("Enter batch ID: ")
    params['inputs'] = input("Enter input texts (one per line, empty line to finish):\n").split('\n')
    params['request_id'] = input_int("Enter request ID: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    params['max_new_tokens'] = input_int("Enter max new tokens: ")
    params['ignore_eos_token'] = input_bool("Ignore EOS token? (y/n): ")
    params['prefill_logprobs'] = input_bool("Return prefill logprobs? (y/n): ")
    params['top_n_tokens'] = input_int("Enter top n tokens: ")
    return params

def get_decode_params() -> Dict[str, Any]:
    """Get parameters for Decode request."""
    params = {}
    params['batch_ids'] = input_ints("Enter batch IDs (space-separated): ")
    params['batch_size'] = input_int("Enter batch size: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    return params

def get_send_params_params() -> Dict[str, Any]:
    """Get parameters for SendParams request."""
    params = {}
    params['dst_rank'] = input_int("Enter destination rank: ")
    return params

def get_recv_params_params() -> Dict[str, Any]:
    """Get parameters for RecvParams request."""
    params = {}
    params['src_rank'] = input_int("Enter source rank: ")
    return params

def get_load_params_params() -> Dict[str, Any]:
    """Get parameters for LoadParams request."""
    params = {}
    params['load_case'] = input_int("Enter load case (0: host, 1: disk): ")
    params['model_name'] = input("Enter model name: ")
    if input_bool("Do you want to specify model path? (y/n): "):
        params['model_path'] = input("Enter model path: ")
    return params

def get_prefill_v2_params() -> Dict[str, Any]:
    """Get parameters for PrefillV2 request."""
    params = {}
    params['batch_id'] = input_int("Enter batch ID: ")
    params['inputs'] = input("Enter input texts (one per line, empty line to finish):\n").split('\n')
    params['request_id'] = input_int("Enter request ID: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    params['max_new_tokens'] = input_int("Enter max new tokens: ")
    params['ignore_eos_token'] = input_bool("Ignore EOS token? (y/n): ")
    params['prefill_logprobs'] = input_bool("Return prefill logprobs? (y/n): ")
    params['top_n_tokens'] = input_int("Enter top n tokens: ")
    params['forward_case'] = input_int("Enter forward case: ")
    
    if input_bool("Do you want to specify pipeline parallel info? (y/n): "):
        print("Start layer options: embedding, lm_head, tfm")
        start_layer = input("Enter start layer: ")
        if start_layer == "tfm":
            params['start_layer'] = input_int("Enter transformer layer number: ")
        else:
            params['start_layer'] = start_layer
        params['num_layer_per_rank'] = input_ints("Enter number of layers per rank (space-separated): ")
    
    if input_bool("Do you want to specify pipe peer? (y/n): "):
        params['pipe_peer'] = input_int("Enter pipe peer: ")
    
    return params

def get_decode_v2_params() -> Dict[str, Any]:
    """Get parameters for DecodeV2 request."""
    params = {}
    params['batch_ids'] = input_ints("Enter batch IDs (space-separated): ")
    params['batch_size'] = input_int("Enter batch size: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    return params

def get_migrate_params() -> Dict[str, Any]:
    """Get parameters for Migrate request."""
    params = {}
    params['batch_id'] = input_int("Enter batch ID: ")
    params['inputs'] = input("Enter input texts (one per line, empty line to finish):\n").split('\n')
    params['request_id'] = input_int("Enter request ID: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    params['max_new_tokens'] = input_int("Enter max new tokens: ")
    params['ignore_eos_token'] = input_bool("Ignore EOS token? (y/n): ")
    params['src_ranks'] = input_ints("Enter source ranks (space-separated): ")
    params['dst_ranks'] = input_ints("Enter destination ranks (space-separated): ")
    return params

def get_immigrate_params() -> Dict[str, Any]:
    """Get parameters for Immigrate request."""
    params = {}
    params['batch_id'] = input_int("Enter batch ID: ")
    params['inputs'] = input("Enter input texts (one per line, empty line to finish):\n").split('\n')
    params['request_id'] = input_int("Enter request ID: ")
    params['max_tokens'] = input_int("Enter max tokens: ")
    params['max_new_tokens'] = input_int("Enter max new tokens: ")
    params['ignore_eos_token'] = input_bool("Ignore EOS token? (y/n): ")
    params['src_ranks'] = input_ints("Enter source ranks (space-separated): ")
    params['dst_ranks'] = input_ints("Enter destination ranks (space-separated): ")
    return params

def get_wait_rdma_done_params() -> Dict[str, Any]:
    """Get parameters for WaitRdmaDone request."""
    return {}

def get_reset_status_params() -> Dict[str, Any]:
    """Get parameters for ResetStatus request."""
    return {}

def get_set_status_ready_params() -> Dict[str, Any]:
    """Get parameters for SetStatusReady request."""
    return {}

def get_relay_params() -> Dict[str, Any]:
    """Get parameters for Relay request."""
    params = {}
    params['rank'] = input_int("Enter rank: ")
    params['relax_not_head'] = input_bool("Relax not head? (y/n): ")
    return params

def get_nvl_broadcast_params() -> Dict[str, Any]:
    """Get parameters for NvlBroadcast request."""
    params = {}
    params['broadcast_ranks'] = input_ints("Enter broadcast ranks (space-separated): ")
    return params

def get_rdma_broadcast_params() -> Dict[str, Any]:
    """Get parameters for RdmaBroadcast request."""
    params = {}
    params['broadcast_ranks'] = input_ints("Enter broadcast ranks (space-separated): ")
    return params

def get_tanz_broadcast_params() -> Dict[str, Any]:
    """Get parameters for RdmaBroadcast request."""
    params = {}
    params['src_ranks'] = input_ints("Tanz broadcast source ranks (space-separated): ")
    params['dst_ranks'] = input_ints("Tanz broadcast destination ranks (space-separated): ")
    return params

# Map of request names to their parameter getter functions
PARAM_GETTERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "info": get_info_params,
    "health": get_health_params,
    "service_discovery": get_service_discovery_params,
    "clear_cache": get_clear_cache_params,
    "filter_batch": get_filter_batch_params,
    "warmup": get_warmup_params,
    "prefill": get_prefill_params,
    "decode": get_decode_params,
    "send_params": get_send_params_params,
    "recv_params": get_recv_params_params,
    "load_params": get_load_params_params,
    "prefill_v2": get_prefill_v2_params,
    "decode_v2": get_decode_v2_params,
    "migrate": get_migrate_params,
    "immigrate": get_immigrate_params,
    "wait_rdma_done": get_wait_rdma_done_params,
    "reset_status": get_reset_status_params,
    "set_status_ready": get_set_status_ready_params,
    "relay": get_relay_params,
    "nvl_broadcast": get_nvl_broadcast_params,
    "rdma_broadcast": get_rdma_broadcast_params,
    "tanz_broadcast": get_tanz_broadcast_params,
}

def send_info_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send Info request and print response."""
    request = generate_pb2.InfoRequest()
    response = stub.Info(request)
    print("Info Response:")
    print(f"  requires_padding: {response.requires_padding}")
    print(f"  dtype: {response.dtype}")
    print(f"  device_type: {response.device_type}")
    print(f"  window_size: {response.window_size}")
    print(f"  speculate: {response.speculate}")

def send_health_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send Health request and print response."""
    request = generate_pb2.HealthRequest()
    response = stub.Health(request)
    print("Health Response:")
    print(f"  state: {response.state}")

def send_service_discovery_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send ServiceDiscovery request and print response."""
    request = generate_pb2.ServiceDiscoveryRequest()
    response = stub.ServiceDiscovery(request)
    print("ServiceDiscovery Response:")
    print(f"  urls: {response.urls}")
    print(f"  ranks_view_in_json: {response.ranks_view_in_json}")

def send_clear_cache_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send ClearCache request and print response."""
    request = generate_pb2.ClearCacheRequest()
    if args.batch_id is not None:
        request.id = args.batch_id
    response = stub.ClearCache(request)
    print("ClearCache Response: Success")

def send_filter_batch_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send FilterBatch request and print response."""
    request = generate_pb2.FilterBatchRequest()
    request.batch_id = args.batch_id
    request.request_ids.extend(args.request_ids)
    response = stub.FilterBatch(request)
    print("FilterBatch Response:")
    print(f"  Batch ID: {response.batch.id}")
    print(f"  Request IDs: {response.batch.request_ids}")
    print(f"  Size: {response.batch.size}")
    print(f"  Max Tokens: {response.batch.max_tokens}")

def send_warmup_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Warmup request and print response."""
    request = generate_pb2.WarmupRequest()
    if args.max_input_length is not None:
        request.max_input_length = args.max_input_length
    if args.max_prefill_tokens is not None:
        request.max_prefill_tokens = args.max_prefill_tokens
    if args.max_total_tokens is not None:
        request.max_total_tokens = args.max_total_tokens
    response = stub.Warmup(request)
    print("Warmup Response:")
    if response.max_supported_total_tokens:
        print(f"  max_supported_total_tokens: {response.max_supported_total_tokens}")

def send_prefill_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Prefill request and print response."""
    request = generate_pb2.PrefillRequest()
    batch = request.batch
    batch.id = args.batch_id
    batch.size = len(args.inputs)
    batch.max_tokens = args.max_tokens
    
    for input_text in args.inputs:
        req = batch.requests.add()
        req.id = args.request_id
        req.inputs = input_text
        req.stopping_parameters.max_new_tokens = args.max_new_tokens
        req.stopping_parameters.ignore_eos_token = args.ignore_eos_token
        req.prefill_logprobs = args.prefill_logprobs
        req.top_n_tokens = args.top_n_tokens
    
    response = stub.Prefill(request)
    print("Prefill Response:")
    for gen in response.generations:
        print(f"  Request ID: {gen.request_id}")
        if gen.generated_text:
            print(f"  Generated Text: {gen.generated_text.text}")
            print(f"  Generated Tokens: {gen.generated_text.generated_tokens}")
            print(f"  Finish Reason: {gen.generated_text.finish_reason}")

def send_decode_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Decode request and print response."""
    request = generate_pb2.DecodeRequest()
    for batch_id in args.batch_ids:
        batch = request.batches.add()
        batch.id = batch_id
        batch.size = args.batch_size
        batch.max_tokens = args.max_tokens
    
    response = stub.Decode(request)
    print("Decode Response:")
    for gen in response.generations:
        print(f"  Request ID: {gen.request_id}")
        if gen.generated_text:
            print(f"  Generated Text: {gen.generated_text.text}")
            print(f"  Generated Tokens: {gen.generated_text.generated_tokens}")
            print(f"  Finish Reason: {gen.generated_text.finish_reason}")

def send_send_params_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send SendParams request and print response."""
    request = generate_pb2.SendParamsRequest()
    request.dst = args.dst_rank
    response = stub.SendParams(request)
    print("SendParams Response: Success")

def send_recv_params_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send RecvParams request and print response."""
    request = generate_pb2.RecvParamsRequest()
    request.src = args.src_rank
    response = stub.RecvParams(request)
    print("RecvParams Response: Success")

def send_load_params_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send LoadParams request and print response."""
    request = generate_pb2.LoadParamsRequest()
    request.load_case = args.load_case
    request.model_name = args.model_name
    if args.model_path:
        request.model_path = args.model_path
    response = stub.LoadParams(request)
    print("LoadParams Response: Success")

def send_prefill_v2_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send PrefillV2 request and print response."""
    request = generate_pb2.PrefillV2Request()
    batch = request.batch
    batch.id = args.batch_id
    batch.size = len(args.inputs)
    batch.max_tokens = args.max_tokens
    
    for input_text in args.inputs:
        req = batch.requests.add()
        req.id = args.request_id
        req.inputs = input_text
        req.stopping_parameters.max_new_tokens = args.max_new_tokens
        req.stopping_parameters.ignore_eos_token = args.ignore_eos_token
        req.prefill_logprobs = args.prefill_logprobs
        req.top_n_tokens = args.top_n_tokens
    
    request.forward_case = args.forward_case
    if args.pp_info:
        pp_info = request.pp_info
        if args.start_layer == "embedding":
            pp_info.embedding_layer = 1
        elif args.start_layer == "lm_head":
            pp_info.lm_head = 1
        else:
            pp_info.tfm_layer = int(args.start_layer)
        pp_info.num_layer_per_rank.extend(args.num_layer_per_rank)
    
    if args.pipe_peer is not None:
        request.pipe_peer = args.pipe_peer
    
    response = stub.PrefillV2(request)
    print("PrefillV2 Response:")
    for gen in response.generations:
        print(f"  Request ID: {gen.request_id}")
        if gen.generated_text:
            print(f"  Generated Text: {gen.generated_text.text}")
            print(f"  Generated Tokens: {gen.generated_text.generated_tokens}")
            print(f"  Finish Reason: {gen.generated_text.finish_reason}")

def send_decode_v2_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send DecodeV2 request and print response."""
    request = generate_pb2.DecodeV2Request()
    for batch_id in args.batch_ids:
        batch = request.batches.add()
        batch.id = batch_id
        batch.size = args.batch_size
        batch.max_tokens = args.max_tokens
    
    response = stub.DecodeV2(request)
    print("DecodeV2 Response:")
    for gen in response.generations:
        print(f"  Request ID: {gen.request_id}")
        if gen.generated_text:
            print(f"  Generated Text: {gen.generated_text.text}")
            print(f"  Generated Tokens: {gen.generated_text.generated_tokens}")
            print(f"  Finish Reason: {gen.generated_text.finish_reason}")

def send_migrate_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Migrate request and print response."""
    request = generate_pb2.MigrateRequest()
    batch = request.batch
    batch.id = args.batch_id
    batch.size = len(args.inputs)
    batch.max_tokens = args.max_tokens
    
    for input_text in args.inputs:
        req = batch.requests.add()
        req.id = args.request_id
        req.inputs = input_text
        req.stopping_parameters.max_new_tokens = args.max_new_tokens
        req.stopping_parameters.ignore_eos_token = args.ignore_eos_token
    
    request.src.extend(args.src_ranks)
    request.dst.extend(args.dst_ranks)
    
    response = stub.Migrate(request)
    print("Migrate Response:")
    print(f"  Batch ID: {response.batch.id}")
    print(f"  Request IDs: {response.batch.request_ids}")
    print(f"  Size: {response.batch.size}")
    print(f"  Max Tokens: {response.batch.max_tokens}")

def send_immigrate_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Immigrate request and print response."""
    request = generate_pb2.ImmigrateRequest()
    batch = request.batch
    batch.id = args.batch_id
    batch.size = len(args.inputs)
    batch.max_tokens = args.max_tokens
    
    for input_text in args.inputs:
        req = batch.requests.add()
        req.id = args.request_id
        req.inputs = input_text
        req.stopping_parameters.max_new_tokens = args.max_new_tokens
        req.stopping_parameters.ignore_eos_token = args.ignore_eos_token
    
    request.src.extend(args.src_ranks)
    request.dst.extend(args.dst_ranks)
    
    response = stub.Immigrate(request)
    print("Immigrate Response:")
    print(f"  Batch ID: {response.batch.id}")
    print(f"  Request IDs: {response.batch.request_ids}")
    print(f"  Size: {response.batch.size}")
    print(f"  Max Tokens: {response.batch.max_tokens}")

def send_wait_rdma_done_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send WaitRdmaDone request and print response."""
    request = generate_pb2.WaitRdmaDoneRequest()
    response = stub.WaitRdmaDone(request)
    print("WaitRdmaDone Response: Success")

def send_reset_status_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send ResetStatus request and print response."""
    request = generate_pb2.ResetStatusRequest()
    response = stub.ResetStatus(request)
    print("ResetStatus Response: Success")

def send_set_status_ready_request(stub: generate_pb2_grpc.TextGenerationServiceStub) -> None:
    """Send SetStatusReady request and print response."""
    request = generate_pb2.SetStatusReadyRequest()
    response = stub.SetStatusReady(request)
    print("SetStatusReady Response: Success")

def send_relay_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send Relay request and print response."""
    request = generate_pb2.RelayRequest()
    request.rank = args.rank
    request.relax_not_head = args.relax_not_head
    response = stub.Relay(request)
    print("Relay Response:")
    if response.batch_id:
        print(f"  Batch ID: {response.batch_id}")
    if response.seq_num:
        print(f"  Sequence Number: {response.seq_num}")

def send_nvl_broadcast_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send NvlBroadcast request and print response."""
    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend(args.src_ranks)
    request.dst_ranks.extend(args.dst_ranks)
    response = stub.NvlBroadcast(request)
    print("NvlBroadcast Response: Success")

def send_rdma_broadcast_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send RdmaBroadcast request and print response."""
    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend(args.src_ranks)
    request.dst_ranks.extend(args.dst_ranks)
    response = stub.RdmaBroadcast(request)
    print("RdmaBroadcast Response: Success")

def send_tanz_broadcast_request(stub: generate_pb2_grpc.TextGenerationServiceStub, args: argparse.Namespace) -> None:
    """Send TanzBroadcast request and print response."""
    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend(args.src_ranks)
    request.dst_ranks.extend(args.dst_ranks)
    response = stub.TanzBroadcast(request)
    print("TanzBroadcast Response: Success")
    return response

# Map of request names to their handler functions
REQUEST_HANDLERS: Dict[str, Callable[[generate_pb2_grpc.TextGenerationServiceStub, argparse.Namespace], None]] = {
    "info": lambda stub, _: send_info_request(stub),
    "health": lambda stub, _: send_health_request(stub),
    "service_discovery": lambda stub, _: send_service_discovery_request(stub),
    "clear_cache": send_clear_cache_request,
    "filter_batch": send_filter_batch_request,
    "warmup": send_warmup_request,
    "prefill": send_prefill_request,
    "decode": send_decode_request,
    "send_params": send_send_params_request,
    "recv_params": send_recv_params_request,
    "load_params": send_load_params_request,
    "prefill_v2": send_prefill_v2_request,
    "decode_v2": send_decode_v2_request,
    "migrate": send_migrate_request,
    "immigrate": send_immigrate_request,
    "wait_rdma_done": lambda stub, _: send_wait_rdma_done_request(stub),
    "reset_status": lambda stub, _: send_reset_status_request(stub),
    "set_status_ready": lambda stub, _: send_set_status_ready_request(stub),
    "relay": send_relay_request,
    "nvl_broadcast": send_nvl_broadcast_request,
    "rdma_broadcast": send_rdma_broadcast_request,
    "tanz_broadcast": send_tanz_broadcast_request,
}

def benchmark_nccl(all_stubs, all_ranks: List[int]):
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    import concurrent.futures

    # NOTE Tanz broadcast first
    request_type = 'nvl_broadcast'
    tp_size = TP_SIZE[0]
    if tp_size == 1:
        src_ranks_list = [[0], [2], [4], [6]]
        dst_ranks_list = [[1], [3], [5], [7]]
    elif tp_size == 2:
        src_ranks_list = [[0, 1], [6, 7]]
        dst_ranks_list = [[2, 3], [4, 5]]
    message = ""
    for (src, dst) in zip(src_ranks_list, dst_ranks_list):
        message += "({}={})".format(src, dst)
    print("Nvl broadcast: {}".format(message))

    handler = REQUEST_HANDLERS[request_type]

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for (src_ranks, dst_ranks) in zip(src_ranks_list, dst_ranks_list):
            rpc_ranks = src_ranks + dst_ranks
            rpc_args = {"src_ranks": src_ranks, "dst_ranks": dst_ranks}
            args = Args(**rpc_args)
            futures = futures | {
                executor.submit(handler, rank_stub[1], args): rank_stub
                for rank_stub in filter(
                    lambda rank_stub: rank_stub[0] in rpc_ranks, zip(all_ranks, all_stubs)
                )
            }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))

def benchmark_rdma_chain(all_stubs, all_ranks: List[int]):
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    import concurrent.futures

    # NOTE Tanz broadcast first
    request_type = 'rdma_broadcast'
    tp_size = TP_SIZE[0]
    if tp_size == 1:
        src_ranks_list = [[0]]
        dst_ranks_list = [[1], [3], [5], [7]]
    elif tp_size == 2:
        src_ranks_list = [[2, 3]]
        dst_ranks_list = [[4, 5], [6, 7]]
    message = "({})".format(src_ranks_list)
    for dst in dst_ranks_list:
        message += "~+~({})".format(dst)
    print("Rdma broadcast: {}".format(message))

    handler = REQUEST_HANDLERS[request_type]
    from functools import reduce
    import operator
    src_ranks = reduce(operator.add, src_ranks_list)
    dst_ranks = reduce(operator.add, dst_ranks_list)
    rpc_ranks = src_ranks + dst_ranks

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # BUG; FIXME
        rpc_args = {"src_ranks": src_ranks, "dst_ranks": dst_ranks}
        args = Args(**rpc_args)
        futures = futures | {
            executor.submit(handler, rank0_stub1[1], args): rank0_stub1
            for rank0_stub1 in filter(
                lambda rank0_stub1: rank0_stub1[0] in rpc_ranks, zip(all_ranks, all_stubs)
            )
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))

    request_type = 'wait_rdma_done'
    handler = REQUEST_HANDLERS[request_type]
    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        args = Args()
        futures = futures | {
            executor.submit(handler, rank_stub[1], args): rank_stub
            for rank_stub in filter(
                lambda rank_stub: rank_stub[0] in rpc_ranks, zip(all_ranks, all_stubs)
            )
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))

def benchmark_tanz(all_stubs, all_ranks: List[int]):
    src_ranks = [2, 3]
    # src_ranks = [1, 3]
    dst_ranks = [6, 7]
    print("Unit test for Tanz: src_ranks={}; dst_ranks={}".format(src_ranks, dst_ranks))
    request_type = "tanz_broadcast"

    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    rpc_args = {"src_ranks": src_ranks, "dst_ranks": dst_ranks}
    args = Args(**rpc_args)

    handler = REQUEST_HANDLERS[request_type]
    import concurrent.futures

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rpc_ranks = src_ranks + dst_ranks
        futures = {
            executor.submit(handler, rank_stub[1], args): rank_stub
            for rank_stub in filter(
                lambda rank_stub: rank_stub[0] in rpc_ranks, zip(all_ranks, all_stubs)
            )
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))

def benchmark_tanz_nccl(all_stubs, all_ranks: List[int]):
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    import concurrent.futures

    # NOTE Tanz broadcast first
    request_type = 'tanz_broadcast'
    src_ranks = [0, 2]
    # src_ranks = [1, 3]
    dst_ranks = [4, 6]
    # dst_ranks = [4, 6]
    print("Tanz broadcast: src_ranks={}; dst_ranks={}".format(src_ranks, dst_ranks))

    rpc_args = {"src_ranks": src_ranks, "dst_ranks": dst_ranks}
    args = Args(**rpc_args)
    handler = REQUEST_HANDLERS[request_type]

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rpc_ranks = src_ranks + dst_ranks
        futures = {
            executor.submit(handler, rank_stub[1], args): rank_stub
            for rank_stub in filter(
                lambda rank_stub: rank_stub[0] in rpc_ranks, zip(all_ranks, all_stubs)
            )
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))
    
    # NOTE Nccl broadcast for the rest
    request_type = 'nvl_broadcast'
    src_ranks = [4]
    # src_ranks = [5]
    dst_ranks = [5, 7]
    print("Tanz broadcast: src_ranks={}; dst_ranks={}".format(src_ranks, dst_ranks))

    rpc_args = {"src_ranks": src_ranks, "dst_ranks": dst_ranks}
    args = Args(**rpc_args)
    handler = REQUEST_HANDLERS[request_type]

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rpc_ranks = src_ranks + dst_ranks
        futures = {
            executor.submit(handler, rank_stub[1], args): rank_stub
            for rank_stub in filter(
                lambda rank_stub: rank_stub[0] in rpc_ranks, zip(all_ranks, all_stubs)
            )
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            unwrap.append(result)
    print("RPC resp: {}".format(len(unwrap)))


def main():
    parser = argparse.ArgumentParser(description="GRPC Client for TextGenerationService")
    parser.add_argument("--host", default='localhost')
    parser.add_argument("--port", type=int, default='50051')
    parser.add_argument("-n", required=True, type=int, help="mpi world size")
    args = parser.parse_args()

    all_ranks = [rank for rank in range(0, 8)]

    # XXX: a magical way to use Box::<int> in python
    TP_SIZE[0] = 1

    try:
        # Create stub
        all_stubs = []
        for i in range(args.n):
            all_stubs.append(create_stub("{}:{}".format(args.host, args.port + i)))

        from time import time
        begin = time()
        # benchmark_rdma_chain(all_stubs, all_ranks)
        # end = time()
        # print("Elapse broadcast: {}ms".format((end - begin) * 1000))
        # begin = end
        # benchmark_nccl(all_stubs, all_ranks)
        # end = time()
        # print("Elapse broadcast: {}ms".format((end - begin) * 1000))
        # begin = end
        benchmark_tanz(all_stubs, all_ranks)
        # end = time()
        # print("Elapse broadcast: {}ms".format((end - begin) * 1000))
        # begin = end
        # benchmark_tanz_nccl(all_stubs, all_ranks)
        end = time()
        print("Elapse broadcast: {}ms".format((end - begin) * 1000))

    except grpc.RpcError as e:
        print(f"RPC failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
