#!/usr/bin/env python3
"""NVLink and RDMA broadcast tests for TP=4, 8 ranks."""
import grpc
import concurrent.futures
import sys
from time import time

sys.path.insert(0, '.')
import generate_pb2
import generate_pb2_grpc


def create_stub(server_address):
    channel = grpc.insecure_channel(server_address)
    return generate_pb2_grpc.TextGenerationServiceStub(channel)


def benchmark_nccl(all_stubs, all_ranks):
    """NVLink broadcast: replica [0,1,2,3] -> replica [4,5,6,7] with TP=4."""
    src_ranks = [0, 1, 2, 3]
    dst_ranks = [4, 5, 6, 7]
    rpc_ranks = src_ranks + dst_ranks
    print("NVLink broadcast: src={} -> dst={}".format(src_ranks, dst_ranks))

    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend(src_ranks)
    request.dst_ranks.extend(dst_ranks)

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(rank_stub[1].NvlBroadcast, request): rank_stub[0]
            for rank_stub in filter(lambda rs: rs[0] in rpc_ranks, zip(all_ranks, all_stubs))
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            rank = futures[future]
            try:
                result = future.result()
                print("  Rank {}: OK".format(rank))
                unwrap.append(result)
            except grpc.RpcError as e:
                print("  Rank {}: FAILED - {}".format(rank, e.details()))
                raise
    print("RPC resp: {} (expected {})".format(len(unwrap), len(rpc_ranks)))


def benchmark_rdma_chain(all_stubs, all_ranks):
    """RDMA broadcast: replica [0,1,2,3] -> replica [4,5,6,7] with TP=4."""
    src_ranks = [0, 1, 2, 3]
    dst_ranks = [4, 5, 6, 7]
    rpc_ranks = src_ranks + dst_ranks
    print("RDMA broadcast: src={} -> dst={}".format(src_ranks, dst_ranks))

    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend(src_ranks)
    request.dst_ranks.extend(dst_ranks)

    unwrap = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(rank_stub[1].RdmaBroadcast, request): rank_stub[0]
            for rank_stub in filter(lambda rs: rs[0] in rpc_ranks, zip(all_ranks, all_stubs))
        }
        print("RPC req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            rank = futures[future]
            try:
                result = future.result()
                print("  Rank {}: OK".format(rank))
                unwrap.append(result)
            except grpc.RpcError as e:
                print("  Rank {}: FAILED - {}".format(rank, e.details()))
                raise
    print("RPC resp: {}".format(len(unwrap)))

    # WaitRdmaDone
    request2 = generate_pb2.WaitRdmaDoneRequest()
    unwrap2 = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(rank_stub[1].WaitRdmaDone, request2): rank_stub[0]
            for rank_stub in filter(lambda rs: rs[0] in rpc_ranks, zip(all_ranks, all_stubs))
        }
        print("WaitRdmaDone req: {}".format(len(futures)))
        for future in concurrent.futures.as_completed(futures):
            rank = futures[future]
            try:
                result = future.result()
                print("  Rank {}: OK".format(rank))
                unwrap2.append(result)
            except grpc.RpcError as e:
                print("  Rank {}: FAILED - {}".format(rank, e.details()))
                raise
    print("WaitRdmaDone resp: {}".format(len(unwrap2)))


def main():
    all_ranks = list(range(8))
    all_stubs = [create_stub("localhost:{}".format(50051 + i)) for i in range(8)]

    try:
        begin = time()
        print("=== NVLink Broadcast Test ===")
        benchmark_nccl(all_stubs, all_ranks)
        end = time()
        print("Elapsed NVLink: {:.1f}ms\n".format((end - begin) * 1000))

        begin = time()
        print("=== RDMA Broadcast Test ===")
        benchmark_rdma_chain(all_stubs, all_ranks)
        end = time()
        print("Elapsed RDMA: {:.1f}ms\n".format((end - begin) * 1000))

        print("=== ALL TESTS PASSED ===")
    except grpc.RpcError as e:
        print("RPC failed: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
