#!/usr/bin/env python3
"""Targeted broadcast tests for refactored proto services."""
import grpc
import concurrent.futures
import generate_pb2
import generate_pb2_grpc

def main():
    # Connect to all 8 ranks
    stubs = []
    for i in range(8):
        channel = grpc.insecure_channel(f"localhost:{50051+i}")
        stubs.append(generate_pb2_grpc.TextGenerationServiceStub(channel))

    # Test 1: Health check
    print("=== Health Check ===")
    for i, stub in enumerate(stubs):
        try:
            resp = stub.Health(generate_pb2.HealthRequest())
            print(f"  Rank {i}: {resp.state}")
        except Exception as e:
            print(f"  Rank {i}: FAILED - {e}")

    # Test 2: NVLink broadcast (within TP group: src=[0], dst=[1])
    print("\n=== NVLink Broadcast Test (src=[0], dst=[1]) ===")
    request = generate_pb2.BroadcastRequest()
    request.src_ranks.extend([0])
    request.dst_ranks.extend([1])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(stubs[r].NvlBroadcast, request): r for r in [0, 1]}
        for future in concurrent.futures.as_completed(futures):
            rank = futures[future]
            try:
                future.result()
                print(f"  Rank {rank}: OK")
            except grpc.RpcError as e:
                print(f"  Rank {rank}: FAILED - {e.details()}")

    # Test 3: RDMA broadcast (cross TP groups: src=[0], dst=[4])
    print("\n=== RDMA Broadcast Test (src=[0], dst=[4]) ===")
    request2 = generate_pb2.BroadcastRequest()
    request2.src_ranks.extend([0])
    request2.dst_ranks.extend([4])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(stubs[r].RdmaBroadcast, request2): r for r in [0, 4]}
        for future in concurrent.futures.as_completed(futures):
            rank = futures[future]
            try:
                future.result()
                print(f"  Rank {rank}: OK")
            except grpc.RpcError as e:
                print(f"  Rank {rank}: FAILED - {e.details()}")

    print("\n=== All tests done ===")

if __name__ == "__main__":
    main()
