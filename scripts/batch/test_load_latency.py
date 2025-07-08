from time import sleep
from runner import (
    freeze,
    dump_bash_scripts,
    launch_client,
    launch_router,
    launch_server,
)
from parser import instantiate_template, timestamp
from plot import plot_main
import toml
import os
import argparse
import subprocess

import grpc
import asyncio
import generate_pb2 as pb
import generate_pb2_grpc as pb_grpc
import time

# If True, print colorful messages
colorful = True
# Sleep time in seconds after all GPUs are released before next round of tests
sleep_after_release = 60


def _released() -> bool:
    commands = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    output = (
        subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )
    return len(output) == 0


def wait_until_released(sleep_after_release):
    print_info("Wait until all GPUs are released")
    while True:
        if _released():
            print_info("All GPUs are released")
            sleep(sleep_after_release)
            break
        else:
            print_info("Waiting...")
            sleep(1)


def print_info(s: str):
    if colorful is False:
        print(f"[{timestamp()}] {s}", flush=True)
    else:
        bold_green = "\033[1;32m"
        reset = "\033[0m"
        print(f"{bold_green}[{timestamp()}] {s}{reset}", flush=True)


def print_error(s: str):
    if colorful is False:
        print(f"[{timestamp()}] {s}", flush=True)
    else:
        bold_red = "\033[1;31m"
        reset = "\033[0m"
        print(f"{bold_red}[{timestamp()}] {s}{reset}", flush=True)


def run_instances(archive_home, dry_run):
    configs = None
    instances_config_path = os.path.join(archive_home, "instances.toml")
    checkpoint_file = os.path.join(archive_home, "checkpoint.txt")
    checkpoint = set()
    # Create checkpoint failed to skip redudant test cases
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line in f:
                checkpoint.add(line.strip())
    else:
        open(checkpoint_file, "w").close()

    # Create error file to log failed tests information
    err_file = os.path.join(archive_home, "error.log")
    if os.path.exists(err_file) is not True:
        open(err_file, "w").close()

    # Read test cases configuration
    with open(instances_config_path, "r") as f:
        configs = toml.load(f)
    total = 0
    for _, instance_path_list in configs.items():
        total += len(instance_path_list)

    # Run each test case one by one
    count = 0
    for _, instance_path_list in configs.items():
        for instance_path in instance_path_list:
            count += 1
            print_info(f"[{count}/{total}] Running {instance_path}")

            # Skip is this configuration has already been executed
            if instance_path in checkpoint:
                print_info(f"{instance_path} skipped")
                continue
            instance_config = os.path.join(instance_path, "instance.toml")
            base_port = 50051 + 10 * count
            print_info(f"GRPC stubs base port: {base_port}")
            dump_bash_scripts(instance_config, base_port)
            if dry_run is True:
                print_info(f"dry-run enabled: skipping instance execution")
                continue
            # Launch server
            server, server_stdout_monitor, server_stderr_monitor = None, None, None
            try:
                print_info(f"Launching server.")
                server, server_stdout_monitor, server_stderr_monitor = launch_server(
                    instance_config, base_port
                )
            except Exception as e:
                print_error("Test failed: Failed to launch server")
                with open(err_file, "a") as f:
                    f.write(f"{timestamp()} Test failed\n")
                    f.write(f"{instance_path}\n")
                    f.write(f"{e}\n\n")
                if server is not None:
                    server.terminate()
                    server_stdout_monitor.join()
                    server_stderr_monitor.join()
                server, server_stdout_monitor, server_stderr_monitor = None, None, None
                print_error("Server terminated unexpectedly.")
                wait_until_released(sleep_after_release)
                continue
            
            test_load_ttft(base_port)

            test_nvlink_ttft(base_port)
            server.terminate()
            server_stdout_monitor.join()
            server_stderr_monitor.join()
            print_info("Server terminated.")
            wait_until_released(sleep_after_release)
            # Launch router and client
            # router, router_monitor, client, client_monitor = None, None, None, None
            # success = True
            # try:
            #     print_info(f"Launching router.")
            #     router, router_monitor = launch_router(instance_config)
            #     print_info(f"Launching client.")
            #     client, client_monitor = launch_client(instance_config)
            #     while True:
            #         if server.poll() is not None:
            #             print_error("Server terminated unexpectedly")
            #             raise Exception("Server runtime error")
            #         elif router.poll() is not None:
            #             print_error("Router terminated unexpectedly")
            #             raise Exception("Router runtime error")
            #         else:
            #             client_code = client.poll()
            #             if client_code is None:
            #                 sleep(1)
            #             elif client_code is not None and client_code == 0:
            #                 print_info("Client exited")
            #                 break
            #             else:
            #                 raise Exception("Client returncode error")
            # except Exception as e:
            #     with open(err_file, "a") as f:
            #         f.write(f"Test failed\n")
            #         f.write(f"{instance_path}\n")
            #         f.write(f"{e}\n\n")
            #     success = False
            # finally:
            #     if router is not None:
            #         router.terminate()
            #         router_monitor.join()
            #     if client is not None:
            #         client.terminate()
            #         client_monitor.join()
            #     if server is not None:
            #         server.terminate()
            #         server_stdout_monitor.join()
            #         server_stderr_monitor.join()
            #         print_info("Server terminated.")
            #         wait_until_released(sleep_after_release)
            #     if success is True:
            #         print_info("Test passed")
            #     else:
            #         print_error("Test failed")
            #         continue

            # with open(f"{instance_path}/client_prologue.log", "r") as f:
            #     prologue = f.read()
            # with open(f"{instance_path}/router.log", "a") as f:
            #     f.write(prologue)
            # plot_main(
            #     f"{instance_path}/router.log",
            #     f"{instance_path}/client.jsonl",
            #     f"{instance_path}/fig.pdf",
            # )
            # # Append log
            # checkpoint.add(instance_path)
            # with open(checkpoint_file, "a") as f:
            #     f.write(f"{instance_path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates",
        nargs="+",
        help="Path to the template file(s)",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to archive home",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint and run all instances",
        default=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the script without executing any instances",
        default=False,
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Enable colorful output",
        default=False,
    )
    args = parser.parse_args()

    if (args.templates == None and args.checkpoint == None) or (
        args.templates != None and args.checkpoint != None
    ):
        parser.error("Either `--templates` or `--checkpoint` should be enabled")

    global colorful
    colorful = args.color

    if args.templates is not None:
        configs = dict[str, list[str]]()
        archive_home = os.path.join("./log_home", f"{timestamp()}_eval")
        print_info(f"Generating configurations to: {archive_home}")
        for template_path in args.templates:
            temp = instantiate_template(template_path, archive_home)
            for key, val in temp.items():
                if key not in configs:
                    configs[key] = []
                configs[key].extend(val)
        with open(os.path.join(archive_home, "instances.toml"), "w") as f:
            toml.dump(configs, f)
        freeze(archive_home)
    else:
        archive_home = args.checkpoint
        checkpoint_path = os.path.join(archive_home, "checkpoint.txt")
        if args.force and os.path.exists(checkpoint_path):
            print_info("--force: removing checkpoint file")
            os.remove(checkpoint_path)
    run_instances(archive_home, args.dry_run)

def send_params_task(base_port, dst_rank: int):
    SERVER = "localhost:" + str(base_port)
    ch = grpc.insecure_channel(SERVER)
    stub = pb_grpc.TextGenerationServiceStub(ch)
    req = pb.SendParamsRequest(dst=dst_rank)
    return stub.SendParams.future(req)

def recv_params_task(base_port, src_rank: int):
    SERVER = "localhost:" + str(base_port)
    ch = grpc.insecure_channel(SERVER)
    stub = pb_grpc.TextGenerationServiceStub(ch)
    req = pb.RecvParamsRequest(src=src_rank)
    return stub.RecvParams.future(req)

def prefillv2(base_port, request):
    SERVER = "localhost:" + str(base_port)
    ch = grpc.insecure_channel(SERVER)
    stub = pb_grpc.TextGenerationServiceStub(ch)
    
    stub.PrefillV2(request)

def make_fake_prefillv2_request():
    req = pb.Request(
        id=1,
        inputs="Hello, world!",
        truncate=10,
        parameters=pb.NextTokenChooserParameters(
            temperature=1.0, top_k=1, top_p=1.0, typical_p=1.0, do_sample=False, seed=0, repetition_penalty=1.0, watermark=False
        ),
        stopping_parameters=pb.StoppingCriteriaParameters(
            max_new_tokens=10, stop_sequences=[], ignore_eos_token=True
        ),
        prefill_logprobs=False,
        top_n_tokens=0,
        input_tokens=[0]
    )
    batch = pb.Batch(
        requests=[req],
        id=1,
        size=1,
        max_tokens=10,
    )

    prefillv2_req = pb.PrefillV2Request(
        batch=batch,
        forward_case=0,
    )
    return prefillv2_req


def load_params(base_port: int, path: str, model_name: str):
    SERVER = "localhost:" + str(base_port)
    ch = grpc.insecure_channel(SERVER)
    stub = pb_grpc.TextGenerationServiceStub(ch)
    
    req = pb.LoadParamsRequest(
        load_case=pb.LOAD_FROM_DISK,
        model_name=model_name,
        model_path=path,
    )
    return stub.LoadParams(req)

def test_load_ttft(base_port: int):
    print_info("Test LoadParams From Disk")

    start_time = time.time()

    load_params(base_port, "/mnt/disk/zkx/output_llama/llama3-8b", "llama3-8b")
    print_info("LoadParams...")
    print_info("Make Requests")
    req = make_fake_prefillv2_request()
    method = "LoadParams from Disk"
    prefillv2(base_port, req)
    print_info("Request Done!")
    
    end_time = time.time()

    duration = end_time - start_time
    print_info(f"RANK[{0}], method: \"{method}\" TTFT: {duration:.3f} seconds]")



def test_nvlink_ttft(base_port:int):
    print_info("Test SendParams Using NVLink")
    reset_status(base_port + 1)
    start_time = time.time()

    send_future = send_params_task(base_port, 1)
    recv_future = recv_params_task(base_port + 1, 0)
    
    send_future.result()
    recv_future.result()

    print_info("Make Requests")
    req = make_fake_prefillv2_request()
    prefillv2(base_port + 1, req)
    print_info("Request Done!")
    end_time = time.time()

    method = "NVLink P2P SendParams"
    duration = end_time - start_time
    print_info(f"RANK[{1}], method: \"{method}\" TTFT: {duration:.3f} seconds]")

def reset_status(base_port: int):
    SERVER = "localhost:" + str(base_port)
    ch = grpc.insecure_channel(SERVER)
    stub = pb_grpc.TextGenerationServiceStub(ch)
    req = pb.ResetStatusRequest()
    return stub.ResetStatus(req)


if __name__ == "__main__":
    main()


