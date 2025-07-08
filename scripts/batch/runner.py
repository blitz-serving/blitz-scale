import json
import os
import shutil
import subprocess
import threading
from time import sleep
from typing import Any
import toml
import re
import rank_to_rnic


def _remove_ansi_sequences(text) -> str:
    ansi_escape = re.compile(r"\x1b\[([0-9;]*m)")
    return ansi_escape.sub("", text)


def _monitor_router(stderr, log_path):
    with open(log_path, "w") as f:
        for line in stderr:
            f.write(_remove_ansi_sequences(line))
            f.flush()


def _monitor_client(stdout, log_path, prologue_path):
    with open(log_path, "w") as f:
        for line in stdout:
            f.write(_remove_ansi_sequences(line))
            f.flush()
            if "Client start" in line:
                prologue = open(prologue_path, "w")
                prologue.write(_remove_ansi_sequences(line))
                prologue.flush()
                prologue.close()


def _monitor_server_stdout(stdout, log_path, expected, event: threading.Event):
    i = 0
    with open(log_path, "w") as f:
        for line in stdout:
            f.write(line)
            f.flush()
            if "Start gRPC server" in line:
                i += 1
                if i == expected:
                    event.set()


def _monitor_server_stderr(stderr, log_path):
    with open(log_path, "w") as f:
        for line in stderr:
            f.write(_remove_ansi_sequences(line))
            f.flush()


def _get_extra_envs(configuration_path: str) -> dict[str, str]:
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    cuda_visible_devices: list[int] = config["global"]["cuda_devices"]
    envs = config["extra-envs"]
    envs["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in cuda_visible_devices])

    mapping = rank_to_rnic.rank_to_rnic_name()
    rank = 0
    for gpu, ibv_name in mapping.items():
        envs["RNIC_NAMES_FOR_RANK_" + str(rank)] = ibv_name
        rank += 1

    return envs


def _get_git_info() -> str:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8")
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode("utf-8")
        modified_files = subprocess.check_output(
            ["git", "status", "--porcelain"]
        ).decode("utf-8")
        return f"Branch: {branch}\nCommit: {commit}\nStatus:\n{modified_files}"
    except subprocess.CalledProcessError as e:
        return "Error occurred while getting git info"


def _get_diff() -> str:
    try:
        result = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return "Error occurred while getting git diff"


def _copy_untracked_files(destination):
    try:
        if not os.path.exists(destination):
            os.makedirs(destination)
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
        )
        untracked_files = result.stdout.splitlines()

        for file_path in untracked_files:
            dest_path = os.path.join(destination, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)
    except Exception as _:
        print("Failed to copy untracked files, but it's no big deal. :)")


def freeze(archive_dir: str):
    with open(os.path.join(archive_dir, "git-info.txt"), "w") as f:
        f.write(_get_git_info())
    with open(os.path.join(archive_dir, "git-diff.patch"), "w") as f:
        f.write(_get_diff())
    _copy_untracked_files(os.path.join(archive_dir, "untracked"))


def _init_config(
    cuda_devices: list[int],
    num_gpus_per_node: int,
    tp_size: int,
    inter_size: int,
    init_prefill_num: int,
    init_decode_num: int,
) -> dict[str, Any]:
    init_states = []
    replicas = []
    machines = []
    if len(cuda_devices) % num_gpus_per_node != 0:
        raise Exception("Number of GPUs cannot be divided by num_gpus_per_node evenly")
    if len(cuda_devices) % tp_size != 0:
        raise Exception("Number of GPUs cannot be divided by tp_size evenly")
    num_replicas = inter_size * len(cuda_devices) // tp_size
    for i in range(num_replicas):
        if i < init_prefill_num:
            states = ["Prefill"] * tp_size
        elif i < init_prefill_num + init_decode_num:
            states = ["Decode"] * tp_size
        else:
            states = ["Inactive"] * tp_size
        ranks = [x for x in range(i * tp_size, (i + 1) * tp_size)]
        rank_on_machine = [m // num_gpus_per_node for m in ranks]
        init_states.extend(states)
        replicas.append(ranks)
        machines.extend(rank_on_machine)
    return {
        "init_states": init_states,
        "replicas": replicas,
        "machines": machines,
    }


# def reset_server(configuration_path: str, base_port=50051):
#     import grpc
#     from grpc import RpcError
#     import generate_pb2
#     import generate_pb2_grpc
#     config = None
#     with open(configuration_path, "r") as f:
#         config = toml.load(f)
#     envs = _get_extra_envs(configuration_path)
#     envs.update(os.environ.copy())
#     cuda_devices: list[int] = config["global"]["cuda_devices"]
#     if config["server"]["inter_node"] == False:
#         inter_size = 1
#     else:
#         inter_size = len(config["server"]["config"])
#     server_init_config = _init_config(
#         cuda_devices,
#         config["global"]["num_gpus_per_node"],
#         config["model"]["tp_size"],
#         inter_size,
#         config["router"]["min_prefill_num"],
#         config["router"]["min_decode_num"],
#     )
#     stubs = []
#     _stubs = []
#     if config["server"]["inter_node"] == False:
#         nodes = None
#     else:
#         nodes = config["server"]["config"]
#     if nodes is not None:
#         for _, ip in nodes.items():
#             for i in range(len(cuda_devices)):
#                 stubs.append(f"http://{ip}:{base_port + i}")
#                 _stubs.append(f"{ip}:{base_port + i}")
#     else:
#         for i in range(len(cuda_devices)):
#             stubs.append(f"http://localhost:{base_port + i}")
#             _stubs.append(f"localhost:{base_port + i}")
#     # Reset server status
#     init_states = server_init_config["init_states"]
#     for i, state in enumerate(init_states):
#         with grpc.insecure_channel(_stubs[i]) as channel:
#             try:
#                 grpc.channel_ready_future(channel).result(timeout=10)
#             except RpcError as e:
#                 raise RuntimeError(f"Failed to connect to the stub {_stubs[i]}") from e
#             if state == "Inactive":
#                 generate_pb2_grpc.TextGenerationServiceStub(channel).ResetStatus(
#                     generate_pb2.ResetStatusRequest()
#                 )
#             else:
#                 generate_pb2_grpc.TextGenerationServiceStub(channel).SetStatusReady(
#                     generate_pb2.SetStatusReadyRequest()
#                 )
#     sleep(3)


def _dump_server_bash_script(configuration_path: str, base_port):
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    export_envs = "\n".join([f"export {key}={val}" for key, val in envs.items()]) + "\n"
    envs.update(os.environ.copy())

    # inter_size and nodes
    if config["server"]["inter_node"] == False:
        nodes = None
    else:
        nodes = config["server"]["config"]

    # Dump server initialization config
    cuda_devices: list[int] = config["global"]["cuda_devices"]
    if config["server"]["inter_node"] == False:
        inter_size = 1
    else:
        inter_size = len(config["server"]["config"])
    server_init_config = _init_config(
        cuda_devices,
        config["global"]["num_gpus_per_node"],
        config["model"]["tp_size"],
        inter_size,
        config["router"]["min_prefill_num"],
        config["router"]["min_decode_num"],
    )
    with open(os.path.join(archive_dir, "config-server.json"), "w") as f:
        f.write(json.dumps(server_init_config, indent=4))

    stubs = []
    if nodes is not None:
        for _, ip in nodes.items():
            for i in range(len(cuda_devices)):
                stubs.append(f"http://{ip}:{base_port + i}")
    else:
        for i in range(len(cuda_devices)):
            stubs.append(f"http://localhost:{base_port + i}")
    with open(os.path.join(archive_dir, "config-stubs.json"), "w") as f:
        f.write(json.dumps(stubs, indent=4))
    if config["server"]["inter_node"] == True:
        _dump_host_file(
            slots_num=len(cuda_devices),
            workdir=archive_dir,
            server_config=config["server"]["config"],
        )
        inter_launch_command = [
            os.path.abspath("./build/release/bin/run_server_disaggregative"),
            "-T",
            str(config["model"]["tokenizer"]),
            "-V",
            "''",
            "-TP",
            str(config["model"]["tp_size"]),
            "--host",
            "0.0.0.0",
            "--port",
            str(base_port),
            "--model-path",
            "''",
            "--model-name",
            str(config["model"]["model_name"]),
            "-P",
            "fp16",
            "--config",
            os.path.join(archive_dir, "config-server.json"),
            "--num-total-blocks",
            str(config["model"]["num_available_blocks"]),
            "--ibv-rate",
            str(config["server"]["ibv_rate"]),
        ]
        with open(os.path.join(archive_dir, "inter_launch_command.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(export_envs)
            f.write(" ".join(inter_launch_command) + "\n")
        inter_launch_command = [
            "mpirun",
            "--allow-run-as-root",
            "--hostfile",
            os.path.abspath(f"{archive_dir}/hostfile.txt"),
            "--map-by",
            "slot",
            "--mca",
            "btl_tcp_if_include=eth0",
            str(len(config["server"]["config"]) * len(cuda_devices)),
            "bash",
            os.path.abspath(f"{archive_dir}/inter_launch_command.sh"),
        ]
        with open(os.path.join(archive_dir, "launch_server.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(export_envs)
            f.write(" ".join(inter_launch_command) + "\n")
    else:
        # Dump launch_server.sh and launch
        intra_launch_command = [
            "mpirun",
            "-n",
            str(len(cuda_devices)),
            "--allow-run-as-root",
            os.path.abspath("./build/release/bin/run_server_disaggregative"),
            "-T",
            str(config["model"]["tokenizer"]),
            "-V",
            "''",
            "-TP",
            str(config["model"]["tp_size"]),
            "--host",
            "0.0.0.0",
            "--port",
            str(base_port),
            "--model-path",
            "''",
            "--model-name",
            str(config["model"]["model_name"]),
            "-P",
            "fp16",
            "--config",
            os.path.join(archive_dir, "config-server.json"),
            "--num-total-blocks",
            str(config["model"]["num_available_blocks"]),
            "--ibv-rate",
            str(config["server"]["ibv_rate"]),
        ]
        with open(os.path.join(archive_dir, "launch_server.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(export_envs)
            f.write(" ".join(intra_launch_command) + "\n")


def launch_server(
    configuration_path: str, base_port=50051
) -> tuple[subprocess.Popen, threading.Thread, threading.Thread]:
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    envs.update(os.environ.copy())

    cuda_devices: list[int] = config["global"]["cuda_devices"]
    expected = 0
    if config["server"]["inter_node"] == True:
        expected = len(config["server"]["config"]) * len(cuda_devices)
        inter_launch_command = [
            "mpirun",
            "--allow-run-as-root",
            "--hostfile",
            os.path.abspath(f"{archive_dir}/hostfile.txt"),
            "--map-by",
            "slot",
            "--mca",
            "btl_tcp_if_include=eth0",
            str(expected),
            "bash",
            os.path.abspath(f"{archive_dir}/inter_launch_command.sh"),
        ]
        server_process = subprocess.Popen(
            inter_launch_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=envs,
        )
    else:
        expected = len(cuda_devices)
        intra_launch_command = [
            "mpirun",
            "-n",
            str(expected),
            "--allow-run-as-root",
            os.path.abspath("./build/release/bin/run_server_disaggregative"),
            "-T",
            str(config["model"]["tokenizer"]),
            "-V",
            "''",
            "-TP",
            str(config["model"]["tp_size"]),
            "--host",
            "0.0.0.0",
            "--port",
            str(base_port),
            "--model-path",
            "''",
            "--model-name",
            str(config["model"]["model_name"]),
            "-P",
            "fp16",
            "--config",
            os.path.join(archive_dir, "config-server.json"),
            "--num-total-blocks",
            str(config["model"]["num_available_blocks"]),
            "--ibv-rate",
            str(config["server"]["ibv_rate"]),
        ]
        server_process = subprocess.Popen(
            intra_launch_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=envs,
        )
    event = threading.Event()
    stdout_monitor = threading.Thread(
        target=_monitor_server_stdout,
        args=(
            server_process.stdout,
            os.path.join(archive_dir, "server.log"),
            expected,
            event,
        ),
    )
    stdout_monitor.start()
    stderr_monitor = threading.Thread(
        target=_monitor_server_stderr,
        args=(
            server_process.stderr,
            os.path.join(archive_dir, "server.error.log"),
        ),
    )
    stderr_monitor.start()
    if event.wait(timeout=60) == False:
        raise Exception("Server failed to start")
    sleep(3)
    return server_process, stdout_monitor, stderr_monitor


def _dump_client_bash_script(configuration_path: str):
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    export_envs = "\n".join([f"export {key}={val}" for key, val in envs.items()]) + "\n"
    envs.update(os.environ.copy())

    # Dump build_client.sh and build
    build_command = [
        "cargo",
        "build",
        "--release",
        "--package",
        "request-sim",
        "--bin",
        "client",
    ]
    with open(os.path.join(archive_dir, "build_client.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(build_command) + "\n")

    # Dump launch_client.sh and launch
    launch_command = [
        "./target/release/client",
        "--tokenizer",
        str(config["model"]["tokenizer"]),
        "--endpoint",
        "http://localhost:{}/generate".format(config["router"]["port"]),
        "--protocol",
        "st",
        "--replay-mode",
        "--scale-factor",
        str(config["dataset"]["scale_factor"]),
        "--dataset-type",
        "mock",
        "--dataset-path",
        str(config["dataset"]["dataset_path"]),
        "--time-in-secs",
        str(config["dataset"]["time_in_secs"]),
        "--truncate",
        "4095",
        "--output-path",
        os.path.join(archive_dir, "client.jsonl"),
    ]
    with open(os.path.join(archive_dir, "launch_client.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(export_envs)
        f.write(" ".join(launch_command) + "\n")


def launch_client(configuration_path: str) -> tuple[subprocess.Popen, threading.Thread]:
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    envs.update(os.environ.copy())

    build_command = [
        "cargo",
        "build",
        "--release",
        "--package",
        "request-sim",
        "--bin",
        "client",
    ]
    res = subprocess.run(build_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise Exception("Failed to build client")

    launch_command = [
        "./target/release/client",
        "--tokenizer",
        str(config["model"]["tokenizer"]),
        "--endpoint",
        "http://localhost:{}/generate".format(config["router"]["port"]),
        "--protocol",
        "st",
        "--replay-mode",
        "--scale-factor",
        str(config["dataset"]["scale_factor"]),
        "--dataset-type",
        "mock",
        "--dataset-path",
        str(config["dataset"]["dataset_path"]),
        "--time-in-secs",
        str(config["dataset"]["time_in_secs"]),
        "--truncate",
        "4095",
        "--output-path",
        os.path.join(archive_dir, "client.jsonl"),
    ]
    client_process = subprocess.Popen(
        launch_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=envs,
    )
    monitor_thread = threading.Thread(
        target=_monitor_client,
        args=(
            client_process.stdout,
            os.path.join(archive_dir, "client.log"),
            os.path.join(archive_dir, "client_prologue.log"),
        ),
    )
    monitor_thread.start()
    sleep(1)
    return client_process, monitor_thread


def _dump_router_bash_script(configuration_path: str):
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    export_envs = "\n".join([f"export {key}={val}" for key, val in envs.items()]) + "\n"
    envs.update(os.environ.copy())

    # Dump build_router.sh and build
    feature = config["router"]["feature"]
    build_command = [
        "cargo",
        "build",
        "--release",
        "--package",
        "router_v2",
        "--no-default-features",
        "--features",
        feature,
    ]
    with open(os.path.join(archive_dir, "build_router.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(build_command) + "\n")

    # Dump router initialization config
    if config["server"]["inter_node"] == False:
        inter_size = 1
    else:
        inter_size = len(config["server"]["config"])
    router_init_config = _init_config(
        config["global"]["cuda_devices"],
        config["global"]["num_gpus_per_node"],
        config["model"]["tp_size"],
        inter_size,
        config["router"]["min_prefill_num"],
        config["router"]["min_decode_num"],
    )
    with open(os.path.join(archive_dir, "config-router.json"), "w") as f:
        f.write(json.dumps(router_init_config, indent=4))

    # Dump launch_router.sh and launch
    launch_command = [
        "./target/release/router_v2",
        "--hostname",
        "localhost",
        "--port",
        str(config["router"]["port"]),
        "--use-tokenizer",
        "--deployment",
        "disaggregation",
        "--tokenizer-name",
        config["model"]["model_path"],
        "--client-config",
        os.path.join(archive_dir, "config-stubs.json"),
        "--deployment-config-path",
        os.path.join(archive_dir, "config-router.json"),
        "--log-path",
        os.path.join(archive_dir, "router.log"),
        "--max-input-length",
        "4090",
        "--max-total-tokens",
        "4096",
        "--max-concurrent-requests",
        "4096",
        "--tokens-prefilled-per-sec",
        str(config["model"]["tokens_prefilled_per_sec"]),
        "--tokens-transferred-per-sec",
        str(config["model"]["tokens_transferred_per_sec"]),
        "--max-blocks-per-replica",
        str(config["model"]["num_available_blocks"]),
        "--max-prefill-num",
        str(config["router"]["max_prefill_num"]),
        "--max-decode-num",
        str(config["router"]["max_decode_num"]),
        "--min-prefill-num",
        str(config["router"]["min_prefill_num"]),
        "--min-decode-num",
        str(config["router"]["min_decode_num"]),
        "--prefill-lower-bound",
        str(config["router"]["prefill_lower_bound"]),
        "--prefill-upper-bound",
        str(config["router"]["prefill_upper_bound"]),
        "--decode-lower-bound",
        str(config["router"]["decode_lower_bound"]),
        "--decode-upper-bound",
        str(config["router"]["decode_upper_bound"]),
        "--migration-lower-bound",
        str(config["router"]["migration_lower_bound"]),
        "--migration-upper-bound",
        str(config["router"]["migration_upper_bound"]),
        "--scale-down-threshold-millis",
        str(config["router"]["scale_down_threshold_millis"]),
        "--num-hidden-layers",
        str(config["model"]["num_hidden_layers"]),
        "--num-gpus-per-node",
        str(config["global"]["num_gpus_per_node"]),
        "--mock-load-millis",
        str(config["router"]["mock_load_millis"]),
        "--mock-transfer-millis",
        str(config["router"]["mock_transfer_millis"]),
        "--tensor-parallel-size",
        str(config["model"]["tp_size"]),
        "--model-name",
        config["model"]["model_name"],
        "--model-path",
        config["model"]["model_path"],
        "--parameter-size",
        str(config["model"]["parameter_size"]),
    ]
    with open(os.path.join(archive_dir, "launch_router.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        f.write(export_envs)
        f.write(" ".join(launch_command) + "\n")


def launch_router(configuration_path: str) -> tuple[subprocess.Popen, threading.Thread]:
    # Read config
    config = None
    with open(configuration_path, "r") as f:
        config = toml.load(f)
    archive_dir = config["global"]["archive_dir"]
    envs = _get_extra_envs(configuration_path)
    envs.update(os.environ.copy())

    feature = config["router"]["feature"]
    build_command = [
        "cargo",
        "build",
        "--release",
        "--package",
        "router_v2",
        "--no-default-features",
        "--features",
        feature,
    ]
    builder = subprocess.run(
        build_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if builder.returncode != 0:
        raise Exception("Failed to build router")

    launch_command = [
        "./target/release/router_v2",
        "--hostname",
        "localhost",
        "--port",
        str(config["router"]["port"]),
        "--use-tokenizer",
        "--deployment",
        "disaggregation",
        "--tokenizer-name",
        config["model"]["model_path"],
        "--client-config",
        os.path.join(archive_dir, "config-stubs.json"),
        "--deployment-config-path",
        os.path.join(archive_dir, "config-router.json"),
        "--log-path",
        os.path.join(archive_dir, "router.log"),
        "--max-input-length",
        "4090",
        "--max-total-tokens",
        "4096",
        "--max-concurrent-requests",
        "4096",
        "--tokens-prefilled-per-sec",
        str(config["model"]["tokens_prefilled_per_sec"]),
        "--tokens-transferred-per-sec",
        str(config["model"]["tokens_transferred_per_sec"]),
        "--max-blocks-per-replica",
        str(config["model"]["num_available_blocks"]),
        "--max-prefill-num",
        str(config["router"]["max_prefill_num"]),
        "--max-decode-num",
        str(config["router"]["max_decode_num"]),
        "--min-prefill-num",
        str(config["router"]["min_prefill_num"]),
        "--min-decode-num",
        str(config["router"]["min_decode_num"]),
        "--prefill-lower-bound",
        str(config["router"]["prefill_lower_bound"]),
        "--prefill-upper-bound",
        str(config["router"]["prefill_upper_bound"]),
        "--decode-lower-bound",
        str(config["router"]["decode_lower_bound"]),
        "--decode-upper-bound",
        str(config["router"]["decode_upper_bound"]),
        "--migration-lower-bound",
        str(config["router"]["migration_lower_bound"]),
        "--migration-upper-bound",
        str(config["router"]["migration_upper_bound"]),
        "--scale-down-threshold-millis",
        str(config["router"]["scale_down_threshold_millis"]),
        "--num-hidden-layers",
        str(config["model"]["num_hidden_layers"]),
        "--num-gpus-per-node",
        str(config["global"]["num_gpus_per_node"]),
        "--mock-load-millis",
        str(config["router"]["mock_load_millis"]),
        "--mock-transfer-millis",
        str(config["router"]["mock_transfer_millis"]),
        "--tensor-parallel-size",
        str(config["model"]["tp_size"]),
        "--model-name",
        config["model"]["model_name"],
        "--model-path",
        config["model"]["model_path"],
        "--parameter-size",
        str(config["model"]["parameter_size"]),
    ]
    router_process = subprocess.Popen(
        launch_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=envs,
    )
    monitor_thread = threading.Thread(
        target=_monitor_router,
        args=(
            router_process.stderr,
            os.path.join(archive_dir, "router.error.log"),
        ),
    )
    monitor_thread.start()
    sleep(3)
    return router_process, monitor_thread


def dump_bash_scripts(configuration_path: str, base_port=50051):
    _dump_server_bash_script(configuration_path, base_port)
    _dump_client_bash_script(configuration_path)
    _dump_router_bash_script(configuration_path)


def _dump_host_file(slots_num, workdir, server_config):
    with open(f"{workdir}/hostfile.txt", "w") as f:
        for hostname, _ in server_config.items():
            f.write(f"{hostname} slots={slots_num} max_slots={slots_num}\n")


if __name__ == "__main__":
    print(_init_config([0, 1, 2, 3, 4, 5, 6, 7], 4, 2, 2, 1, 1))
