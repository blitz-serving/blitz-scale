import datetime
import numpy as np
import pandas as pd
import copy
import matplotlib.dates as mdates
from uniform import *


def plot_waiting_prefill(ax, waiting_prefill, start_time):
    def _process_raw_waiting(line):
        timestamp_str = line[0]
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")

        time_diff = timestamp - start_time
        ms_diff = time_diff.total_seconds() * 1000
        return ms_diff, line[1]

    new_waiting_prefill = []
    for line in waiting_prefill:
        new_waiting_prefill.append(_process_raw_waiting(line))
    new_waiting_prefill = np.array(new_waiting_prefill)
    ax.plot(
        new_waiting_prefill[:, 0],
        new_waiting_prefill[:, 1],
        lw=0.7,
        label="waiting prefill",
    )


def plot_scale_events(
    events,
    scale_events,
    start_time,
    end_time,
    ax=None,
    ax2=None,
    usable_replica_num=None,
):
    def _process_raw_scale_events(timestamp_str, event):
        assert event is not None
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")

        time_diff = timestamp - start_time

        ms_diff = time_diff.total_seconds() * 1000
        scale_event = (ms_diff, event)

        return scale_event if scale_event else None

    new_scale_events = []
    for timestamp_str, event in scale_events:
        ret = _process_raw_scale_events(timestamp_str, event)
        if ret and ret[0] >= 0:
            new_scale_events.append(ret)
    up_label, down_label, com_label, init_label, mutation_label = (
        False,
        False,
        False,
        False,
        False,
    )
    prefill_replica_num = [{"timestamp": 0, "replica_num": 1}]
    decode_replica_num = [{"timestamp": 0, "replica_num": 1}]
    all_replica_num = [{"timestamp": 0, "replica_num": 2}]
    prefill_replica_num_count = 1
    decode_replica_num_count = 1
    filter_events = copy.deepcopy(events)
    filter_events.extend(
        ["prefill up", "prefill down", "decode down", "mutation", "decode up"]
    )

    for event_time, event_type in new_scale_events:
        make_label = False
        if event_type not in filter_events:
            continue
        if event_type == "prefill up":
            color = "g"
            marker = "^"
            label = "scale up"
            prefill_replica_num_count += 1
            prefill_replica_num.append(
                {"timestamp": event_time, "replica_num": prefill_replica_num_count}
            )
            all_replica_num.append(
                {
                    "timestamp": event_time,
                    "replica_num": prefill_replica_num_count + decode_replica_num_count,
                }
            )
            if up_label is False:
                make_label = True
                up_label = True
        if event_type == "decode up":
            color = "g"
            marker = "^"
            label = "scale up"
            decode_replica_num_count += 1
            decode_replica_num.append(
                {"timestamp": event_time, "replica_num": decode_replica_num_count}
            )
            all_replica_num.append(
                {
                    "timestamp": event_time,
                    "replica_num": prefill_replica_num_count + decode_replica_num_count,
                }
            )
            if up_label is False:
                make_label = True
                up_label = True
        elif event_type == "prefill down":
            color = "r"
            marker = "v"
            label = "scale down"
            prefill_replica_num_count -= 1
            prefill_replica_num.append(
                {"timestamp": event_time, "replica_num": prefill_replica_num_count}
            )
            all_replica_num.append(
                {
                    "timestamp": event_time,
                    "replica_num": prefill_replica_num_count + decode_replica_num_count,
                }
            )
            if down_label is False:
                make_label = True
                down_label = True
        elif event_type == "decode down":
            color = "r"
            marker = "v"
            label = "scale down"
            decode_replica_num_count -= 1
            decode_replica_num.append(
                {"timestamp": event_time, "replica_num": decode_replica_num_count}
            )
            all_replica_num.append(
                {
                    "timestamp": event_time,
                    "replica_num": prefill_replica_num_count + decode_replica_num_count,
                }
            )
            if down_label is False:
                make_label = True
                down_label = True
        elif event_type == "complete":
            color = "blue"
            marker = "o"
            label = "scale up complete"
            if com_label is False:
                make_label = True
                com_label = True
        elif event_type == "init":
            color = "black"
            marker = "h"
            label = "init time"
            if init_label is False:
                make_label = True
                init_label = True
        elif event_type == "mutation":
            color = "cyan"
            label = "mutation"
            decode_replica_num_count += 1
            prefill_replica_num_count -= 1
            decode_replica_num.append(
                {"timestamp": event_time, "replica_num": decode_replica_num_count}
            )
            prefill_replica_num.append(
                {"timestamp": event_time, "replica_num": prefill_replica_num_count}
            )
            all_replica_num.append(
                {
                    "timestamp": event_time,
                    "replica_num": prefill_replica_num_count + decode_replica_num_count,
                }
            )
            if mutation_label is False:
                make_label = True
                mutation_label = True
        elif event_type == "down complete":
            pass
        else:
            continue
        if ax and event_type in events:
            if make_label:
                ax.axvline(
                    x=event_time, label=label, color=color, linestyle="--", alpha=0.5
                )
            else:
                ax.axvline(x=event_time, color=color, linestyle="--", alpha=0.5)
        # ax.plot(event_duration, 0, marker=marker, color=color, markersize=10)
    if ax2 is not None:
        assert usable_replica_num

        # pn = [item["replica_num"] for item in prefill_replica_num]
        # dn = [item["replica_num"] for item in decode_replica_num]
        # print(f"prefill num: {pn}\ndecode num: {dn}")
        prefill_replica_num.append(
            {
                "timestamp": (end_time - start_time).total_seconds() * 1000,
                "replica_num": prefill_replica_num[-1]["replica_num"],
            }
        )
        decode_replica_num.append(
            {
                "timestamp": (end_time - start_time).total_seconds() * 1000,
                "replica_num": decode_replica_num[-1]["replica_num"],
            }
        )
        all_replica_num.append(
            {
                "timestamp": (end_time - start_time).total_seconds() * 1000,
                "replica_num": all_replica_num[-1]["replica_num"],
            }
        )

        # print(f"prefill: {prefill_replica_num}")
        # print(f"decode: {decode_replica_num}")

        prefill_replica_num_df = pd.DataFrame.from_records(prefill_replica_num)
        decode_replica_num_df = pd.DataFrame.from_records(decode_replica_num)
        all_replica_num_df = pd.DataFrame.from_records(all_replica_num)

        prefill_replica_num_df["timestamp"] = pd.to_datetime(
            prefill_replica_num_df["timestamp"], unit="ms"
        )
        decode_replica_num_df["timestamp"] = pd.to_datetime(
            decode_replica_num_df["timestamp"], unit="ms"
        )
        all_replica_num_df["timestamp"] = pd.to_datetime(
            all_replica_num_df["timestamp"], unit="ms"
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
        ax2.step(
            prefill_replica_num_df["timestamp"],
            prefill_replica_num_df["replica_num"],
            where="post",
            label="Prefill Replica",
            linestyle="-",
            color="g",
            alpha=0.7,
        )
        ax2.step(
            decode_replica_num_df["timestamp"],
            decode_replica_num_df["replica_num"],
            where="post",
            label="Decode Replica",
            linestyle="-",
            color="r",
            alpha=0.7,
        )
        ax2.step(
            all_replica_num_df["timestamp"],
            all_replica_num_df["replica_num"],
            where="post",
            label="All Replica",
            linestyle="-",
            color="b",
            alpha=0.7,
        )

        total_time = (
            all_replica_num_df["timestamp"].max()
            - all_replica_num_df["timestamp"].min()
        )
        total_area = total_time * usable_replica_num
        used_area = np.trapz(
            all_replica_num_df["replica_num"], all_replica_num_df["timestamp"]
        )
        if used_area == 0:
            usage_percentage = 0
        else:
            usage_percentage = (used_area / total_area) * 100

        ax2.text(
            0.95,
            0.95,
            f"Resource Usage: {usage_percentage:.2f}%",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        ax2.set_ylim(0, usable_replica_num)

        ax2.set_ylabel("Replica Number")
        ax2.legend(loc="upper left")


def plot_thpts(
    ax,
    thpt_metrics,
    start_time,
    case="prefill",
    calculated_by_cli_data=False,
    cli_data: pd.DataFrame = None,
):
    def _process_raw_thpt_metrics(line, case):
        assert case == "prefill" or case == "decode" or case == "both"
        timestamp_str = line[0]
        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        time_diff = timestamp - start_time
        ms_diff = time_diff.total_seconds() * 1000
        if ms_diff >= 125000:
            return None, None
        if case == "prefill":
            return (ms_diff, line[1])
        else:
            return (ms_diff, line[2])

    if calculated_by_cli_data:
        assert cli_data is not None
        thpt_df = calculate_uniform_prefill_thpt(
            start_time=start_time, cli_df=cli_data, input_length=2000
        )
        # thpt_df = calculate_uniform_prefill_thpt_v2(cli_df=cli_data, input_length=2000)
        ax.set_ylim(0, 60000)
        # ax.scatter(thpt_df["s_time"], thpt_df["thpt"], label=f"{case} thpt", alpha=0.6, s=10)
        ax.plot(thpt_df["s_time"], thpt_df["thpt"], lw=0.7, label=f"{case} thpt")
    else:
        new_thpt_metrics = []
        for line in thpt_metrics:
            ret = _process_raw_thpt_metrics(line, case)
            if ret[0] is None:
                break
            new_thpt_metrics.append(ret)
        new_thpt_metrics = np.array(new_thpt_metrics)
        ax.plot(
            new_thpt_metrics[:, 0], new_thpt_metrics[:, 1], lw=0.7, label=f"{case} thpt"
        )


def prepare_data(
    file_name, add_queue_time: bool = False, zoom_out_millis=None, latency_scale=1
) -> pd.DataFrame:

    import json

    with open(file_name, "r") as f:
        lines = f.readlines()

    cli_data = [json.loads(line) for line in lines]
    cli_df = pd.DataFrame(cli_data)
    # thpt_df = pd.read_csv(prefill_thpt)

    columns = [
        "first_token_time",
        "s_time",
        "e_time",
        "input_length",
        "inference_time",
        "max_time_between_tokens",
        "avg_time_between_tokens",
        "p95_time_between_tokens",
        "queue_time",
    ]

    cli_df = cli_df[cli_df["status"] == "200"]
    # print(f"client_df: {len(cli_df)}")
    cli_df[columns] = cli_df[columns].astype(float)
    cli_df["calculation_time"] = cli_df["first_token_time"] - cli_df["queue_time"]
    if add_queue_time:
        cli_df["s_time"] = cli_df["s_time"] + cli_df["queue_time"]
    cli_df = cli_df.sort_values(by="s_time")
    if zoom_out_millis:
        min_ts = cli_df["s_time"].min()
        cli_df = cli_df[cli_df["s_time"] <= min_ts + zoom_out_millis]

    cli_df["s_time"] = cli_df["s_time"] - cli_df["s_time"].min()
    cli_df = cli_df[cli_df["first_token_time"] > 0]
    # cli_df = cli_df[cli_df["avg_time_between_tokens"] > 0]
    if latency_scale != 1:
        cli_df["first_token_time"] = cli_df["first_token_time"] * latency_scale
        cli_df["inference_time"] = cli_df["inference_time"] * latency_scale
        cli_df["max_time_between_tokens"] = (
            cli_df["max_time_between_tokens"] * latency_scale
        )
        cli_df["avg_time_between_tokens"] = (
            cli_df["avg_time_between_tokens"] * latency_scale
        )
        cli_df["p95_time_between_tokens"] = (
            cli_df["p95_time_between_tokens"] * latency_scale
        )

    return cli_df


def extract_log(log_file, zoom_out_millis=None):
    p2d_migration_dict = {}
    p2p_migration_dict = {}

    def _extract_action(log_line):
        timestamp = log_line.split()[0]
        if "Prefill => MutatingToDecode" in log_line:
            action = "mutation"
        elif "ShuttingPrefill => Prefill" in log_line:
            return None, None
        elif "MutatingToDecode => Decode" in log_line:
            return None, None
        elif "=> Prefill" in log_line:
            action = "prefill up"
        elif "=> Decode" in log_line:
            action = "decode up"
        elif "ShuttingPrefill => ShuttingNull" in log_line:
            action = "prefill down"
        elif "ShuttingDecode => ShuttingNull" in log_line:
            action = "decode down"
        elif "Client start" in log_line:
            action = "init"
        else:
            action = None
            return None, None
        return timestamp, action

    def _extract_metrics(log_line):
        import re

        match = re.search(r"prefill_tokens: (\d+), decode_tokens: (\d+)", log_line)
        timestamp = log_line.split()[0]
        if match:
            prefill_tokens = int(match.group(1))
            decode_tokens = int(match.group(2))
            return timestamp, prefill_tokens, decode_tokens
        else:
            return None, -1, -1

    def _extract_waiting_prefill(log_line):
        import re

        match = re.search(r"Cur waiting prefill tokens: (\d+)", log_line)
        timestamp = log_line.split()[0]
        if match:
            return timestamp, int(match.group(1))
        else:
            return None, -1

    def _extract_migration(log_line):
        import re

        start_pd_match = re.search(r"Batch \[(\d+)\] p2d migration starts", log_line)
        end_pd_match = re.search(r"Batch \[(\d+)\] p2d migration ends", log_line)
        start_pp_match = re.search(r"Batch \[(\d+)\] p2p migration starts", log_line)
        end_pp_match = re.search(r"Batch \[(\d+)\] p2p migration ends", log_line)

        timestamp = log_line.split()[0]
        timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        if start_pd_match:
            batch_num = int(start_pd_match.group(1))
            p2d_migration_dict[batch_num] = [timestamp]
        elif end_pd_match:
            batch_num = int(end_pd_match.group(1))
            assert len(p2d_migration_dict[batch_num]) == 1
            p2d_migration_dict[batch_num].append(timestamp)
        elif start_pp_match:
            batch_num = int(start_pp_match.group(1))
            p2p_migration_dict[batch_num] = [timestamp]
        elif end_pp_match:
            batch_num = int(end_pp_match.group(1))
            assert len(p2p_migration_dict[batch_num]) == 1
            p2p_migration_dict[batch_num].append(timestamp)

    if log_file == "":
        return []
    with open(log_file, "r") as file:
        log_data = file.readlines()
    start_time = datetime.datetime.strptime(
        log_data[-1].split()[0], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    end_time = datetime.datetime.strptime(
        log_data[-2].split()[0], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    for line in log_data:
        _extract_migration(line)
    action_result = []
    metrics = []
    waiting_prefill = []
    for line in log_data:
        res = _extract_metrics(line)
        if res[0]:
            timestamp = datetime.datetime.strptime(res[0], "%Y-%m-%dT%H:%M:%S.%fZ")
            if (
                zoom_out_millis
                and (timestamp - start_time).total_seconds() * 1000 > zoom_out_millis
            ):
                break
        if res[0]:
            metrics.append(res)
        res = _extract_action(line)
        if res[0]:
            action_result.append(res)
        res = _extract_waiting_prefill(line)
        if res[0]:
            waiting_prefill.append(res)

    return (
        start_time,
        end_time,
        action_result,
        metrics,
        waiting_prefill,
        p2d_migration_dict,
        p2p_migration_dict if len(p2p_migration_dict) > 0 else None,
    )


def plot_ttft(ax, df, src_type="st"):
    start_time = df["s_time"].min()
    if src_type == "st":
        ax.plot(
            df["s_time"] - start_time,
            df["first_token_time"],
            lw=0.5,
            label="latency",
        )
        # ax.plot(df["s_time"] - start_time, df["queue_time"], lw=0.7, label="queue_time")
        # ax.set_ylim(0, 800)
    elif src_type == "distserve":
        ax.plot(
            df["s_time"] - start_time,
            (df["first_token_time"] + df["queue_time"]) * 1000,
            lw=0.5,
            label="latency",
        )
        ax.plot(
            df["s_time"] - start_time,
            df["queue_time"] * 1000,
            lw=0.7,
            label="queue_time",
        )


def plot_max_tbt(ax, df):
    start_time = df["s_time"].min()
    ax.plot(
        df["s_time"] - start_time,
        df["max_time_between_tokens"],
        lw=0.7,
        label="max tbt",
    )


def plot_avg_tbt(ax, df):
    start_time = df["s_time"].min()
    ax.plot(
        df["s_time"] - start_time,
        df["avg_time_between_tokens"],
        lw=0.7,
        label="avg tbt",
    )


def plot_p95_tbt(ax, df):
    start_time = df["s_time"].min()
    ax.plot(
        df["s_time"] - start_time,
        df["p95_time_between_tokens"],
        lw=0.7,
        label="p95 tbt",
    )


def plot_migration(ax, migration_dict, label):
    batches = list(migration_dict.keys())
    x = [migration_dict[batch][0] for batch in batches]
    heights = (
        (batch[1] - batch[0]).total_seconds() * 1000
        for batch in migration_dict.values()
    )

    ax.plot(x, heights, alpha=0.7, label=label)
    ax.set_ylabel("migration time (ms)")
    ax.legend()


def plot_decode_slo_violation(ax, client_df, key, watermark, scale_type: str):
    assert key in ["avg", "p95", "max"]

    client_df["timestamp"] = pd.to_datetime(client_df["s_time"], unit="ms")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))
    # start_time = client_df["s_time"].min()
    latency_key = f"{key}_time_between_tokens"

    ax.bar(
        client_df["timestamp"],
        client_df[latency_key],
        width=0.000001,
        alpha=0.3,
        color="red",
        label=f"{key} TBT slo violation",
    )

    # ax.fill_between(
    #     client_df["timestamp"],
    #     watermark,
    #     client_df[latency_key],
    #     where=(client_df[latency_key] > watermark),
    #     color="red",
    #     alpha=0.3,
    #     label=f"{key} TBT slo violation",
    # )
    violation_percentage = (client_df[latency_key] > watermark).mean() * 100

    ax.text(
        0.95,
        0.95,
        f"SLO violation rate: {violation_percentage:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    ax.axhline(y=watermark, color="r", linestyle="--", label="SLO bar")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"{scale_type} Decode SLO Violation")
    ax.legend(loc="upper left")


def plot_prefill_slo_violation(ax, client_df, watermark, scale_type: str):
    start_time = client_df["s_time"].min()
    client_df["timestamp"] = pd.to_datetime(client_df["s_time"], unit="ms")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))

    # ax.fill_between(
    #     client_df["timestamp"],
    #     watermark,
    #     client_df["first_token_time"],
    #     where=(client_df["first_token_time"] > watermark),
    #     color="red",
    #     alpha=0.3,
    #     label="TTFT SLO violation",
    # )

    ax.bar(
        client_df["timestamp"],
        client_df["first_token_time"],
        width=0.000001,
        color="red",
        alpha=0.3,
        label=f"TTFT slo violation",
    )

    # violated = client_df[
    #     (client_df["first_token_time"] > 3 * client_df["calculation_time"])
    #     & (client_df["first_token_time"] > 300)
    # ]
    # violated = client_df[
    #     (client_df["first_token_time"] > client_df["input_length"] * 0.375)
    #     & (client_df["first_token_time"] > 300)
    # ]

    violated = client_df[client_df["first_token_time"] > 200]
    violation_percentage = len(violated) / len(client_df) * 100

    # violate = []
    # for i in range(0, len(client_df)):
    #     violate.append(
    #         float(client_df["first_token_time"].iloc[i])
    #         > max(float(client_df["input_length"].iloc[i]) * 0.38, watermark)
    #     )
    # violation_percentage = np.array(violate).mean() * 100

    ax.text(
        0.95,
        0.95,
        f"SLO violation rate: {violation_percentage:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    ax.axhline(y=watermark, color="r", linestyle="--", label="SLO bar")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"{scale_type} Prefill SLO Violation")
    ax.legend(loc="upper left")
