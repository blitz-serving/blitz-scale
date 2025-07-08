import datetime
import numpy as np
import pandas as pd
import copy
import matplotlib.dates as mdates


def extract_collocation_log(log_file, zoom_out_millis=None):
    p2d_migration_dict = {}
    p2p_migration_dict = {}

    def _extract_action(log_line):
        timestamp = log_line.split()[0]
        if "Scale up complete" in log_line:
            action = "complete"
        elif (
            "Trigger" in log_line and "scale up" in log_line
        ) or "scale up..." in log_line:
            action = "up"
        elif "Trigger replica" in log_line and  "scale down" in log_line:
            action = "down"
        elif "Client start" in log_line:
            action = "init"
        elif "to normal" in log_line:
            action = "relive"
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
    dt = datetime.datetime.strptime(log_data[0].split()[0], "%Y-%m-%dT%H:%M:%S.%fZ")
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
                and (timestamp - dt).total_seconds() * 1000 > zoom_out_millis
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
        dt,
        action_result,
        metrics,
        waiting_prefill,
        p2d_migration_dict,
        p2p_migration_dict if len(p2p_migration_dict) > 0 else None,
    )
    
    
    
def plot_collocation_scale_events(
    events, scale_events, start_time, ax=None, ax2=None, usable_replica_num=None
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
    replica_num = [{"timestamp": 0, "replica_num": 1}]
    replica_num_count = 1
    filter_events = copy.deepcopy(events)
    filter_events.extend(["complete", "prefill down", "decode down", "mutation"])

    for event_time, event_type in new_scale_events:
        make_label = False
        if event_type not in filter_events:
            continue
        if event_type == "up":
            color = "g"
            marker = "^"
            label = "scale up"
            replica_num_count += 1
            replica_num.append(
                {"timestamp": event_time, "replica_num": replica_num_count}
            )
            if up_label is False:
                make_label = True
                up_label = True
        elif event_type == "down":
            color = "r"
            marker = "v"
            label = "scale down"
            replica_num_count -= 1
            replica_num.append(
                {"timestamp": event_time, "replica_num": replica_num_count}
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
        elif event_type == "relive":
            color = "cyan"
            label = "mutation"
            replica_num_count += 1
            replica_num.append(
                {"timestamp": event_time, "replica_num": replica_num_count}
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

        rn = [item["replica_num"] for item in replica_num]
        replica_num_df = pd.DataFrame.from_records(replica_num)
        
        replica_num_df['timestamp'] = pd.to_datetime(replica_num_df['timestamp'], unit='ms')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax2.step(
            replica_num_df["timestamp"],
            replica_num_df["replica_num"],
            where="post",
            label="Replica Number",
            linestyle="-",
            color="g",
            alpha = 0.7
        )
        total_time = (
            replica_num_df["timestamp"].max()
            - replica_num_df["timestamp"].min()
        )
        total_area = total_time * usable_replica_num
        used_area = np.trapz(
            replica_num_df["replica_num"], replica_num_df["timestamp"]
        )
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