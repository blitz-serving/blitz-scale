import re
import pandas as pd

p_up_pattern = r".*\[(\d+)\]->\[(\d+)\] => Prefill"
d_up_pattern = r".*\[(\d+)\]->\[(\d+)\] => Decode"  # scale naive only
d_mutate_pattern = r".*\[(\d+)\] => Mutate"  # blitz only
p_down_pattern = r".*\[(\d+)\] => Trigger Shutting Prefill"  # trigger shuttingPrefill
d_down_pattern = r".*\[(\d+)\] => Trigger Shutting Decode"  # trigger shuttingDecode


def read_log(log_path):
    with open(log_path, "r") as f:
        # read .log file
        lines = f.readlines()

    scale_events = []
    for line in lines:
        if "=> Prefill" in line:
            match = re.match(p_up_pattern, line)
            if match:
                timestamp = line.split(" ")[0]
                scale_events.append(
                    {
                        "event": "prefill_up",
                        "timestamp": timestamp,
                        "src": match.group(1),
                        "dst": match.group(2),
                    }
                )
        elif "=> Decode" in line:
            match = re.match(d_up_pattern, line)
            if match:
                timestamp = line.split(" ")[0]
                scale_events.append(
                    {
                        "event": "decode_up",
                        "timestamp": timestamp,
                        "src": match.group(1),
                        "dst": match.group(2),
                    }
                )
        elif "=> Mutate" in line:
            match = re.match(d_mutate_pattern, line)
            if match:
                timestamp = line.split(" ")[0]
                scale_events.append(
                    {
                        "event": "mutate",
                        "timestamp": timestamp,
                        "src": match.group(1),
                        "dst": match.group(1),
                    }
                )
        elif "=> Shut Prefill" in line:
            match = re.match(p_down_pattern, line)
            if match:
                timestamp = line.split(" ")[0]
                scale_events.append(
                    {
                        "event": "prefill_down",
                        "timestamp": timestamp,
                        "src": match.group(1),
                        "dst": match.group(1),
                    }
                )
        elif "=> Shut Decode" in line:
            match = re.match(d_down_pattern, line)
            if match:
                timestamp = line.split(" ")[0]
                scale_events.append(
                    {
                        "event": "decode_down",
                        "timestamp": timestamp,
                        "src": match.group(1),
                        "dst": match.group(1),
                    }
                )
        elif "Client start" in line:
            timestamp = line.split(" ")[0]
            scale_events.append(
                {
                    "event": "client_start",
                    "timestamp": timestamp,
                    "src": None,
                    "dst": None,
                }
            )
    df = pd.DataFrame(scale_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # find the first client_start event
    start_time = df[df["event"] == "client_start"]["timestamp"].min()

    df["s_time"] = (df["timestamp"] - start_time).dt.total_seconds() * 1000
    start_time = df[df["event"] != "client_start"]["s_time"].min()
    df["s_time"] = df["s_time"] - start_time
    df["s_time"] = df["s_time"].astype(int)
    return df


def write_csv(df, output_path):
    # write all columns except 'timestamp'
    df = df.drop(columns=["timestamp"])
    # drop client_start events
    df = df[df["event"] != "client_start"]
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise ("Usage: python extract_scale_events.py <log_path> <output_path>")
    log_path = sys.argv[1]
    output_path = sys.argv[2]
    df = read_log(log_path)
    write_csv(df, output_path)
