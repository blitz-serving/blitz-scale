import numpy as np
import pandas as pd
import copy


def calculate_uniform_prefill_thpt(
    start_time, cli_df, input_length: int, output_length: int = 1
):
    new_cli_df = copy.deepcopy(cli_df)
    new_cli_df["s_time"] = new_cli_df["s_time"] + new_cli_df["first_token_time"]
    new_cli_df = new_cli_df.sort_values(by="s_time")
    thpts = []

    s_times = new_cli_df["s_time"].values
    for i in range(1, len(s_times) - 1):
        thpt = 2000 * input_length / (s_times[i + 1] - s_times[i - 1])
        thpts.append({"s_time": s_times[i], "thpt": thpt})

    ret = pd.DataFrame.from_dict(thpts)
    return ret


def calculate_uniform_prefill_thpt_v2(cli_df, input_length: int):
    new_cli_df = copy.deepcopy(cli_df)
    new_cli_df["s_time"] = new_cli_df["s_time"]
    new_cli_df = new_cli_df.sort_values(by="s_time")

    largest_ts = new_cli_df["e_time"].max()

    time_range = np.arange(0, largest_ts + 1, dtype=float)
    thpts = [0.0] * len(time_range)

    for i in range(len(new_cli_df) - 1):
        start = int(new_cli_df.iloc[i]["s_time"])
        end = int(new_cli_df.iloc[i]["e_time"])
        if start != end:
            thpt = input_length * 1000 / (end - start)
            for j in range(start, end):
                thpts[j] += thpt
    ret = pd.DataFrame({"s_time": time_range, "thpt": thpts})
    return ret
