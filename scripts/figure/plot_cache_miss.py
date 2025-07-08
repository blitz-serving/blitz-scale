import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def aggregate_dataset(dataset_path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["second"] = df["timestamp"].astype(int) // 1000
    result = (
        df.groupby("second")
        .agg(
            num_requests_per_sec=("timestamp", "size"),
            prefill=("prefill", "sum"),
        )
        .reset_index()
    )
    result.rename(columns={"second": "timestamp"}, inplace=True)
    result["node_needed"] = np.ceil(result["prefill"] * 4 / 13000).astype(int)

    return result


def get_cache_miss_df(dataset_df) -> pd.DataFrame:
    df2 = dataset_df
    scale_up_count = 0
    cache_miss_count = 0

    cache_miss_df = pd.DataFrame(
        columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"]
    )

    max_cache_miss_rate = 0

    for i in range(1, len(df2)):
        current_time = df2.loc[i, "timestamp"]
        current_node_needed = df2.loc[i, "node_needed"]
        previous_node_needed = df2.loc[i - 1, "node_needed"]

        if current_node_needed > previous_node_needed:
            scale_up = current_node_needed - previous_node_needed
            scale_up_count += scale_up

            past_five_seconds_max = df2.loc[
                (df2["timestamp"] > current_time - 5 * 60)
                & (df2["timestamp"] < current_time),
                "node_needed",
            ].max()
            if past_five_seconds_max < current_node_needed:
                cache_miss = current_node_needed - past_five_seconds_max
                cache_miss_count += cache_miss
                new_row = pd.DataFrame(
                    [[current_time, scale_up, cache_miss, current_node_needed]],
                    columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"],
                )
                cache_miss_df = pd.concat([cache_miss_df, new_row], ignore_index=True)
                if cache_miss > 1:
                    max_cache_miss_rate = max(
                        max_cache_miss_rate, cache_miss / current_node_needed
                    )
            else:
                new_row = pd.DataFrame(
                    [[current_time, scale_up, 0, current_node_needed]],
                    columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"],
                )
                cache_miss_df = pd.concat([cache_miss_df, new_row], ignore_index=True)
    cache_miss_df["cache_miss_rate"] = (
        cache_miss_df["cache_miss_n"] / cache_miss_df["scale_up_n"]
    )
    print("Max cache miss rate {}%".format(max_cache_miss_rate * 100))
    print("Cache miss rate {}%".format(cache_miss_count / scale_up_count * 100))
    return cache_miss_df


def plot_cache_miss() -> pd.DataFrame:
    dataset_df = aggregate_dataset("temp/BurstGPT2.csv")
    cache_miss_df = get_cache_miss_df(dataset_df)

    cache_miss_df = fill_timestamps(cache_miss_df)
    cache_miss_df["cache_miss_rate"].fillna(0, inplace=True)
    cache_miss_df["cache_miss_n"].fillna(0, inplace=True)
    cache_miss_df["node_needed"].fillna(0, inplace=True)

    dataset_df = fill_timestamps(dataset_df)
    dataset_df["node_needed"].fillna(0, inplace=True)
    dataset_df["num_requests_per_sec"].fillna(0, inplace=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax0.plot(
        dataset_df["timestamp"],
        dataset_df["num_requests_per_sec"],
        color="blue",
    )
    ax0.set_title("Number of Requests per Second")
    ax0.set_xlabel("Timestamp")
    ax0.set_ylabel("Requests per Second")

    ax1.plot(
        dataset_df["timestamp"],
        dataset_df["node_needed"],
        color="green",
    )
    ax1.plot(
        cache_miss_df["timestamp"],
        cache_miss_df["cache_miss_n"],
        color="red",
    )
    ax1.set_title("Nodes Needed")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Node needed")

    print(cache_miss_df)
    fig.savefig("temp/BurstGPT2_cache_miss_114514.pdf")


def fill_timestamps(df) -> pd.DataFrame:
    all_timestamps = pd.Series(
        range(df["timestamp"].min() - 1, df["timestamp"].max() + 2)
    )
    full_df = pd.DataFrame({"timestamp": all_timestamps})
    merged_df = full_df.merge(df, on="timestamp", how="left")
    return merged_df


if __name__ == "__main__":
    plot_cache_miss()
