import pandas as pd


def calculate_cache_miss_rate(dataset_path, tokens_prefilled_per_second):
    df1 = pd.read_csv(dataset_path)
    df1["timestamp"] = df1["timestamp"] // 1000
    df1.groupby("timestamp").size().reset_index(name="count").to_csv(
        "temp/BurstGPT2_count.csv", index=False
    )

    # df2 = df1.groupby("timestamp")["prefill"].sum().reset_index()
    # df2["node_needed"] = (
    #     df2["prefill"] + tokens_prefilled_per_second - 1
    # ) // tokens_prefilled_per_second

    # scale_up_count = 0
    # cache_miss_count = 0

    # cache_miss_df = pd.DataFrame(
    #     columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"]
    # )

    # for i in range(1, len(df2)):
    #     current_time = df2.loc[i, "timestamp"]
    #     current_node_needed = df2.loc[i, "node_needed"]
    #     previous_node_needed = df2.loc[i - 1, "node_needed"]

    #     if current_node_needed > previous_node_needed:
    #         scale_up = current_node_needed - previous_node_needed
    #         scale_up_count += scale_up

    #         past_five_seconds_max = df2.loc[
    #             (df2["timestamp"] > current_time - 5 * 60)
    #             & (df2["timestamp"] < current_time),
    #             "node_needed",
    #         ].max()
    #         if past_five_seconds_max < current_node_needed:
    #             cache_miss = current_node_needed - past_five_seconds_max
    #             cache_miss_count += cache_miss
    #             new_row = pd.DataFrame(
    #                 [[current_time, scale_up, cache_miss, current_node_needed]],
    #                 columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"],
    #             )
    #             cache_miss_df = pd.concat([cache_miss_df, new_row], ignore_index=True)
    #         else:
    #             new_row = pd.DataFrame(
    #                 [[current_time, scale_up, 0, current_node_needed]],
    #                 columns=["timestamp", "scale_up_n", "cache_miss_n", "node_needed"],
    #             )
    #             cache_miss_df = pd.concat([cache_miss_df, new_row], ignore_index=True)

    # print(dataset_path)
    # print("Total scale up times:", scale_up_count)
    # print("Total cache miss times:", cache_miss_count)
    # print("Cache miss rate {}%".format(cache_miss_count / scale_up_count * 100))
    # cache_miss_df.to_csv("temp/BurstGPT2_cache_miss.csv", index=False)


if __name__ == "__main__":
    tokens_prefilled_per_second = 13000
    # dataset_path = "temp/AzureConv2023-original.csv"
    # calculate_cache_miss_rate(dataset_path, tokens_prefilled_per_second)
    # dataset_path = "temp/AzureCode2023-original.csv"
    # calculate_cache_miss_rate(dataset_path, tokens_prefilled_per_second)
    dataset_path = "temp/BurstGPT2.csv"
    calculate_cache_miss_rate(dataset_path, tokens_prefilled_per_second)
