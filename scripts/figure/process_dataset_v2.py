import matplotlib.pyplot as plt
import pandas as pd

millis_per_minute = 60 * 1000
millis_per_hour = 60 * millis_per_minute
millis_per_day = 24 * millis_per_hour

# `./dataset_home` is a softlink to `/nvme/blitz/trace_home/processed`
# ln -s /nvme/blitz/trace_home/processed ./dataset_home
origin_csv = "dataset_home/AzureCode2023-3min.csv"
output_csv = "dataset_home/AzureCode2023-3min.csv"
output_png = "dataset_home/AzureCode2023-3min.png"

scale = 1
selected_s_time = int(53055)
selected_e_time = int(3 * millis_per_minute)


def process_dataset(origin_csv, output_csv, scale, selected_s_time, selected_e_time):
    timestamps = []
    values = []
    header = ""
    with open(origin_csv, newline="", encoding="utf-8") as file:
        header = file.readline()
        for line in file:
            line = line.strip().split(",")
            ts = int(line[0])
            if ts >= selected_s_time and int(line[0]) <= selected_e_time:
                timestamps.append(int(line[0]) - selected_s_time)
            if ts >= selected_s_time:
                if len(values) < scale * len(timestamps):
                    values.append([int(line[1]), int(line[2])])
                else:
                    break
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        n = len(values)
        file.write(header)
        for i in range(n * scale):
            index = i // scale
            file.write(f"{timestamps[index]},{values[i % n][0]},{values[i % n][1]}\n")


def plot(output_csv, output_png):
    cli_data = pd.read_csv(output_csv)
    seconds = pd.to_datetime(cli_data["timestamp"], unit="ms").dt.floor("s")
    counts = seconds.value_counts().sort_index()
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 6))
    ax0.bar(counts.index, counts.values, width=0.000005)
    ax0.set_title("Number of requests per Second")
    ax0.set_ylabel("Number of requests")

    cli_data["second"] = cli_data["timestamp"] // 1000

    prefill_result = cli_data.groupby("second")["prefill"].sum().reset_index()
    ax1.bar(prefill_result["second"], prefill_result["prefill"])
    ax1.set_ylabel("Total prefill tokens")

    decode_result = cli_data.groupby("second")["decode"].sum().reset_index()
    ax2.bar(decode_result["second"], decode_result["decode"])
    ax2.set_ylabel("Total decode tokens")
    plt.savefig(output_png)


if __name__ == "__main__":
    process_dataset(origin_csv, output_csv, scale, selected_s_time, selected_e_time)
    plot(output_csv, output_png)
