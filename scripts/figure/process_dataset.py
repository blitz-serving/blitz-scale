import pandas as pd
import matplotlib.pyplot as plt

base_dataset = "./output/concat.csv"


begin_after = 0
scale_factor = 1
ts_min = 0
ts_max = 999999999
cli_data = pd.read_csv(base_dataset)
drop_rate = 0
picked = f"./output/concat"

# scale factor
cli_data["timestamp"] = (
    (cli_data["timestamp"].astype(int) + begin_after) / scale_factor
).astype(int)

# filter timestamp
cli_data = cli_data[
    (cli_data["timestamp"] >= ts_min) & (cli_data["timestamp"] <= ts_max)
]

# drop request randomly and keep the remaining requests' relative order
if drop_rate > 0:
    cli_data = cli_data.sample(frac=1 - drop_rate, random_state=42).sort_values(
        "timestamp"
    )

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

print(f"{picked}.pdf")
fig.savefig(f"{picked}.pdf")
print(f"{picked}.csv")
cli_data.to_csv(f"{picked}.csv", index=False)
