import json
import pandas as pd
import matplotlib.pyplot as plt


def is_violated(entry) -> bool:
    return entry["first_token_time"] > max(entry["input_length"] * 0.375, 300)


def is_violated_2(entry) -> bool:
    return (
        entry["first_token_time"] / entry["calculation_time"] > 3
        and entry["first_token_time"] > 300
    )


def process_client_log(client_log_path) -> float:
    with open(client_log_path, "r") as f:
        total_entries = f.readlines()
        total_entries = [json.loads(entry) for entry in total_entries]
        total_entries = [
            entry for entry in total_entries if int(entry["s_time"]) < 30000
        ]
    violated_entries = []

    for entry in total_entries:
        entry = {k: int(v) for k, v in entry.items()}
        entry["calculation_time"] = entry["first_token_time"] - entry["queue_time"]
        if is_violated_2(entry):
            violated_entries.append(
                {
                    "request_id": entry["request_id"],
                    "s_time": entry["s_time"],
                    "e_time": entry["e_time"],
                    "queue_time": entry["queue_time"],
                    "first_token_time": entry["first_token_time"],
                    "calculation_time": entry["calculation_time"],
                    "input_length": entry["input_length"],
                }
            )

    total = pd.to_datetime(
        pd.DataFrame(total_entries)["s_time"].astype(int), unit="ms"
    ).dt.floor("s")
    violated = pd.to_datetime(
        pd.DataFrame(violated_entries)["s_time"], unit="ms"
    ).dt.floor("s")

    total_counts = total.value_counts().sort_index()
    violated_counts = violated.value_counts().sort_index()
    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))

    ax0.bar(total_counts.index, total_counts.values, width=0.000005, color="blue")
    ax0.bar(violated_counts.index, violated_counts.values, width=0.000005, color="red")

    ax0.set_title("Number of requests per Second")
    ax0.set_ylabel("Number of requests")

    # print(f"temp/violated_of_total.pdf")
    fig.savefig(f"temp/violated_of_total.pdf")
    plt.close(fig)

    violation_rate = len(violated_entries) / len(total_entries)
    # print(f"Violation rate: {violation_rate}")
    # for entry in violated_entries:
    #     print(entry)
    return violation_rate


if __name__ == "__main__":
    prefix = "20241127-2352-azurecode-pick-9"
    for m in ["llama2_7b", "llama2_7b_gqa"]:
        model_name = m
        for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            prefill_upper_bound = p
            for t in [0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]:
                mock_scale_millis = t
                line = []
                for i in [1, 2, 3, 4]:
                    if model_name == "llama2_7b":
                        tag_path = f"mha/prefill-{prefill_upper_bound}/mock-{mock_scale_millis}/{i}"
                    else:
                        tag_path = f"gqa/prefill-{prefill_upper_bound}/mock-{mock_scale_millis}/{i}"
                    tag_name = tag_path.rsplit("/", 1)[0].replace("/", "-")
                    client_log_path = f"./log/{prefix}/{tag_path}/client.jsonl"
                    violation_rate = process_client_log(client_log_path) * 100
                    line.append(violation_rate)
                sorted_arr = sorted(line)
                trimmed_arr = sorted_arr[1:-1]
                average = "{:.2f}".format(sum(trimmed_arr) / len(trimmed_arr))
                print(f"{tag_name},{average}")

    # for i in [1, 2, 3, 4, 5]:
    #     violation_rates = []
    #     for t in [0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]:
    #         mock_scale_millis = t
    #         tag = f"mock-{mock_scale_millis}-prefill0.4-{i}"
    #         client_log_path = f"./log/20241126-1518-azurecode-pick-9/{tag}/client.jsonl"
    #         rate = process_client_log(client_log_path)
    #         violation_rates.append(rate)
    #     print(violation_rates)
    #     print("===============================================================")

    # log_home = "./log/20241125-1942-azurecode-pick-9"
    # upper_bound = "0.4"
    # print(f"{log_home}/mock-0-prefill{upper_bound}/client.jsonl")
    # process_client_log(f"{log_home}/mock-0-prefill{upper_bound}/client.jsonl")
    # print("===============================================================")
    # print(f"{log_home}/mock-1000-prefill{upper_bound}/client.jsonl")
    # process_client_log(f"{log_home}/mock-1000-prefill{upper_bound}/client.jsonl")

    pass
