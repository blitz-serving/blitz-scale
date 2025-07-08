import json
import pandas as pd

log_path1 = "/root/code/blitz-remake/log_home/20241209-2127-intra-node-eval-Full-llama2_7b/azure-code-percentage-60-peak-3/xserve_hierarchical/client.jsonl"
log_path2 = "/root/code/blitz-remake/log_home/20241209-2127-intra-node-eval-Full-llama2_7b/azure-code-percentage-60-peak-3/baseline_cache/client.jsonl"

requests1 = []
requests2 = []

with open(log_path1, "r") as f:
    for line in f:
        requests1.append(json.loads(line))

with open(log_path2, "r") as f:
    for line in f:
        requests2.append(json.loads(line))

requests1 = pd.DataFrame(requests1)
requests2 = pd.DataFrame(requests2)

requests1["request_id"] = requests1["request_id"].astype(int)
requests2["request_id"] = requests2["request_id"].astype(int)

requests1 = requests1.sort_values("request_id")
requests2 = requests2.sort_values("request_id")


requests = pd.merge(
    requests1[["request_id", "s_time"]],
    requests2[["request_id", "s_time"]],
    on="request_id",
    suffixes=("_xserve_hierarchical", "_baseline_cache"),
)

# output the requests to csv file
requests.to_csv("requests.csv", index=False)
