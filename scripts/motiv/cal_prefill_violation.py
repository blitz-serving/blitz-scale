import json
import os
import sys
from typing import List, Dict


def calculatePrefillSloViolation(
    client_log_dir: str, prefill_slo_threshold_ms: int
) -> float:
    # client.jsonl
    # {"avg_time_between_tokens":xxx, "first_token_time":xxx}

    data = []

    # traverse all jsonl files in the directory
    for client_log_path in os.listdir(client_log_dir):
        total_prefill_requests = 0
        violated_requests = 0
        scale_up_time = int(client_log_path.split("/")[-1].split(".")[0])
        with open(os.path.join(client_log_dir, client_log_path)) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "first_token_time" in record:
                        s_time = int(record["s_time"])
                        if s_time > 5500:
                            continue
                        total_prefill_requests += 1
                        input_length = int(record["input_length"])
                        if (
                            int(record["first_token_time"])
                            > prefill_slo_threshold_ms + input_length * 0.08
                        ):
                            violated_requests += 1
                except json.JSONDecodeError:
                    continue

        if total_prefill_requests == 0:
            return 0.0

        violation_rate = (violated_requests / total_prefill_requests) * 100
        # print(f"Total prefill requests: {total_prefill_requests}")
        # print(f"Violated requests: {violated_requests}")
        data.append((scale_up_time, violation_rate))

    # sort by scale_up_time
    data.sort(key=lambda x: x[0])

    # print the data
    # print all scale up time in one line
    print(",".join([str(x[0]) for x in data]))
    print(",".join([f"{x[1]:.2f}" for x in data]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cal_slo_violation.py <client_log_dir>")
        sys.exit(1)

    client_log_dir = sys.argv[1]
    for prefill_slo_threshold_ms in [300, 350, 400, 450, 500]:
        print(f"\nCalculating SLO violation for {prefill_slo_threshold_ms}ms")
        calculatePrefillSloViolation(client_log_dir, prefill_slo_threshold_ms)
