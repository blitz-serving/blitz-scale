import random


if __name__ == "__main__":
    dataset = []
    with open("/nvme/workdir/wht/processed-dataset/BurstGPT2.csv", "r") as f:
        f.readline()
        for line in f:
            timestamp, prefill, decode = line.split(",")
            dataset.append((int(prefill), int(decode)))

    original_timestamps = []
    with open("/nvme/workdir/wht/processed-dataset/AzureCode2023.csv", "r") as f:
        f.readline()
        for line in f:
            original_timestamps.append(int(line.split(",")[0]))

    output_to = "/nvme/workdir/wht/processed-dataset/burstgpt-w-code-trace.csv"
    with open(output_to, "w") as f:
        f.write("timestamp,prefill,decode\n")
        for i, timestamp in enumerate(original_timestamps):
            input_length, output_length = random.choice(dataset)
            f.write(f"{timestamp},{input_length},{output_length}\n")

    print(f"Output to {output_to}")
