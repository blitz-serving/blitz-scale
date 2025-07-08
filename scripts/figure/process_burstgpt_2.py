with open("/nvme/workdir/wht/processed-dataset/BurstGPT1.csv", "w") as o:
    with open("/nvme/workdir/wht/dataset/BurstGPT_without_fails_1.csv", "r") as f:
        o.write("timestamp,prefill,decode\n")
        f.readline()
        for line in f:
            entry = line.split(",")
            timestamp = int(float(entry[0])) * 1000
            prefill = int(entry[2])
            decode = int(entry[3])
            o.write(f"{timestamp},{prefill},{decode}\n")
