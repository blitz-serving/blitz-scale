import matplotlib.pyplot as plt


def get_violation_rates(csv_path) -> dict[str, float]:
    violation_rates = {}
    with open(csv_path, "r") as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(",")
            config = parts[0]
            violation_rate = float(parts[1])
            violation_rates[config] = violation_rate
    return violation_rates


if __name__ == "__main__":
    csv_path = "./20241127-2352-azurecode-pick-9.csv"
    violation_rates = get_violation_rates(csv_path)

    color_list = ["b", "g", "r", "c", "m", "y", "k"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for attention in ["mha", "gqa"]:
        if attention == "mha":
            ax = ax0
            ax.set_title("MHA")
        else:
            ax = ax1
            ax.set_title("GQA")
        for index, prefill_upper_bound in enumerate(
            [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        ):
            color = color_list[index]
            time_list = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
            violation_list = []
            for mock_scale_millis in time_list:
                config_name = f"{attention}-prefill-{prefill_upper_bound}-mock-{mock_scale_millis}"
                violation_list.append(violation_rates[config_name])
            ax.plot(
                time_list,
                violation_list,
                color=color,
                label=f"prefill-{prefill_upper_bound}",
            )
        ax.legend()
        ax.set_ylabel("Violation Rate")
        ax.set_xlabel("Mock Scale Millis")
    plt.savefig("violation_rate.pdf")
