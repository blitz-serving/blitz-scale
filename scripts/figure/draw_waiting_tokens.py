import re
import matplotlib.pyplot as plt
import pandas as pd

import re
import matplotlib.pyplot as plt
from datetime import datetime

# 初始化列表来存储时间戳和对应的token值
timestamps = []
tokens = []

log_path = "log/azurecode-3.0x-prefill0.7/mock-scale-millis-0/router.log"

client_start = "2024-11-24T02:36:12.692006Z  INFO client: Client start"
start_pattern = (
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+INFO\s+client: Client start"
)
match = re.search(start_pattern, client_start)
timestamp_str = match.group(1)
base_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")


# 打开日志文件并逐行读取
with open(log_path, "r") as file:
    for line in file:
        pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z).*?Cur waiting prefill tokens: (\d+)"
        match = re.search(pattern, line)
        if match:
            timestamp_str = match.group(1)
            token_value = int(match.group(2))
            other_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp = (other_timestamp - base_timestamp).total_seconds() / 60
            timestamps.append(timestamp)
            tokens.append(token_value)

# 使用matplotlib绘制图表
plt.figure(figsize=(10, 5))
plt.plot(timestamps, tokens)
plt.title("Cur Waiting Prefill Tokens Over Time")
plt.xlabel("Time")
plt.ylabel("Prefill Tokens")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temp/waiting_tokens.pdf")
