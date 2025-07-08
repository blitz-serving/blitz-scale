import sys
import re
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

REPLICA_STATE_ORDER = [
    'Decode',
    'ShuttingNull',
    'Prefill',
    'Inactive',
]

COLOR_MAP = {
    "Prefill": "#4C72B0",  # Blau
    "Decode": "#C44E52",   # Rot
    "Inactive": "#D3D3D3", # 
    "ShuttingNull": "#CCB974"     # gold
}

# 自定义分类映射函数（示例配置，请根据实际需求修改）
def category_mapper(line):
    p_replica_state = r"prefill|decode|inactive"
    m_replica_state = re.search(pattern=p_replica_state, string=line, flags=re.IGNORECASE)
    # XXX: ShuttingNull
    if m_replica_state is None:
        p_shutting = r'(shutting)[a-z]*'
        m_replica_state0 = re.search(pattern=p_shutting, string=line, flags=re.IGNORECASE)
        assert m_replica_state0.group().lower() == 'shuttingnull'
        m_replica_state = m_replica_state0
    
    assert not (m_replica_state is None)
    match m_replica_state.group().lower():
        case 'prefill':
            return 'Prefill'
        case 'decode':
            return 'Decode'
        case 'inactive':
            return 'Inactive'
        case 'shuttingnull':
            return 'ShuttingNull'
    
    raise RuntimeError

def parse_log():
    p_timestamp_replica_index = r'^([\d-]+T[\d:]+)\.\d+Z.*?Replica<(\d+)>'
    stats = defaultdict(lambda: defaultdict(int))
    categories = set()

    for line in sys.stdin:
        timestamp_0_replica_index_1 = re.match(p_timestamp_replica_index, line)
        assert timestamp_0_replica_index_1
        timestamp = datetime.strptime(timestamp_0_replica_index_1[1], "%Y-%m-%dT%H:%M:%S")
        ts_key = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        replica_index = timestamp_0_replica_index_1[2]
        category = category_mapper(line)

        stats[ts_key][category] += 1
        categories.add(category)

    return stats, sorted(stats.keys())

def plot_results(stats, timestamps):
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df[REPLICA_STATE_ORDER]  # 按自定义顺序排列列
    
    # 绘制堆叠柱状图
    plt.figure(figsize=(15, 7))
    ax = df.plot.bar(
        stacked=True,
        width=0.9,          # 柱宽
        align='center',     # 居中对齐
        color=[COLOR_MAP[cat] for cat in REPLICA_STATE_ORDER],
    )
    
    # 优化坐标轴
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.set_axisbelow(True)  # 网格线在柱状图下方
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 自定义图例
    ax.legend(
        title="Categories",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )
    
    plt.tight_layout()
    plt.savefig('output.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    stats, timestamps = parse_log()
    plot_results(stats, timestamps)