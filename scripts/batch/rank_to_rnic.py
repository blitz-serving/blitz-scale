#!/usr/bin/env python3
import subprocess, re, json, sys


# select all ib device that can be used in the container
# and find the most matching ib for each GPU rank
# using "nvidia-smi topo -m" to find(PIX > PXB > NODE > PHB=3 > SYS=4 > LOC=5)
def get_visible_rdma_devices():
    """Use `ibv_devinfo -l` to get usable RNICs in this container"""
    try:
        output = subprocess.check_output(['ibv_devinfo', '-l'], text=True)
    except FileNotFoundError:
        sys.exit("ibv_devinfo not found in container. Please install OFED or rdma-core.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ibv_devinfo error: {e.output}")

    devs = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            devs.append(line)
    return set(devs)  # e.g. {"mlx5_0", "mlx5_1"}

def rank_to_rnic_name():
    # ---------- 获取容器中可用网卡 ----------
    visible_rdmas = get_visible_rdma_devices()

    # ---------- 1. 拿到拓扑文本 ----------
    try:
        topo_txt = subprocess.check_output(['nvidia-smi', 'topo', '-m'],
                                           text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        sys.exit(f'Error running nvidia-smi: {e.output}')

    lines = topo_txt.splitlines()

    # ---------- 2. 解析 NIC Legend ----------
    legend = {}
    legend_mode = False
    for ln in lines:
        if ln.startswith('NIC Legend'):
            legend_mode = True
            continue
        if legend_mode:
            m = re.match(r'\s*(NIC\d+):\s+(\S+)', ln)
            if m and m.group(2) in visible_rdmas:   # 只保留容器可用的
                legend[m.group(1)] = m.group(2)

    if not legend:
        sys.exit("No usable RNICs found in container.")

    # ---------- 3. 找表头 ----------
    header = None
    for ln in lines:
        if 'GPU0' in ln and 'NIC0' in ln:
            header = re.split(r'\s+', ln.strip())
            break
    if not header:
        sys.exit('Failed to locate header line with GPU0 / NIC0')

    gpu_cols = [i for i, h in enumerate(header) if h.startswith('GPU')]
    nic_cols = [i for i, h in enumerate(header) if h.startswith('NIC') and h in legend]

    # ---------- 4. 定义距离优先级 ----------
    rank_score = dict(PIX=0, PXB=1, NODE=2, PHB=3, SYS=4, LOC=5)

    # ---------- 5. 匹配 GPU → 最近合法 NIC ----------
    mapping = {}
    for ln in lines:
        if re.match(r'\s*GPU\d+', ln):
            toks = re.split(r'\s+', ln.strip())
            gname = toks[0]
            best = (None, 99)
            for idx in nic_cols:
                if idx >= len(toks):
                    continue
                dist = toks[idx]
                score = rank_score.get(dist, 99)
                if score < best[1]:
                    nheader = header[idx - 1]              # e.g. NIC6
                    nic_real = legend.get(nheader, None)
                    if nic_real:
                        best = (nic_real, score)
            mapping[gname] = best[0] if best[0] else "N/A"

    return mapping

if __name__ == "__main__":
    mapping = rank_to_rnic_name()
    print(json.dumps(mapping, indent=2))
