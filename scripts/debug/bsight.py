import argparse
import inquirer
import re
import sys
from typing import List, Dict

# 定义可监控的行为及其对应的匹配模式
ACTION_PATTERNS = {
    "state_transfer": r"=>",  # 状态转移 => instance Replica<?>
    "nvlink": r"\+>\+>\+>+",  # nvlink: +>+>+>+
    "rdma": r"~\+~",  # rdma: buhaha
    "kvcache": r"(?:-~>|<~-)",  # kvcache: -~> 或 <~-
    "scaling": r"P:\d+, D:\d+",  # 扩缩容决策结果
    "unload": r">>=",
}

ACTION_NAMES = {
    "state_transfer": "状态转移",
    "nvlink": "nvlink传参",
    "rdma": "rdma传参",
    "kvcache": "kvcache传输",
    "scaling": "扩缩容决策结果",
    "model-unload": "模型取消加载",
}


def prompt_user():
    """交互式提示用户选择配置"""
    questions = [
        inquirer.Checkbox(
            "instances",
            message="请选择要监控的instance号(0-8)(空格选择/取消, Enter确认)",
            choices=[str(i) for i in range(0, 9)],  # 0-8
        ),
        inquirer.Checkbox(
            "actions",
            message="请选择要监控的行为(空格选择/取消, Enter确认)",
            choices=list(ACTION_PATTERNS.keys()),
        ),
        inquirer.Text("input_file", message="请输入要分析的输入文件路径"),
    ]
    return inquirer.prompt(questions)


def build_instance_pattern(instances: List[str]) -> str:
    """构建instance匹配模式"""
    if not instances:
        return r"(?:Rank<\d+>|Replica<\d+>|Machine<\d+>::<{[\d,]+}>)"

    instance_pattern = "|".join(instances)

    # 匹配 Rank<1>, Replica<1> 或 Machine<0>::<0,1,2> 中的instance
    return rf"""
        (?:
            Rank<({instance_pattern})>
            |Replica<({instance_pattern})>
            |Machine<\d+>::<[^>]*\b({instance_pattern})\b[^>]*>
        )
    """


def build_pattern(instances: List[str], actions: List[str]) -> str:
    """构建精确的匹配模式"""
    # instances.extend(actions)
    # print(instances)
    # return instances
    # 处理instance部分
    instance_pattern = build_instance_pattern(instances)

    # 处理action部分
    if actions:
        action_patterns = [ACTION_PATTERNS[action] for action in actions]
    else:
        action_patterns = list(ACTION_PATTERNS.values())

    # 组合模式：instance模式在前，action模式在后
    patterns = []
    patterns.append(instance_pattern)
    patterns.extend(action_patterns)
    print(f"patterns: {patterns}")
    return patterns


def filter_log(input_file: str, output_file: str, patterns: List[str]):
    """过滤日志并写入输出文件"""
    # print(f"pattern: {pattern}")
    compiled_patterns = []
    for pattern in patterns:
        compiled_patterns.append(re.compile(pattern, re.VERBOSE))

    print(compiled_patterns)
    # 使用re.VERBOSE允许正则表达式中的注释和空白
    # compiled_pattern = re.compile(pattern, re.VERBOSE)

    match_count = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):

        for line in infile:
            for pattern in compiled_patterns:
                print(line)
                if pattern.search(line):
                    outfile.write(line)
                    match_count += 1
                    break

    print(f"✓ 找到 {match_count} 条匹配记录")
    print(f"✓ 结果已保存到: {output_file}")


def migration(log_lines, **kwargs):
    batches = {}
    p1 = r"-~>|<~-"
    r1 = re.compile(p1)
    p2 = r"Batch\[\d+\]"
    r2 = re.compile(p2)
    for line in log_lines:
        m1 = r1.findall(line)
        if len(m1) > 0:
            m2 = r2.findall(line)
            for batch in m2:
                v = batches.get(batch)
                match v:
                    case None:
                        batches[batch] = [line]
                    case ls:
                        ls.append(line)
    print(batches.keys())
    for batch, log in batches.items():
        p3 = r"fst|snd"
        r3 = re.compile(p3)
        from functools import reduce
        partial_layers = reduce(
            lambda init, l: init or len(r3.findall(l)) > 0, log, False
        )
        match partial_layers:
            # Full layer migration
            case True:
                if len(log) != 12:
                    print("=" * 8 + " ERROR " + "=" * 8)
                    for l in log:
                        print(l, end="")
            # Partial layer migration
            case False:
                if len(log) != 6:
                    print("=" * 8 + " ERROR " + "=" * 8)
                    for l in log:
                        print(l, end="")

def zigzag(log_lines, **kwargs):
    rank = kwargs['rank']
    p0 = r"Rank<{}>".format(rank)
    r0 = re.compile(p0)
    p1 = r"RPC ZagPrefill Batch\[\d+\]"
    r1 = re.compile(p1)
    p2 = r"schedule :>"
    r2 = re.compile(p2)
    p3 = r"batch.id=\d+ fwd layer"
    r3 = re.compile(p3)
    p4 = r"\d+"
    r4 = re.compile(p4)
    p5 = r"Batch\[\d+\]"
    r5 = re.compile(p5)
    p6 = r"ZagPrefill Batch\[\d+\] ready to return"
    r6 = re.compile(p6)

    step = 0
    batches = {}

    for l in log_lines:
        if len(r0.findall(l)) > 0:
            # Rank<{}>
            l1 = r1.findall(l)
            l2 = r2.findall(l)
            l3 = r3.findall(l)
            l6 = r6.findall(l)
            if len(l1) > 0:
                # RPC req
                batch_id = int(*r4.findall(*r5.findall(*l1)))
                batches[batch_id] = (step, '[' + '·' * step + '|')
            elif len(l2) > 0:
                # scheduler
                step += 1
            elif len(l6) > 0:
                # return
                batch_id = int(*r4.findall(*l6))
                _, display = batches.get(batch_id)
                display += '|'
                batches[batch_id] = (_, display)
            elif len(l3) > 0:
                batch_id = int(*r4.findall(*l3))
                last_step, display = batches.get(batch_id)
                new_cons = display + '-' * (step - last_step - 1) + '*'
                batches[batch_id] = (step, new_cons)
    
    batches = dict(sorted(batches.items(), key=lambda it: it[0]))
    for batch_id, (_, display) in batches.items():
        print("Rank<{}> Batch[{}]\t".format(rank, batch_id) + display + ']')

TERM_HANDLE = {
    "migration": migration,
    "zigzag": zigzag
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique matches from grep output."
    )
    parser.add_argument(
        "-t", "--term", required=True, help="Choose the interesting part of log"
    )
    parser.add_argument(
        "--rank", default=None, help="Choose the interesting part of log"
    )
    args = parser.parse_args()

    # 获取用户选择
    # answers = prompt_user()
    # 构建正则表达式模式
    # patterns = build_pattern(answers['instances'], answers['actions'])
    # 过滤日志
    # filter_log(answers['input_file'], f"{'.'.join(answers['input_file'].split('.')[:-1])}.new.log", patterns)

    # 从标准输入读取所有行
    input_lines = sys.stdin.readlines()
    handle = args.term
    kwargs = vars(args)
    kwargs.pop("term")
    TERM_HANDLE[handle](input_lines, **kwargs)


if __name__ == "__main__":
    main()
