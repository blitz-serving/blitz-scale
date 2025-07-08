import inquirer
import re
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
    instances_choices = ["all"]
    instances_choices.extend([str(i) for i in range(0, 9)])
    questions = [
        inquirer.Checkbox(
            "instances",
            message="请选择要监控的instance号(0-8)(空格选择/取消, Enter确认)",
            choices=instances_choices,  # 0-8
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

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            for pattern in compiled_patterns:
                if pattern.search(line):
                    outfile.write(line)
                    match_count += 1
                    break

    print(f"✓ 找到 {match_count} 条匹配记录")
    print(f"✓ 结果已保存到: {output_file}")


def main():
    print("=== 日志监控分析工具 ===")
    print("请选择要监控的配置:\n")

    # 获取用户选择
    answers = prompt_user()
    if answers["instances"][0] == "all":
        answers["instances"] = [str(i) for i in range(9)]

    # 构建正则表达式模式
    patterns = build_pattern(answers["instances"], answers["actions"])

    # 过滤日志
    filter_log(
        answers["input_file"],
        f"{'.'.join(answers['input_file'].split('.')[:-1])}.new.log",
        patterns,
    )


if __name__ == "__main__":
    main()
