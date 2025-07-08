import sys
from datetime import datetime
from functools import reduce

def timestamp_to_int(_timestamp_string: str) -> int:
    cons = _timestamp_string.split(sep=':')
    return int(reduce(lambda t, s: t + s, map(lambda s: s.replace('.', ''), cons)))

def main():
    lines = sys.stdin.readlines()
    lines = [line.strip() for line in lines]

    sorted_lines = sorted(lines, key=lambda line: timestamp_to_int(line.split(sep=' ')[0]))
    # Parse lines and sort by timestamp

    for line in sorted_lines:
        print(line)

if __name__ == "__main__":
    main()
