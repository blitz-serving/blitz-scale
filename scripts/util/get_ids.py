import json

ids = []
with open("./temp/client.jsonl", "r") as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        ids.append(int(data["request_id"]))
ids.sort()

for i, rid in enumerate(ids):
    assert i == rid, f"expected {i} but got {rid}"
