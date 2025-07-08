from typing import Any, Tuple
import toml
import os
import copy
from datetime import datetime
import pytz


def timestamp() -> str:
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d%H%M")


def load_template(template_path: str) -> Tuple[dict[str, Any], dict[str, Any]]:
    template = None
    with open(template_path, "r") as f:
        template = toml.load(f)
    list_members = {}
    for key, value in template["router"].items():
        if isinstance(value, list):
            list_members[key] = template["router"].pop(key)
    return template, list_members


def instantiate_template_router(
    template: dict[str, dict],
    list_members: dict[str, Any],
    path: str,
    instances: list[str],
):
    if len(list_members) == 0:
        instances.append(path)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "instance.toml"), "w") as f:
            template_copied = copy.deepcopy(template)
            template_copied["global"]["archive_dir"] = path
            toml.dump(template_copied, f)
    else:
        list_members = copy.deepcopy(list_members)
        key, val = list_members.popitem()
        for idx, item in enumerate(val):
            if isinstance(item, dict):
                template["router"].update(item)
                tag = f"{key}-{idx}"
            else:
                template["router"][key] = item
                tag = f"{key}-{item}"
            instantiate_template_router(
                template,
                list_members,
                os.path.join(path, tag),
                instances,
            )


def instantiate_template(template_path: str, archive_home: str) -> dict[str, list[str]]:
    template = {}
    list_members = {}
    with open(template_path, "r") as f:
        template = toml.load(f)
    archive_dir = archive_home
    for key, val in template["router"].items():
        if isinstance(val, list):
            list_members[key] = template["router"][key]
    for key in list_members.keys():
        template["router"].pop(key)

    template_copied = copy.deepcopy(template)
    if template_copied["selection"].get("tuples") is not None:
        template_copied["selection"].pop("tuples")
    if template_copied["selection"].get("models") is not None:
        template_copied["selection"].pop("models")
    if template_copied["selection"].get("datasets") is not None:
        template_copied["selection"].pop("datasets")
    if template_copied["selection"].get("features") is not None:
        template_copied["selection"].pop("features")
    template_copied.pop("models")
    template_copied.pop("datasets")
    template_copied.pop("features")

    instance_map = {}
    if template["selection"].get("tuples") is not None:
        for tuple in template["selection"]["tuples"]:
            model = tuple["model"]
            dataset = tuple["dataset"]
            feature = tuple["feature"]
            instances = []
            template_copied["model"] = template["models"][model]
            template_copied["model"]["model_name"] = model
            template_copied["dataset"] = template["datasets"][dataset]
            template_copied["router"]["feature"] = template["features"][feature]
            path = os.path.join(archive_dir, f"{model}-{dataset}-{feature}")
            instantiate_template_router(template_copied, list_members, path, instances)
            instance_map.setdefault(model, []).extend(instances)
        with open(os.path.join(archive_dir, "instances.toml"), "w") as f:
            toml.dump(instance_map, f)
        return instance_map

    instance_map = {}
    for model in template["selection"]["models"]:
        template_copied["model"] = template["models"][model]
        template_copied["model"]["model_name"] = model
        for dataset in template["selection"]["datasets"]:
            template_copied["dataset"] = template["datasets"][dataset]
            for feature in template["selection"]["features"]:
                instances = []
                template_copied["router"]["feature"] = template["features"][feature]
                path = os.path.join(archive_dir, f"{model}-{dataset}-{feature}")
                instantiate_template_router(
                    template_copied, list_members, path, instances
                )
                instance_map.setdefault(model, []).extend(instances)
    with open(os.path.join(archive_dir, "instances.toml"), "w") as f:
        toml.dump(instance_map, f)
    return instance_map


if __name__ == "__main__":
    template_path = "config/test_distserve.toml"
    archive_home = os.path.join("./log_home", timestamp())
    instantiate_template(template_path, archive_home)
