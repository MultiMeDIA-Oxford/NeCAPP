"""Load YAML configuration files and merge inherited settings."""

import yaml


def load_config(path, default_path=None):
    """Load a YAML file and recursively merge optional inherited configuration."""
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg_special = yaml.safe_load(f)

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r", encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Merge nested configuration mappings in place."""
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
