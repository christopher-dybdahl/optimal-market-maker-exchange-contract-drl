import json


def load_cfg(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def load_train_cfg(path):
    cfg = load_cfg(path)
    return cfg
