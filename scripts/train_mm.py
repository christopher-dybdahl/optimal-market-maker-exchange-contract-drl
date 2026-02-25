# fmt: off
from argparse import ArgumentParser

import torch
from utils import load_train_cfg


def main():
    train_cfg = load_train_cfg(path="config/train_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=train_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"], help="Batch size")
    parser.add_argument("--save_dir", type=str, default=train_cfg["save_dir"], help="Path to save the model")
    args = parser.parse_args()

    save_dir = args.save_dir

    device = torch.device(args.device)
