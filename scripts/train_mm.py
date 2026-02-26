# fmt: off
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from utils import load_train_cfg

from optimal_market_maker_exchange_contract_drl import Logger


def main():
    train_cfg = load_train_cfg(path="config/train_config.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=train_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"], help="Batch size")
    parser.add_argument("--model_name", type=str, default=train_cfg["model_name"], help="Path to save the model")
    parser.add_argument("--verbose", action="store_true", default=train_cfg["verbose"], help="Print training progress")
    parser.add_argument("--train", action="store_true", default=train_cfg["train"], help="Train the model")
    parser.add_argument("--load_if_exists", action="store_true", default=train_cfg["load_if_exists"], help="Load model if it exists")
    args = parser.parse_args()

    # Initialise general directories
    PROJ_DIR = Path.cwd().parent
    MODELS_DIR = PROJ_DIR / "saved_models" / "market_maker"

    # Initialise particular directory and create if it doesn't exist
    model_dir = MODELS_DIR / args.model_name + "" # TODO: Create suitable default name suffix
    os.makedirs(model_dir, exist_ok=True)

    # If loading an existing model and training.log exists, append to it instead of overwriting
    training_log_path = model_dir / "training.log"
    overwrite_log = args.train and not (args.load_if_exists and os.path.exists(training_log_path))

    # Initialise logger object
    logger = Logger(save_dir=model_dir, filename="training.log", verbose=args.verbose, overwrite=overwrite_log)

    device = torch.device(args.device)
