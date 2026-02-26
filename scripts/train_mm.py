import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from utils import load_cfg, load_train_cfg

from optimal_market_maker_exchange_contract_drl import (
    Logger,
    Market,
    MarketMaker,
    make_market_params,
)


def main():
    # Initialise general directories
    PROJ_DIR = Path.cwd()
    MODELS_DIR = PROJ_DIR / "saved_models" / "market_maker"
    CONFIG_DIR = PROJ_DIR / "config"

    # fmt: off
    train_cfg = load_train_cfg(path=CONFIG_DIR / "train_cfg.json")
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=train_cfg["device"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=train_cfg["batch_size"], help="Batch size")
    parser.add_argument("--model_name", type=str, default=train_cfg["model_name"], help="Prefix of model save directory")
    parser.add_argument("--verbose", action="store_true", default=train_cfg["verbose"], help="Print training progress")
    parser.add_argument("--train", action="store_true", default=train_cfg["train"], help="Train the model")
    parser.add_argument("--load_if_exists", action="store_true", default=train_cfg["load_if_exists"], help="Load model if it exists")
    args = parser.parse_args()
    # fmt: on

    # Initialise particular directory and create if it doesn't exist
    save_dir = MODELS_DIR / (
        str(args.model_name) + "model_1"
    )  # TODO: Create suitable default name suffix
    os.makedirs(save_dir, exist_ok=True)

    # If loading an existing model and training.log exists, append to it instead of overwriting
    training_log_path = save_dir / "training.log"
    overwrite_log = args.train and not (
        args.load_if_exists and os.path.exists(training_log_path)
    )

    # Initialise logger object
    logger = Logger(
        save_dir=save_dir,
        filename="training.log",
        verbose=args.verbose,
        overwrite=overwrite_log,
    )

    device = torch.device(args.device)

    # Retrieve market config and initialise market object
    try:
        if args.load_if_exists and os.path.exists(save_dir / "market_cfg.json"):
            market_cfg_path = save_dir / "market_cfg.json"
        else:
            market_cfg_path = CONFIG_DIR / "market_cfg.json"

        logger.log(f"Loading market config from {market_cfg_path}")
        market_cfg = load_cfg(market_cfg_path)
    except Exception as e:
        logger.log(f"Error loading market config: {e}")
        raise LookupError

    market_params = make_market_params(**market_cfg, device=device)
    market = Market(
        market_params=market_params,
        device=device,
        batch_size=args.batch_size,
        dtype=torch.float32,
    )

    # Retrieve market maker config and initialise market maker object
    try:
        if args.load_if_exists and os.path.exists(save_dir / "market_maker_cfg.json"):
            mm_cfg_path = save_dir / "market_maker_cfg.json"
        else:
            mm_cfg_path = CONFIG_DIR / "market_maker_cfg.json"

        logger.log(f"Loading market maker config from {mm_cfg_path}")
        mm_cfg = load_cfg(mm_cfg_path)
    except Exception as e:
        logger.log(f"Error loading market maker config: {e}")
        raise LookupError

    mm = MarketMaker(
        market=market,
        mm_cfg=mm_cfg,
        device=device,
        batch_size=args.batch_size,
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
