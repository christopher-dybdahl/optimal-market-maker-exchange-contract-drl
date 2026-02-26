import json
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

    ######################
    ### INITIALISATION ###
    ######################

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
        f"{args.model_name + '_' if args.model_name else ''}model"
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

    # Log run configuration
    # fmt: off
    logger.log("Run configuration")
    logger.log(f"  device      : {args.device}")
    logger.log(f"  batch_size  : {args.batch_size}")
    logger.log(f"  model_name  : {args.model_name!r}")
    logger.log(f"  train       : {args.train}")
    logger.log(f"  load_if_exists: {args.load_if_exists} ({'will attempt to load saved configs' if args.load_if_exists else 'will use configs from config/'})")
    logger.log("=" * 70)
    # fmt: on

    device = torch.device(args.device)

    # Retrieve market config and initialise market object
    try:
        if args.load_if_exists and os.path.exists(save_dir / "market_cfg.json"):
            market_cfg_path = save_dir / "market_cfg.json"
            source = "saved"
        else:
            market_cfg_path = CONFIG_DIR / "market_cfg.json"
            source = "config/"

        logger.log(f"Loading market config from ({source}) {market_cfg_path}")
        market_cfg = load_cfg(market_cfg_path)
    except Exception as e:
        logger.log(f"Error loading market config: {e}")
        raise LookupError

    market_params = make_market_params(**market_cfg, device=device)

    # fmt: off
    logger.log("Market params:")
    logger.log(f"  A (limit, dark)    : {market_params.A.tolist()}")
    logger.log(f"  c (theta/sigma)    : {market_params.c.tolist()}")
    logger.log(f"  Gamma (limit, dark): {market_params.Gamma.tolist()}")
    logger.log(f"  sigma              : {market_params.sigma.item():.6g}")
    logger.log(f"  tick_size          : {market_params.tick_size.item():.6g}")
    logger.log(f"  half_tick          : {market_params.half_tick.item():.6g}")
    logger.log(f"  S_tilde_0          : {market_params.S_tilde_0}")
    logger.log(f"  V_l                : {market_params.V_l.tolist()}")
    logger.log(f"  V_d                : {market_params.V_d.tolist()}")
    logger.log(f"  eps                : {market_params.eps.item():.6g}")
    logger.log("=" * 70)
    # fmt: on

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
            source = "saved"
        else:
            mm_cfg_path = CONFIG_DIR / "market_maker_cfg.json"
            source = "config/"

        logger.log(f"Loading market maker config from ({source}) {mm_cfg_path}")
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

    # Save configs
    train_cfg_path = save_dir / "train_cfg.json"
    market_cfg_path = save_dir / "market_cfg.json"
    mm_cfg_path = save_dir / "market_maker_cfg.json"
    logger.log("Saving configuration files...")
    with open(train_cfg_path, "w") as f:
        json.dump(train_cfg, f, indent=4)
    with open(market_cfg_path, "w") as f:
        json.dump(market_cfg, f, indent=4)
    with open(mm_cfg_path, "w") as f:
        json.dump(mm_cfg, f, indent=4)
    logger.log(f"Configuration saved to {save_dir}")
    logger.log("=" * 70)

    ######################
    ###### TRAINING ######
    ######################

    if args.train:
        logger.log("Train")


if __name__ == "__main__":
    main()
