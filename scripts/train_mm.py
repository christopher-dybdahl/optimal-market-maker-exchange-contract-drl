import glob
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
    plot_loss,
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
    parser.add_argument("--device",        type=str,   default=train_cfg["device"],        help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size",    type=int,   default=train_cfg["batch_size"],    help="Batch size")
    parser.add_argument("--model_name",    type=str,   default=train_cfg["model_name"],    help="Prefix of model save directory")
    parser.add_argument("--verbose",       action="store_true", default=train_cfg["verbose"], help="Print training progress")
    parser.add_argument("--train",         action="store_true", default=train_cfg["train"],   help="Train the model")
    parser.add_argument("--load_if_exists",action="store_true", default=train_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--epochs",        type=int,   default=train_cfg["epochs"],        help="Number of epochs to train")
    parser.add_argument("--lr",            type=float, default=train_cfg["lr"],            help="Adam learning rate")
    parser.add_argument("--save_per",      type=int,   default=train_cfg["save_per"],      help="Save checkpoint every N epochs")
    parser.add_argument("--log_per",       type=int,   default=train_cfg["log_per"],       help="Log loss every N epochs")
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
    logger.log("=" * 70)
    logger.log("Run configuration")
    logger.log(f"  device        : {args.device}")
    logger.log(f"  batch_size    : {args.batch_size}")
    logger.log(f"  model_name    : {args.model_name!r}")
    logger.log(f"  train         : {args.train}")
    logger.log(f"  load_if_exists: {args.load_if_exists} ({'will attempt to load latest checkpoint' if args.load_if_exists else 'will start from scratch'})")
    logger.log(f"  epochs        : {args.epochs}")
    logger.log(f"  lr            : {args.lr}")
    logger.log(f"  save_per      : {args.save_per}")
    logger.log(f"  log_per       : {args.log_per}")
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

    market = Market(**market_cfg, batch_size=args.batch_size, dtype=torch.float32)
    market.to(device=device)

    # fmt: off
    logger.log("Market params:")
    logger.log(f"  A (lit, dark)      : {market.A.tolist()}")
    logger.log(f"  c (theta/sigma)    : {market.c.tolist()}")
    logger.log(f"  Gamma (lit, dark)  : {market.Gamma.tolist()}")
    logger.log(f"  sigma              : {market.sigma.item():.6g}")
    logger.log(f"  tick_size          : {market.tick_size.item():.6g}")
    logger.log(f"  half_tick          : {market.half_tick.item():.6g}")
    logger.log(f"  eps                : {market.eps.item():.6g}")
    logger.log("=" * 70)
    # fmt: on

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
        batch_size=args.batch_size,
        dtype=torch.float32,
    ).to(device=device)

    # Locate latest checkpoint (highest epoch number) if load_if_exists
    start_epoch = 1
    optimizer_state = None
    prior_losses = []
    ckpt_pattern = str(save_dir / "checkpoint_epoch_*.pt")
    existing_ckpts = sorted(
        glob.glob(ckpt_pattern),
        key=lambda p: int(Path(p).stem.replace("checkpoint_epoch_", "")),
    )

    if args.load_if_exists and existing_ckpts:
        latest_ckpt = Path(existing_ckpts[-1])
        logger.log(f"Loading checkpoint: {latest_ckpt}")
        epochs_trained, optimizer_state, prior_losses = mm.load(
            path=latest_ckpt, device=device
        )
        start_epoch = epochs_trained + 1
        logger.log(
            f"Resuming from epoch {start_epoch} (previously trained for {epochs_trained} epochs)"
        )
    else:
        if args.load_if_exists:
            logger.log("No existing checkpoint found - starting from scratch")
        else:
            logger.log("load_if_exists=False - starting from scratch")
    logger.log("=" * 70)

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
        logger.log("Training")
        all_losses = mm.fit(
            epochs=args.epochs,
            lr=args.lr,
            save_dir=save_dir,
            save_per=args.save_per,
            log_per=args.log_per,
            logger=logger,
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            prior_losses=prior_losses,
        )
    else:
        all_losses = prior_losses

    ######################
    ###### PLOTTING ######
    ######################

    if all_losses:
        loss_fig_path = save_dir / "loss.png"
        loss_fig = plot_loss(losses=all_losses, save_path=loss_fig_path)
        logger.log(f"Loss plot saved to {loss_fig_path}")
        loss_fig.show()


if __name__ == "__main__":
    main()
