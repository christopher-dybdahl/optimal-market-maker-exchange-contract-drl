import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from utils import load_cfg, load_train_cfg

from optimal_market_maker_exchange_contract_drl import (
    Exchange,
    Logger,
    Market,
    MarketMaker,
    plot_exchange_losses,
)


def main():

    ######################
    ### INITIALISATION ###
    ######################

    # Initialise general directories
    PROJ_DIR = Path.cwd()
    MODELS_DIR = PROJ_DIR / "saved_models" / "exchange"
    CONFIG_DIR = PROJ_DIR / "config"

    # fmt: off
    train_cfg = load_train_cfg(path=CONFIG_DIR / "train_exchange_cfg.json")
    parser = ArgumentParser()
    parser.add_argument("--device",        type=str,   default=train_cfg["device"],        help="Device to use for training (cpu or cuda)")
    parser.add_argument("--batch_size",    type=int,   default=train_cfg["batch_size"],    help="Batch size")
    parser.add_argument("--model_name",    type=str,   default=train_cfg["model_name"],    help="Prefix of model save directory")
    parser.add_argument("--verbose",       action="store_true", default=train_cfg["verbose"], help="Print training progress")
    parser.add_argument("--train",         action="store_true", default=train_cfg["train"],   help="Train the model")
    parser.add_argument("--load_if_exists",action="store_true", default=train_cfg["load_if_exists"], help="Load model if it exists")
    parser.add_argument("--load_best_mm", action="store_true", default=train_cfg["load_best_mm"], help="Load best MM checkpoint instead of latest")
    parser.add_argument("--epochs",        type=int,   default=train_cfg["epochs"],        help="Number of epochs to train")
    parser.add_argument("--lr_v",          type=float, default=train_cfg["lr_v"],          help="Learning rate for critic (value)")
    parser.add_argument("--lr_z",          type=float, default=train_cfg["lr_z"],          help="Learning rate for actor (exploitation)")
    parser.add_argument("--lr_z_explore",  type=float, default=train_cfg["lr_z_explore"],  help="Learning rate for actor (exploration)")
    parser.add_argument("--n_critic_steps",type=int,   default=train_cfg["n_critic_steps"],help="Critic update steps per actor step")
    parser.add_argument("--save_per",      type=int,   default=train_cfg["save_per"],      help="Save checkpoint every N epochs")
    parser.add_argument("--log_per",       type=int,   default=train_cfg["log_per"],       help="Log loss every N epochs")
    args = parser.parse_args()
    # fmt: on

    # Initialise particular directory and create if it doesn't exist
    save_dir = MODELS_DIR / (f"{args.model_name + '_' if args.model_name else ''}model")
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
    logger.log(f"  lr_v          : {args.lr_v}")
    logger.log(f"  lr_z          : {args.lr_z}")
    logger.log(f"  lr_z_explore  : {args.lr_z_explore}")
    logger.log(f"  n_critic_steps: {args.n_critic_steps}")
    logger.log(f"  save_per      : {args.save_per}")
    logger.log(f"  log_per       : {args.log_per}")
    logger.log("=" * 70)
    # fmt: on

    device = torch.device(args.device)

    # Load exchange config
    try:
        if args.load_if_exists and os.path.exists(save_dir / "exchange_cfg.json"):
            exchange_cfg_path = save_dir / "exchange_cfg.json"
            source = "saved"
        else:
            exchange_cfg_path = CONFIG_DIR / "exchange_cfg.json"
            source = "config/"

        logger.log(f"Loading exchange config from ({source}) {exchange_cfg_path}")
        exchange_cfg = load_cfg(exchange_cfg_path)
    except Exception as e:
        logger.log(f"Error loading exchange config: {e}")
        raise LookupError

    # Resolve MM model directory (relative paths are relative to project root)
    mm_model_dir = Path(exchange_cfg["mm_model_dir"])
    if not mm_model_dir.is_absolute():
        mm_model_dir = PROJ_DIR / mm_model_dir

    logger.log(f"Loading pre-trained MarketMaker from {mm_model_dir}")

    # Load market config from the saved MM model directory
    try:
        market_cfg_path = mm_model_dir / "market_cfg.json"
        logger.log(f"Loading market config from {market_cfg_path}")
        market_cfg = load_cfg(market_cfg_path)
    except Exception as e:
        logger.log(f"Error loading market config from MM model dir: {e}")
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

    # Load market maker config from the saved MM model directory
    try:
        mm_cfg_path = mm_model_dir / "market_maker_cfg.json"
        logger.log(f"Loading market maker config from {mm_cfg_path}")
        mm_cfg = load_cfg(mm_cfg_path)
    except Exception as e:
        logger.log(f"Error loading market maker config from MM model dir: {e}")
        raise LookupError

    mm = MarketMaker(
        market=market,
        mm_cfg=mm_cfg,
        batch_size=args.batch_size,
        dtype=torch.float32,
    ).to(device=device)

    # Load MM checkpoint (best or latest)
    mm_best_path = mm_model_dir / "best_model.pt"
    if args.load_best_mm and mm_best_path.exists():
        mm_ckpt_path = mm_best_path
    else:
        mm_ckpts = sorted(
            mm_model_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.replace("checkpoint_epoch_", "")),
        )
        if not mm_ckpts:
            logger.log(f"ERROR: No MM checkpoints found in {mm_model_dir}")
            raise FileNotFoundError(f"No MM checkpoints in {mm_model_dir}")
        mm_ckpt_path = Path(mm_ckpts[-1])
    logger.log(f"Loading MM checkpoint: {mm_ckpt_path}")
    mm_epochs, _, _, _ = mm.load(path=mm_ckpt_path, device=device)
    logger.log(f"MM trained for {mm_epochs} epochs")

    # Freeze MM parameters (the exchange optimises its own networks)
    for p in mm.parameters():
        p.requires_grad_(False)
    mm.eval()
    logger.log("MM parameters frozen")
    logger.log("=" * 70)

    # Log exchange params
    # fmt: off
    logger.log("Exchange params:")
    logger.log(f"  eta                : {exchange_cfg['eta']}")
    logger.log(f"  c_l                : {exchange_cfg['c_l']}")
    logger.log(f"  c_d                : {exchange_cfg['c_d']}")
    logger.log(f"  T                  : {exchange_cfg['T']}")
    logger.log(f"  N                  : {exchange_cfg['N']}")
    logger.log(f"  actor_architecture : {exchange_cfg['actor_architecture']}")
    logger.log(f"  actor_layers       : {exchange_cfg['actor_layers']}")
    logger.log(f"  critic_architecture: {exchange_cfg['critic_architecture']}")
    logger.log(f"  critic_layers      : {exchange_cfg['critic_layers']}")
    logger.log("=" * 70)
    # fmt: on

    # Build Exchange
    exchange = Exchange(
        market=market,
        mm=mm,
        exchange_cfg=exchange_cfg,
        batch_size=args.batch_size,
        dtype=torch.float32,
    ).to(device=device)

    # Locate latest exchange checkpoint if load_if_exists
    start_epoch = 1
    optimizer_states = None
    prior_losses = {"value": [], "policy": [], "exploration": []}
    best_loss = None
    existing_ckpts = sorted(
        save_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.replace("checkpoint_epoch_", "")),
    )

    if args.load_if_exists and existing_ckpts:
        latest_ckpt = Path(existing_ckpts[-1])
        logger.log(f"Loading exchange checkpoint: {latest_ckpt}")
        epochs_trained, optimizer_states, prior_losses, best_loss = exchange.load(
            path=latest_ckpt, device=device
        )
        start_epoch = epochs_trained + 1
        logger.log(
            f"Resuming from epoch {start_epoch} "
            f"(previously trained for {epochs_trained} epochs)"
        )
    else:
        if args.load_if_exists:
            logger.log("No existing exchange checkpoint found - starting from scratch")
        else:
            logger.log("load_if_exists=False - starting from scratch")
    logger.log("=" * 70)

    # Save configs
    logger.log("Saving configuration files...")
    with open(save_dir / "train_exchange_cfg.json", "w") as f:
        json.dump(train_cfg, f, indent=4)
    with open(save_dir / "market_cfg.json", "w") as f:
        json.dump(market_cfg, f, indent=4)
    with open(save_dir / "market_maker_cfg.json", "w") as f:
        json.dump(mm_cfg, f, indent=4)
    with open(save_dir / "exchange_cfg.json", "w") as f:
        json.dump(exchange_cfg, f, indent=4)
    logger.log(f"Configuration saved to {save_dir}")
    logger.log("=" * 70)

    ######################
    ###### TRAINING ######
    ######################

    if args.train:
        logger.log("Training")
        all_losses, best_loss = exchange.fit(
            epochs=args.epochs,
            lr_v=args.lr_v,
            lr_z=args.lr_z,
            lr_z_explore=args.lr_z_explore,
            n_critic_steps=args.n_critic_steps,
            save_dir=save_dir,
            save_per=args.save_per,
            log_per=args.log_per,
            logger=logger,
            start_epoch=start_epoch,
            optimizer_states=optimizer_states,
            prior_losses=prior_losses,
            best_loss=best_loss,
        )
    else:
        all_losses = prior_losses

    ######################
    ###### PLOTTING ######
    ######################

    if any(all_losses.get(k) for k in ["value", "policy", "exploration"]):
        loss_fig_path = save_dir / "loss.png"
        loss_fig = plot_exchange_losses(losses=all_losses, save_path=loss_fig_path)
        logger.log(f"Loss plot saved to {loss_fig_path}")
        loss_fig.show()


if __name__ == "__main__":
    main()
