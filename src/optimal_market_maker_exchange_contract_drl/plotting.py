from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_controls(
    mm,
    plot_z_d: float,
    plot_q: float,
    n_points: int = 400,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot optimal market-maker controls as a function of z^{a,l} (bottom axis)
    and z^{b,l} (top axis, reversed), with fixed q = plot_q and
    z^{a,d} = z^{b,d} = plot_z_d.

    The sweep parameterises:
        z^{a,l} = t,    t in [-z_bar, z_bar]   (bottom axis, left -> right)
        z^{b,l} = -t                            (top axis, right -> left)
        z^{a,d} = z^{b,d} = plot_z_d           (fixed)
        q                = plot_q               (fixed)

    Args:
        mm:         trained MarketMaker
        plot_q:     inventory value to hold fixed
        plot_z_d:   dark-pool z value to hold fixed for both sides
        n_points:   number of points along the sweep
        save_path:  if provided, save the figure to this path

    Returns:
        matplotlib Figure
    """
    device = mm.gamma.device
    dtype = mm.gamma.dtype
    z_bar = mm.z_bar.item()

    was_training = mm.training
    mm.eval()

    with torch.no_grad():
        # z^{a,l} sweeps from -z_bar to z_bar; z^{b,l} = -z^{a,l}
        t = torch.linspace(-z_bar, z_bar, n_points, device=device, dtype=dtype)
        z_d = torch.full((n_points,), plot_z_d, device=device, dtype=dtype)
        q = torch.full((n_points,), plot_q, device=device, dtype=dtype)

        # z4: (N, 4) = [z^{a,l}, z^{b,l}, z^{a,d}, z^{b,d}]
        z4 = torch.stack([t, -t, z_d, z_d], dim=1)  # (N, 4)

        ell4 = mm.controls(z4=z4, q=q).cpu().numpy()  # (N, 4)

    if was_training:
        mm.train()

    ell_al = ell4[:, 0]  # ell^{a,l}
    ell_bl = ell4[:, 1]  # ell^{b,l}
    ell_ad = ell4[:, 2]  # ell^{a,d}
    ell_bd = ell4[:, 3]  # ell^{b,d}
    x = t.cpu().numpy()

    fig, ax_bot = plt.subplots(figsize=(7, 5))

    ax_bot.plot(x, ell_al, linestyle=":", label="ask lit")
    ax_bot.plot(x, ell_bl, linestyle="--", label="bid lit")
    ax_bot.plot(x, ell_ad, linestyle="-.", label="ask dark")
    ax_bot.plot(x, ell_bd, linestyle=(0, (3, 1, 1, 1, 1, 1)), label="bid dark")

    ax_bot.set_xlabel(r"$z^{a,l}$")
    ax_bot.set_ylabel(r"Optimal volume $\ell$")
    ax_bot.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax_bot.legend(loc="upper center")

    # Top axis: z^{b,l} = -z^{a,l}, so limits are reversed
    ax_top = ax_bot.twiny()
    ax_top.set_xlim(-z_bar, z_bar)  # same data range as bottom
    ax_top.invert_xaxis()  # reversed so z^{b,l} goes z_bar -> -z_bar
    ax_top.set_xlabel(r"$z^{b,l}$")

    fig.suptitle(
        rf"Optimal controls  ($q={plot_q:.3g}$, $z^{{a,d}}=z^{{b,d}}={plot_z_d:.3g}$)",
        fontsize=10,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_incentives_vs_inventory(
    exchange,
    step: int,
    n_points: int = 400,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot exchange optimal incentives z*(q) as a function of inventory q
    at a given time step.

    Args:
        exchange:   trained Exchange
        step:       time step index (0..N-1)
        n_points:   number of points along the inventory sweep
        save_path:  if provided, save the figure to this path

    Returns:
        matplotlib Figure
    """
    device = exchange.eta.device
    dtype = exchange.eta.dtype
    q_bar = exchange.mm.q_bar.item()

    was_training = exchange.training
    exchange.eval()

    with torch.no_grad():
        q = torch.linspace(-q_bar, q_bar, n_points, device=device, dtype=dtype)
        z4 = exchange.policy(q, step).cpu().numpy()  # (N, 4)

    if was_training:
        exchange.train()

    x = q.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, z4[:, 0], linestyle=":", label="ask lit")
    ax.plot(x, z4[:, 1], linestyle="--", label="bid lit")
    ax.plot(x, z4[:, 2], linestyle="-.", label="ask dark")
    ax.plot(x, z4[:, 3], linestyle=(0, (3, 1, 1, 1, 1, 1)), label="bid dark")

    ax.set_xlabel("Inventory")
    ax.set_title("Incentives")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_optimal_volumes_vs_inventory(
    exchange,
    step: int,
    n_points: int = 400,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot market maker optimal volumes under the exchange's optimal incentives,
    as a function of inventory q at a given time step.

    For each q, computes z*(q) from the exchange policy, then ell*(z*(q), q)
    from the market maker controls.

    Args:
        exchange:   trained Exchange
        step:       time step index (0..N-1)
        n_points:   number of points along the inventory sweep
        save_path:  if provided, save the figure to this path

    Returns:
        matplotlib Figure
    """
    device = exchange.eta.device
    dtype = exchange.eta.dtype
    q_bar = exchange.mm.q_bar.item()

    was_training = exchange.training
    exchange.eval()

    with torch.no_grad():
        q = torch.linspace(-q_bar, q_bar, n_points, device=device, dtype=dtype)
        z4 = exchange.policy(q, step)              # (N, 4)
        ell4 = exchange.mm.controls(z4=z4, q=q).cpu().numpy()  # (N, 4)

    if was_training:
        exchange.train()

    x = q.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, ell4[:, 0], linestyle=":", label="ask lit")
    ax.plot(x, ell4[:, 1], linestyle="--", label="bid lit")
    ax.plot(x, ell4[:, 2], linestyle="-.", label="ask dark")
    ax.plot(x, ell4[:, 3], linestyle=(0, (3, 1, 1, 1, 1, 1)), label="bid dark")

    ax.set_xlabel("Inventory")
    ax.set_title("Optimal volumes")
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_loss(
    losses: list[float],
    smoothing: int = 100,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot training loss over epochs.

    Args:
        losses:     per-epoch loss values
        smoothing:  window size for rolling average (0 to disable)
        save_path:  if provided, save the figure to this path

    Returns:
        matplotlib Figure
    """
    epochs = np.arange(1, len(losses) + 1)
    loss_arr = np.array(losses)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, loss_arr, alpha=0.3, linewidth=0.5, label="raw")

    if smoothing > 0 and len(losses) >= smoothing:
        kernel = np.ones(smoothing) / smoothing
        smoothed = np.convolve(loss_arr, kernel, mode="valid")
        ax.plot(
            epochs[smoothing - 1 :],
            smoothed,
            linewidth=1.5,
            label=f"rolling avg ({smoothing})",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_exchange_losses(
    losses: dict[str, list[float]],
    smoothing: int = 100,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot three exchange training losses side-by-side.

    Args:
        losses:     dict with keys "value", "policy", "exploration"
        smoothing:  window size for rolling average (0 to disable)
        save_path:  if provided, save the figure to this path

    Returns:
        matplotlib Figure
    """
    titles = {
        "value": "Critic (value)",
        "policy": "Actor (exploitation)",
        "exploration": "Actor (exploration)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key in zip(axes, ["value", "policy", "exploration"]):
        vals = losses.get(key, [])
        if not vals:
            ax.set_title(titles[key])
            continue

        epochs = np.arange(1, len(vals) + 1)
        arr = np.array(vals)

        ax.plot(epochs, arr, alpha=0.3, linewidth=0.5, label="raw")

        if smoothing > 0 and len(vals) >= smoothing:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.convolve(arr, kernel, mode="valid")
            ax.plot(
                epochs[smoothing - 1 :],
                smoothed,
                linewidth=1.5,
                label=f"rolling avg ({smoothing})",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(titles[key])
        ax.legend()
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    fig.suptitle("Exchange Training Losses")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
