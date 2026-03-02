from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import Market
from .nnets import FCnet
from .utils import Logger


class MarketMaker(nn.Module):
    def __init__(
        self,
        market: Market,
        mm_cfg: dict,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Training params
        self.B = batch_size
        self.mm_cfg = mm_cfg

        # Market object
        self.market = market

        # Market Maker params
        self.rho = mm_cfg["rho"]
        self.register_buffer(
            "gamma", torch.tensor(self.mm_cfg["gamma"], dtype=dtype)
        )  # Risk aversion parameter
        self.register_buffer(
            "q_bar", torch.tensor(self.mm_cfg["q_bar"], dtype=dtype)
        )  # Single side risk limit

        # Pre-computed (1, 1, 3) broadcast constants for channels {l, d_lat, d_non}
        self.register_buffer(
            "_Gamma3",
            self.market.Gamma[[0, 1, 1]].view(1, 1, 3).to(dtype=dtype),
        )  # (1, 1, 3)  [Gamma_l, Gamma_d, Gamma_d]
        self.register_buffer(
            "_spread3",
            self.market.half_tick
            * torch.tensor([1.0, 1.0, 0.0], dtype=dtype).view(1, 1, 3),
        )  # (1, 1, 3)  half_tick * phi_lat(kappa) per channel

        # Episode state — None until reset_state() is called
        self.register_buffer("Q", None)  # (B,)
        self.register_buffer("N_l", None)  # (B, 2, #V_l)
        self.register_buffer("N_d", None)  # (B, 2, #V_d)
        self.register_buffer("N_d_agg", None)  # (B, 2, 2) [side, kappa={lat,non}]
        self.register_buffer("N_agg", None)  # (B, 2)   [side, pool={l,d}]

        # Initialise neural network
        if self.mm_cfg["architecture"] == "fc":
            self.net = FCnet(
                q_bar=self.q_bar,
                layers=self.mm_cfg["layers"],
                activation=self.mm_cfg["activation"],
            )
        else:
            raise ValueError

    def reset_state(self) -> None:
        device = self.gamma.device
        dtype = self.gamma.dtype

        # Initialize inventory
        self.Q = torch.zeros((self.B,), device=device, dtype=dtype)  # (B, )

        # Counts per volume N^{i, k}
        self.N_l = torch.zeros(
            (self.B, 2, self.market.V_l.numel()),
            device=device,
            dtype=torch.int64,
        )  # (B, 2, #V_l)
        self.N_d = torch.zeros(
            (self.B, 2, self.market.V_d.numel()),
            device=device,
            dtype=torch.int64,
        )  # (B, 2, #V_d)

        # Aggregated counts latency/non-latency N^{i, lat/non-lat}
        self.N_d_agg = torch.zeros(
            (self.B, 2, 2), device=device, dtype=torch.int64
        )  # (B, 2, 2) [side={a, b}, kappa={lat, non}]

        # Aggregated counts N^{i, j}
        self.N_agg = torch.zeros(
            (self.B, 2), device=device, dtype=torch.int64
        )  # (B, 2, 2) [side={a, b}, pool={l, d}]

    def save(
        self,
        path: str,
        epochs_trained: int,
        optimizer: torch.optim.Optimizer = None,
        losses: list[float] = None,
    ) -> None:
        ckpt = {
            "mm_state_dict": self.state_dict(),
            "epochs_trained": epochs_trained,
            "dtype": str(self.gamma.dtype).replace("torch.", ""),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "losses": losses if losses is not None else [],
        }
        torch.save(ckpt, path)

    def load(
        self, path: str, device: torch.device
    ) -> tuple[int, dict | None, list[float]]:
        """
        Load model weights and training state from a checkpoint.

        Returns:
            (epochs_trained, optimizer_state_dict, losses) — pass the latter two to
            fit(..., optimizer_state=optimizer_state_dict, prior_losses=losses)
            to resume training correctly.
        """
        ckpt = torch.load(path, map_location="cpu")

        dtype_str = ckpt.get("dtype", "float32")
        dtype = getattr(torch, dtype_str)

        self.to(device=device, dtype=dtype)
        self.load_state_dict(ckpt["mm_state_dict"])

        # reset episode state on the right device
        self.reset_state()

        return (
            ckpt.get("epochs_trained", 0),
            ckpt.get("optimizer_state_dict", None),
            ckpt.get("losses", []),
        )

    def _lam_eff_c(self, ell: torch.Tensor) -> torch.Tensor:
        lam = self.market.lam_base(ell[:, :, 0])  # (B, 2, 2) [side={a, b}, pool={l, d}]

        # phi^d(i, kappa)
        phi_d = self.market.phi_d(
            ell[:, :, 0]
        )  # (B, 2, 2) [side={a, b}, kappa={lat, non}]

        # Expand to (B, 2, 3): [lam_l, lam_d * phi_d_lat, lam_d * phi_d_non]
        return torch.cat([lam[:, :, 0:1], lam[:, :, 1:2] * phi_d], dim=2)  # (B, 2, 3)

    def _lam_eff(self, ell: torch.Tensor) -> torch.Tensor:
        lam = self.market.lam_base(ell[:, :, 0])  # (B, 2, 2) [side={a, b}, pool={l, d}]
        inv_mask = (self.market.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        lam = lam * inv_mask[:, :, None].to(lam.dtype)  # (B, 2, 2)

        # phi^d(i, kappa)
        phi_d = self.market.phi_d(
            ell[:, :, 0]
        )  # (B, 2, 2) [side={a, b}, kappa={lat, non}]

        # Expand to (B, 2, 3): [lam_l, lam_d * phi_d_lat, lam_d * phi_d_non]
        return torch.cat([lam[:, :, 0:1], lam[:, :, 1:2] * phi_d], dim=2)  # (B, 2, 3)

    def _hamiltonian(
        self, ell: torch.Tensor, z: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        # Tile ell and z to (B, 2, 3): channels {l, d_lat, d_non}
        # ell^{i,d} is identical for both dark kappa; z^{i,d} likewise
        ell3 = torch.cat([ell, ell[:, :, 1:2]], dim=2)  # (B, 2, 3)
        z3 = torch.cat([z, z[:, :, 1:2]], dim=2)  # (B, 2, 3)

        # arg^{i,c} = z^{i,c} + ell^{i,c}*(spread^c + phi(i)*Gamma^c*q) - Gamma^c*(ell^{i,c})^2
        arg = (
            z3
            + ell3
            * (
                self._spread3
                + self.market.phi.unsqueeze(-1) * self._Gamma3 * q.view(self.B, 1, 1)
            )
            - self._Gamma3 * ell3**2
        )  # (B, 2, 3)

        return (1.0 / self.gamma) * (
            (1.0 - torch.exp(-self.gamma * arg)) * self._lam_eff_c(ell)
        ).sum(dim=(1, 2))  # (B,)

    def _pack_z(self, z4: torch.Tensor) -> torch.Tensor:
        """
        z4: (B, 4) = [z^{a,l}, z^{b,l}, z^{a,d}, z^{b,d}]
        returns z: (B, 2, 2) with axes [side={a,b}, pool={l,d}]
        """
        z_al = z4[:, 0]
        z_bl = z4[:, 1]
        z_ad = z4[:, 2]
        z_bd = z4[:, 3]
        return torch.stack(
            [
                torch.stack([z_al, z_ad], dim=-1),
                torch.stack([z_bl, z_bd], dim=-1),
            ],
            dim=1,
        )  # (B, 2, 2)

    def _pack_ell(self, ell4: torch.Tensor) -> torch.Tensor:
        """
        ell4: (B, 4) = [ell^{a,l}, ell^{b,l}, ell^{a,d}, ell^{b,d}]
        returns ell: (B, 2, 2) with axes [side={a,b}, pool={l,d}]
        """
        ell_al = ell4[:, 0]
        ell_bl = ell4[:, 1]
        ell_ad = ell4[:, 2]
        ell_bd = ell4[:, 3]
        return torch.stack(
            [
                torch.stack([ell_al, ell_ad], dim=-1),
                torch.stack([ell_bl, ell_bd], dim=-1),
            ],
            dim=1,
        )  # (B, 2, 2)

    def _nn_input(
        self, z4: torch.Tensor, q: torch.Tensor, z_bar: float
    ) -> torch.Tensor:
        """
        Normalize inputs:
          z in [-z_bar, z_bar]^4 -> z/z_bar
          q in [-q_bar, q_bar]   -> q/q_bar
        returns y: (B, 5)
        """
        zn = z4 / z_bar
        qn = q / self.q_bar
        return torch.cat([zn, qn[:, None]], dim=1)

    def _inventory_penalty(self, ell: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Penalty from the paper:

          (q + ell^{b,l} + ell^{b,d} - q_bar)_+
        + (q - ell^{a,l} - ell^{a,d} + q_bar)_-

        Implement (x)_+ = relu(x), (x)_- = relu(-x)
        Returns: (B,)
        """
        ell_b_sum = ell[:, 1, 0] + ell[:, 1, 1]
        ell_a_sum = ell[:, 0, 0] + ell[:, 0, 1]

        t_plus = F.relu(q + ell_b_sum - self.q_bar)
        t_minus = F.relu(-(q - ell_a_sum + self.q_bar))
        return t_plus + t_minus

    def fit(
        self,
        z_bar: float,
        epochs: int,
        lr: float,
        seed: int = None,
        save_dir: Path = None,
        save_per: int = 50,
        log_per: int = 10,
        logger: Logger = None,
        start_epoch: int = 1,
        optimizer_state: dict = None,
        prior_losses: list[float] = None,
    ) -> list[float]:
        """
        Train the market maker policy network.

        Epochs are numbered globally: epoch start_epoch, start_epoch+1, ...,
        start_epoch+epochs-1.  Checkpoints are saved to
        save_dir/checkpoint_epoch_{N}.pt whenever N % save_per == 0, and
        always at the final epoch.

        Each epoch draws self.B i.i.d. (z, q) samples — the same batch size
        used everywhere else in the simulation.

        Args:
            z_bar:           half-range for uniform z sampling
            epochs:          number of epochs to run in this call
            lr:              Adam learning rate
            seed:            optional RNG seed for reproducibility
            save_dir:        directory to write checkpoints into (None = no saving)
            save_per:        save a checkpoint every this many epochs
            log_per:         log loss every this many epochs
            logger:          Logger instance (prints + writes to file)
            start_epoch:     global epoch index of the first epoch in this run
            optimizer_state: if provided, restore Adam state (e.g. momentum)
                             from a previous run before training begins

        Returns:
            List of per-epoch loss values (length == epochs).
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = self.gamma.device
        dtype = self.gamma.dtype

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        if optimizer_state is not None:
            opt.load_state_dict(optimizer_state)

        final_epoch = start_epoch + epochs - 1
        losses: list[float] = list(prior_losses) if prior_losses else []

        def _log(msg: str) -> None:
            if logger is not None:
                logger.log(msg)

        _log(f"Training epochs {start_epoch} -> {final_epoch} (lr={lr}, B={self.B})")

        for step, epoch in enumerate(range(start_epoch, start_epoch + epochs)):
            # Sample q ~ Uniform([-q_bar, q_bar])
            q = (
                2.0 * torch.rand((self.B,), device=device, dtype=dtype) - 1.0
            ) * self.q_bar
            # Sample z ~ Uniform([-z_bar, z_bar]^4)
            z4 = (
                2.0 * torch.rand((self.B, 4), device=device, dtype=dtype) - 1.0
            ) * z_bar

            y = self._nn_input(z4=z4, q=q, z_bar=z_bar)  # (B, 5)
            ell4 = self.net(y)  # (B, 4)
            ell = self._pack_ell(ell4)  # (B, 2, 2)
            z = self._pack_z(z4)  # (B, 2, 2)

            h = self._hamiltonian(ell=ell, z=z, q=q)  # (B,)
            pen = self._inventory_penalty(ell=ell, q=q)  # (B,)

            loss = -(h - self.rho * pen).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)

            if epoch % log_per == 0 or epoch == final_epoch:
                _log(
                    f"  Epoch {epoch:>{len(str(final_epoch))}}/{final_epoch} | loss: {loss_val:.6f}"
                )

            if save_dir is not None and (epoch % save_per == 0 or epoch == final_epoch):
                ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save(
                    path=ckpt_path, epochs_trained=epoch, optimizer=opt, losses=losses
                )
                _log(f"  Checkpoint saved -> {ckpt_path}")

        _log(
            f"Training complete. Final epoch: {final_epoch} | "
            f"avg loss (last {min(log_per, epochs)} epochs): "
            f"{sum(losses[-min(log_per, epochs) :]) / min(log_per, epochs):.6f}"
        )

        return losses

    def update_state(self, v: torch.Tensor) -> None:
        # TODO: Implement update of
        # self.N_l
        # self.N_d
        # self.N_d_lat_agg
        # self.N_d_non_agg
        # self.N_l_agg
        # self.N_d_agg
        # self.Q
        pass

    def post_liquidity(self) -> None:
        pass
