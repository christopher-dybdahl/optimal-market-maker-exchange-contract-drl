from pathlib import Path

import torch
import torch.nn as nn

from .agent import MarketMaker
from .dynamics import Market
from .nnets import FCnet
from .utils import Logger


class Exchange(nn.Module):
    def __init__(
        self,
        market: Market,
        mm: MarketMaker,
        exchange_cfg: dict,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.B = batch_size
        self.exchange_cfg = exchange_cfg
        self.market = market
        self.mm = mm

        # Time discretisation
        self.N = exchange_cfg["N"]
        self.register_buffer("T_total", torch.tensor(exchange_cfg["T"], dtype=dtype))
        self.register_buffer(
            "dt", torch.tensor(exchange_cfg["T"] / exchange_cfg["N"], dtype=dtype)
        )

        # Exchange params
        # η in the paper (exchange's risk aversion)
        self.register_buffer("eta", torch.tensor(exchange_cfg["eta"], dtype=dtype))
        # Fixed fees per market order: (1, 1, 2) [lit, dark]
        self.register_buffer(
            "c",
            torch.tensor([exchange_cfg["c_l"], exchange_cfg["c_d"]], dtype=dtype).view(
                1, 1, 2
            ),
        )

        # Precomputed constants for ε computation in (B, 2, 3) channels {l, d_lat, d_non}
        # Market impact: (1, 1, 3) = [Gamma_l, Gamma_d, Gamma_d]
        self.register_buffer(
            "_Gamma3",
            self.market.Gamma[[0, 1, 1]].view(1, 1, 3).to(dtype=dtype),
        )
        # Spread per channel: (1, 1, 3) = [T/2, T/2, 0]
        # phi^lat(kappa) = 1 for lit & dark_lat, 0 for dark_non
        self.register_buffer(
            "_spread3",
            self.market.half_tick
            * torch.tensor([1.0, 1.0, 0.0], dtype=dtype).view(1, 1, 3),
        )

        # Neural networks, one per time step
        if exchange_cfg["actor_architecture"] == "fc":
            self.actor_nets = nn.ModuleList(
                [
                    FCnet(
                        layers=exchange_cfg["actor_layers"],
                        activation=exchange_cfg["actor_activation"],
                        output_activation=exchange_cfg["actor_output_activation"],
                    )
                    for _ in range(self.N)
                ]
            )
        else:
            raise ValueError

        if exchange_cfg["critic_architecture"] == "fc":
            self.critic_nets = nn.ModuleList(
                [
                    FCnet(
                        layers=exchange_cfg["critic_layers"],
                        activation=exchange_cfg["critic_activation"],
                        output_activation=exchange_cfg["critic_output_activation"],
                    )
                    for _ in range(self.N)
                ]
            )
        else:
            raise ValueError

    def value(self, q: torch.Tensor, i: int) -> torch.Tensor:
        """
        Value function v_t(q).

        Args:
            q: (B,) inventory (any batch shape)
            i: time step index (0..N).
               i = N returns the terminal condition v(T, q) = -1.

        Returns:
            (B,) value at each q
        """
        if i == self.N:
            return -torch.ones_like(q)

        q_norm = (q / self.mm.q_bar).unsqueeze(-1)  # (B, 1)

        if self.exchange_cfg["critic_architecture"] == "fc":
            return self.critic_nets[i](q_norm).squeeze(-1)  # (B,)
        else:
            raise ValueError

    def policy(self, q: torch.Tensor, i: int) -> torch.Tensor:
        """
        Incentive policy z_t(q).

        Args:
            q: (B,) inventory
            i: time step index (0..N-1)

        Returns:
            (B, 4) incentives [z^{a,l}, z^{b,l}, z^{a,d}, z^{b,d}]
                   in [-z_bar, z_bar]
        """
        q_norm = (q / self.mm.q_bar).unsqueeze(-1)  # (B, 1)

        if self.exchange_cfg["actor_architecture"] == "fc":
            return self.actor_nets[i](q_norm) * self.mm.z_bar  # (B, 4)
        else:
            raise ValueError

    def _hamiltonian(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        ell: torch.Tensor,
        i: int,
    ) -> torch.Tensor:
        """
        Exchange Hamiltonian U^c(z, q, ell*, v_t).

        From Section 5.1.2:

          U^c = v(q) [η/2 σ²γ(z^S+q)² + η²σ²/2 (z^S)²]
              + Σ_{i,j} λ^{i,j} [ exp(η(z^{i,j} - c^j ℓ^{i,j})) v(q')
                                 - v(q) (1 + η ε^{i,j}) ]

        where z^S = -γ/(γ+η) q,  q' = q - φ(i)ℓ^{i,j},
              ε^{i,l} = (1/γ)(1 - exp(-γ arg^{i,l})),
              ε^{i,d} = (1/γ) Σ_κ φ^d(i,κ) (1 - exp(-γ arg^{i,d,κ})),
              arg^{i,j,κ} = z^{i,j} + ℓ^{i,j}(φ^lat(κ) T/2 + φ(i)Γ^j q) - Γ^j(ℓ^{i,j})².

        Args:
            z:   (B, 2, 2) incentives [side={a,b}, pool={l,d}]
            q:   (B,) inventory
            ell: (B, 2, 2) MM's optimal volumes [side={a,b}, pool={l,d}]
            i:   time step index for the value function

        Returns:
            (B,) Hamiltonian value per sample
        """
        B = q.shape[0]
        gamma = self.mm.gamma  # MM's risk aversion γ
        eta = self.eta  # Exchange's risk aversion η
        sigma = self.market.sigma

        # ---- Value function evaluations ----
        # φ(i) = +1 for ask, -1 for bid → (1, 2, 1)
        phi = self.market.phi.unsqueeze(-1)

        # Shifted inventories: q' = q - φ(i) ℓ^{i,j}  → (B, 2, 2)
        q_shifted = q.view(B, 1, 1) - phi * ell

        # Batch-evaluate v_t at q and all 4 shifted q values
        q_all = torch.cat([q, q_shifted.reshape(B * 4)])  # (5B,)
        v_all = self.value(q_all, i)  # (5B,)
        v_q = v_all[:B]  # (B,)
        v_shifted = v_all[B:].reshape(B, 2, 2)  # (B, 2, 2)

        # ---- Drift term ----
        # z^S = -γ/(γ+η) q  (optimal inventory incentive, solved in closed form)
        z_S = -(gamma / (gamma + eta)) * q  # (B,)

        drift = v_q * (
            (eta / 2) * sigma**2 * gamma * (z_S + q) ** 2
            + (eta**2 * sigma**2 / 2) * z_S**2
        )  # (B,)

        # ---- ε: certainty equivalent per (i, j) with κ summation for dark ----
        # Expand z and ell to (B, 2, 3): channels {l, d_lat, d_non}
        z3 = torch.cat([z[:, :, 0:1], z[:, :, 1:2], z[:, :, 1:2]], dim=2)
        ell3 = torch.cat([ell[:, :, 0:1], ell[:, :, 1:2], ell[:, :, 1:2]], dim=2)

        # arg^{i,c} = z^{i,c} + ℓ^{i,c}(spread^c + φ(i)Γ^c q) − Γ^c(ℓ^{i,c})²
        arg = (
            z3
            + ell3 * (self._spread3 + phi * self._Gamma3 * q.view(B, 1, 1))
            - self._Gamma3 * ell3**2
        )  # (B, 2, 3)

        eps_raw = (1.0 / gamma) * (1.0 - torch.exp(-gamma * arg))  # (B, 2, 3)

        # Collapse dark κ channels: ε^{i,d} = Σ_κ φ^d(i,κ) · ε_raw^{i,d,κ}
        phi_d = self.market.phi_d(ell[:, :, 0])  # (B, 2, 2) [side, kappa={lat, non}]
        eps_d = (phi_d * eps_raw[:, :, 1:3]).sum(dim=2, keepdim=True)  # (B, 2, 1)

        # ε: (B, 2, 2) = [ε^{i,l}, ε^{i,d}]
        eps = torch.cat([eps_raw[:, :, 0:1], eps_d], dim=2)  # (B, 2, 2)

        # ---- Arrival rates ----
        lam = self.market.lam_base(ell[:, :, 0])  # (B, 2, 2) [side, pool]

        # ---- Fee-adjusted exponential ----
        exp_term = torch.exp(eta * (z - self.c * ell))  # (B, 2, 2)

        # ---- Jump terms ----
        jump = lam * (
            exp_term * v_shifted - v_q.view(B, 1, 1) * (1.0 + eta * eps)
        )  # (B, 2, 2)

        return drift + jump.sum(dim=(1, 2))  # (B,)

    def save(
        self,
        path: str,
        epochs_trained: int,
        optimizers: dict[str, torch.optim.Optimizer] = None,
        losses: dict[str, list[float]] = None,
    ) -> None:
        ckpt = {
            "exchange_state_dict": self.state_dict(),
            "epochs_trained": epochs_trained,
            "dtype": str(self.eta.dtype).replace("torch.", ""),
            "optimizer_states": {k: opt.state_dict() for k, opt in optimizers.items()}
            if optimizers is not None
            else None,
            "losses": losses
            if losses is not None
            else {"value": [], "policy": [], "exploration": []},
        }
        torch.save(ckpt, path)

    def load(
        self, path: str, device: torch.device
    ) -> tuple[int, dict | None, dict[str, list[float]]]:
        """
        Load exchange weights and training state from a checkpoint.

        Returns:
            (epochs_trained, optimizer_states, losses) — pass the latter two
            to fit() to resume training correctly.
        """
        ckpt = torch.load(path, map_location="cpu")

        dtype_str = ckpt.get("dtype", "float32")
        dtype = getattr(torch, dtype_str)

        self.to(device=device, dtype=dtype)
        self.load_state_dict(ckpt["exchange_state_dict"], strict=False)

        return (
            ckpt.get("epochs_trained", 0),
            ckpt.get("optimizer_states", None),
            ckpt.get("losses", {"value": [], "policy": [], "exploration": []}),
        )

    def fit(
        self,
        epochs: int,
        lr_v: float,
        lr_z: float,
        lr_z_explore: float,
        seed: int = None,
        save_dir: Path = None,
        save_per: int = 50,
        log_per: int = 10,
        logger: Logger = None,
        start_epoch: int = 1,
        optimizer_states: dict = None,
        prior_losses: dict[str, list[float]] = None,
    ) -> dict[str, list[float]]:
        """
        Train the exchange actor-critic networks.

        Three optimizers:
          opt_v:         updates critic_nets   (lr_v)
          opt_z:         updates actor_nets    (lr_z)       — exploitation
          opt_z_explore: updates actor_nets    (lr_z_explore) — exploration

        Each epoch loops backward from t_{N-1} to t_0 performing:
          1. Critic update:       fit v_i(q) to Bellman target
                                  v_{i+1}(q) + Δt·H(z, q, ℓ, v_{i+1})
          2. Actor exploitation:  maximise H(z, q, ℓ, v_i)  w.r.t. z
          3. Actor exploration:   push policy toward perturbations
                                  that improve H

        Returns:
            dict with keys "value", "policy", "exploration",
            each a list of per-epoch losses.
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = self.eta.device
        dtype = self.eta.dtype

        # --- Three optimizers ---
        opt_v = torch.optim.Adam(self.critic_nets.parameters(), lr=lr_v)
        opt_z = torch.optim.Adam(self.actor_nets.parameters(), lr=lr_z)
        opt_z_explore = torch.optim.Adam(self.actor_nets.parameters(), lr=lr_z_explore)

        if optimizer_states is not None:
            opt_v.load_state_dict(optimizer_states["opt_v"])
            opt_z.load_state_dict(optimizer_states["opt_z"])
            opt_z_explore.load_state_dict(optimizer_states["opt_z_explore"])

        optimizers = {
            "opt_v": opt_v,
            "opt_z": opt_z,
            "opt_z_explore": opt_z_explore,
        }

        final_epoch = start_epoch + epochs - 1
        losses: dict[str, list[float]] = {
            "value": list(prior_losses.get("value", [])) if prior_losses else [],
            "policy": list(prior_losses.get("policy", [])) if prior_losses else [],
            "exploration": list(prior_losses.get("exploration", []))
            if prior_losses
            else [],
        }

        def _log(msg: str) -> None:
            if logger is not None:
                logger.log(msg)

        _log(
            f"Training epochs {start_epoch} -> {final_epoch} "
            f"(lr_v={lr_v}, lr_z={lr_z}, lr_z_explore={lr_z_explore}, B={self.B})"
        )

        for epoch in range(start_epoch, start_epoch + epochs):
            # Sample q ~ Uniform([-q_bar, q_bar])
            q = (
                2.0 * torch.rand((self.B,), device=device, dtype=dtype) - 1.0
            ) * self.mm.q_bar

            epoch_v = 0.0
            epoch_z = 0.0
            epoch_e = 0.0

            for i in range(self.N - 1, -1, -1):
                # --- 1. Critic (value) update ---
                # Target is fully detached: v_{i+1}(q) + Δt·H(z, q, ℓ, v_{i+1})
                with torch.no_grad():
                    z4 = self.policy(q, i)
                    ell4 = self.mm.controls(z4=z4, q=q)
                    z = self.mm._pack_z(z4)
                    ell = self.mm._pack_ell(ell4)
                    target = self.value(q, i + 1) + self.dt * self._hamiltonian(
                        z=z, q=q, ell=ell, i=i + 1
                    )

                v_i = self.value(q, i)
                v_loss = ((v_i - target) ** 2).mean()

                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                opt_v.step()
                epoch_v += v_loss.item()

                # --- 2. Actor exploitation update ---
                # Maximise H w.r.t. z  →  minimise -H
                z4 = self.policy(q, i)
                ell4 = self.mm.controls(z4=z4, q=q)
                z = self.mm._pack_z(z4)
                ell = self.mm._pack_ell(ell4)
                h = self._hamiltonian(z=z, q=q, ell=ell, i=i)
                z_loss = -h.mean()

                opt_z.zero_grad(set_to_none=True)
                z_loss.backward()
                opt_z.step()
                epoch_z += z_loss.item()

                # --- 3. Actor exploration update ---
                # Re-evaluate after exploitation step updated the actor
                z4 = self.policy(q, i)
                ell4 = self.mm.controls(z4=z4, q=q)
                z = self.mm._pack_z(z4)
                ell = self.mm._pack_ell(ell4)

                perturbation = torch.randn_like(z4)  # (B, 4)
                z4_pert = z4 + perturbation
                ell4_pert = self.mm.controls(z4=z4_pert, q=q)
                z_pert = self.mm._pack_z(z4_pert)
                ell_pert = self.mm._pack_ell(ell4_pert)

                h_base = self._hamiltonian(z=z, q=q, ell=ell, i=i)
                h_pert = self._hamiltonian(z=z_pert, q=q, ell=ell_pert, i=i)
                e_loss = -(h_pert - h_base).mean()

                opt_z_explore.zero_grad(set_to_none=True)
                e_loss.backward()
                opt_z_explore.step()
                epoch_e += e_loss.item()

            # Record epoch-averaged losses
            losses["value"].append(epoch_v / self.N)
            losses["policy"].append(epoch_z / self.N)
            losses["exploration"].append(epoch_e / self.N)

            if epoch % log_per == 0 or epoch == final_epoch:
                _log(
                    f"  Epoch {epoch:>{len(str(final_epoch))}}/{final_epoch} | "
                    f"v_loss: {losses['value'][-1]:.6f} | "
                    f"z_loss: {losses['policy'][-1]:.6f} | "
                    f"e_loss: {losses['exploration'][-1]:.6f}"
                )

            if save_dir is not None and (epoch % save_per == 0 or epoch == final_epoch):
                ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save(
                    path=ckpt_path,
                    epochs_trained=epoch,
                    optimizers=optimizers,
                    losses=losses,
                )
                _log(f"  Checkpoint saved -> {ckpt_path}")

        _log(
            f"Training complete. Final epoch: {final_epoch} | "
            f"avg v_loss (last {min(log_per, epochs)}): "
            f"{sum(losses['value'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f} | "
            f"avg z_loss: "
            f"{sum(losses['policy'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f} | "
            f"avg e_loss: "
            f"{sum(losses['exploration'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f}"
        )

        return losses
