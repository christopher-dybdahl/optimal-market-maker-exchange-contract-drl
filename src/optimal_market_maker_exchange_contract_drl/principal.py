from pathlib import Path

import torch
import torch.nn as nn

from .agent import MarketMaker
from .dynamics import Market
from .nnets import BatchedFCnet, FCnet
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
            * torch.tensor([1.0, 1.0, 0.0], dtype=dtype, device=self.market.half_tick.device).view(1, 1, 3),
        )

        # Normalized time grid: t_norm[i] = i / N ∈ [0, 1) for fc_time input
        self.register_buffer(
            "_t_norm", torch.arange(self.N, dtype=dtype) / self.N
        )

        # Neural networks
        if exchange_cfg["actor_architecture"] == "fc":
            self.actor_nets = BatchedFCnet(
                n=self.N,
                layers=exchange_cfg["actor_layers"],
                activation=exchange_cfg["actor_activation"],
                output_activation=exchange_cfg["actor_output_activation"],
            )
        elif exchange_cfg["actor_architecture"] == "fc_time":
            self.actor_nets = FCnet(
                layers=exchange_cfg["actor_layers"],
                activation=exchange_cfg["actor_activation"],
                output_activation=exchange_cfg["actor_output_activation"],
            )
        else:
            raise ValueError

        if exchange_cfg["critic_architecture"] == "fc":
            self.critic_nets = BatchedFCnet(
                n=self.N,
                layers=exchange_cfg["critic_layers"],
                activation=exchange_cfg["critic_activation"],
                output_activation=exchange_cfg["critic_output_activation"],
            )
        elif exchange_cfg["critic_architecture"] == "fc_time":
            self.critic_nets = FCnet(
                layers=exchange_cfg["critic_layers"],
                activation=exchange_cfg["critic_activation"],
                output_activation=exchange_cfg["critic_output_activation"],
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
            return self.critic_nets.forward_single(q_norm, i).squeeze(-1)  # (B,)
        elif self.exchange_cfg["critic_architecture"] == "fc_time":
            t = self._t_norm[i].expand(q.shape[0], 1)               # (B, 1)
            return self.critic_nets(torch.cat([q_norm, t], dim=-1)).squeeze(-1)
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
            return self.actor_nets.forward_single(q_norm, i) * self.mm.z_bar  # (B, 4)
        elif self.exchange_cfg["actor_architecture"] == "fc_time":
            t = self._t_norm[i].expand(q.shape[0], 1)               # (B, 1)
            return self.actor_nets(torch.cat([q_norm, t], dim=-1)) * self.mm.z_bar
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

    # ------------------------------------------------------------------
    # Batched methods — evaluate all N time-step networks in one call
    # ------------------------------------------------------------------

    def _value_all(self, q: torch.Tensor) -> torch.Tensor:
        """Evaluate all N critic networks at q.  Returns (N, B)."""
        q_norm = (q / self.mm.q_bar).unsqueeze(-1)              # (B, 1)

        if self.exchange_cfg["critic_architecture"] == "fc":
            x = q_norm.unsqueeze(0).expand(self.N, -1, -1)      # (N, B, 1)
            return self.critic_nets(x).squeeze(-1)               # (N, B)
        elif self.exchange_cfg["critic_architecture"] == "fc_time":
            B = q.shape[0]
            q_exp = q_norm.expand(self.N, -1, -1).reshape(self.N * B, 1)   # (N*B, 1)
            t_exp = self._t_norm.unsqueeze(1).expand(-1, B).reshape(self.N * B, 1)
            x = torch.cat([q_exp, t_exp], dim=-1)               # (N*B, 2)
            return self.critic_nets(x).squeeze(-1).reshape(self.N, B)
        else:
            raise ValueError

    def _value_all_shifted(self, q_shifted: torch.Tensor) -> torch.Tensor:
        """
        Evaluate critic_nets[i] at the four shifted-q values for each step i.

        Args:
            q_shifted: (N, B, 2, 2) — shifted inventories per (side, pool)

        Returns:
            (N, B, 2, 2) — critic values at each shifted q
        """
        N, B = q_shifted.shape[:2]

        if self.exchange_cfg["critic_architecture"] == "fc":
            qs_norm = (q_shifted.reshape(N, B * 4) / self.mm.q_bar).unsqueeze(-1)
            return self.critic_nets(qs_norm).squeeze(-1).reshape(N, B, 2, 2)
        elif self.exchange_cfg["critic_architecture"] == "fc_time":
            qs_norm = (q_shifted.reshape(N * B * 4) / self.mm.q_bar).unsqueeze(-1)  # (N*B*4, 1)
            t_exp = (
                self._t_norm.view(N, 1, 1, 1)
                .expand(-1, B, 2, 2)
                .reshape(N * B * 4, 1)
            )
            x = torch.cat([qs_norm, t_exp], dim=-1)              # (N*B*4, 2)
            return self.critic_nets(x).squeeze(-1).reshape(N, B, 2, 2)
        else:
            raise ValueError

    def _policy_all(self, q: torch.Tensor) -> torch.Tensor:
        """Evaluate all N actor networks at q.  Returns (N, B, 4)."""
        q_norm = (q / self.mm.q_bar).unsqueeze(-1)              # (B, 1)

        if self.exchange_cfg["actor_architecture"] == "fc":
            x = q_norm.unsqueeze(0).expand(self.N, -1, -1)      # (N, B, 1)
            return self.actor_nets(x) * self.mm.z_bar            # (N, B, 4)
        elif self.exchange_cfg["actor_architecture"] == "fc_time":
            B = q.shape[0]
            q_exp = q_norm.expand(self.N, -1, -1).reshape(self.N * B, 1)
            t_exp = self._t_norm.unsqueeze(1).expand(-1, B).reshape(self.N * B, 1)
            x = torch.cat([q_exp, t_exp], dim=-1)               # (N*B, 2)
            return (self.actor_nets(x) * self.mm.z_bar).reshape(self.N, B, 4)
        else:
            raise ValueError

    def _controls_all(
        self, z4_all: torch.Tensor, q_rep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MM controls for all N time steps.

        Args:
            z4_all: (N, B, 4)
            q_rep:  (N*B,)

        Returns:
            z_all  (N, B, 2, 2),
            ell_all (N, B, 2, 2)
        """
        N, B = z4_all.shape[:2]
        z4_flat = z4_all.reshape(N * B, 4)
        ell4_flat = self.mm.controls(z4=z4_flat, q=q_rep)
        z_all = self.mm._pack_z(z4_flat).reshape(N, B, 2, 2)
        ell_all = self.mm._pack_ell(ell4_flat).reshape(N, B, 2, 2)
        return z_all, ell_all

    def _hamiltonian_all(
        self,
        z_all: torch.Tensor,
        q: torch.Tensor,
        ell_all: torch.Tensor,
        v_q: torch.Tensor,
        v_shifted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched Hamiltonian for all N time steps.

        Same maths as _hamiltonian(), but with pre-computed v values
        so the caller can choose which time-index to evaluate.

        Args:
            z_all:     (N, B, 2, 2) incentives
            q:         (B,)         inventory
            ell_all:   (N, B, 2, 2) MM optimal volumes
            v_q:       (N, B)       value at q for each step
            v_shifted: (N, B, 2, 2) value at shifted q for each step

        Returns:
            (N, B)
        """
        N, B = z_all.shape[:2]
        gamma = self.mm.gamma
        eta = self.eta
        sigma = self.market.sigma

        # ---- Drift ----
        z_S = -(gamma / (gamma + eta)) * q                          # (B,)
        drift = v_q * (
            (eta / 2) * sigma ** 2 * gamma * (z_S + q) ** 2
            + (eta ** 2 * sigma ** 2 / 2) * z_S ** 2
        )  # (N, B) via broadcast

        # ---- ε computation ---- expand to 3 channels {l, d_lat, d_non}
        phi = self.market.phi.unsqueeze(-1)                          # (1, 2, 1)
        z3 = torch.cat([z_all[..., 0:1], z_all[..., 1:2],
                         z_all[..., 1:2]], dim=-1)                   # (N, B, 2, 3)
        ell3 = torch.cat([ell_all[..., 0:1], ell_all[..., 1:2],
                           ell_all[..., 1:2]], dim=-1)

        arg = (
            z3
            + ell3 * (self._spread3 + phi * self._Gamma3
                      * q.view(1, B, 1, 1))
            - self._Gamma3 * ell3 ** 2
        )  # (N, B, 2, 3)

        eps_raw = (1.0 / gamma) * (1.0 - torch.exp(-gamma * arg))

        ell_l_flat = ell_all[..., 0].reshape(N * B, 2)              # (N*B, 2)
        phi_d = self.market.phi_d(ell_l_flat).reshape(N, B, 2, 2)
        eps_d = (phi_d * eps_raw[..., 1:3]).sum(dim=-1, keepdim=True)
        eps = torch.cat([eps_raw[..., 0:1], eps_d], dim=-1)          # (N, B, 2, 2)

        # ---- Arrival rates & fee-adjusted exponential ----
        lam = self.market.lam_base(ell_l_flat).reshape(N, B, 2, 2)
        exp_term = torch.exp(eta * (z_all - self.c * ell_all))

        # ---- Jump ----
        jump = lam * (
            exp_term * v_shifted
            - v_q.view(N, B, 1, 1) * (1.0 + eta * eps)
        )

        return drift + jump.sum(dim=(-2, -1))                       # (N, B)

    def save(
        self,
        path: str,
        epochs_trained: int,
        optimizers: dict[str, torch.optim.Optimizer] = None,
        losses: dict[str, list[float]] = None,
        best_loss: float = None,
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
            "best_loss": best_loss,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load(
        self, path: str, device: torch.device
    ) -> tuple[int, dict | None, dict[str, list[float]], float | None]:
        """
        Load exchange weights and training state from a checkpoint.

        Returns:
            (epochs_trained, optimizer_states, losses, best_loss) — pass these
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
            ckpt.get("best_loss", None),
        )

    def fit(
        self,
        epochs: int,
        lr_v: float,
        lr_z: float,
        lr_z_explore: float,
        n_critic_steps: int = 5,
        seed: int = None,
        save_dir: Path = None,
        save_per: int = 50,
        log_per: int = 10,
        logger: Logger = None,
        start_epoch: int = 1,
        optimizer_states: dict = None,
        prior_losses: dict[str, list[float]] = None,
        best_loss: float = None,
    ) -> tuple[dict[str, list[float]], float]:
        """
        Train the exchange actor-critic networks.

        All N time steps are updated in parallel each epoch via batched
        forward/backward passes (one per phase), instead of a sequential
        loop over time steps.

        Per epoch:
          Phase 1-2 (repeated n_critic_steps times):
            compute Bellman targets and update all N critics
          Phase 3:  update all N actors (exploitation) in one pass
          Phase 4:  update all N actors (exploration) in one pass

        The inner critic loop compensates for the parallel (Jacobi) update
        structure: the paper's sequential backward sweep propagates the
        terminal condition v(T)=-1 in one pass, but parallel updates need
        multiple passes for the same information to propagate.

        Returns:
            Tuple of (losses, best_loss) where losses is a dict with keys
            "value", "policy", "exploration" (each a list of per-epoch losses)
            and best_loss is the best (lowest) value loss seen.
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = self.eta.device
        dtype = self.eta.dtype

        # --- Three optimizers: critic, actor-exploit, actor-explore ---
        # Separate Adam for exploit vs explore because they receive
        # fundamentally different gradient types (first-order backprop vs
        # zeroth-order score function) and have independent learning rates.
        opt_v = torch.optim.Adam(self.critic_nets.parameters(), lr=lr_v)
        opt_z = torch.optim.Adam(self.actor_nets.parameters(), lr=lr_z)
        opt_z_explore = torch.optim.Adam(self.actor_nets.parameters(), lr=lr_z_explore)

        if optimizer_states is not None:
            opt_v.load_state_dict(optimizer_states["opt_v"])
            opt_z.load_state_dict(optimizer_states["opt_z"])
            if "opt_z_explore" in optimizer_states:
                opt_z_explore.load_state_dict(optimizer_states["opt_z_explore"])

        optimizers = {"opt_v": opt_v, "opt_z": opt_z, "opt_z_explore": opt_z_explore}

        final_epoch = start_epoch + epochs - 1
        losses: dict[str, list[float]] = {
            "value": list(prior_losses.get("value", [])) if prior_losses else [],
            "policy": list(prior_losses.get("policy", [])) if prior_losses else [],
            "exploration": list(prior_losses.get("exploration", []))
            if prior_losses
            else [],
        }
        best_loss = best_loss if best_loss is not None else float("inf")

        def _log(msg: str) -> None:
            if logger is not None:
                logger.log(msg)

        _log(
            f"Training epochs {start_epoch} -> {final_epoch} "
            f"(lr_v={lr_v}, lr_z={lr_z}, lr_z_explore={lr_z_explore}, B={self.B})"
        )

        N, B = self.N, self.B
        phi = self.market.phi.unsqueeze(-1)  # (1, 2, 1) — broadcasts to (N, B, 2, *)

        for epoch in range(start_epoch, start_epoch + epochs):
            # Sample q ~ Uniform([-q_bar, q_bar])
            q = (
                2.0 * torch.rand((B,), device=device, dtype=dtype) - 1.0
            ) * self.mm.q_bar
            q_rep = q.unsqueeze(0).expand(N, -1).reshape(N * B)  # (N*B,)

            # ==============================================================
            # Phases 1-2: Critic inner loop
            #
            # Repeat target computation + critic update n_critic_steps times.
            # Each pass recomputes targets from the UPDATED critics, so the
            # terminal condition v(T)=-1 propagates backward through
            # multiple steps within a single epoch.
            # ==============================================================
            for _cv in range(n_critic_steps):
                # Phase 1: Compute Bellman targets  (no_grad)
                with torch.no_grad():
                    z4_all = self._policy_all(q)                         # (N, B, 4)
                    z_all, ell_all = self._controls_all(z4_all, q_rep)

                q_shifted = q.view(1, B, 1, 1) - phi * ell_all      # (N, B, 2, 2)

                v_q_all = self._value_all(q)                         # (N, B)
                v_shifted_all = self._value_all_shifted(q_shifted)   # (N, B, 2, 2)

                terminal_q = -torch.ones(1, B, device=device, dtype=dtype)
                terminal_qs = -torch.ones(1, B, 2, 2, device=device, dtype=dtype)
                v_q_next = torch.cat([v_q_all[1:], terminal_q])
                v_shifted_next = torch.cat([v_shifted_all[1:], terminal_qs])

                H_target = self._hamiltonian_all(
                    z_all, q, ell_all, v_q_next, v_shifted_next
                )
                targets = v_q_next + self.dt * H_target              # (N, B)

                # Phase 2: Critic update
                v_pred = self._value_all(q)                              # (N, B)
                v_loss = ((v_pred - targets) ** 2).mean()

                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_nets.parameters(), max_norm=1.0
                )
                opt_v.step()

            # ==============================================================
            # Phase 3: Actor exploitation — maximise H w.r.t. z
            # ==============================================================
            # Freeze critic params so backward skips their gradients,
            # but input-gradients (∂v/∂q_shifted) still flow to the actor.
            self.critic_nets.requires_grad_(False)

            z4_all = self._policy_all(q)                             # (N, B, 4)
            z_all, ell_all = self._controls_all(z4_all, q_rep)
            q_shifted = q.view(1, B, 1, 1) - phi * ell_all

            with torch.no_grad():
                v_q_det = self._value_all(q)                         # (N, B)
            v_shifted_live = self._value_all_shifted(q_shifted)      # in graph

            H_exploit = self._hamiltonian_all(
                z_all, q, ell_all, v_q_det, v_shifted_live
            )
            z_loss = -H_exploit.mean()

            opt_z.zero_grad(set_to_none=True)
            z_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_nets.parameters(), max_norm=1.0
            )
            opt_z.step()

            # ==============================================================
            # Phase 4: Actor exploration — score function estimator
            #
            # Paper: ω += μ̂^z · (1/K) Σ_k ε_k · ∇_ω z_t(q_k)
            #                         · [U^c(z+ε,...) − U^c(z,...)]
            #
            # The advantage (H_pert − H_base) is a detached scalar weight;
            # gradients flow only through the policy output z4_all via the
            # surrogate loss:  −(advantage · ε · z).mean()
            # ==============================================================
            z4_all = self._policy_all(q)                             # post-exploit
            perturbation = torch.randn_like(z4_all)
            z4_pert = (z4_all.detach() + perturbation).clamp(
                -self.mm.z_bar, self.mm.z_bar
            )

            with torch.no_grad():
                z_all, ell_all = self._controls_all(
                    z4_all.detach(), q_rep
                )
                z_pert_all, ell_pert_all = self._controls_all(z4_pert, q_rep)

                qs_base = q.view(1, B, 1, 1) - phi * ell_all
                qs_pert = q.view(1, B, 1, 1) - phi * ell_pert_all

                v_q_det = self._value_all(q)
                v_sh_base = self._value_all_shifted(qs_base)
                v_sh_pert = self._value_all_shifted(qs_pert)

                H_base = self._hamiltonian_all(
                    z_all, q, ell_all, v_q_det, v_sh_base
                )
                H_pert = self._hamiltonian_all(
                    z_pert_all, q, ell_pert_all, v_q_det, v_sh_pert
                )
                advantage = H_pert - H_base                          # (N, B)

            # Score function surrogate: −advantage · ε · z  (gradient ascent on H)
            e_loss = -(
                advantage * (perturbation * z4_all).sum(dim=-1)
            ).mean()

            opt_z_explore.zero_grad(set_to_none=True)
            e_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_nets.parameters(), max_norm=1.0
            )
            opt_z_explore.step()

            # Unfreeze critic for next epoch
            self.critic_nets.requires_grad_(True)

            # ==============================================================
            # Logging / checkpointing
            # ==============================================================
            losses["value"].append(v_loss.item())
            losses["policy"].append(z_loss.item())
            losses["exploration"].append(e_loss.item())

            v_loss_val = losses["value"][-1]
            if save_dir is not None and v_loss_val < best_loss:
                best_loss = v_loss_val
                best_path = save_dir / "best_model.pt"
                self.save(
                    path=best_path,
                    epochs_trained=epoch,
                    optimizers=optimizers,
                    losses=losses,
                    best_loss=best_loss,
                )
                _log(f"  New best model (v_loss={best_loss:.6f}) saved -> {best_path}")
            elif v_loss_val < best_loss:
                best_loss = v_loss_val

            if epoch % log_per == 0 or epoch == final_epoch:
                _log(
                    f"  Epoch {epoch:>{len(str(final_epoch))}}/{final_epoch} | "
                    f"v_loss: {losses['value'][-1]:.6f} | "
                    f"z_loss: {losses['policy'][-1]:.6f} | "
                    f"e_loss: {losses['exploration'][-1]:.6f}"
                )

            if save_dir is not None and (
                epoch % save_per == 0 or epoch == final_epoch
            ):
                ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save(
                    path=ckpt_path,
                    epochs_trained=epoch,
                    optimizers=optimizers,
                    losses=losses,
                    best_loss=best_loss,
                )
                _log(f"  Checkpoint saved -> {ckpt_path}")

        _log(
            f"Training complete. Final epoch: {final_epoch} | "
            f"avg v_loss (last {min(log_per, epochs)}): "
            f"{sum(losses['value'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f} | "
            f"avg z_loss: "
            f"{sum(losses['policy'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f} | "
            f"avg e_loss: "
            f"{sum(losses['exploration'][-min(log_per, epochs) :]) / min(log_per, epochs):.6f} | "
            f"best v_loss: {best_loss:.6f}"
        )

        return losses, best_loss
