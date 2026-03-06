import torch
import torch.nn as nn

from .agent import MarketMaker
from .dynamics import Market
from .nnets import FCnet


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

        # Precomputed market impact: (1, 1, 2) = [Gamma_l, Gamma_d]
        self.register_buffer("_Gamma", self.market.Gamma.view(1, 1, 2).to(dtype=dtype))

        # ----- Neural networks: one per time step -----
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

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def value(self, q: torch.Tensor, t: int) -> torch.Tensor:
        """
        Value function v_t(q).

        Args:
            q: (*,) inventory (any batch shape)
            t: time step index (0..N).
               t = N returns the terminal condition v(T, q) = -1.

        Returns:
            (*,) value at each q
        """
        if t == self.N:
            return -torch.ones_like(q)

        q_norm = (q / self.mm.q_bar).unsqueeze(-1)  # (*, 1)

        if self.exchange_cfg["critic_architecture"] == "fc":
            return self.critic_nets[t](q_norm).squeeze(-1)  # (*,)
        else:
            raise ValueError

    def policy(self, q: torch.Tensor, t: int) -> torch.Tensor:
        """
        Incentive policy z_t(q).

        Args:
            q: (B,) inventory
            t: time step index (0..N-1)

        Returns:
            (B, 4) incentives [z^{a,l}, z^{b,l}, z^{a,d}, z^{b,d}]
                   in [-z_bar, z_bar]
        """
        q_norm = (q / self.mm.q_bar).unsqueeze(-1)  # (B, 1)

        if self.exchange_cfg["actor_architecture"] == "fc":
            return self.actor_nets[t](q_norm) * self.mm.z_bar  # (B, 4)
        else:
            raise ValueError

    # ------------------------------------------------------------------
    #  Hamiltonian
    # ------------------------------------------------------------------

    def _hamiltonian(
        self,
        z: torch.Tensor,
        q: torch.Tensor,
        ell: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Exchange Hamiltonian U^c(z, q, ell*, v_t).

        From Section 5.1.2:

          U^c = v(q) [η/2 σ²γ(z^S+q)² + η²σ²/2 (z^S)²]
              + Σ_{i,j} λ^{i,j} [ exp(η(z^{i,j} - c^j ℓ^{i,j})) v(q')
                                 - v(q) (1 + η ε^{i,j}) ]

        where z^S = -γ/(γ+η) q,  q' = q - φ(i)ℓ^{i,j},
              ε^{i,j} = (1/γ)(1 - exp(-γ arg^{i,j})),
              arg^{i,j} = z^{i,j} + ℓ^{i,j}(T/2 + φ(i)Γ^j q) - Γ^j(ℓ^{i,j})².

        Args:
            z:   (B, 2, 2) incentives [side={a,b}, pool={l,d}]
            q:   (B,) inventory
            ell: (B, 2, 2) MM's optimal volumes [side={a,b}, pool={l,d}]
            t:   time step index for the value function

        Returns:
            (B,) Hamiltonian value per sample
        """
        B = q.shape[0]
        gamma = self.mm.gamma  # MM's risk aversion γ
        eta = self.eta  # Exchange's risk aversion η
        sigma = self.market.sigma
        half_tick = self.market.half_tick

        # ---- Value function evaluations ----
        # φ(i) = +1 for ask, -1 for bid → (1, 2, 1)
        phi = self.market.phi.unsqueeze(-1)

        # Shifted inventories: q' = q - φ(i) ℓ^{i,j}  → (B, 2, 2)
        q_shifted = q.view(B, 1, 1) - phi * ell

        # Batch-evaluate v_t at q and all 4 shifted q values
        q_all = torch.cat([q, q_shifted.reshape(B * 4)])  # (5B,)
        v_all = self.value(q_all, t)  # (5B,)
        v_q = v_all[:B]  # (B,)
        v_shifted = v_all[B:].reshape(B, 2, 2)  # (B, 2, 2)

        # ---- Drift term ----
        # z^S = -γ/(γ+η) q  (optimal inventory incentive, solved in closed form)
        z_S = -(gamma / (gamma + eta)) * q  # (B,)

        drift = v_q * (
            (eta / 2) * sigma**2 * gamma * (z_S + q) ** 2
            + (eta**2 * sigma**2 / 2) * z_S**2
        )  # (B,)

        # ---- ε: certainty equivalent per (i, j) ----
        # arg^{i,j} = z + ℓ(T/2 + φ(i)Γ^j q) − Γ^j ℓ²
        arg = (
            z
            + ell * (half_tick + phi * self._Gamma * q.view(B, 1, 1))
            - self._Gamma * ell**2
        )  # (B, 2, 2)

        eps = (1.0 / gamma) * (1.0 - torch.exp(-gamma * arg))  # (B, 2, 2)

        # ---- Arrival rates ----
        lam = self.market.lam_base(ell[:, :, 0])  # (B, 2, 2) [side, pool]

        # ---- Fee-adjusted exponential ----
        exp_term = torch.exp(eta * (z - self.c * ell))  # (B, 2, 2)

        # ---- Jump terms ----
        jump = lam * (
            exp_term * v_shifted - v_q.view(B, 1, 1) * (1.0 + eta * eps)
        )  # (B, 2, 2)

        return drift + jump.sum(dim=(1, 2))  # (B,)
