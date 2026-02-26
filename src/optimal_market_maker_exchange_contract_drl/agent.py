import torch

from .dynamics import Market


class MarketMaker:
    def __init__(
        self,
        market: Market,
        mm_cfg: dict,
        device: torch.device,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        # Training params
        self.device = device
        self.B = batch_size
        self.dtype = dtype

        # Market object
        self.market = market

        # Market params
        self.market_params = self.market.market_params

        # Market Maker params
        self.gamma = torch.tensor(
            mm_cfg["gamma"], device=device, dtype=dtype
        )  # Risk aversion parameter
        self.q_bar = torch.tensor(
            mm_cfg["q_bar"], device=device, dtype=dtype
        )  # Single side risk limit

        # Initialize inventory
        self.Q = torch.zeros((self.B,), device=device, dtype=dtype)  # (B, )

        # Counts per volume N^{i, k}
        self.N_l = torch.zeros(
            (self.B, 2, self.market_params.V_l.numel()),
            device=device,
            dtype=torch.int64,
        )  # (B, 2, #V_l)
        self.N_d = torch.zeros(
            (self.B, 2, self.market_params.V_d.numel()),
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

        # Current posted volumes
        self.ell_idx = torch.zeros(
            (self.B, 2, 2), device=device, dtype=torch.int64
        )  # (B, 2, 2) [side={a, b}, pool={l, d}]
        self.ell_val = torch.zeros(
            (self.B, 2, 2), device=device, dtype=dtype
        )  # (B, 2, 2) [side={a, b}, pool={l, d}]

        # Pre-computed (1, 1, 3) broadcast constants for channels {l, d_lat, d_non}
        self._Gamma3 = self.market_params.Gamma[[0, 1, 1]].view(1, 1, 3)  # (1, 1, 3)
        self._spread3 = self.market_params.half_tick * torch.tensor(
            [1.0, 1.0, 0.0], device=device, dtype=dtype
        ).view(1, 1, 3)  # (1, 1, 3)  half_tick * phi_lat per channel

    def lam_eff(self) -> torch.Tensor:
        lam = self.market.lam_base(
            self.ell_val[:, :, 0]
        )  # (B, 2, 2) [side={a, b}, pool={l, d}]
        inv_mask = (self.market.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        lam = lam * inv_mask[:, :, None].to(lam.dtype)  # (B, 2, 2)

        # phi^d(i, kappa)
        phi_d = self.market.phi_d(
            self.ell_val[:, :, 0]
        )  # (B, 2, 2) [side={a, b}, kappa={lat, non}]

        # Expand to (B, 2, 3): [lam_l, lam_d * phi_d_lat, lam_d * phi_d_non]
        return torch.cat([lam[:, :, 0:1], lam[:, :, 1:2] * phi_d], dim=2)  # (B, 2, 3)

    def hamiltonian(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Tile ell and z to (B, 2, 3): channels {l, d_lat, d_non}
        # ell^{i,d} is identical for both dark kappa; z^{i,d} likewise
        ell3 = torch.cat([self.ell_val, self.ell_val[:, :, 1:2]], dim=2)  # (B, 2, 3)
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
            (1.0 - torch.exp(-self.gamma * arg)) * self.lam_eff()
        ).sum(dim=(1, 2))  # (B,)

    def update_state(self, v: torch.Tensor) -> None:
        # TODO: Implement update of
        # self.N_l
        # self.N_d
        # self.N_d_lat_agg
        # self.N_d_non_agg
        # self.N_l_agg
        # self.N_d_agg
        # self.ell_idx
        # self.ell_val
        # self.Q
        pass
