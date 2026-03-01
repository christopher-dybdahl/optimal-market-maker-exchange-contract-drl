import torch
import torch.nn as nn

from .dynamics import Market
from .nnets import FCnet


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

    def save(self, path: str) -> None:
        ckpt = {
            "mm_state_dict": self.state_dict(),
            "dtype": str(self.gamma.dtype).replace("torch.", ""),
        }
        torch.save(ckpt, path)

    def load(self, path: str, device: torch.device) -> None:
        ckpt = torch.load(path, map_location="cpu")

        dtype_str = ckpt.get("dtype", "float32")
        dtype = getattr(torch, dtype_str)

        self.to(device=device, dtype=dtype)
        self.load_state_dict(ckpt["mm_state_dict"])

        # reset episode state on the right device
        self.reset_state()

    def lam_eff(self, ell: torch.Tensor) -> torch.Tensor:
        lam = self.market.lam_base(ell[:, :, 0])  # (B, 2, 2) [side={a, b}, pool={l, d}]
        inv_mask = (self.market.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        lam = lam * inv_mask[:, :, None].to(lam.dtype)  # (B, 2, 2)

        # phi^d(i, kappa)
        phi_d = self.market.phi_d(
            ell[:, :, 0]
        )  # (B, 2, 2) [side={a, b}, kappa={lat, non}]

        # Expand to (B, 2, 3): [lam_l, lam_d * phi_d_lat, lam_d * phi_d_non]
        return torch.cat([lam[:, :, 0:1], lam[:, :, 1:2] * phi_d], dim=2)  # (B, 2, 3)

    def hamiltonian(
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
            (1.0 - torch.exp(-self.gamma * arg)) * self.lam_eff(ell)
        ).sum(dim=(1, 2))  # (B,)

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
