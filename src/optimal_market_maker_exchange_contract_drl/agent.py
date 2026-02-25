import torch

from .dynamics import Market


class MarketMaker:
    def __init__(
        self,
        market: Market,
        market_maker_cfg: dict,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        # Training params
        self.B = market_maker_cfg["batch_size"]
        self.device = device
        self.dtype = dtype

        # Market object
        self.market = market

        # Market params
        self.market_params = self.market.market_params

        # Market Maker params
        self.gamma = torch.tensor(
            market_maker_cfg["gamma"], device=device, dtype=dtype
        )  # Risk aversion parameter
        self.q_bar = torch.tensor(
            market_maker_cfg["q_bar"], device=device, dtype=dtype
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
        )  # (B, 2, 2)

        # Aggregated counts N^{i, j}
        self.N_agg = torch.zeros(
            (self.B, 2), device=device, dtype=torch.int64
        )  # (B, 2, 2)

        # Current posted volumes
        self.ell_idx = torch.zeros(
            (self.B, 2, 2), device=device, dtype=torch.int64
        )  # (B, 2, 2)
        self.ell_val = torch.zeros(
            (self.B, 2, 2), device=device, dtype=dtype
        )  # (B, 2, 2)

    def lam_eff(self) -> torch.Tensor:
        lam = self.market.lam_base(self.ell_val[:, :, 0])

        inv_mask = (self.market.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        return lam * inv_mask[:, :, None].to(lam.dtype)

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
