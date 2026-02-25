import torch


class Market:
    def __init__(
        self,
        market_params,
        price_cfg,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        # Training params
        self.B = price_cfg["batch_size"]
        self.device = device
        self.dtype = dtype

        # Market params
        self.market_params = market_params

        # Price states
        self.S_tilde_0 = price_cfg["S_tilde_0"]  # Initial mid-price of underlying asset
        self.S_tilde = torch.full(
            (self.B,), self.S_tilde_0, device=device, dtype=dtype
        )  # Current unaffected mid-price of underlying asset
        self.S = torch.full(
            (self.B,), self.S_tilde_0, device=device, dtype=dtype
        )  # Current unaffected mid-price of underlying asset
        self.P_l = torch.full(
            (self.B, 2),
            [
                self.S_tilde_0 + market_params.tick_size,
                self.S_tilde_0 - market_params.tick_size,
            ],
            device=device,
            dtype=dtype,
        )  # Current mid-price of underlying asset on lit pool
        self.P_d = torch.zeros(
            (self.B, 2, 2), device=device, dtype=dtype
        )  # Current mid-price of underlying asset on dark pool
        self.P_d[..., 0, 0] = self.S_tilde_0 + market_params.half_tick  # (a, lat)
        self.P_d[..., 1, 0] = self.S_tilde_0 - market_params.half_tick  # (b, lat)
        self.P_d[..., 0, 1] = self.S_tilde_0  # (a, non-lat)
        self.P_d[..., 1, 1] = self.S_tilde_0  # (b, non-lat)

        # Stochastic increments
        self.dW = torch.zeros(
            (self.B,), device=device, dtype=dtype
        )  # Current brownian diffusion increment
        self.dN_l = torch.zeros(
            (self.B, 2, self.V_l.numel()), device=device, dtype=torch.int64
        )  # (B, 2, #V_l)
        self.dN_d = torch.zeros(
            (self.B, 2, self.V_d.numel()), device=device, dtype=torch.int64
        )  # (B, 2, #V_d)

        self.v = torch.zeros(
            (self.B,), device=device, dtype=dtype
        )  # Current brownian diffusion increment
