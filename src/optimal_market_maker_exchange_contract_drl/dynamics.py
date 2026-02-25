import torch


class Price:
    def __init__(
        self,
        market_params,
        price_cfg,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        # Market params
        self.market_params = market_params

        # Price params
        self.S_tilde_0 = torch.tensor(
            price_cfg["S_tilde_0"], device=device, dtype=dtype
        )  # Initial price of underlying asset

        # Training params
        self.B = price_cfg["batch_size"]
        self.device = device
        self.dtype = dtype
