import torch

from .dynamics import Market


class Exchange:
    def __init__(
        self,
        market: Market,
        exchange_cfg: dict,
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

        # Exchange params
        self.eta = torch.tensor(
            exchange_cfg["eta"], device=device, dtype=dtype
        )  # Risk aversion parameter
