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
        # TODO: Check if dimension for following tensor is correct
        self.c = torch.tensor(
            [exchange_cfg["c_l"], exchange_cfg["c_d"]], device=device, dtype=dtype
        )  # (2, ) Fixed fees for each market order
