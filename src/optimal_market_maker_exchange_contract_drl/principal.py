import torch
import torch.nn as nn

from .dynamics import Market


class Exchange(nn.Module):
    def __init__(
        self,
        market: Market,
        exchange_cfg: dict,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Training params
        self.B = batch_size

        # Market object
        self.market = market

        # Exchange params
        self.register_buffer(
            "eta", torch.tensor(exchange_cfg["eta"], dtype=dtype)
        )  # Risk aversion parameter
        # TODO: Check if dimension for following tensor is correct
        self.register_buffer(
            "c", torch.tensor([exchange_cfg["c_l"], exchange_cfg["c_d"]], dtype=dtype)
        )  # (2,) Fixed fees for each market order [lit, dark]
