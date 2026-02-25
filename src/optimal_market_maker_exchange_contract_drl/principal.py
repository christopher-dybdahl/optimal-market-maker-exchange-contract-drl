import numpy as np
import torch


class Exchange:
    def __init__(
        self,
        market_params,
        exchange_cfg,
        V_l: np.array,
        V_d: np.array,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        # Training params
        self.B = exchange_cfg["batch_size"]
        self.device = device
        self.dtype = dtype

        # Market params
        self.market_params = market_params
        self.V_l = torch.from_numpy(V_l).to(
            device=device, dtype=dtype
        )  # Valid volumes in lit pool
        self.V_d = torch.from_numpy(V_d).to(
            device=device, dtype=dtype
        )  # Valid volumes in dark pool

        # Exchange params
        self.eta = torch.tensor(
            exchange_cfg["eta"], device=device, dtype=dtype
        )  # Risk aversion parameter
