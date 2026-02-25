import numpy as np
import torch

from .functions import compute_lam_base


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

        # Training params
        self.B = exchange_cfg["batch_size"]
        self.device = device
        self.dtype = dtype

        # Initialize inventory
        self.Q = torch.zeros(self.B, device=device, dtype=dtype)  # (B)

        # Counts N^{i, j, k}
        self.N_l = torch.zeros(
            self.B, 2, self.V_l.numel(), device=device, dtype=torch.int64
        )
        self.N_d = torch.zeros(
            self.B, 2, self.V_d.numel(), device=device, dtype=torch.int64
        )

        # Current posted volumes
        self.ell_idx = torch.zeros(
            self.B, 2, 2, device=device, dtype=torch.int64
        )  # (B, 2, 2)
        self.ell_val = torch.zeros(
            self.B, 2, 2, device=device, dtype=dtype
        )  # (B, 2, 2)

        # Phi(i) for vectorized operations
        self.phi = torch.tensor([1.0, -1.0], device=device, dtype=dtype).view(1, 2)

    def lam_base(self) -> torch.Tensor:
        ell_l = self.ell_val[:, :, 0]  # (B, 2)
        return compute_lam_base(ell_l, self.market_params)

    def lam_eff(self) -> torch.Tensor:
        lam = self.lam_base()

        inv_mask = (self.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        return lam * inv_mask[:, :, None].to(lam.dtype)
