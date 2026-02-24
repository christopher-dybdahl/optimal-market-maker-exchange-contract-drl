import numpy as np
import torch

from .functions import compute_lam_base


class MarketMaker:
    def __init__(
        self,
        intensity_params,
        market_maker_cfg,
        V_l: np.array,
        V_d: np.array,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.intensity_params = intensity_params  # Intensity
        self.q_bar = torch.tensor(
            market_maker_cfg["q_bar"], device=device, dtype=dtype
        )  # Single side risk limit
        self.V_l = torch.from_numpy(V_l).to(
            device=device, dtype=dtype
        )  # Valid volumes in lit pool
        self.V_d = torch.from_numpy(V_d).to(
            device=device, dtype=dtype
        )  # Valid volumes in dark pool
        self.B = market_maker_cfg["batch_size"]
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
        return compute_lam_base(ell_l, self.intensity_params)

    def lam_eff(self) -> torch.Tensor:
        lam = self.lam_base()

        inv_mask = (self.phi * self.Q.view(self.B, 1)) > (-self.q_bar)  # (B, 2)
        return lam * inv_mask[:, :, None].to(lam.dtype)
