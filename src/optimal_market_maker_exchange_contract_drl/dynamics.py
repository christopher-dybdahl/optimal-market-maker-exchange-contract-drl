import torch
import torch.nn as nn


class Market(nn.Module):
    def __init__(
        self,
        A_l: float,
        A_d: float,
        Gamma_l: float,
        Gamma_d: float,
        theta_l: float,
        theta_d: float,
        tick_size: float,
        sigma: float,
        eps: float,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.B = batch_size

        # fmt: off
        # Fixed market parameters
        self.register_buffer("A",         torch.tensor([A_l, A_d],                         dtype=dtype))  # (2,)
        self.register_buffer("c",         torch.tensor([theta_l / sigma, theta_d / sigma], dtype=dtype))  # (2,)  theta^j / sigma
        self.register_buffer("Gamma",     torch.tensor([Gamma_l, Gamma_d],                 dtype=dtype))  # (2,)
        self.register_buffer("half_tick", torch.tensor(tick_size * 0.5,                    dtype=dtype))  # scalar  T/2
        self.register_buffer("sigma",     torch.tensor(sigma,                              dtype=dtype))  # scalar
        self.register_buffer("tick_size", torch.tensor(tick_size,                          dtype=dtype))  # scalar
        self.register_buffer("eps",       torch.tensor(eps,                                dtype=dtype))  # scalar

        # Vectorised sign constants
        self.register_buffer("phi",     torch.tensor([1.0, -1.0], dtype=dtype).view(1, 2))       # (1, 2)   phi(a)=1, phi(b)=-1
        self.register_buffer("phi_lat", torch.tensor([1.0, 0.0],  dtype=dtype).view(1, 1, 2))    # (1, 1, 2) phi^lat(lat)=1, phi^lat(non)=0

    def imbalance(self, ell_l: torch.Tensor) -> torch.Tensor:
        den = ell_l.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return ell_l / den  # (B, 2)

    def psi(self, ell_l: torch.Tensor) -> torch.Tensor:
        """psi^{i,j}: imbalance weighting for lit/dark intensity. (B, 2, 2)"""
        I_a, I_b = self.imbalance(ell_l).unbind(dim=-1)  # each (B,)
        return torch.stack(
            [
                torch.stack([I_a, I_b], dim=-1),  # i=a: [psi^{a,l}, psi^{a,d}]
                torch.stack([I_b, I_a], dim=-1),  # i=b: [psi^{b,l}, psi^{b,d}]
            ],
            dim=-2,
        )  # (B, 2, 2)

    def phi_d(self, ell_l: torch.Tensor) -> torch.Tensor:
        """phi^d(i, kappa): dark pool routing probabilities. (B, 2, 2) [side, kappa={lat,non}]"""
        I_a, I_b = self.imbalance(ell_l).unbind(dim=-1)  # each (B,)
        return torch.stack(
            [
                torch.stack([I_b, I_a], dim=-1),  # i=a: [lat->I_b, non->I_a]
                torch.stack([I_a, I_b], dim=-1),  # i=b: [lat->I_a, non->I_b]
            ],
            dim=-2,
        )  # (B, 2, 2)

    def lam_base(self, ell_l: torch.Tensor) -> torch.Tensor:
        """Base arrival intensities lambda^{i,j}. (B, 2, 2) [side, pool={l,d}]"""
        psi = self.psi(ell_l)  # (B, 2, 2)
        A = self.A.view(1, 1, 2)  # (1, 1, 2)
        c = self.c.view(1, 1, 2)  # (1, 1, 2)
        lam_temp = A * torch.exp(-c * psi)  # (B, 2, 2)
        ell_nonzero = (ell_l[:, 0] + ell_l[:, 1]) > 0  # (B,)
        return torch.where(ell_nonzero[:, None, None], lam_temp, self.eps)  # (B, 2, 2)
