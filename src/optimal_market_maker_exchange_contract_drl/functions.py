from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
def IntensityParams():
    A: torch.Tensor
    c: torch.Tensor
    eps: torch.Tensor


def make_intensity_params(
    A_l: float,
    A_d: float,
    theta_l: float,
    theta_d: float,
    sigma: float,
    eps: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> IntensityParams:
    A = torch.tensor([A_l, A_d], device=device, dtype=dtype)
    c = torch.tensor([theta_l / sigma, theta_d / sigma], device=device, dtype=dtype)
    eps = torch.tensor(eps, device=device, dtype=dtype)
    return IntensityParams(A=A, c=c, eps=eps)


def compute_imbalance(ell_l: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ell_a = ell_l[..., 0]
    ell_b = ell_l[..., 1]
    den = (ell_a + ell_b).clamp_min(1e-12)

    return ell_a / den, ell_b / den


def compute_psi(ell_l: torch.tensor) -> torch.Tensor:
    I_a, I_b = compute_imbalance(ell_l)

    psi = torch.empty((*ell_l.shape[:-1], 2, 2), device=ell_l.device, dtype=ell_l.dtype)
    psi[..., 0, 0] = I_a  # (a, l)
    psi[..., 1, 0] = I_b  # (b, l)
    psi[..., 0, 1] = I_b  # (a, d)
    psi[..., 1, 1] = I_a  # (b, d)

    return psi


def compute_lam_base(
    ell_l: torch.Tensor, intensity_params: IntensityParams
) -> torch.Tensor:
    psi = compute_psi(ell_l)  # (B, 2, 2)

    A = intensity_params.A.view(1, 1, 2)
    c = intensity_params.c.view(1, 1, 2)

    lam_temp = A * torch.exp(-c * psi)  # (B , 2, 2)

    ell_nonzero = (ell_l[:, 0] + ell_l[:, 1]) > 0
    lam_base = torch.where(ell_nonzero[:, None, None], lam_temp, intensity_params.eps)
    return lam_base
