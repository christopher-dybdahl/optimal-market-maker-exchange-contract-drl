from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MarketParams:
    A: torch.Tensor  # (2, )
    c: torch.Tensor  # (2, )
    Gamma: torch.Tensor  # (2, )
    half_tick: torch.Tensor  # scalar
    sigma: torch.Tensor  # scalar
    V_l: torch.Tensor  # (#V_l, )
    V_d: torch.Tensor  # (#V_d, )
    tick_size: torch.Tensor  # scalar
    eps: torch.Tensor  # scalar


def make_market_params(
    A_l: float,
    A_d: float,
    Gamma_l: float,
    Gamma_d: float,
    theta_l: float,
    theta_d: float,
    tick_size: float,
    sigma: float,
    V_l: np.array,
    V_d: np.array,
    eps: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> MarketParams:
    A = torch.tensor([A_l, A_d], device=device, dtype=dtype)
    c = torch.tensor([theta_l / sigma, theta_d / sigma], device=device, dtype=dtype)
    Gamma = torch.tensor([Gamma_l, Gamma_d], device=device, dtype=dtype)
    half_tick = torch.tensor(tick_size * 0.5, device=device, dtype=dtype)
    sigma = torch.tensor(sigma, device=device, dtype=dtype)
    V_l = torch.from_numpy(V_l).to(device=device, dtype=dtype)
    V_d = torch.from_numpy(V_d).to(device=device, dtype=dtype)
    tick_size = torch.tensor(tick_size, device=device, dtype=dtype)
    eps = torch.tensor(eps, device=device, dtype=dtype)
    return MarketParams(
        A=A,
        c=c,
        Gamma=Gamma,
        half_tick=half_tick,
        sigma=sigma,
        V_l=V_l,
        V_d=V_d,
        tick_size=tick_size,
        eps=eps,
    )


class Market:
    def __init__(
        self,
        market_params: MarketParams,
        market_cfg: dict,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        # Training params
        self.B = market_cfg["batch_size"]
        self.device = device
        self.dtype = dtype

        # Market params
        self.market_params = market_params

        # Price states
        self.S_tilde_0 = market_cfg[
            "S_tilde_0"
        ]  # Initial unaffected mid-price of underlying asset
        self.S_tilde = torch.full(
            (self.B,), self.S_tilde_0, device=device, dtype=dtype
        )  # (B, ) Current unaffected mid-price of underlying asset
        self.S = torch.full(
            (self.B,), self.S_tilde_0, device=device, dtype=dtype
        )  # (B, ) Current mid-price of underlying asset
        self.P_l = torch.full(
            (self.B, 2),
            [
                self.S_tilde_0 + self.market_params.tick_size,
                self.S_tilde_0 - self.market_params.tick_size,
            ],
            device=device,
            dtype=dtype,
        )  # (B, 2) Current market price of underlying asset on lit pool
        self.P_d = torch.zeros(
            (self.B, 2, 2), device=device, dtype=dtype
        )  # (B, 2, 2) Current market price of underlying asset on dark pool
        self.P_d[..., 0, 0] = self.S_tilde_0 + self.market_params.half_tick  # (a, lat)
        self.P_d[..., 1, 0] = self.S_tilde_0 - self.market_params.half_tick  # (b, lat)
        self.P_d[..., 0, 1] = self.S_tilde_0  # (a, non-lat)
        self.P_d[..., 1, 1] = self.S_tilde_0  # (b, non-lat)

        # Stochastic increments
        self.dW = torch.zeros(
            (self.B,), device=device, dtype=dtype
        )  # (B, ) Current brownian diffusion increment
        self.dN_l = torch.zeros(
            (self.B, 2, self.market_params.V_l.numel()),
            device=device,
            dtype=torch.int64,
        )  # (B, 2, #V_l)
        self.dN_d = torch.zeros(
            (self.B, 2, self.market_params.V_d.numel()),
            device=device,
            dtype=torch.int64,
        )  # (B, 2, #V_d)

        # Stochastic latency
        self.v = torch.zeros((self.B, 2), device=device, dtype=torch.int8)  # (B, 2)

        # phi(i) for vectorized operations
        self.phi = torch.tensor(
            [1.0, -1.0], device=device, dtype=dtype, requires_grad=False
        ).view(1, 2)

        # phi^lat(k) for vectorized operations
        self.phi_lat = torch.tensor(
            [1.0, 0.0], device=device, dtype=dtype, requires_grad=False
        ).view(1, 1, 2)

    def imbalance(self, ell_l: torch.tensor) -> torch.Tensor:
        den = ell_l.sum(dim=-1, keepdim=True).clamp_min(self.market_params.eps)
        return ell_l / den

    def psi(self, ell_l: torch.Tensor) -> torch.Tensor:
        I_a, I_b = self.imbalance(ell_l).unbind(dim=-1)  # each (B, )

        row0 = torch.stack([I_a, I_b], dim=-1)  # (B, 2)
        row1 = torch.stack([I_b, I_a], dim=-1)  # (B, 2)
        return torch.stack([row0, row1], dim=-2)  # (B, 2, 2)

    def phi_d(self, ell_l: torch.Tensor) -> torch.Tensor:
        I_a, I_b = self.imbalance(ell_l).unbind(dim=-1)  # each (B, )

        row0 = torch.stack([I_b, I_a], dim=-1)  # (B, 2)  a: [lat, non]
        row1 = torch.stack([I_a, I_b], dim=-1)  # (B, 2)  b: [lat, non]
        return torch.stack([row0, row1], dim=-2)  # (B, 2, 2)

    def lam_base(self, ell_l: torch.Tensor) -> torch.Tensor:
        psi = self.psi(ell_l)  # (B, 2, 2)

        A = self.market_params.A.view(1, 1, 2)
        c = self.market_params.c.view(1, 1, 2)

        lam_temp = A * torch.exp(-c * psi)  # (B , 2, 2)

        ell_nonzero = (ell_l[:, 0] + ell_l[:, 1]) > 0
        lam_base = torch.where(
            ell_nonzero[:, None, None], lam_temp, self.market_params.eps
        )
        return lam_base

    def step(self) -> None:
        # TODO: Implement one step which updates
        # self.dW
        # self.dN_l
        # self.dN_d
        # self.v
        # self.S_tilde
        # and returns None or necessary objects?
        pass

    def post_liquidity(self) -> None:
        self.step()
        # TODO: Take in posted liquidity by market maker and update
        # self.S
        # and returns None or necessary objects?
        pass
