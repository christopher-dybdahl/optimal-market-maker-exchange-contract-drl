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
        S_tilde_0: float,
        V_l: list,
        V_d: list,
        eps: float,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.B = batch_size
        self.S_tilde_0 = S_tilde_0  # plain float kept for reset arithmetic

        # fmt: off
        # Fixed market parameters
        self.register_buffer("A",         torch.tensor([A_l, A_d],                         dtype=dtype))  # (2,)
        self.register_buffer("c",         torch.tensor([theta_l / sigma, theta_d / sigma], dtype=dtype))  # (2,)  theta^j / sigma
        self.register_buffer("Gamma",     torch.tensor([Gamma_l, Gamma_d],                 dtype=dtype))  # (2,)
        self.register_buffer("half_tick", torch.tensor(tick_size * 0.5,                    dtype=dtype))  # scalar  T/2
        self.register_buffer("sigma",     torch.tensor(sigma,                              dtype=dtype))  # scalar
        self.register_buffer("tick_size", torch.tensor(tick_size,                          dtype=dtype))  # scalar
        self.register_buffer("V_l",       torch.tensor(V_l,                                dtype=dtype))  # (#V_l,)
        self.register_buffer("V_d",       torch.tensor(V_d,                                dtype=dtype))  # (#V_d,)
        self.register_buffer("eps",       torch.tensor(eps,                                dtype=dtype))  # scalar

        # Vectorised sign constants
        self.register_buffer("phi",     torch.tensor([1.0, -1.0], dtype=dtype).view(1, 2))       # (1, 2)   phi(a)=1, phi(b)=-1
        self.register_buffer("phi_lat", torch.tensor([1.0, 0.0],  dtype=dtype).view(1, 1, 2))    # (1, 1, 2) phi^lat(lat)=1, phi^lat(non)=0

        # Simulation state (None until reset_state() is called)
        self.register_buffer("S_tilde", None)  # (B,)   unaffected mid-price
        self.register_buffer("S",       None)  # (B,)   mid-price with market impact
        self.register_buffer("P_l",     None)  # (B, 2) lit pool best ask/bid prices
        self.register_buffer("P_d",     None)  # (B, 2, 2) dark pool prices [side, kappa={lat,non}]
        self.register_buffer("dW",      None)  # (B,)
        self.register_buffer("dN_l",    None)  # (B, 2, #V_l)
        self.register_buffer("dN_d",    None)  # (B, 2, #V_d)
        self.register_buffer("v",       None)  # (B, 2) latency Bernoulli draws
        # fmt: on

    def reset_state(self) -> None:
        device = self.A.device
        dtype = self.A.dtype

        self.S_tilde = torch.full((self.B,), self.S_tilde_0, device=device, dtype=dtype)
        self.S = torch.full((self.B,), self.S_tilde_0, device=device, dtype=dtype)

        P_l = torch.empty((self.B, 2), device=device, dtype=dtype)
        P_l[:, 0] = self.S_tilde_0 + self.tick_size  # ask
        P_l[:, 1] = self.S_tilde_0 - self.tick_size  # bid
        self.P_l = P_l

        P_d = torch.empty((self.B, 2, 2), device=device, dtype=dtype)
        P_d[:, 0, 0] = self.S_tilde_0 + self.half_tick  # (a, lat)
        P_d[:, 1, 0] = self.S_tilde_0 - self.half_tick  # (b, lat)
        P_d[:, 0, 1] = self.S_tilde_0  # (a, non-lat)
        P_d[:, 1, 1] = self.S_tilde_0  # (b, non-lat)
        self.P_d = P_d

        self.dW = torch.zeros((self.B,), device=device, dtype=dtype)
        self.dN_l = torch.zeros(
            (self.B, 2, self.V_l.numel()), device=device, dtype=torch.int64
        )
        self.dN_d = torch.zeros(
            (self.B, 2, self.V_d.numel()), device=device, dtype=torch.int64
        )
        self.v = torch.zeros((self.B, 2), device=device, dtype=torch.int8)

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

    def step(self) -> None:
        # TODO: update dW, dN_l, dN_d, v, S_tilde
        pass

    def post_liquidity(self) -> None:
        self.step()
        # TODO: update S
        pass
