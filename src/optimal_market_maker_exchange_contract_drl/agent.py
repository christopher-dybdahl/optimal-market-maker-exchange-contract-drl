from typing import Tuple

import numpy as np

from .functions import Lam


class MarketMaker:
    def __init__(self, market_maker_cfg, V_l: np.array, V_d: np.array):
        self.q_bar = market_maker_cfg["q_bar"]  # Single side risk limit

        # Intensity
        self.lam = Lam(
            A=market_maker_cfg["A"],
            theta=market_maker_cfg["theta"],
            sigma=market_maker_cfg["sigma"],
            epsilon=market_maker_cfg["epsilon"],
        )

        # Filled trades
        self.V_l = V_l  # Valid volumes in lit pool
        self.V_d = V_d  # Valid volumes in dark pool
        self.N_a_l = np.zeros_like(V_l)  # Filled ask trades in lit pool
        self.N_b_l = np.zeros_like(V_l)  # Filled buy trades in lit  pool
        self.N_a_d = np.zeros_like(V_d)  # Filled ask trades in dark pool
        self.N_b_d = np.zeros_like(V_d)  # Filled buy trades in dark pool

    def get_Q(self):
        # Aggregated sum of volumes filled
        return (self.V_l @ self.N_b_l - self.V_l @ self.N_a_l) + (
            self.V_d @ self.N_b_d - self.V_d @ self.N_a_d
        )

    def lam(self, i: str, j: str, L_l: Tuple[float, float]):
        # Intensity of processes N^{i, j ,k}
        def theta(i):
            if i == "a":
                return 1
            elif i == "b":
                return 0
            else:
                raise ValueError(f"Invalid input: i = {i}")

        if (theta(i) * self.get_Q() > -self.q_bar) and ():
            return self.lam.eval(i, j, L_l)
        else:
            return 0
