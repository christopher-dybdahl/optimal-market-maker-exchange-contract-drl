from abc import ABC


class MarketMakerHamiltonian(ABC):
    def __init__(self, gamma: dict, lamb: float, Gamma: dict):
        self.gamma = gamma
        self.lamb = lamb
        self.Gamma = Gamma
