import torch
import torch.nn as nn

_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


class FCnet(nn.Module):
    def __init__(
        self,
        q_bar: torch.Tensor,
        layers: list[int],
        activation: str = "elu",
    ):
        super().__init__()

        act = _ACTS[activation.lower()]

        mods: list[nn.Module] = []
        for in_f, out_f in zip(layers[:-2], layers[1:-1]):
            mods.append(nn.Linear(in_f, out_f))
            mods.append(act())

        mods.append(nn.Linear(layers[-2], layers[-1]))
        mods.append(nn.Sigmoid())

        self.net = nn.Sequential(*mods)

    def forward(self, y):
        return self.net(y) * self.q_bar
