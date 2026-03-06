import torch.nn as nn

_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "affine": nn.Identity,
}

# Recommended gain for Xavier init per activation family
_GAINS = {
    "relu": nn.init.calculate_gain("relu"),
    "tanh": nn.init.calculate_gain("tanh"),
    "elu": 1.0,  # ELU ≈ linear for x>0; gain ~1.0 is standard
    "gelu": 1.0,
    "sigmoid": nn.init.calculate_gain("sigmoid"),
    "affine": 1.0,
}


class FCnet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        activation: str,
        output_activation: str,
    ):
        super().__init__()

        act = _ACTS[activation.lower()]
        gain = _GAINS.get(activation.lower(), 1.0)

        out_act_cls = _ACTS[output_activation.lower()]
        out_gain = _GAINS.get(output_activation.lower(), 1.0)

        mods: list[nn.Module] = []
        for in_f, out_f in zip(layers[:-2], layers[1:-1]):
            linear = nn.Linear(in_f, out_f)
            nn.init.xavier_uniform_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
            mods.append(linear)
            mods.append(act())

        # Output layer: small weights + zero bias so output activation starts
        # near its midpoint (e.g. sigmoid → ~0.5, so initial ell ≈ q_bar/2)
        out_linear = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(out_linear.weight, gain=0.1 * out_gain)
        nn.init.zeros_(out_linear.bias)
        mods.append(out_linear)
        mods.append(out_act_cls())

        self.net = nn.Sequential(*mods)

    def forward(self, y):
        return self.net(y)
