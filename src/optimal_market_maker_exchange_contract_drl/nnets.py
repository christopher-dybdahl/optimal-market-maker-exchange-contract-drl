import torch
import torch.nn as nn

_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "hardtanh": nn.Hardtanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "affine": nn.Identity,
}

# Recommended gain for Xavier init per activation family
_GAINS = {
    "relu": nn.init.calculate_gain("relu"),
    "tanh": nn.init.calculate_gain("tanh"),
    "hardtanh": nn.init.calculate_gain("tanh"),
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


class BatchedFCnet(nn.Module):
    """
    N independent copies of the same FC architecture, stored as batched
    weight tensors and evaluated simultaneously via torch.bmm.
    """

    def __init__(
        self,
        n: int,
        layers: list[int],
        activation: str,
        output_activation: str,
    ):
        super().__init__()
        self.n = n
        self.num_layers = len(layers) - 1

        self.act = _ACTS[activation.lower()]()
        self.out_act = _ACTS[output_activation.lower()]()

        gain = _GAINS.get(activation.lower(), 1.0)
        out_gain = _GAINS.get(output_activation.lower(), 1.0)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for idx, (in_f, out_f) in enumerate(zip(layers[:-1], layers[1:])):
            is_last = idx == self.num_layers - 1
            g = (0.1 * out_gain) if is_last else gain

            W = torch.empty(n, out_f, in_f)
            b = torch.zeros(n, out_f)
            for k in range(n):
                nn.init.xavier_uniform_(W[k], gain=g)

            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all N networks simultaneously.

        Args:
            x: (N, B, in_features)

        Returns:
            (N, B, out_features)
        """
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
            if idx < self.num_layers - 1:
                x = self.act(x)
            else:
                x = self.out_act(x)
        return x

    def forward_single(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Evaluate only the i-th network.

        Args:
            x: (B, in_features)
            i: network index

        Returns:
            (B, out_features)
        """
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W[i].t() + b[i]
            if idx < self.num_layers - 1:
                x = self.act(x)
            else:
                x = self.out_act(x)
        return x
