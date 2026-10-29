"""Map encoded spatial coordinates to vascular occupancy values."""

import torch
import torch.nn as nn


class DensityNetwork(nn.Module):
    """Predict occupancy from encoded coordinates with the existing MLP architecture."""

    def __init__(
        self,
        encoder,
        bound=None,
        num_layers=8,
        hidden_dim=256,
        skips=[4],
        out_dim=1,
        last_activation="sigmoid",
    ):
        super().__init__()
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        self.bound = bound

        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)]
            + [
                nn.Linear(hidden_dim, hidden_dim)
                if i not in skips
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                for i in range(1, num_layers - 1, 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers - 1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "softplus":
            self.activations.append(nn.Softplus(beta=1.0, threshold=20.0))
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        """Encode spatial coordinates and apply the existing skip-connected occupancy MLP."""
        x = self.encoder(x, self.bound)

        input_pts = x[..., : self.in_dim]

        for i in range(len(self.layers)):
            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)

        return x
