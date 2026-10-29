"""Select the configured reconstruction network."""

from .network import DensityNetwork


def get_network(type):
    """Resolve the configured network type."""
    if type == "mlp":
        return DensityNetwork

    else:
        raise NotImplementedError("Unknown network type!")
