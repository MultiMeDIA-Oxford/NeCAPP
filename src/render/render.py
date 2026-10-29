"""Evaluate coordinate networks in memory-bounded chunks."""

import torch


def run_network(inputs, fn, netchunk):
    """Flatten coordinates, evaluate consecutive chunks, and restore the input grid shape."""
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    out_flat = torch.cat(
        [fn(uvt_flat[i : i + netchunk]) for i in range(0, uvt_flat.shape[0], netchunk)], 0
    )
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    return out
