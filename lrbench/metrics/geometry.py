from __future__ import annotations

import torch


def euclidean_path_energy(path: torch.Tensor) -> dict[str, float]:
    """Discrete Euclidean path energy: sum ||delta z||^2 over the path."""
    diffs = path[1:] - path[:-1]
    energy_per_item = (diffs * diffs).sum(dim=-1).sum(dim=0)
    return {
        "energy_mean": float(energy_per_item.mean().item()),
        "energy_max": float(energy_per_item.max().item()),
        "energy_min": float(energy_per_item.min().item()),
    }
