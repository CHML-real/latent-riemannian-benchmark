from __future__ import annotations

import torch


def tensor_stability(path: torch.Tensor) -> dict[str, float | bool]:
    """Basic numerical stability checks."""
    nan_mask = torch.isnan(path)
    inf_mask = torch.isinf(path)
    total = path.numel()
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())
    return {
        "has_nan": nan_count > 0,
        "has_inf": inf_count > 0,
        "nan_rate": float(nan_count / total),
        "inf_rate": float(inf_count / total),
    }
