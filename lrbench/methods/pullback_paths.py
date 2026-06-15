from __future__ import annotations

import torch

from lrbench.methods.euclidean_line import euclidean_line


def pullback_bump_detour_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    detour_y: float = 1.5,
    num_points: int = 128,
) -> torch.Tensor:
    """Piecewise path that bends around a high-curvature decoder bump.

    The endpoints are expected to lie roughly on the x-axis. The detour goes via
    a midpoint at x=(x0+x1)/2 and y=detour_y, avoiding the expensive origin.
    """
    if z0.shape != z1.shape or z0.shape[-1] != 2:
        raise ValueError(f"expected z0/z1 shape [batch, 2], got {z0.shape}, {z1.shape}")
    if num_points < 3:
        raise ValueError("num_points must be at least 3")

    mid = 0.5 * (z0 + z1)
    mid = mid.clone()
    mid[:, 1] = detour_y

    n1 = num_points // 2 + 1
    n2 = num_points - n1 + 1
    first = euclidean_line(z0, mid, num_points=n1)
    second = euclidean_line(mid, z1, num_points=n2)
    return torch.cat([first[:-1], second], dim=0)
