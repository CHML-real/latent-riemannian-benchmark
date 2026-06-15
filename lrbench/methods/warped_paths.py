from __future__ import annotations

import torch

from lrbench.methods.euclidean_line import euclidean_line


def warped_detour_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    detour_x: float = 1.0,
    num_points: int = 96,
) -> torch.Tensor:
    """Construct a simple metric-aware detour path in 2D.

    For the default warped metric G=diag(exp(alpha*x), exp(-alpha*x)),
    vertical motion is cheaper at positive x. If endpoints require a large
    vertical move at negative x, a detour through positive x can have lower
    Riemannian length/energy than the Euclidean straight segment.

    The path is piecewise linear:
        z0 -> (detour_x, z0_y) -> (detour_x, z1_y) -> z1

    Args:
        z0: Tensor [batch, 2].
        z1: Tensor [batch, 2].
        detour_x: x coordinate used for the middle corridor.
        num_points: Total number of output points, including endpoints.

    Returns:
        Tensor [num_points, batch, 2].
    """
    if z0.shape != z1.shape:
        raise ValueError(f"z0 and z1 must have the same shape, got {z0.shape} and {z1.shape}")
    if z0.ndim != 2 or z0.shape[-1] != 2:
        raise ValueError(f"expected z0/z1 shape [batch, 2], got {z0.shape}")
    if num_points < 4:
        raise ValueError("num_points must be >= 4")

    batch = z0.shape[0]
    mid1 = z0.clone()
    mid1[:, 0] = torch.as_tensor(detour_x, device=z0.device, dtype=z0.dtype)
    mid2 = z1.clone()
    mid2[:, 0] = torch.as_tensor(detour_x, device=z0.device, dtype=z0.dtype)

    n1 = max(2, num_points // 3 + 1)
    n2 = max(2, num_points // 3 + 1)
    n3 = max(2, num_points - n1 - n2 + 2)

    p1 = euclidean_line(z0, mid1, n1)
    p2 = euclidean_line(mid1, mid2, n2)[1:]
    p3 = euclidean_line(mid2, z1, n3)[1:]
    path = torch.cat([p1, p2, p3], dim=0)

    # Adjust for any off-by-one introduced by integer splitting.
    if path.shape[0] > num_points:
        keep = torch.linspace(0, path.shape[0] - 1, num_points, device=z0.device)
        idx = keep.round().to(torch.long)
        path = path[idx]
    elif path.shape[0] < num_points:
        pad = z1.view(1, batch, 2).expand(num_points - path.shape[0], batch, 2)
        path = torch.cat([path, pad], dim=0)

    path[0] = z0
    path[-1] = z1
    return path
