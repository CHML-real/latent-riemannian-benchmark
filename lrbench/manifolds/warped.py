from __future__ import annotations

import torch


def diagonal_warped_metric(z: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Return a 2D position-dependent diagonal Riemannian metric.

    The metric is

        G(x, y) = diag(exp(alpha * x), exp(-alpha * x))

    This makes vertical motion expensive at negative x and cheap at positive x.
    It is a minimal synthetic proxy for latent spaces where local geometry changes
    depending on position.

    Args:
        z: Tensor with shape (..., 2).
        alpha: Warp strength.

    Returns:
        Tensor with shape (..., 2, 2).
    """
    if z.shape[-1] != 2:
        raise ValueError(f"warped metric expects last dimension 2, got {z.shape}")

    x = z[..., 0]
    g_xx = torch.exp(alpha * x)
    g_yy = torch.exp(-alpha * x)

    G = torch.zeros(*z.shape[:-1], 2, 2, device=z.device, dtype=z.dtype)
    G[..., 0, 0] = g_xx
    G[..., 1, 1] = g_yy
    return G


def metric_eigen_stats(z: torch.Tensor, alpha: float = 2.0) -> dict[str, float]:
    """Report SPD/eigenvalue statistics for the warped metric."""
    G = diagonal_warped_metric(z, alpha=alpha)
    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals[..., 0]
    max_eig = eigvals[..., -1]
    condition = max_eig / torch.clamp(min_eig, min=1e-12)
    spd_violation = (min_eig <= 0).to(torch.float32)
    return {
        "metric_min_eigenvalue_mean": float(min_eig.mean().item()),
        "metric_min_eigenvalue_min": float(min_eig.min().item()),
        "metric_max_eigenvalue_mean": float(max_eig.mean().item()),
        "metric_condition_number_mean": float(condition.mean().item()),
        "metric_condition_number_max": float(condition.max().item()),
        "spd_violation_rate": float(spd_violation.mean().item()),
    }
