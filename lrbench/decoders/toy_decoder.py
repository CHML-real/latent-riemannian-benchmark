from __future__ import annotations

import torch


def gaussian_bump_decoder(
    z: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
) -> torch.Tensor:
    """A small nonlinear decoder f: R^2 -> R^4 for pullback metric tests.

    The first two output channels preserve coordinates, which keeps the
    pullback metric positive definite. The third channel is a Gaussian bump
    centered at the origin. Paths crossing the origin therefore become
    expensive under G(z) = J(z)^T J(z). The fourth channel adds a mild coupling
    term so the Jacobian is not purely radial.
    """
    if z.shape[-1] != 2:
        raise ValueError(f"expected last dimension 2, got {z.shape}")

    x = z[..., 0]
    y = z[..., 1]
    r2 = x * x + y * y
    bump = beta * torch.exp(-r2 / (sigma * sigma))
    coupling = cross * x * y
    return torch.stack([x, y, bump, coupling], dim=-1)
