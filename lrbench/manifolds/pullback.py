from __future__ import annotations

import torch

from lrbench.decoders.toy_decoder import gaussian_bump_decoder


def decoder_jacobian(
    z: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
) -> torch.Tensor:
    """Compute batched Jacobian J(z) for the toy decoder.

    Returns shape [..., output_dim=4, latent_dim=2].
    """
    if z.shape[-1] != 2:
        raise ValueError(f"expected last dimension 2, got {z.shape}")

    z_req = z.detach().clone().requires_grad_(True)
    x = gaussian_bump_decoder(z_req, beta=beta, sigma=sigma, cross=cross)

    rows = []
    for out_idx in range(x.shape[-1]):
        grad = torch.autograd.grad(
            x[..., out_idx].sum(),
            z_req,
            retain_graph=True,
            create_graph=False,
        )[0]
        rows.append(grad)
    return torch.stack(rows, dim=-2)


def toy_pullback_metric(
    z: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    damping: float = 1e-4,
) -> torch.Tensor:
    """Compute G(z) = J(z)^T J(z) + damping * I for the toy decoder."""
    J = decoder_jacobian(z, beta=beta, sigma=sigma, cross=cross)
    G = torch.matmul(J.transpose(-1, -2), J)
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    return G + damping * eye


def pullback_metric_eigen_stats(
    z: torch.Tensor,
    beta: float = 6.0,
    sigma: float = 0.7,
    cross: float = 0.2,
    damping: float = 1e-4,
) -> dict[str, float]:
    """Eigenvalue, condition number, and SPD diagnostics for the pullback metric."""
    G = toy_pullback_metric(z, beta=beta, sigma=sigma, cross=cross, damping=damping)
    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals[..., 0]
    max_eig = eigvals[..., -1]
    cond = max_eig / torch.clamp(min_eig, min=1e-12)
    return {
        "pullback_min_eigenvalue_mean": float(min_eig.mean().item()),
        "pullback_min_eigenvalue_min": float(min_eig.min().item()),
        "pullback_max_eigenvalue_mean": float(max_eig.mean().item()),
        "pullback_condition_number_mean": float(cond.mean().item()),
        "pullback_condition_number_max": float(cond.max().item()),
        "pullback_spd_violation_rate": float((min_eig <= 0).float().mean().item()),
    }
