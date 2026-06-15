from __future__ import annotations

import torch

from lrbench.manifolds.warped import diagonal_warped_metric, metric_eigen_stats
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.warped_paths import warped_detour_path
from lrbench.metrics.accuracy import endpoint_error
from lrbench.metrics.warped_energy import warped_path_energy, warped_path_length


def test_warped_metric_is_spd():
    z = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    G = diagonal_warped_metric(z, alpha=2.0)
    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals > 0)
    stats = metric_eigen_stats(z, alpha=2.0)
    assert stats["spd_violation_rate"] == 0.0


def test_warped_detour_preserves_endpoints():
    z0 = torch.tensor([[-1.0, -4.0]], dtype=torch.float32)
    z1 = torch.tensor([[-1.0, 4.0]], dtype=torch.float32)
    path = warped_detour_path(z0, z1, detour_x=1.0, num_points=96)
    metrics = endpoint_error(path, z0, z1)
    assert metrics["endpoint_error_mean"] < 1e-6


def test_warped_detour_has_lower_energy_than_straight_path():
    z0 = torch.tensor([[-1.0, -4.0]], dtype=torch.float32)
    z1 = torch.tensor([[-1.0, 4.0]], dtype=torch.float32)
    straight = euclidean_line(z0, z1, num_points=96)
    detour = warped_detour_path(z0, z1, detour_x=1.0, num_points=96)
    e_straight = warped_path_energy(straight, alpha=2.0)["warped_energy_mean"]
    e_detour = warped_path_energy(detour, alpha=2.0)["warped_energy_mean"]
    assert e_detour < e_straight


def test_warped_detour_has_lower_length_than_straight_path():
    z0 = torch.tensor([[-1.0, -4.0]], dtype=torch.float32)
    z1 = torch.tensor([[-1.0, 4.0]], dtype=torch.float32)
    straight = euclidean_line(z0, z1, num_points=96)
    detour = warped_detour_path(z0, z1, detour_x=1.0, num_points=96)
    l_straight = warped_path_length(straight, alpha=2.0)["warped_length_mean"]
    l_detour = warped_path_length(detour, alpha=2.0)["warped_length_mean"]
    assert l_detour < l_straight
