from __future__ import annotations

import torch

from lrbench.decoders.toy_decoder import gaussian_bump_decoder
from lrbench.manifolds.pullback import toy_pullback_metric
from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.pullback_paths import pullback_bump_detour_path
from lrbench.metrics.accuracy import endpoint_error
from lrbench.metrics.pullback_energy import pullback_path_energy, pullback_path_length


def test_toy_decoder_output_shape():
    z = torch.zeros(5, 2)
    x = gaussian_bump_decoder(z)
    assert x.shape == (5, 4)


def test_pullback_metric_is_spd():
    z = torch.randn(64, 2)
    G = toy_pullback_metric(z)
    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals > 0)


def test_pullback_detour_preserves_endpoints():
    z0 = torch.tensor([[-2.0, 0.0], [-2.0, 0.0]])
    z1 = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    path = pullback_bump_detour_path(z0, z1, detour_y=1.5, num_points=65)
    metrics = endpoint_error(path, z0, z1)
    assert metrics["endpoint_error_mean"] < 1e-6


def test_pullback_detour_has_lower_energy_than_straight_path():
    z0 = torch.tensor([[-2.0, 0.0]] * 8)
    z1 = torch.tensor([[2.0, 0.0]] * 8)
    straight = euclidean_line(z0, z1, num_points=128)
    detour = pullback_bump_detour_path(z0, z1, detour_y=1.5, num_points=128)
    straight_energy = pullback_path_energy(straight)["pullback_energy_mean"]
    detour_energy = pullback_path_energy(detour)["pullback_energy_mean"]
    assert detour_energy < straight_energy


def test_pullback_detour_has_lower_length_than_straight_path():
    z0 = torch.tensor([[-2.0, 0.0]] * 8)
    z1 = torch.tensor([[2.0, 0.0]] * 8)
    straight = euclidean_line(z0, z1, num_points=128)
    detour = pullback_bump_detour_path(z0, z1, detour_y=1.5, num_points=128)
    straight_length = pullback_path_length(straight)["pullback_length_mean"]
    detour_length = pullback_path_length(detour)["pullback_length_mean"]
    assert detour_length < straight_length
