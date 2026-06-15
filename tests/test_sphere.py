from __future__ import annotations

import torch

from lrbench.manifolds.sphere import normalize_to_sphere, radial_unit_error
from lrbench.methods.sphere_great_circle import sphere_great_circle
from lrbench.metrics.accuracy import endpoint_error, sphere_distance_error


def test_sphere_great_circle_endpoints_are_preserved():
    torch.manual_seed(42)
    z0 = normalize_to_sphere(torch.randn(16, 3))
    z1 = normalize_to_sphere(torch.randn(16, 3))
    path = sphere_great_circle(z0, z1, num_points=32)
    metrics = endpoint_error(path, z0, z1)
    assert metrics["endpoint_error_mean"] < 1e-6


def test_sphere_great_circle_stays_on_unit_sphere():
    torch.manual_seed(42)
    z0 = normalize_to_sphere(torch.randn(16, 3))
    z1 = normalize_to_sphere(torch.randn(16, 3))
    path = sphere_great_circle(z0, z1, num_points=32)
    metrics = radial_unit_error(path)
    assert metrics["sphere_radial_error_max"] < 1e-6


def test_sphere_distance_error_near_zero():
    torch.manual_seed(42)
    z0 = normalize_to_sphere(torch.randn(16, 3))
    z1 = normalize_to_sphere(torch.randn(16, 3))
    path = sphere_great_circle(z0, z1, num_points=64)
    metrics = sphere_distance_error(path, z0, z1)
    assert metrics["sphere_distance_error_mean"] < 1e-4
