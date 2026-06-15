import torch

from lrbench.methods.euclidean_line import euclidean_line
from lrbench.metrics.accuracy import endpoint_error, euclidean_distance_error


def test_euclidean_line_endpoint_error_zero():
    z0 = torch.zeros(4, 3)
    z1 = torch.ones(4, 3)
    path = euclidean_line(z0, z1, num_points=8)
    metrics = endpoint_error(path, z0, z1)
    assert metrics["endpoint_error_mean"] == 0.0


def test_euclidean_line_distance_error_near_zero():
    z0 = torch.zeros(4, 3)
    z1 = torch.ones(4, 3)
    path = euclidean_line(z0, z1, num_points=8)
    metrics = euclidean_distance_error(path, z0, z1)
    assert metrics["distance_error_mean"] < 1e-6
