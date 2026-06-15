from __future__ import annotations

import torch

from lrbench.methods.euclidean_line import euclidean_line
from lrbench.methods.lrw_geodesic_anatomy import path_axis_candidates, summarize_anatomy, unpack_lrw_output


def test_path_axis_candidates_detect_tbd_endpoint_valid() -> None:
    z0 = torch.tensor([[0.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    path = euclidean_line(z0, z1, num_points=5)
    candidates = path_axis_candidates(path, z0, z1, requested_n_points=5)
    best = min((c for c in candidates if c.get("valid_shape")), key=lambda c: c["endpoint_error_mean"])
    assert best["endpoint_error_mean"] == 0.0
    assert best["candidate_name"] == "as_TBD"


def test_path_axis_candidates_detect_btd_transpose_endpoint_valid() -> None:
    z0 = torch.tensor([[0.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    path_tbd = euclidean_line(z0, z1, num_points=5)
    path_btd = path_tbd.transpose(0, 1)
    candidates = path_axis_candidates(path_btd, z0, z1, requested_n_points=5)
    best = min((c for c in candidates if c.get("valid_shape")), key=lambda c: c["endpoint_error_mean"])
    assert best["endpoint_error_mean"] == 0.0
    assert best["candidate_name"] == "as_BTD_transpose_0_1"


def test_summarize_anatomy_reports_valid_axis() -> None:
    z0 = torch.tensor([[0.0, 0.0]])
    z1 = torch.tensor([[1.0, 0.0]])
    path = euclidean_line(z0, z1, num_points=5)
    calls = [{"call_name": "dummy", "status": "success", "axis_candidates": path_axis_candidates(path, z0, z1, 5)}]
    summary = summarize_anatomy(calls, endpoint_tolerance=1e-6)
    assert summary["anatomy_axis_endpoint_valid_count"] >= 1
    assert summary["anatomy_best_axis_endpoint_error"] == 0.0
    assert summary["anatomy_verdict"] == "axis_interpretation_pass"


def test_unpack_lrw_output_tuple() -> None:
    x = torch.zeros(2, 1, 3)
    raw, info = unpack_lrw_output((x, {"loss": 1.0}))
    assert raw is x
    assert info["raw_output_type"] == "tuple"
    assert info["tuple_len"] == 2
    assert info["solver_info"]["loss"] == 1.0
