from __future__ import annotations

from lrbench.methods.lrw_source_probe import _contains_flags, _hash_text, default_source_targets, probe_source_target


def test_contains_flags_detects_key_solver_terms() -> None:
    text = "def geodesic_distance(self, z0, z1):\n    return self.step_size * torch.norm(z1 - z0)\n"
    flags = _contains_flags(text)
    assert flags["contains_step_size"] is True
    assert flags["contains_geodesic_distance"] is True
    assert flags["contains_torch_norm"] is True


def test_hash_text_is_stable_short_fingerprint() -> None:
    assert _hash_text("abc") == _hash_text("abc")
    assert len(_hash_text("abc")) == 16


def test_default_source_targets_include_solver_and_slerp() -> None:
    ids = {target.target_id for target in default_source_targets()}
    assert "geodesic_solver_interpolate" in ids
    assert "geodesic_solver_geodesic_distance" in ids
    assert "bvp_solver_geodesic_path" in ids
    assert "slerp_path_function" in ids


def test_probe_source_target_failure_is_structured() -> None:
    target = default_source_targets()[0]
    bad = type(target)("bad_target", "definitely_missing_module_xyz", ("Nope",))
    entry = probe_source_target(bad)
    assert entry["status"] == "failure"
    assert "error" in entry
