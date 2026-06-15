from __future__ import annotations

from pathlib import Path

from lrbench.methods.lrw_adapter import probe_lrw_package
from lrbench.runners.run_lrw_probe import run_lrw_probe


def test_lrw_probe_reports_missing_candidate_as_unavailable():
    result = probe_lrw_package(["definitely_not_a_real_lrw_module_987654321"])
    assert result["available"] is False
    assert result["module_name"] is None
    assert isinstance(result["error"], str)


def test_lrw_probe_can_detect_existing_python_module():
    result = probe_lrw_package(["json"])
    assert result["available"] is True
    assert result["module_name"] == "json"
    assert isinstance(result["public_symbols"], list)


def test_lrw_probe_runner_writes_skipped_result_for_missing_package(tmp_path: Path):
    config_path = tmp_path / "lrw_probe_missing.yaml"
    output_path = tmp_path / "lrw_probe_missing.json"
    config_path.write_text(
        "\n".join(
            [
                "benchmark_id: lrw_probe_missing_test",
                "benchmark_layer: adapter",
                "target: latent_riemannian_world",
                "method: lrw_package_probe",
                f"output_path: {output_path.as_posix()}",
                "module_candidates:",
                "  - definitely_not_a_real_lrw_module_987654321",
            ]
        ),
        encoding="utf-8",
    )

    result = run_lrw_probe(str(config_path))
    assert result["status"] == "skipped"
    assert result["available"] is False
    assert result["skip"]["is_skipped"] is True
    assert output_path.exists()
