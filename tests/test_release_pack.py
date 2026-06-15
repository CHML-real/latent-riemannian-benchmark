from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_release_metadata_files_exist():
    for path in ["CHANGELOG.md", "CITATION.cff", "LICENSE", "docs/release/RELEASE_NOTES_v1.1.2.md"]:
        assert Path(path).exists(), path


def test_citation_metadata_mentions_project_and_version():
    text = read("CITATION.cff")
    assert "Latent Riemannian Benchmark" in text
    assert 'version: "1.1.3"' in text
    assert "license: MIT" in text


def test_license_is_mit():
    text = read("LICENSE")
    assert "MIT License" in text
    assert "THE SOFTWARE IS PROVIDED" in text


def test_issue_templates_exist_and_explain_evidence_failures():
    paths = [
        ".github/ISSUE_TEMPLATE/benchmark-result.md",
        ".github/ISSUE_TEMPLATE/lrw-adapter-evidence.md",
        ".github/ISSUE_TEMPLATE/bug-report.md",
    ]
    for path in paths:
        text = read(path)
        assert "endpoint-preserving geodesic" in text or "LRW" in text


def test_release_notes_include_expected_failure_warning():
    text = read("docs/release/RELEASE_NOTES_v1.1.2.md")
    assert "evidence outputs, not harness errors" in text
    assert "PullbackMetric.metric_tensor" in text
    assert "BVPSolver.geodesic_path" in text
