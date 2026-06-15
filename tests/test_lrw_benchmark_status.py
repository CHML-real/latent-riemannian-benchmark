from pathlib import Path


def test_lrw_benchmark_status_doc_exists():
    path = Path("docs/LRW_BENCHMARK_STATUS.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "collaborator-facing benchmark suite" in text
    assert "official LRW benchmark" in text
    assert "does not vendor, copy, or relicense" in text


def test_readme_mentions_lrw_collaborator_status():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "collaborator-facing benchmark suite for LRW" in text
    assert "official LRW benchmark suite" in text
    assert "evidence outputs, not harness errors" in text
