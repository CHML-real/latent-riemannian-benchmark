from pathlib import Path


def test_public_docs_exist_and_explain_expected_failures():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "evidence outputs, not harness errors" in readme
    assert "GEO_F01_ENDPOINT_MISS" in readme

    docs = root / "docs"
    for name in ["LRW_EVIDENCE_SUMMARY.md", "FAILURE_MODES.md", "REPRODUCIBILITY.md"]:
        path = docs / name
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert len(text) > 500


def test_reproducibility_doc_lists_evidence_runner():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "run_lrw_evidence_report" in text
    assert "results/reports/lrw_evidence_report.md" in text
