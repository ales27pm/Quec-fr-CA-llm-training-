import json
from pathlib import Path

from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.corpus_readiness import audit_corpus_readiness
from qfr_pipeline.modern_corpus import acquire_modern_corpus

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_ckan_fixture_acquisition_writes_deterministic_records(tmp_path: Path) -> None:
    out = tmp_path / "dq.jsonl"
    report = tmp_path / "dq_report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=25,
    )
    rows = _read_jsonl(out)
    assert payload["records_written"] == len(rows) == 2
    assert [row["package_id"] for row in rows] == ["pkg-001", "pkg-002"]
    assert all(row["adapter"] == "donnees_quebec_ckan" for row in rows)
    assert all(row["normalized_text_sha256"] for row in rows)


def test_ckan_adapter_skips_unknown_license_package(tmp_path: Path) -> None:
    out = tmp_path / "dq.jsonl"
    report = tmp_path / "dq_report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=25,
    )
    skipped = payload["skipped_packages"]
    assert any(item["package_id"] == "pkg-003" for item in skipped)
    assert any(item["reason"].startswith("license_not_allowed") for item in skipped)


def test_ckan_report_has_package_license_org_domain_summaries(tmp_path: Path) -> None:
    out = tmp_path / "dq.jsonl"
    report = tmp_path / "dq_report.json"
    acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=25,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["package_count_seen"] == 3
    assert payload["package_count_selected"] == 2
    assert payload["package_count_skipped"] >= 1
    assert payload["license_summary"].get("open_compatible") == 2
    assert "Ville de Québec" in payload["organization_summary"]
    assert payload["domain_summary"].get("public_administration") == 2


def test_ckan_max_documents_zero_no_fetch_and_empty_output(tmp_path: Path, monkeypatch) -> None:
    def _no_fetch(*args, **kwargs):
        raise AssertionError("urlopen should not be called when --max-documents 0")

    monkeypatch.setattr("qfr_pipeline.modern_corpus.urlopen", _no_fetch)
    out = tmp_path / "dq_empty.jsonl"
    report = tmp_path / "dq_empty_report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=0,
    )
    assert payload["records_written"] == 0
    assert out.read_text(encoding="utf-8") == ""
    assert any(item["reason"] == "dry_run_max_documents_zero" for item in payload["skipped_sources"])


def test_assnat_without_include_noncommercial_skips_source(tmp_path: Path) -> None:
    out = tmp_path / "assnat_skip.jsonl"
    report = tmp_path / "assnat_skip_report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.assnat_seed.template.yaml",
        out,
        report,
        include_noncommercial=False,
        max_documents=10,
    )
    assert payload["records_written"] == 0
    assert any(item["reason"] == "noncommercial_requires_explicit_flag" for item in payload["skipped_sources"])


def test_assnat_with_include_noncommercial_fixture_writes_records(tmp_path: Path) -> None:
    out = tmp_path / "assnat.jsonl"
    report = tmp_path / "assnat_report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.assnat_seed.template.yaml",
        out,
        report,
        include_noncommercial=True,
        max_documents=10,
    )
    rows = _read_jsonl(out)
    assert payload["records_written"] == len(rows)
    assert len(rows) >= 3
    assert all(row["adapter"] == "assnat_journal_debats" for row in rows)
    assert all(row["license_status"] == "noncommercial_only" for row in rows)


def test_assnat_filters_navigation_junk_text(tmp_path: Path) -> None:
    out = tmp_path / "assnat.jsonl"
    report = tmp_path / "assnat_report.json"
    acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.assnat_seed.template.yaml",
        out,
        report,
        include_noncommercial=True,
        max_documents=10,
    )
    text = out.read_text(encoding="utf-8").casefold()
    assert "accueil" not in text
    assert "plan du site" not in text
    assert "partager" not in text


def test_assnat_does_not_crawl_links(tmp_path: Path) -> None:
    out = tmp_path / "assnat.jsonl"
    report = tmp_path / "assnat_report.json"
    acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.assnat_seed.template.yaml",
        out,
        report,
        include_noncommercial=True,
        max_documents=10,
    )
    rows = _read_jsonl(out)
    assert rows
    assert all(row["source_url"] == "file://fixtures/modern_corpus/assnat_seed_page.html" for row in rows)


def test_duplicate_text_suppression_works(tmp_path: Path) -> None:
    manifest = tmp_path / "dup_manifest.yaml"
    manifest.write_text(
        """
kind: modern_corpus_acquisition_manifest
schema_version: "1.0"
primary_language: fr-CA
sources:
  - source_id: assnat_dup
    name: AssNat Dup
    source_type: official_html
    acquisition_status: active
    license_status: noncommercial_only
    commercial_use: permission_required
    allowed_for_training: true
    allowed_for_evaluation: false
    holdout_only: false
    min_text_chars: 80
    adapter:
      name: assnat_journal_debats
      seed_urls:
        - file://fixtures/modern_corpus/assnat_seed_page.html
        - file://fixtures/modern_corpus/assnat_seed_page.html
""",
        encoding="utf-8",
    )
    out = tmp_path / "dup.jsonl"
    report = tmp_path / "dup_report.json"
    payload = acquire_modern_corpus(
        manifest,
        out,
        report,
        include_noncommercial=True,
        max_documents=10,
    )
    rows = _read_jsonl(out)
    assert payload["records_written"] == len(rows)
    unique_hashes = {row["normalized_text_sha256"] for row in rows}
    assert len(unique_hashes) == len(rows)


def test_no_workspace_or_runner_path_leakage(tmp_path: Path) -> None:
    out = tmp_path / "dq.jsonl"
    report = tmp_path / "dq_report.json"
    acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=25,
    )
    text = out.read_text(encoding="utf-8") + report.read_text(encoding="utf-8")
    assert "/workspace" not in text
    assert "/home/runner" not in text


def test_cli_fail_on_empty_fails_only_when_expected(tmp_path: Path) -> None:
    runner = CliRunner()
    empty_out = tmp_path / "empty.jsonl"
    empty_report = tmp_path / "empty.json"
    r1 = runner.invoke(
        app,
        [
            "acquire-modern-corpus",
            "--manifest",
            "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
            "--out",
            str(empty_out),
            "--report",
            str(empty_report),
            "--max-documents",
            "0",
            "--fail-on-empty",
        ],
    )
    assert r1.exit_code != 0

    ok_out = tmp_path / "ok.jsonl"
    ok_report = tmp_path / "ok.json"
    r2 = runner.invoke(
        app,
        [
            "acquire-modern-corpus",
            "--manifest",
            "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
            "--out",
            str(ok_out),
            "--report",
            str(ok_report),
            "--max-documents",
            "25",
            "--fail-on-empty",
        ],
    )
    assert r2.exit_code == 0


def test_readiness_audit_includes_modern_and_institutional_ratios(tmp_path: Path) -> None:
    out = tmp_path / "dq.jsonl"
    report = tmp_path / "dq_report.json"
    acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition.donnees_quebec.template.yaml",
        out,
        report,
        max_documents=25,
    )
    readiness_out = tmp_path / "readiness.json"
    readiness = audit_corpus_readiness(out, readiness_out)
    assert "modern_source_ratio" in readiness
    assert "institutional_source_ratio" in readiness
    assert readiness["modern_source_ratio"] > 0.0
    assert readiness["institutional_source_ratio"] > 0.0
