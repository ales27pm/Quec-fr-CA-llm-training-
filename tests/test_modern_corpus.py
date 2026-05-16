import json
from pathlib import Path

import pytest
from qfr_pipeline.modern_corpus import acquire_modern_corpus, validate_modern_corpus_manifest
from qfr_pipeline.schemas import ModernCorpusAcquisitionManifest

ROOT = Path(__file__).resolve().parents[1]


def test_valid_manifest_passes() -> None:
    validate_modern_corpus_manifest(ROOT / "manifests/modern_corpus_acquisition_manifest.template.yaml")


def test_duplicate_source_id_fails() -> None:
    payload = {
        "kind": "modern_corpus_acquisition_manifest",
        "schema_version": "1.0",
        "primary_language": "fr-CA",
        "sources": [
            {"source_id": "dup", "name": "A", "source_type": "catalog_only", "acquisition_status": "catalog_only", "license_status": "open_compatible", "commercial_use": "allowed", "allowed_for_training": False, "adapter": {"name": "noop"}},
            {"source_id": "dup", "name": "B", "source_type": "catalog_only", "acquisition_status": "catalog_only", "license_status": "open_compatible", "commercial_use": "allowed", "allowed_for_training": False, "adapter": {"name": "noop"}},
        ],
    }
    with pytest.raises(Exception):
        ModernCorpusAcquisitionManifest.model_validate(payload)


def test_holdout_training_allowed_fails() -> None:
    payload = {
        "kind": "modern_corpus_acquisition_manifest",
        "schema_version": "1.0",
        "primary_language": "fr-CA",
        "sources": [
            {"source_id": "qfrcola_holdout", "name": "QFrCoLA", "source_type": "evaluation_holdout", "acquisition_status": "holdout_only", "license_status": "holdout_only", "commercial_use": "prohibited", "allowed_for_training": True, "adapter": {"name": "holdout_registry"}},
        ],
    }
    with pytest.raises(Exception):
        ModernCorpusAcquisitionManifest.model_validate(payload)


def test_catalog_and_holdout_behavior(tmp_path: Path) -> None:
    out = tmp_path / "harvest.jsonl"
    report = tmp_path / "report.json"
    payload = acquire_modern_corpus(
        ROOT / "manifests/modern_corpus_acquisition_manifest.template.yaml",
        out,
        report,
        max_documents=0,
    )
    assert payload["ok"]
    assert any(item["reason"] == "catalog_only" for item in payload["skipped_sources"])
    assert "qfrcola_holdout" in payload["holdout_registry"]
    assert all(json.loads(line)["holdout_only"] is False for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    report_text = report.read_text(encoding="utf-8")
    assert "/workspace" not in report_text
    assert "/home/runner" not in report_text


def test_assnat_file_adapter_deterministic(tmp_path: Path) -> None:
    html = tmp_path / "assnat.html"
    html.write_text("<html><body><p>Intervention parlementaire québécoise suffisamment longue pour test déterministe.</p></body></html>", encoding="utf-8")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        """
kind: modern_corpus_acquisition_manifest
schema_version: "1.0"
primary_language: fr-CA
sources:
  - source_id: assnat_test
    name: AssNat Test
    source_type: official_html
    acquisition_status: active
    license_status: noncommercial_only
    commercial_use: permission_required
    allowed_for_training: true
    adapter:
      name: assnat_journal_debats
      seed_urls:
        - file://{path}
""".format(path=html.as_posix()),
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    res = acquire_modern_corpus(manifest, out, report, include_noncommercial=True, max_documents=1)
    assert res["records_written"] >= 1
