import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.corpus_sources import ingest_corpus_sources, validate_corpus_source_manifest, write_ingestion_report
from qfr_pipeline.paths import ROOT
from qfr_pipeline.schemas import CorpusSourceManifest


def _manifest_data():
    return yaml.safe_load((ROOT / "manifests/corpus_source_manifest.template.yaml").read_text(encoding="utf-8"))


def test_valid_corpus_manifest_validates():
    m = validate_corpus_source_manifest(ROOT / "manifests/corpus_source_manifest.template.yaml")
    assert m.primary_language == "fr-CA"


def test_duplicate_source_id_fails():
    d = _manifest_data()
    d["sources"][1]["source_id"] = d["sources"][0]["source_id"]
    with pytest.raises(Exception):
        CorpusSourceManifest.model_validate(d)


def test_non_fr_ca_fails():
    d = _manifest_data()
    d["primary_language"] = "fr-FR"
    with pytest.raises(Exception):
        CorpusSourceManifest.model_validate(d)


def test_holdout_training_fails():
    d = _manifest_data()
    d["sources"][0]["contains_holdout_material"] = True
    with pytest.raises(Exception):
        CorpusSourceManifest.model_validate(d)


def test_quarantine_training_fails():
    d = _manifest_data()
    d["sources"][0]["quality_tier"] = "quarantine"
    with pytest.raises(Exception):
        CorpusSourceManifest.model_validate(d)


def test_personal_data_requires_review():
    d = _manifest_data()
    d["sources"][1]["requires_review"] = False
    with pytest.raises(Exception):
        CorpusSourceManifest.model_validate(d)


def test_ingestion_skip_rules_and_deterministic_output(tmp_path: Path):
    out = tmp_path / "harvest.jsonl"
    report = ingest_corpus_sources(ROOT / "manifests/corpus_source_manifest.template.yaml", out)
    assert report.ok and report.sources_ingested == 1
    reasons = {x["reason"] for x in report.skipped_sources}
    assert "future_remote" in reasons and "requires_review" in reasons
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines and all("/workspace" not in line_text and "/home/runner" not in line_text for line_text in lines)


def test_review_required_can_be_included(tmp_path: Path):
    out = tmp_path / "harvest.jsonl"
    report = ingest_corpus_sources(ROOT / "manifests/corpus_source_manifest.template.yaml", out, include_review_required=True)
    assert report.sources_ingested == 2


def test_cli_validate_and_ingest_pass(tmp_path: Path):
    rc = CliRunner()
    report = tmp_path / "report.json"
    out = tmp_path / "harvest.jsonl"
    r1 = rc.invoke(app, ["validate-corpus-sources", "--manifest", "manifests/corpus_source_manifest.template.yaml"])
    r2 = rc.invoke(app, ["ingest-corpus-sources", "--manifest", "manifests/corpus_source_manifest.template.yaml", "--out", str(out), "--report", str(report), "--min-chars", "20"])
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert r1.exit_code == 0 and r2.exit_code == 0 and payload["ok"]
    assert "/workspace" not in json.dumps(payload) and "/home/runner" not in json.dumps(payload)


def test_malformed_local_path_fails_validation(tmp_path: Path):
    d = _manifest_data()
    d["sources"][0]["path"] = "/tmp/bad.txt"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception):
        validate_corpus_source_manifest(p)


def test_write_report_has_no_absolute_paths(tmp_path: Path):
    out = tmp_path / "harvest.jsonl"
    rep = tmp_path / "report.json"
    report = ingest_corpus_sources(ROOT / "manifests/corpus_source_manifest.template.yaml", out)
    write_ingestion_report(report, rep)
    txt = rep.read_text(encoding="utf-8")
    assert "/workspace" not in txt and "/home/runner" not in txt
