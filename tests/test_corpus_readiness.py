import json
from pathlib import Path

from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.corpus_readiness import audit_corpus_readiness


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_empty_corpus_insufficient(tmp_path: Path) -> None:
    inp = tmp_path / "empty.jsonl"
    inp.write_text("", encoding="utf-8")
    out = tmp_path / "report.json"
    report = audit_corpus_readiness(inp, out)
    assert report["readiness_level"] == "insufficient"
    assert "no_records" in report["blocking_reasons"]


def test_smoke_threshold(tmp_path: Path) -> None:
    inp = tmp_path / "smoke.jsonl"
    _write_rows(inp, [{"source_id": "s1", "text": "a" * 2_000_000, "domain": "d1", "register": "r1", "license_status": "open_compatible", "commercial_use": "allowed"}])
    out = tmp_path / "report.json"
    report = audit_corpus_readiness(inp, out)
    assert report["readiness_level"] in {"smoke_test", "pilot_lora_candidate", "production_blocked"}


def test_duplicates_detection(tmp_path: Path) -> None:
    inp = tmp_path / "dup.jsonl"
    row = {"source_id": "s1", "text": "Bonjour   Québec", "domain": "d1", "register": "r1", "license_status": "open_compatible", "commercial_use": "allowed"}
    _write_rows(inp, [row, {**row, "text": "bonjour québec"}])
    out = tmp_path / "report.json"
    report = audit_corpus_readiness(inp, out)
    assert report["duplicates_normalized"] >= 1


def test_cli_fail_below(tmp_path: Path) -> None:
    runner = CliRunner()
    inp = tmp_path / "small.jsonl"
    _write_rows(inp, [{"source_id": "s1", "text": "petit", "domain": "d1", "register": "r1", "license_status": "open_compatible", "commercial_use": "allowed"}])
    out = tmp_path / "out.json"
    res = runner.invoke(app, ["audit-corpus-readiness", "--input", str(inp), "--out", str(out), "--fail-below", "pilot_lora_candidate"])
    assert res.exit_code != 0
