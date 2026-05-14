import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs
from qfr_pipeline.minimal_pairs import generate_minimal_pairs
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.validation import validate_lp_context_manifest, validate_release_gates


def test_release_gate_schema_validation():
    assert validate_release_gates(RELEASE_GATES_PATH).asr.wer_max == pytest.approx(0.0665)


def test_lp9_context_manifest_validates():
    m = validate_lp_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    assert m.lp_id == 9


def test_lp9_generator_produces_grammar_valid_records():
    recs, report = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    assert report.ok
    assert recs


def test_generator_rejects_le_fin_de_semaine():
    quality = validate_minimal_pairs([{"id": "x", "lp_id": 9, "good": "Le fin de semaine commence bientôt.", "bad": "Le week-end commence bientôt.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}])
    assert any(i.code == "forbidden_ngram" for i in quality.issues)


def test_generator_rejects_good_side_week_end():
    quality = validate_minimal_pairs([{"id": "x", "lp_id": 9, "good": "Le week-end commence bientôt.", "bad": "La fin de semaine commence bientôt.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}])
    assert any(i.code == "neutralization_good" for i in quality.issues)


def test_duplicate_pair_detection():
    base = {"lp_id": 9, "good": "La fin de semaine commence bientôt.", "bad": "Le week-end commence bientôt.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}
    q = validate_minimal_pairs([{"id": "a", **base}, {"id": "b", **base}])
    assert any(i.code == "duplicate_pair" for i in q.issues)


def test_validate_minimal_pairs_cli_pass(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    report = tmp_path / "report.json"
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    out.write_text("\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["validate-minimal-pairs", "--input", str(out), "--context", "rules/lp9_lexical_semantics.contexts.yaml", "--report", str(report)])
    assert r.exit_code == 0


def test_validate_minimal_pairs_cli_fail(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    report = tmp_path / "report.json"
    bad.write_text(json.dumps({"id": "x", "lp_id": 9, "good": "Le week-end.", "bad": "Le week-end.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}) + "\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["validate-minimal-pairs", "--input", str(bad), "--context", "rules/lp9_lexical_semantics.contexts.yaml", "--report", str(report)])
    assert r.exit_code != 0


def test_generate_minimal_pairs_writes_jsonl_and_report(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    report = tmp_path / "quality.json"
    r = CliRunner().invoke(app, ["generate-minimal-pairs", "--rule", "rules/lp_rule_manifest.template.yaml", "--context", "rules/lp9_lexical_semantics.contexts.yaml", "--out", str(out), "--report", str(report)])
    assert r.exit_code == 0 and out.exists() and report.exists()


def test_cli_agents_refresh_no_subprocess():
    assert "subprocess" not in (ROOT / "src/qfr_pipeline/cli.py").read_text(encoding="utf-8")
