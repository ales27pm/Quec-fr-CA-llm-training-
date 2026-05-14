import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs, validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, load_context_manifest
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.validation import validate_lp_context_manifest, validate_release_gates


def test_release_gate_schema_validation():
    assert validate_release_gates(RELEASE_GATES_PATH).asr.wer_max == pytest.approx(0.0665)


def test_lp9_context_manifest_validates():
    m = validate_lp_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    assert m.lp_id == 9


def test_lp9_generated_records_pass_context_bound_validation():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([r.__dict__ for r in recs], manifest, str(ROOT / "rules/lp9_lexical_semantics.contexts.yaml"))
    assert report.ok


def test_context_binding_forged_contrast_fails():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["contrast_id"] = "forged"
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([bad], manifest)
    assert any(i.code == "context_unknown_contrast" for i in report.issues)


def test_context_binding_wrong_pattern_fails():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["metadata"] = dict(bad["metadata"])
    bad["metadata"]["positive_pattern"] = "PATTERN_FORGÉ"
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([bad], manifest)
    assert any(i.code == "context_positive_pattern_mismatch" for i in report.issues)


def test_context_binding_unauthorized_good_fails():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["good"] = bad["good"] + " EXTRA"
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([bad], manifest)
    assert any(i.code == "context_unauthorized_pair" for i in report.issues)


def test_context_binding_wrong_stable_id_fails():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["id"] = "lp9-deadbeef00"
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([bad], manifest)
    assert any(i.code == "context_stable_id_mismatch" for i in report.issues)


def test_context_binding_register_term_type_source_failures():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["register"] = "informal"
    bad["term_type"] = "idiom"
    bad["source_context"] = "wrong.yaml"
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([bad], manifest, source_context="rules/lp9_lexical_semantics.contexts.yaml")
    codes = {i.code for i in report.issues}
    assert "context_register_mismatch" in codes and "context_term_type_mismatch" in codes and "context_source_path_mismatch" in codes


def test_generator_rejects_le_fin_de_semaine():
    quality = validate_minimal_pairs([{"id": "x", "lp_id": 9, "good": "Le fin de semaine commence bientôt.", "bad": "Le week-end commence bientôt.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}])
    assert any(i.code == "forbidden_ngram" for i in quality.issues)


def test_generator_rejects_good_side_week_end():
    quality = validate_minimal_pairs([{"id": "x", "lp_id": 9, "good": "Le week-end commence bientôt.", "bad": "La fin de semaine commence bientôt.", "metadata": {"positive_pattern": "fin de semaine", "negative_pattern": "week-end"}}])
    assert any(i.code == "neutralization_good" for i in quality.issues)


def test_validate_minimal_pairs_cli_fail_on_forged_record(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    report = tmp_path / "report.json"
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    forged = recs[0].__dict__.copy()
    forged["contrast_id"] = "forged"
    out.write_text(json.dumps(forged, ensure_ascii=False) + "\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["validate-minimal-pairs", "--input", str(out), "--context", "rules/lp9_lexical_semantics.contexts.yaml", "--report", str(report)])
    assert r.exit_code != 0


def test_generate_minimal_pairs_writes_jsonl_and_report(tmp_path: Path):
    out = tmp_path / "pairs.jsonl"
    report = tmp_path / "quality.json"
    r = CliRunner().invoke(app, ["generate-minimal-pairs", "--rule", "rules/lp_rule_manifest.template.yaml", "--context", "rules/lp9_lexical_semantics.contexts.yaml", "--out", str(out), "--report", str(report)])
    assert r.exit_code == 0 and out.exists() and report.exists()


def test_cli_agents_refresh_no_subprocess():
    assert "subprocess" not in (ROOT / "src/qfr_pipeline/cli.py").read_text(encoding="utf-8")
