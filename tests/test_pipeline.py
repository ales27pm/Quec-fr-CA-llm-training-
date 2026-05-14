import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.contamination import detect_contamination
from qfr_pipeline.minimal_pairs import generate_lp9_minimal_pairs
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.release_report import evaluate_release
from qfr_pipeline.validation import (
    validate_dataset_manifest,
    validate_evaluation_manifest,
    validate_lp_rule_manifest,
    validate_release_gates,
)


def test_release_gate_schema_validation():
    gates = validate_release_gates(RELEASE_GATES_PATH)
    assert gates.asr.wer_max == pytest.approx(0.0665)


def test_eval_manifest_sync_with_release_gates():
    m = validate_evaluation_manifest(ROOT / "eval/evaluation_manifest.template.yaml", RELEASE_GATES_PATH)
    assert m.kind == "evaluation_manifest"


def test_dataset_manifest_rejects_fr_fr_neutralization(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    doc = (ROOT / "manifests/dataset_manifest.template.yaml").read_text(encoding="utf-8").replace("fr-CA", "fr-FR")
    p.write_text(doc, encoding="utf-8")
    with pytest.raises(Exception):
        validate_dataset_manifest(p)


def test_lp_rule_manifest_validates_current_lp9_template():
    m = validate_lp_rule_manifest(ROOT / "rules/lp_rule_manifest.template.yaml")
    assert m.lp_id == 9


def test_contamination_exact_match():
    matches = detect_contamination([{"id": "t1", "text": "Courriel officiel"}], [{"id": "h1", "text": "courriel officiel"}], 0.92)
    assert matches and matches[0].match_type == "exact"


def test_contamination_fuzzy_match():
    matches = detect_contamination([{"id": "t1", "text": "Courriel officiel"}], [{"id": "h1", "text": "Courriel officiel!"}], 0.9)
    assert matches


def test_contamination_clean_case():
    matches = detect_contamination([{"id": "t1", "text": "Bonjour"}], [{"id": "h1", "text": "Au revoir"}], 0.95)
    assert not matches


def test_minimal_pair_generation_from_lp9_rule():
    out = generate_lp9_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml")
    assert out and out[0]["expected"] == "good"


def test_release_report_pass_case(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"asr_wer":0.05,"overall_lp_accuracy":0.9,"lp9_lexical_semantics":0.81,"lp20_orphaned_preposition":0.72,"lp7_standard_negation_post_alignment_drop_ratio":0.02}), encoding="utf-8")
    report = evaluate_release(p, RELEASE_GATES_PATH)
    assert report.passed


def test_release_report_fail_case(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"asr_wer":0.2}), encoding="utf-8")
    report = evaluate_release(p, RELEASE_GATES_PATH)
    assert not report.passed


def test_cli_smoke(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
