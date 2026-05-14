import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs, validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import expand_contrast_templates, generate_minimal_pairs, load_context_manifest
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.paths import repo_relative_path
from qfr_pipeline.schemas import LPContextContrast
from qfr_pipeline.diagnostics import load_eval_rows, load_taxonomies, run_diagnostics
from qfr_pipeline.validation import validate_error_taxonomy_manifest, validate_lp_context_manifest, validate_release_gates


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


def test_lp9_generated_records_count_matches_committed_artifact():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    committed = (ROOT / "data/generated/minimal_pairs.lp9.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(recs) == len(committed)


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


def test_expand_contrast_templates_rejects_mismatched_lengths():
    contrast = LPContextContrast.model_construct(
        contrast_id="x",
        positive_pattern="fin de semaine",
        negative_pattern="week-end",
        register="formal",
        term_type="noun_phrase",
        allowed_contexts=["a"],
        blocked_contexts=["b"],
        good_templates=["{positive_pattern} A", "{positive_pattern} B"],
        bad_templates=["{negative_pattern} A"],
        notes="n",
        source_authority="s",
    )
    with pytest.raises(ValueError, match="Mismatched template counts"):
        expand_contrast_templates(contrast)


def test_context_binding_source_context_canonical_path_match():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    manifest = load_context_manifest(ROOT / "rules/lp9_lexical_semantics.contexts.yaml")
    report = validate_minimal_pairs_against_context([recs[0].__dict__], manifest, source_context=str((ROOT / "rules/lp9_lexical_semantics.contexts.yaml").resolve()))
    assert not any(i.code == "context_source_path_mismatch" for i in report.issues)


def test_lp20_context_manifest_validates():
    m = validate_lp_context_manifest(ROOT / "rules/lp20_orphaned_preposition.contexts.yaml")
    assert m.lp_id == 20 and len(m.contrasts) >= 4
    assert any(c.phenomenon_tags for c in m.contrasts)
    assert any(c.required_good_substrings or c.required_bad_substrings for c in m.contrasts)
    assert any(c.forbidden_good_substrings or c.forbidden_bad_substrings for c in m.contrasts)
    assert any(c.minimal_contrast_focus for c in m.contrasts)


def test_lp20_context_manifest_invalid_fields_rejected(tmp_path: Path):
    data = yaml.safe_load((ROOT / "rules/lp20_orphaned_preposition.contexts.yaml").read_text(encoding="utf-8"))
    first = data["contrasts"][0]
    first["phenomenon_tags"] = []
    first["required_good_substrings"] = ["   "]
    bad_path = tmp_path / "bad_lp20.contexts.yaml"
    bad_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_lp_context_manifest(bad_path)


def test_lp20_generation_and_validation_pass():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp20_orphaned_preposition.contexts.yaml")
    manifest = load_context_manifest(ROOT / "rules/lp20_orphaned_preposition.contexts.yaml")
    report = validate_minimal_pairs_against_context([r.__dict__ for r in recs], manifest, str(ROOT / "rules/lp20_orphaned_preposition.contexts.yaml"))
    assert len(recs) >= 8 and report.ok


def test_lp20_context_binding_failures():
    recs, _ = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", ROOT / "rules/lp20_orphaned_preposition.contexts.yaml")
    manifest = load_context_manifest(ROOT / "rules/lp20_orphaned_preposition.contexts.yaml")
    bad = recs[0].__dict__.copy()
    bad["good"] = "..."
    bad["bad"] = "..."
    bad["id"] = "lp20-deadbeef00"
    report = validate_minimal_pairs_against_context([bad], manifest)
    codes = {i.code for i in report.issues}
    assert "context_unauthorized_pair" in codes and "context_stable_id_mismatch" in codes and "lp20_too_short" in codes


def test_lp20_punctuation_only_and_duplicate_pair_fail():
    records = [
        {"id": "x1", "lp_id": 20, "good": "De quoi parles-tu?", "bad": "De quoi parles-tu !", "metadata": {"positive_pattern": "de", "negative_pattern": ""}},
        {"id": "x2", "lp_id": 20, "good": "De quoi parles-tu?", "bad": "De quoi parles-tu !", "metadata": {"positive_pattern": "de", "negative_pattern": ""}},
    ]
    quality = validate_minimal_pairs(records)
    codes = {i.code for i in quality.issues}
    assert "punctuation_only_change" in codes and "duplicate_pair" in codes

def test_taxonomy_schema_validation_passes():
    assert validate_error_taxonomy_manifest(ROOT / "eval/lp9_error_taxonomy.yaml").lp_id == 9
    assert validate_error_taxonomy_manifest(ROOT / "eval/lp20_error_taxonomy.yaml").lp_id == 20


def test_taxonomy_duplicate_error_code_fails(tmp_path: Path):
    bad = yaml.safe_load((ROOT / "eval/lp9_error_taxonomy.yaml").read_text())
    bad["taxonomy"].append(dict(bad["taxonomy"][0]))
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump(bad, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        validate_error_taxonomy_manifest(p)


def test_diagnostics_aggregation_and_cosine():
    rows = load_eval_rows(ROOT / "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl")
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    report = run_diagnostics(rows, tax)
    assert report.global_summary["total_records"] == 4
    assert report.phenomena["LP9:lexical_semantics"].binary_accuracy == pytest.approx(0.5)
    assert report.phenomena["LP20:orphaned_preposition"].mean_semantic_similarity is not None


def test_diagnostics_unknown_error_code_blocking():
    rows = [{"id": "x", "lp_id": 9, "phenomenon": "lexical_semantics", "is_correct": 0, "error_code": "nope"}, {"id": "y", "lp_id": 20, "phenomenon": "orphaned_preposition", "is_correct": 1}]
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    report = run_diagnostics(rows, tax)
    assert not report.ok


def test_diagnostics_missing_error_code_blocking():
    rows = [{"id": "x", "lp_id": 9, "phenomenon": "lexical_semantics", "is_correct": 0}, {"id": "y", "lp_id": 20, "phenomenon": "orphaned_preposition", "is_correct": 1}]
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    report = run_diagnostics(rows, tax)
    assert any(i.code == "missing_error_code" for i in report.issues)


def test_cli_validate_taxonomy_and_diagnose_eval(tmp_path: Path):
    outj = tmp_path / "d.json"
    outm = tmp_path / "d.md"
    r1 = CliRunner().invoke(app, ["validate-taxonomy", "--taxonomy", "eval/lp9_error_taxonomy.yaml"])
    r2 = CliRunner().invoke(app, ["diagnose-eval", "--input", "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl", "--taxonomy", "eval/lp9_error_taxonomy.yaml", "--taxonomy", "eval/lp20_error_taxonomy.yaml", "--out-json", str(outj), "--out-md", str(outm)])
    assert r1.exit_code == 0 and r2.exit_code == 0 and outj.exists() and outm.exists()


def test_diagnostics_known_blocking_taxonomy_sets_not_ok():
    rows = load_eval_rows(ROOT / "fixtures/diagnostics/lp9_lp20_eval_blocking_sample.jsonl")
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    report = run_diagnostics(rows, tax)
    assert not report.ok and report.global_summary["blocking_error_count"] > 0


def test_diagnostics_known_non_blocking_taxonomy_can_stay_ok():
    rows = load_eval_rows(ROOT / "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl")
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    report = run_diagnostics(rows, tax)
    assert report.ok


def test_diagnostics_invalid_is_correct_values_rejected():
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    for bad in (2, -1, "yes"):
        rows = [{"id": "x", "lp_id": 9, "phenomenon": "lexical_semantics", "is_correct": bad}, {"id": "y", "lp_id": 20, "phenomenon": "orphaned_preposition", "is_correct": 1}]
        report = run_diagnostics(rows, tax)
        assert report.global_summary["malformed_rows"] >= 1 and not report.ok


def test_diagnostics_zero_one_is_correct_supported():
    tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
    rows = [{"id": "x", "lp_id": 9, "phenomenon": "lexical_semantics", "is_correct": 0, "error_code": "normative_register_mismatch"}, {"id": "y", "lp_id": 20, "phenomenon": "orphaned_preposition", "is_correct": 1}]
    report = run_diagnostics(rows, tax)
    assert report.ok


def test_cli_diagnose_eval_fails_on_known_blocking(tmp_path: Path):
    outj = tmp_path / "d.json"
    outm = tmp_path / "d.md"
    r = CliRunner().invoke(app, ["diagnose-eval", "--input", "fixtures/diagnostics/lp9_lp20_eval_blocking_sample.jsonl", "--taxonomy", "eval/lp9_error_taxonomy.yaml", "--taxonomy", "eval/lp20_error_taxonomy.yaml", "--out-json", str(outj), "--out-md", str(outm)])
    assert r.exit_code != 0


def test_release_report_fails_on_blocking_diagnostics(tmp_path: Path):
    djson = tmp_path / "blocking.json"
    dmd = tmp_path / "blocking.md"
    rc = CliRunner()
    r1 = rc.invoke(app, ["diagnose-eval", "--input", "fixtures/diagnostics/lp9_lp20_eval_blocking_sample.jsonl", "--taxonomy", "eval/lp9_error_taxonomy.yaml", "--taxonomy", "eval/lp20_error_taxonomy.yaml", "--out-json", str(djson), "--out-md", str(dmd)])
    assert r1.exit_code != 0
    r2 = rc.invoke(app, ["release-report", "--metrics", "fixtures/valid_metrics.json", "--diagnostics", str(djson), "--out-json", str(tmp_path / "r.json"), "--out-md", str(tmp_path / "r.md")])
    assert r2.exit_code != 0


def test_ci_lp20_commands_no_bare_extra_args():
    ci_path = ROOT / ".github/workflows/ci.yml"
    if not ci_path.exists():
        return
    ci = ci_path.read_text(encoding="utf-8")
    assert "qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp20_orphaned_preposition.contexts.yaml --out data/generated/minimal_pairs.lp20.jsonl --report reports/minimal_pair_quality.lp20.json reports/diagnostics.lp9_lp20.json" not in ci
    assert "qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp20.jsonl --context rules/lp20_orphaned_preposition.contexts.yaml --report reports/minimal_pair_quality.lp20.json reports/diagnostics.lp9_lp20.json" not in ci


def test_workflow_file_exists():
    assert (ROOT / ".github/workflows/ci.yml").exists()


def test_local_validation_script_exists_and_has_required_commands():
    script = (ROOT / "scripts/run_local_validation.sh")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    for cmd in [
        "set -euo pipefail",
        "ruff check .",
        "python -m pytest",
        "python3 tools/update_agents.py --validate",
        "qfr validate",
        "qfr validate-taxonomy --taxonomy eval/lp9_error_taxonomy.yaml",
        "qfr validate-taxonomy --taxonomy eval/lp20_error_taxonomy.yaml",
        "qfr diagnose-eval --input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl",
        "qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp9_lexical_semantics.contexts.yaml",
        "qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp20_orphaned_preposition.contexts.yaml",
        "qfr release-report --metrics fixtures/valid_metrics.json --diagnostics reports/diagnostics.lp9_lp20.json",
    ]:
        assert cmd in text


def test_t11_t15_remain_fully_implemented_after_write():
    status = json.loads((ROOT / "project/status.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t["status"] for t in status["tasks"]}
    assert by_id.get("T11") == "fully_implemented"
    assert by_id.get("T15") == "fully_implemented"


def test_release_candidate_command_succeeds_with_valid_fixtures(tmp_path: Path):
    rc = CliRunner()
    outj = tmp_path / "rc.json"
    outm = tmp_path / "rc.md"
    r = rc.invoke(app, ["release-candidate", "--metrics", "fixtures/valid_metrics.json", "--diagnostics-input", "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl", "--out-json", str(outj), "--out-md", str(outm)])
    assert r.exit_code == 0
    payload = json.loads(outj.read_text(encoding="utf-8"))
    assert payload["ok"]
    assert "LP9:lexical_semantics" in payload["diagnostics_summary"]["phenomena"]
    assert "LP20:orphaned_preposition" in payload["diagnostics_summary"]["phenomena"]


def test_release_candidate_fails_with_blocking_diagnostics_and_writes_partial(tmp_path: Path):
    rc = CliRunner()
    outj = tmp_path / "rc_fail.json"
    outm = tmp_path / "rc_fail.md"
    r = rc.invoke(app, ["release-candidate", "--metrics", "fixtures/valid_metrics.json", "--diagnostics-input", "fixtures/diagnostics/lp9_lp20_eval_blocking_sample.jsonl", "--out-json", str(outj), "--out-md", str(outm)])
    assert r.exit_code != 0
    payload = json.loads(outj.read_text(encoding="utf-8"))
    assert "diagnostics_generation" in payload["blocking_failures"]
    assert len(payload["stages"]) >= 3


def test_release_candidate_has_minimal_pair_summaries_and_no_subprocess_usage():
    text = (ROOT / "src/qfr_pipeline/release_candidate.py").read_text(encoding="utf-8")
    assert "subprocess" not in text
    assert "lp9_records" in text and "lp20_records" in text


def test_ci_and_local_runner_include_release_candidate_and_stale_guard():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    local = (ROOT / "scripts/run_local_validation.sh").read_text(encoding="utf-8")
    assert "qfr release-candidate --metrics fixtures/valid_metrics.json --diagnostics-input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --out-json reports/release_candidate.json --out-md reports/release_candidate.md" in ci
    assert "qfr release-candidate --metrics fixtures/valid_metrics.json --diagnostics-input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --out-json reports/release_candidate.json --out-md reports/release_candidate.md" in local
    assert "reports/release_candidate.json reports/release_candidate.md" in ci
    assert "reports/release_candidate.json reports/release_candidate.md" in local


def test_repo_relative_path_normalizes_root_and_relative():
    assert repo_relative_path(ROOT / "rules/lp9_lexical_semantics.contexts.yaml") == "rules/lp9_lexical_semantics.contexts.yaml"
    assert repo_relative_path(Path("rules/lp9_lexical_semantics.contexts.yaml")) == "rules/lp9_lexical_semantics.contexts.yaml"


def test_release_candidate_and_generated_artifacts_have_no_absolute_paths():
    rc = CliRunner()
    r = rc.invoke(app, ["release-candidate", "--metrics", "fixtures/valid_metrics.json", "--diagnostics-input", "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl", "--out-json", "reports/release_candidate.json", "--out-md", "reports/release_candidate.md"])
    assert r.exit_code == 0
    repo_root = str(ROOT.resolve())
    for path in [
        ROOT / "reports/release_candidate.json",
        ROOT / "reports/release_candidate.md",
        ROOT / "data/generated/minimal_pairs.lp9.jsonl",
        ROOT / "data/generated/minimal_pairs.lp20.jsonl",
        ROOT / "reports/minimal_pair_quality.lp9.json",
        ROOT / "reports/minimal_pair_quality.lp20.json",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "/workspace" not in text
        assert repo_root not in text
    lp9 = (ROOT / "data/generated/minimal_pairs.lp9.jsonl").read_text(encoding="utf-8").splitlines()
    lp20 = (ROOT / "data/generated/minimal_pairs.lp20.jsonl").read_text(encoding="utf-8").splitlines()
    for row in lp9 + lp20:
        payload = json.loads(row)
        assert payload["source_rule"] == "rules/lp_rule_manifest.template.yaml"
        assert payload["source_context"].startswith("rules/")
    q9 = json.loads((ROOT / "reports/minimal_pair_quality.lp9.json").read_text(encoding="utf-8"))
    q20 = json.loads((ROOT / "reports/minimal_pair_quality.lp20.json").read_text(encoding="utf-8"))
    assert q9["context_manifest"] == "rules/lp9_lexical_semantics.contexts.yaml"
    assert q20["context_manifest"] == "rules/lp20_orphaned_preposition.contexts.yaml"
