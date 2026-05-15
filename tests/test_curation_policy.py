import json
import warnings
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.curation_policy import curate_ingested_corpus, score_record, validate_curation_policy_manifest
from qfr_pipeline.paths import ROOT
from qfr_pipeline.schemas import CorpusSourceManifest, CurationPolicyManifest


def _policy_data():
    return yaml.safe_load((ROOT / "manifests/curation_policy_manifest.template.yaml").read_text(encoding="utf-8"))


def test_valid_curation_policy_validates():
    policy = validate_curation_policy_manifest(ROOT / "manifests/curation_policy_manifest.template.yaml")
    assert policy.primary_language == "fr-CA"


def test_non_fr_ca_policy_fails():
    d = _policy_data()
    d["primary_language"] = "fr-FR"
    with pytest.raises(Exception):
        CurationPolicyManifest.model_validate(d)


def test_threshold_ordering_failure_fails():
    d = _policy_data()
    d["scoring"]["thresholds"]["accept_min_score"] = 0.1
    d["scoring"]["thresholds"]["review_min_score"] = 0.8
    with pytest.raises(Exception):
        CurationPolicyManifest.model_validate(d)


def test_dialect_neutralization_rule_fails():
    d = _policy_data()
    d["scoring"]["review_rules"][0]["description"] = "Neutralize toward fr_FR standard parisian"
    with pytest.raises(Exception):
        CurationPolicyManifest.model_validate(d)


def test_score_record_flows_and_duplicate_detection():
    policy = validate_curation_policy_manifest(ROOT / "manifests/curation_policy_manifest.template.yaml")
    seen: set[str] = set()
    a1 = score_record({"text": "Je t'écris un courriel cette fin de semaine."}, policy, seen)
    assert a1.label == "accepted"
    a2 = score_record({"text": "EMAIL WEEK-END VOITURE"}, policy, seen)
    assert a2.label in {"review_required", "quarantine", "rejected"}
    a3 = score_record({"text": "Je t'écris un courriel cette fin de semaine."}, policy, seen)
    assert any("duplicate_text" in reason for reason in a3.reasons)


def test_curate_ingested_corpus_outputs(tmp_path: Path):
    input_jsonl = tmp_path / "harvest.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"text": "Je vais au dépanneur en fin de semaine."}, ensure_ascii=False),
                json.dumps({"text": "EMAIL ET WEEK-END"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = curate_ingested_corpus(input_jsonl, ROOT / "manifests/curation_policy_manifest.template.yaml", tmp_path / "out")
    assert report.records_total == report.accepted + report.review_required + report.quarantine + report.rejected
    for rel in report.outputs.values():
        assert "/workspace" not in rel and "/home/runner" not in rel


def test_cli_validate_and_curate_pass(tmp_path: Path):
    input_jsonl = tmp_path / "harvest.jsonl"
    input_jsonl.write_text(json.dumps({"text": "Chu icitte pantoute."}, ensure_ascii=False) + "\n", encoding="utf-8")
    runner = CliRunner()
    r1 = runner.invoke(app, ["validate-curation-policy", "--policy", "manifests/curation_policy_manifest.template.yaml"])
    out_dir = tmp_path / "curation"
    r2 = runner.invoke(app, ["curate-corpus", "--input", str(input_jsonl), "--policy", "manifests/curation_policy_manifest.template.yaml", "--out-dir", str(out_dir)])
    assert r1.exit_code == 0 and r2.exit_code == 0


def test_corpus_source_register_alias_and_no_shadow_warning():
    d = yaml.safe_load((ROOT / "manifests/corpus_source_manifest.template.yaml").read_text(encoding="utf-8"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest = CorpusSourceManifest.model_validate(d)
    assert manifest.sources[0].language_register in {"formal", "informal", "mixed", "unknown"}
    assert not any("shadows an attribute" in str(w.message) for w in caught)
