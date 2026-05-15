import json
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.curated_split import split_curated_corpus, validate_split_policy_manifest

runner = CliRunner()


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


def test_validate_split_policy_manifest_validates():
    m = validate_split_policy_manifest(Path("manifests/split_policy_manifest.template.yaml"))
    assert m.primary_language == "fr-CA"


def test_split_curated_only_accepted(tmp_path: Path):
    inp = tmp_path / "accepted.jsonl"
    _write_jsonl(inp, [{"text": "a", "curation_label": "accepted", "policy_id": "p", "curation_score": 0.9, "curation_reasons": ["ok"]} for _ in range(10)])
    report = split_curated_corpus(inp, Path("manifests/split_policy_manifest.template.yaml"), tmp_path / "out")
    assert report.ok
    assert report.records_total == 10


def test_split_curated_rejects_forbidden_label(tmp_path: Path):
    inp = tmp_path / "accepted.jsonl"
    _write_jsonl(inp, [{"curation_label": "rejected", "policy_id": "p", "curation_score": 0.1, "curation_reasons": ["x"]}])
    report = split_curated_corpus(inp, Path("manifests/split_policy_manifest.template.yaml"), tmp_path / "out")
    assert not report.ok


def test_cli_split_commands():
    r1 = runner.invoke(app, ["validate-split-policy", "--policy", "manifests/split_policy_manifest.template.yaml"])
    assert r1.exit_code == 0


def test_ci_workflow_contains_split_steps_and_valid_yaml():
    ci_path = Path(".github/workflows/ci.yml")
    content = ci_path.read_text(encoding="utf-8")
    assert "- run: qfr validate-split-policy --policy manifests/split_policy_manifest.template.yaml" in content
    assert "- run: qfr split-curated-corpus --input reports/corpus_curation/accepted.jsonl --policy manifests/split_policy_manifest.template.yaml --out-dir reports/curated_splits" in content
    assert re.search(r"^qfr validate-split-policy", content, flags=re.MULTILINE) is None
    assert re.search(r"^qfr split-curated-corpus", content, flags=re.MULTILINE) is None
    parsed = yaml.safe_load(content)
    assert isinstance(parsed, dict)
