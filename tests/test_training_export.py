from pathlib import Path

from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.training_export import export_training_dataset, validate_training_export_manifest
from qfr_pipeline.release_candidate import run_release_candidate

runner = CliRunner()


def test_valid_training_manifest_validates():
    m = validate_training_export_manifest(Path("manifests/training_export_manifest.template.yaml"))
    assert m.primary_language == "fr-CA"


def test_export_training_dataset_and_hash_determinism(tmp_path: Path):
    r1 = export_training_dataset(Path("manifests/training_export_manifest.template.yaml"), tmp_path / "out1")
    r2 = export_training_dataset(Path("manifests/training_export_manifest.template.yaml"), tmp_path / "out2")
    assert r1.ok and r2.ok
    assert r1.hashes == r2.hashes


def test_cli_training_export_commands_pass(tmp_path: Path):
    r1 = runner.invoke(app, ["validate-training-export", "--manifest", "manifests/training_export_manifest.template.yaml"])
    r2 = runner.invoke(app, ["export-training-dataset", "--manifest", "manifests/training_export_manifest.template.yaml", "--out-dir", str(tmp_path / "out")])
    assert r1.exit_code == 0 and r2.exit_code == 0


def test_release_candidate_contains_training_export(tmp_path: Path):
    report = run_release_candidate(metrics=Path("fixtures/valid_metrics.json"), diagnostics_input=Path("fixtures/diagnostics/lp9_lp20_eval_sample.jsonl"), out_json=tmp_path / "rc.json", out_md=tmp_path / "rc.md")
    assert any(s.name == "training_export_generation" for s in report.stages)
    assert report.training_export_summary is not None


def test_bootstrap_script_exists_no_wrapper():
    content = Path("scripts/bootstrap_dev_env.sh").read_text(encoding="utf-8")
    assert "pip install -e" in content
    assert "./qfr" not in content
