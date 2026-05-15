import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.data_pipeline import curate, edit_normative, harvest, split_train_dev_test, write_training_recipe
from qfr_pipeline.lp7_monitor import monitor_lp7
from qfr_pipeline.paths import ROOT


def test_harvest_filters_and_dedupes(tmp_path: Path):
    out = tmp_path / "h.jsonl"
    kept = harvest([ROOT / "fixtures/data_pipeline/raw_sample_1.txt", ROOT / "fixtures/data_pipeline/raw_sample_2.txt"], out, 20)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert kept == len(lines) == 5


def test_harvest_respects_batch_reset(tmp_path: Path):
    out = tmp_path / "h.jsonl"
    kept = harvest([ROOT / "fixtures/data_pipeline/raw_sample_1.txt", ROOT / "fixtures/data_pipeline/raw_sample_2.txt"], out, 20, dedupe_batch_size=1)
    assert kept > 5


def test_curate_and_edit_and_split_and_recipe(tmp_path: Path):
    h, c, e = tmp_path / "h.jsonl", tmp_path / "c.jsonl", tmp_path / "e.jsonl"
    harvest([ROOT / "fixtures/data_pipeline/raw_sample_1.txt", ROOT / "fixtures/data_pipeline/raw_sample_2.txt"], h, 20)
    kept = curate(h, c, 0.0)
    assert kept >= 3
    changed = edit_normative(c, e)
    txt = e.read_text(encoding="utf-8")
    assert changed >= 1 and "courriel" in txt and "fin de semaine" in txt and "char" in txt
    counts = split_train_dev_test(e, tmp_path / "splits", 42)
    assert counts == split_train_dev_test(e, tmp_path / "splits2", 42)
    write_training_recipe(tmp_path / "splits", tmp_path / "recipe.yaml")
    assert "lp7_post_alignment_max_drop_ratio" in (tmp_path / "recipe.yaml").read_text(encoding="utf-8")


def test_write_training_recipe_rejects_non_mapping_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "dev", "test"):
        (splits / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
    bad = tmp_path / "bad_gates.yaml"
    bad.write_text("alignment: []\n", encoding="utf-8")
    import qfr_pipeline.data_pipeline as dp

    monkeypatch.setattr(dp, "RELEASE_GATES_PATH", bad)
    with pytest.raises(SystemExit, match="alignment"):
        write_training_recipe(splits, tmp_path / "recipe.yaml")


def test_monitor_lp7():
    ok = monitor_lp7(0.95, 0.94, ROOT / "project/release_gates.yaml")
    assert ok["rollback_required"] is False
    bad = monitor_lp7(0.95, 0.70, ROOT / "project/release_gates.yaml")
    assert bad["rollback_required"] is True
    with pytest.raises(ValueError):
        monitor_lp7(0.0, 0.5, ROOT / "project/release_gates.yaml")


def test_legacy_wrapper_and_qfr_cli(tmp_path: Path):
    harvest_out = tmp_path / "h1.jsonl"
    subprocess.run([sys.executable, str(ROOT / "tools/pipeline_ops.py"), "harvest", "--inputs", str(ROOT / "fixtures/data_pipeline/raw_sample_1.txt"), str(ROOT / "fixtures/data_pipeline/raw_sample_2.txt"), "--out", str(harvest_out), "--min-chars", "20"], check=True)
    pkg_out = tmp_path / "h2.jsonl"
    harvest([ROOT / "fixtures/data_pipeline/raw_sample_1.txt", ROOT / "fixtures/data_pipeline/raw_sample_2.txt"], pkg_out, 20)
    assert harvest_out.read_text(encoding="utf-8") == pkg_out.read_text(encoding="utf-8")

    rc = CliRunner()
    out = tmp_path / "lp7.json"
    r = rc.invoke(app, ["monitor-lp7", "--pre", "0.95", "--post", "0.94", "--release-gates", str(ROOT / "project/release_gates.yaml"), "--out", str(out)])
    assert r.exit_code == 0 and out.exists()


def test_harvest_cli_accepts_both_input_forms(tmp_path: Path):
    rc = CliRunner()
    out1 = tmp_path / "h1.jsonl"
    out2 = tmp_path / "h2.jsonl"
    a = str(ROOT / "fixtures/data_pipeline/raw_sample_1.txt")
    b = str(ROOT / "fixtures/data_pipeline/raw_sample_2.txt")
    r1 = rc.invoke(app, ["harvest", "--inputs", a, b, "--out", str(out1), "--min-chars", "20"])
    r2 = rc.invoke(app, ["harvest", "--inputs", a, "--inputs", b, "--out", str(out2), "--min-chars", "20"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_pipeline_ops_is_wrapper_only():
    text = (ROOT / "tools/pipeline_ops.py").read_text(encoding="utf-8")
    assert "def _norm_text" not in text and "FR_CA_MARKERS" not in text
