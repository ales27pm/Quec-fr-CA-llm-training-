import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.legacy_diagnostics import (
    load_legacy_semantic_csv,
    run_legacy_semantic_diagnostics,
)
from qfr_pipeline.paths import ROOT


def _assert_maintained_shape(payload: dict) -> None:
    assert set(payload.keys()) == {"ok", "phenomena", "global_summary", "issues"}


def test_load_legacy_semantic_csv_reads_fixture():
    rows = load_legacy_semantic_csv(
        ROOT / "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv"
    )
    assert len(rows) == 4
    assert set(rows[0]) == {
        "phenomenon",
        "is_correct",
        "embedding_ref",
        "embedding_pred",
        "error_label",
    }


def test_legacy_alias_mapping_to_maintained_shape(tmp_path: Path):
    out_json = tmp_path / "legacy_diag.json"
    payload = run_legacy_semantic_diagnostics(
        ROOT / "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv",
        out_json,
    )
    _assert_maintained_shape(payload)
    assert out_json.exists()
    assert set(payload["phenomena"].keys()) == {
        "LP9:lexical_semantics",
        "LP20:orphaned_preposition",
    }
    assert payload["phenomena"]["LP9:lexical_semantics"]["lp_id"] == 9
    assert payload["phenomena"]["LP20:orphaned_preposition"]["lp_id"] == 20


def test_legacy_aliases_all_supported_variants(tmp_path: Path):
    lp9_aliases = ["LP9", "lp9", "LP9:lexical_semantics", "lexical_semantics"]
    lp20_aliases = ["LP20", "lp20", "LP20:orphaned_preposition", "orphaned_preposition"]
    for lp9_alias in lp9_aliases:
        for lp20_alias in lp20_aliases:
            csv_text = "\n".join(
                [
                    "phenomenon,is_correct,embedding_ref,embedding_pred,error_label",
                    f"{lp9_alias},1,1 0 0,1 0 0,",
                    f"{lp20_alias},1,1 0 0,1 0 0,",
                ]
            )
            in_csv = tmp_path / f"{lp9_alias}_{lp20_alias}.csv"
            out_json = tmp_path / f"{lp9_alias}_{lp20_alias}.json"
            in_csv.write_text(csv_text + "\n", encoding="utf-8")
            payload = run_legacy_semantic_diagnostics(in_csv, out_json)
            assert payload["ok"]
            assert set(payload["phenomena"]) == {
                "LP9:lexical_semantics",
                "LP20:orphaned_preposition",
            }


def test_invalid_legacy_phenomenon_becomes_malformed_issue(tmp_path: Path):
    in_csv = tmp_path / "bad_phenomenon.csv"
    out_json = tmp_path / "bad_phenomenon.json"
    in_csv.write_text(
        "\n".join(
            [
                "phenomenon,is_correct,embedding_ref,embedding_pred,error_label",
                "unknown_thing,1,1 0 0,1 0 0,",
                "LP20,1,1 0 0,1 0 0,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload = run_legacy_semantic_diagnostics(
        in_csv,
        out_json,
        allow_missing_phenomena=True,
    )
    assert not payload["ok"]
    assert payload["global_summary"]["malformed_rows"] >= 1
    assert any(issue["code"] == "malformed_row" for issue in payload["issues"])


def test_legacy_non_blocking_fixture_ok_true(tmp_path: Path):
    payload = run_legacy_semantic_diagnostics(
        ROOT / "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv",
        tmp_path / "non_blocking.json",
    )
    assert payload["ok"]


def test_legacy_blocking_fixture_ok_false(tmp_path: Path):
    payload = run_legacy_semantic_diagnostics(
        ROOT / "fixtures/diagnostics/lp9_lp20_legacy_semantic_blocking.csv",
        tmp_path / "blocking.json",
    )
    assert not payload["ok"]
    assert payload["global_summary"]["blocking_error_count"] > 0


def test_pipeline_ops_diagnose_semantic_legacy_wrapper_succeeds(tmp_path: Path):
    out_json = tmp_path / "tool.json"
    cmd = [
        sys.executable,
        "tools/pipeline_ops.py",
        "diagnose-semantic",
        "--in-csv",
        "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv",
        "--out-json",
        str(out_json),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    _assert_maintained_shape(payload)
    assert payload["ok"]


def test_pipeline_ops_diagnose_semantic_blocking_fixture_signals_failure(tmp_path: Path):
    out_json = tmp_path / "tool_blocking.json"
    cmd = [
        sys.executable,
        "tools/pipeline_ops.py",
        "diagnose-semantic",
        "--in-csv",
        "fixtures/diagnostics/lp9_lp20_legacy_semantic_blocking.csv",
        "--out-json",
        str(out_json),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    assert result.returncode != 0 or payload.get("ok") is False


def test_qfr_diagnose_legacy_csv_matches_pipeline_ops_output(tmp_path: Path):
    tool_out = tmp_path / "tool.json"
    qfr_out = tmp_path / "qfr.json"
    cmd = [
        sys.executable,
        "tools/pipeline_ops.py",
        "diagnose-semantic",
        "--in-csv",
        "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv",
        "--out-json",
        str(tool_out),
    ]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    cli_result = CliRunner().invoke(
        app,
        [
            "diagnose-legacy-csv",
            "--in-csv",
            "fixtures/diagnostics/lp9_lp20_legacy_semantic.csv",
            "--out-json",
            str(qfr_out),
        ],
    )
    assert cli_result.exit_code == 0
    assert json.loads(qfr_out.read_text(encoding="utf-8")) == json.loads(
        tool_out.read_text(encoding="utf-8")
    )


def test_pipeline_ops_no_standalone_semantic_diagnostics_engine():
    text = (ROOT / "tools/pipeline_ops.py").read_text(encoding="utf-8")
    assert "def lp_semantic_diagnostics" not in text
    assert "_cosine_similarity" not in text
