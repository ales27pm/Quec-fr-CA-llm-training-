from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import yaml

from qfr_pipeline.lp9_eval import (
    generate_lp9_eval_prompts,
    load_eval_prompts,
    score_response,
    write_baseline_expected_report,
)
from qfr_pipeline.lp9_micro_pack import generate_lp9_micro_pack
from qfr_pipeline.paths import ROOT

EVAL_SCRIPT_PATH = ROOT / "scripts" / "evaluate_lp9_adapter.py"
spec = importlib.util.spec_from_file_location("qfr_lp9_eval_script", EVAL_SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load script module from {EVAL_SCRIPT_PATH}")
lp9_eval_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp9_eval_script)


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_manifest_with_output_dir(tmp_root: Path, output_dir_rel: str) -> Path:
    manifest = yaml.safe_load(
        (ROOT / "manifests/lp9_lexical_preference_pack.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest["output_dir"] = output_dir_rel
    manifest_path = tmp_root / "lp9_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest_path


def test_eval_prompts_include_required_and_forbidden_terms() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        manifest_path = _write_manifest_with_output_dir(
            tmp_root,
            _repo_rel(tmp_root / "lp9_pack"),
        )
        out = tmp_root / "eval_prompts.jsonl"
        summary = generate_lp9_eval_prompts(manifest_path, out)

        assert summary["ok"]
        rows = load_eval_prompts(out)
        assert rows
        for row in rows:
            assert row["expected_terms"]
            assert row["forbidden_terms"]
            scoring = row["scoring"]
            assert scoring["required_any"]
            assert scoring["forbidden_any"]
            assert isinstance(scoring["exact_preferred_bonus"], str)


def test_score_response_counts_preferred_and_forbidden_terms() -> None:
    scoring = {
        "required_any": ["courriel"],
        "forbidden_any": ["email", "e-mail"],
        "exact_preferred_bonus": "courriel",
    }

    preferred = score_response("Nous envoyons un courriel officiel.", scoring)
    forbidden = score_response("Nous envoyons un email officiel.", scoring)
    both = score_response("Un courriel remplace email dans ce texte.", scoring)

    assert preferred["score"] == 1
    assert preferred["preferred_hit"] is True
    assert preferred["forbidden_hit"] is False

    assert forbidden["score"] == -1
    assert forbidden["preferred_hit"] is False
    assert forbidden["forbidden_hit"] is True

    assert both["score"] == 0
    assert both["preferred_hit"] is True
    assert both["forbidden_hit"] is True


def test_fake_greedy_eval_report_scoring_and_outputs() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        manifest_path = _write_manifest_with_output_dir(
            tmp_root,
            _repo_rel(tmp_root / "lp9_pack"),
        )

        # Build both artifacts to mimic the intended local flow.
        generate_lp9_micro_pack(manifest_path)
        prompts_path = tmp_root / "eval_prompts.jsonl"
        generate_lp9_eval_prompts(manifest_path, prompts_path)

        report_out = tmp_root / "base_vs_adapter_report.json"

        def fake_base(prompt_row: dict, _prompt: str) -> str:
            forbidden = prompt_row["forbidden_terms"][0]
            return f"Je conserve le terme {forbidden}."

        def fake_adapter(prompt_row: dict, _prompt: str) -> str:
            preferred = prompt_row["expected_terms"][0]
            return f"Je privilégie le terme {preferred}."

        report = lp9_eval_script.run_evaluation(
            prompts_path=prompts_path,
            report_out=report_out,
            generate_base=fake_base,
            generate_adapter=fake_adapter,
        )

        generations_path = Path(report["generations_jsonl"])
        assert report["total_prompts"] > 0
        assert report["base_score"] < 0
        assert report["adapter_score"] > 0
        assert report["delta"] > 0
        assert generations_path.exists()
        assert report_out.exists()


def test_lp9_eval_baseline_report_and_path_leakage() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        manifest_path = _write_manifest_with_output_dir(
            tmp_root,
            _repo_rel(tmp_root / "lp9_pack"),
        )
        prompts_path = tmp_root / "eval_prompts.jsonl"
        generate_lp9_eval_prompts(manifest_path, prompts_path)

        baseline_path = tmp_root / "baseline_expected_report.json"
        baseline = write_baseline_expected_report(baseline_path, prompts_path)
        assert baseline["expected_total_score"] < 0

        for path in [prompts_path, baseline_path]:
            text = path.read_text(encoding="utf-8")
            assert "/workspace" not in text
            assert "/home/runner" not in text

        rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows
        assert all(row["task_type"] in {"rewrite", "direct_preference", "choose_best_term", "correction", "open_generation"} for row in rows)
