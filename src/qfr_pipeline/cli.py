from __future__ import annotations

import os
from pathlib import Path

import typer
import yaml
from rich import print

from qfr_pipeline.contamination import detect_contamination
from qfr_pipeline.io import load_json, write_json
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, read_jsonl, write_jsonl
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.release_report import evaluate_release
from qfr_pipeline.validation import validate_dataset_manifest, validate_evaluation_manifest, validate_lp_context_manifest, validate_lp_rule_manifest, validate_release_gates, validate_repository

app = typer.Typer()


def _refresh_dynamic_agents() -> None:
    import importlib.util

    update_agents_path = ROOT / "tools" / "update_agents.py"
    spec = importlib.util.spec_from_file_location("qfr_update_agents", update_agents_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load update_agents module from {update_agents_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.environ.get("QFR_NO_AGENTS_BACKUP") is None:
        os.environ["QFR_NO_AGENTS_BACKUP"] = "1"
    module.write_root_agents()


@app.command("validate")
def validate_repo():
    _refresh_dynamic_agents()
    report = validate_repository(ROOT)
    if not report.ok:
        for i in report.issues:
            print(f"[red]{i.path}[/red] {i.message}")
        raise typer.Exit(code=1)
    print("Validation passed")


@app.command("generate-minimal-pairs")
def generate_minimal_pairs_cmd(rule: Path = typer.Option(..., "--rule"), context: Path = typer.Option(..., "--context"), out: Path = typer.Option(..., "--out"), report: Path | None = typer.Option(None, "--report")):
    _refresh_dynamic_agents()
    pairs, gen_report = generate_minimal_pairs(rule, context)
    if report:
        write_json(report, {"ok": gen_report.ok, "records_generated": gen_report.records_generated, "issues": [i.__dict__ for i in gen_report.issues]})
    if not gen_report.ok:
        raise typer.Exit(code=1)
    write_jsonl(pairs, out)
    print(f"Wrote {len(pairs)} records to {out}")


@app.command("validate-minimal-pairs")
def validate_minimal_pairs_cmd(input: Path = typer.Option(..., "--input"), context: Path = typer.Option(..., "--context"), report: Path = typer.Option(..., "--report")):
    _refresh_dynamic_agents()
    validate_lp_context_manifest(context)
    records = read_jsonl(input)
    quality = validate_minimal_pairs(records)
    write_json(report, {"ok": quality.ok, "total_records": quality.total_records, "issues": [i.__dict__ for i in quality.issues]})
    if not quality.ok:
        raise typer.Exit(code=1)


@app.command("validate-file")
def validate_file(path: Path):
    _refresh_dynamic_agents()
    p = Path(path)
    kind = None
    try:
        raw = p.read_text(encoding="utf-8")
        parsed = load_json(p) if p.suffix == ".json" else yaml.safe_load(raw)
        if isinstance(parsed, dict):
            kind = parsed.get("kind")
    except Exception:
        kind = None
    if kind == "dataset_manifest":
        validate_dataset_manifest(p)
    elif kind == "lp_rule_manifest":
        validate_lp_rule_manifest(p)
    elif kind == "lp_context_manifest":
        validate_lp_context_manifest(p)
    elif kind == "evaluation_manifest":
        validate_evaluation_manifest(p, RELEASE_GATES_PATH)
    elif p.suffix in {".yaml", ".yml"}:
        validate_release_gates(p)
    else:
        raise typer.BadParameter("Unsupported file or missing kind")
    print("File valid")


@app.command("contamination-check")
def contamination_check(train: Path = typer.Option(..., "--train"), holdout: Path = typer.Option(..., "--holdout"), threshold: float = typer.Option(0.92, "--threshold"), out: Path = typer.Option(Path("reports/contamination_report.json"), "--out")):
    _refresh_dynamic_agents()
    matches = detect_contamination(load_json(train), load_json(holdout), threshold)
    write_json(out, {"threshold": threshold, "matches": [m.__dict__ for m in matches]})


@app.command("release-report")
def release_report(metrics: Path = typer.Option(..., "--metrics"), out_json: Path = typer.Option(..., "--out-json"), out_md: Path = typer.Option(..., "--out-md")):
    _refresh_dynamic_agents()
    report = evaluate_release(metrics, RELEASE_GATES_PATH)
    write_json(out_json, report.to_json())
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    if not report.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
