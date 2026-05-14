from pathlib import Path
import os
import subprocess

import typer
import yaml
from rich import print

from qfr_pipeline.contamination import detect_contamination
from qfr_pipeline.io import load_json, write_json
from qfr_pipeline.minimal_pairs import generate_lp9_minimal_pairs, write_jsonl
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.release_report import evaluate_release
from qfr_pipeline.validation import (
    validate_dataset_manifest,
    validate_evaluation_manifest,
    validate_lp_rule_manifest,
    validate_release_gates,
    validate_repository,
)

app = typer.Typer()


def _refresh_dynamic_agents() -> None:
    env = dict(os.environ)
    env["QFR_NO_AGENTS_BACKUP"] = "1"
    subprocess.run(["python3", str(ROOT / "tools" / "update_agents.py"), "--write"], check=True, env=env)


@app.command("validate")
def validate_repo():
    _refresh_dynamic_agents()
    report = validate_repository(ROOT)
    if not report.ok:
        for i in report.issues:
            print(f"[red]{i.path}[/red] {i.message}")
        raise typer.Exit(code=1)
    print("Validation passed")


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
    elif kind == "evaluation_manifest":
        validate_evaluation_manifest(p, RELEASE_GATES_PATH)
    elif p.suffix in {".yaml", ".yml"}:
        validate_release_gates(p)
    else:
        raise typer.BadParameter("Unsupported file or missing kind")
    print("File valid")


@app.command("generate-minimal-pairs")
def generate_minimal_pairs(rule: Path = typer.Option(..., "--rule"), out: Path = typer.Option(..., "--out")):
    _refresh_dynamic_agents()
    pairs = generate_lp9_minimal_pairs(rule)
    write_jsonl(pairs, out)
    print(f"Wrote {len(pairs)} records to {out}")


@app.command("contamination-check")
def contamination_check(train: Path = typer.Option(..., "--train"), holdout: Path = typer.Option(..., "--holdout"), threshold: float = typer.Option(0.92, "--threshold"), out: Path = typer.Option(Path("reports/contamination_report.json"), "--out")):
    _refresh_dynamic_agents()
    train_items = load_json(train)
    holdout_items = load_json(holdout)
    matches = detect_contamination(train_items, holdout_items, threshold)
    write_json(out, {"threshold": threshold, "matches": [m.__dict__ for m in matches]})
    print(f"Found {len(matches)} matches")


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
