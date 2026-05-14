from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import typer
import yaml
from rich import print

from qfr_pipeline.contamination import detect_contamination
from qfr_pipeline.diagnostics import load_eval_rows, load_taxonomies, run_diagnostics, write_diagnostics_json, write_diagnostics_markdown
from qfr_pipeline.io import load_json, write_json
from qfr_pipeline.legacy_diagnostics import run_legacy_semantic_diagnostics
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, load_context_manifest, read_jsonl, write_jsonl
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT, repo_relative_path
from qfr_pipeline.release_report import evaluate_release, ReleaseReport
from qfr_pipeline.release_candidate import run_release_candidate
from qfr_pipeline.validation import validate_dataset_manifest, validate_error_taxonomy_manifest, validate_evaluation_manifest, validate_lp_context_manifest, validate_lp_rule_manifest, validate_release_gates, validate_repository

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


def _build_minimal_pair_report(
    *,
    context: Path,
    context_manifest,
    quality,
    gen_report=None,
) -> dict:
    payload = {
        "ok": quality.ok if gen_report is None else (gen_report.ok and quality.ok),
        "total_records": quality.total_records,
        "issues": [asdict(i) for i in quality.issues],
        "context_manifest": repo_relative_path(context),
        "authorized_pair_count": sum(len(c.good_templates) for c in context_manifest.contrasts),
        "lp_id": context_manifest.lp_id,
        "phenomenon": context_manifest.phenomenon,
        "name": context_manifest.name,
    }
    if gen_report is not None:
        payload["records_generated"] = gen_report.records_generated
        payload["issues"] = [asdict(i) for i in gen_report.issues] + payload["issues"]
    return payload


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
    context_manifest = load_context_manifest(context)
    quality = validate_minimal_pairs_against_context([p.__dict__ for p in pairs], context_manifest, source_context=str(context))
    if report:
        write_json(report, _build_minimal_pair_report(context=context, context_manifest=context_manifest, quality=quality, gen_report=gen_report))
    if (not gen_report.ok) or (not quality.ok):
        raise typer.Exit(code=1)
    write_jsonl(pairs, out)
    print(f"Wrote {len(pairs)} records to {out}")


@app.command("validate-minimal-pairs")
def validate_minimal_pairs_cmd(input: Path = typer.Option(..., "--input"), context: Path = typer.Option(..., "--context"), report: Path = typer.Option(..., "--report")):
    _refresh_dynamic_agents()
    context_manifest = validate_lp_context_manifest(context)
    records = read_jsonl(input)
    quality = validate_minimal_pairs_against_context(records, context_manifest, source_context=str(context))
    write_json(report, _build_minimal_pair_report(context=context, context_manifest=context_manifest, quality=quality))
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
def release_report(metrics: Path = typer.Option(..., "--metrics"), out_json: Path = typer.Option(..., "--out-json"), out_md: Path = typer.Option(..., "--out-md"), diagnostics: Path | None = typer.Option(None, "--diagnostics"), allow_missing_diagnostics_phenomena: bool = typer.Option(False, "--allow-missing-diagnostics-phenomena")):
    _refresh_dynamic_agents()
    report = evaluate_release(metrics, RELEASE_GATES_PATH)
    if diagnostics is not None:
        d = load_json(diagnostics)
        has_lp9 = any(v.get("lp_id") == 9 for v in d.get("phenomena", {}).values())
        has_lp20 = any(v.get("lp_id") == 20 for v in d.get("phenomena", {}).values())
        if not d.get("ok", False):
            raise typer.Exit(code=1)
        if (not allow_missing_diagnostics_phenomena) and (not (has_lp9 and has_lp20)):
            raise typer.Exit(code=1)
        report = ReleaseReport(passed=report.passed, per_gate=report.per_gate, missing_metrics=report.missing_metrics, diagnostics=d)
    write_json(out_json, report.to_json())
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("validate-taxonomy")
def validate_taxonomy(taxonomy: Path = typer.Option(..., "--taxonomy")):
    _refresh_dynamic_agents()
    validate_error_taxonomy_manifest(taxonomy)
    print("Taxonomy valid")


@app.command("diagnose-eval")
def diagnose_eval(input: Path = typer.Option(..., "--input"), taxonomy: list[Path] = typer.Option(..., "--taxonomy"), out_json: Path = typer.Option(..., "--out-json"), out_md: Path = typer.Option(..., "--out-md"), allow_missing_phenomena: bool = typer.Option(False, "--allow-missing-phenomena")):
    _refresh_dynamic_agents()
    taxonomies = load_taxonomies(taxonomy)
    rows = load_eval_rows(input)
    report = run_diagnostics(rows, taxonomies, allow_missing_phenomena=allow_missing_phenomena)
    write_diagnostics_json(report, out_json)
    write_diagnostics_markdown(report, out_md)
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("diagnose-legacy-csv")
def diagnose_legacy_csv(
    in_csv: Path = typer.Option(..., "--in-csv"),
    out_json: Path = typer.Option(..., "--out-json"),
    taxonomy: list[Path] | None = typer.Option(None, "--taxonomy"),
    allow_missing_phenomena: bool = typer.Option(False, "--allow-missing-phenomena"),
):
    _refresh_dynamic_agents()
    try:
        payload = run_legacy_semantic_diagnostics(
            in_csv,
            out_json,
            taxonomy_paths=taxonomy,
            allow_missing_phenomena=allow_missing_phenomena,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not payload["ok"]:
        raise typer.Exit(code=1)


@app.command("release-candidate")
def release_candidate(
    metrics: Path = typer.Option(..., "--metrics"),
    diagnostics_input: Path = typer.Option(..., "--diagnostics-input"),
    out_json: Path = typer.Option(..., "--out-json"),
    out_md: Path = typer.Option(..., "--out-md"),
    allow_missing_diagnostics_phenomena: bool = typer.Option(False, "--allow-missing-diagnostics-phenomena"),
    lp9_context: Path = typer.Option(ROOT / "rules/lp9_lexical_semantics.contexts.yaml", "--lp9-context"),
    lp20_context: Path = typer.Option(ROOT / "rules/lp20_orphaned_preposition.contexts.yaml", "--lp20-context"),
):
    _refresh_dynamic_agents()
    report = run_release_candidate(
        metrics=metrics,
        diagnostics_input=diagnostics_input,
        out_json=out_json,
        out_md=out_md,
        allow_missing_diagnostics_phenomena=allow_missing_diagnostics_phenomena,
        lp9_context=lp9_context,
        lp20_context=lp20_context,
    )
    if not report.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
