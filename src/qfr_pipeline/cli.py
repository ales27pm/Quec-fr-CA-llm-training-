from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import typer
import yaml
from rich import print

from qfr_pipeline.contamination import detect_contamination
from qfr_pipeline.curated_split import split_curated_corpus, validate_split_policy_manifest
from qfr_pipeline.curation_policy import curate_ingested_corpus, validate_curation_policy_manifest
from qfr_pipeline.corpus_sources import ingest_corpus_sources, validate_corpus_source_manifest, write_ingestion_report
from qfr_pipeline.data_pipeline import curate, edit_normative, harvest, split_train_dev_test, write_training_recipe
from qfr_pipeline.lp7_monitor import monitor_lp7
from qfr_pipeline.diagnostics import load_eval_rows, load_taxonomies, run_diagnostics, write_diagnostics_json, write_diagnostics_markdown
from qfr_pipeline.io import load_json, write_json
from qfr_pipeline.legacy_diagnostics import run_legacy_semantic_diagnostics
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, load_context_manifest, read_jsonl, write_jsonl
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT, repo_relative_path
from qfr_pipeline.release_report import evaluate_release, ReleaseReport
from qfr_pipeline.release_candidate import run_release_candidate
from qfr_pipeline.modern_corpus import acquire_modern_corpus, validate_modern_corpus_manifest
from qfr_pipeline.corpus_readiness import audit_corpus_readiness
from qfr_pipeline.training_export import export_training_dataset, validate_training_export_manifest
from qfr_pipeline.validation import validate_corpus_source_manifest as validate_corpus_source_manifest_model, validate_curation_policy_manifest as validate_curation_policy_manifest_model, validate_dataset_manifest, validate_error_taxonomy_manifest, validate_evaluation_manifest, validate_lp_context_manifest, validate_lp_rule_manifest, validate_release_gates, validate_repository, validate_split_policy_manifest as validate_split_policy_manifest_model

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




@app.command("validate-corpus-sources")
def validate_corpus_sources_cmd(manifest: Path = typer.Option(..., "--manifest")):
    _refresh_dynamic_agents()
    validate_corpus_source_manifest_model(manifest)
    validate_corpus_source_manifest(manifest)
    print("Corpus source manifest valid")


@app.command("ingest-corpus-sources")
def ingest_corpus_sources_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    out: Path = typer.Option(..., "--out"),
    report: Path = typer.Option(..., "--report"),
    min_chars: int = typer.Option(20, "--min-chars"),
    include_review_required: bool = typer.Option(False, "--include-review-required"),
):
    _refresh_dynamic_agents()
    payload = ingest_corpus_sources(manifest, out, min_chars=min_chars, include_review_required=include_review_required)
    write_ingestion_report(payload, report)
    print(asdict(payload))
    if not payload.ok:
        raise typer.Exit(code=1)



@app.command("validate-curation-policy")
def validate_curation_policy_cmd(policy: Path = typer.Option(..., "--policy")):
    _refresh_dynamic_agents()
    validate_curation_policy_manifest_model(policy)
    validate_curation_policy_manifest(policy)
    print("Curation policy manifest valid")


@app.command("validate-split-policy")
def validate_split_policy_cmd(policy: Path = typer.Option(..., "--policy")):
    _refresh_dynamic_agents()
    validate_split_policy_manifest_model(policy)
    validate_split_policy_manifest(policy)
    print("Split policy manifest valid")


@app.command("split-curated-corpus")
def split_curated_corpus_cmd(
    input: Path = typer.Option(..., "--input"),
    policy: Path = typer.Option(..., "--policy"),
    out_dir: Path = typer.Option(..., "--out-dir"),
):
    _refresh_dynamic_agents()
    report = split_curated_corpus(input, policy, out_dir)
    print(f"Curated split report: {out_dir / 'split_report.json'}")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("curate-corpus")
def curate_corpus_cmd(
    input: Path = typer.Option(..., "--input"),
    policy: Path = typer.Option(..., "--policy"),
    out_dir: Path = typer.Option(..., "--out-dir"),
):
    _refresh_dynamic_agents()
    report = curate_ingested_corpus(input, policy, out_dir)
    print(asdict(report))
    if not report.ok:
        raise typer.Exit(code=1)

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



@app.command("harvest", context_settings={"allow_extra_args": True, "ignore_unknown_options": False})
def harvest_cmd(ctx: typer.Context, inputs: list[Path] = typer.Option([], "--inputs"), out: Path = typer.Option(..., "--out"), min_chars: int = typer.Option(20, "--min-chars"), dedupe_batch_size: int = typer.Option(0, "--dedupe-batch-size")):
    extra_inputs = [Path(arg) for arg in ctx.args if not str(arg).startswith("-")]
    all_inputs = [*inputs, *extra_inputs]
    if not all_inputs:
        raise typer.BadParameter("At least one input path is required via --inputs")
    print(harvest(all_inputs, out, min_chars, dedupe_batch_size))


@app.command("curate")
def curate_cmd(inp: Path = typer.Option(..., "--in"), out: Path = typer.Option(..., "--out"), min_fr_ca_score: float = typer.Option(0.34, "--min-fr-ca-score")):
    print(curate(inp, out, min_fr_ca_score))


@app.command("edit-normative")
def edit_normative_cmd(inp: Path = typer.Option(..., "--in"), out: Path = typer.Option(..., "--out")):
    print(edit_normative(inp, out))


@app.command("split")
def split_cmd(inp: Path = typer.Option(..., "--in"), out_dir: Path = typer.Option(..., "--out-dir"), seed: int = typer.Option(42, "--seed")):
    print(split_train_dev_test(inp, out_dir, seed))


@app.command("training-recipe")
def training_recipe_cmd(data_dir: Path = typer.Option(..., "--data-dir"), out: Path = typer.Option(..., "--out")):
    write_training_recipe(data_dir, out)
    print(out)


@app.command("monitor-lp7")
def monitor_lp7_cmd(pre: float = typer.Option(..., "--pre"), post: float = typer.Option(..., "--post"), release_gates: Path = typer.Option(RELEASE_GATES_PATH, "--release-gates"), out: Path | None = typer.Option(None, "--out")):
    payload = monitor_lp7(pre, post, release_gates)
    if out is not None:
        write_json(out, payload)
    print(payload)


@app.command("validate-training-export")
def validate_training_export_cmd(manifest: Path = typer.Option(..., "--manifest")):
    _refresh_dynamic_agents()
    validate_training_export_manifest(manifest)
    print("Training export manifest valid")


@app.command("export-training-dataset")
def export_training_dataset_cmd(manifest: Path = typer.Option(..., "--manifest"), out_dir: Path = typer.Option(..., "--out-dir")):
    _refresh_dynamic_agents()
    report = export_training_dataset(manifest, out_dir)
    if not report.ok:
        raise typer.Exit(code=1)
    print(f"Training export artifacts generated in {out_dir}")


@app.command("validate-modern-corpus")
def validate_modern_corpus_cmd(manifest: Path = typer.Option(..., "--manifest")):
    _refresh_dynamic_agents()
    validate_modern_corpus_manifest(manifest)
    print("Modern corpus manifest valid")

@app.command("acquire-modern-corpus")
def acquire_modern_corpus_cmd(manifest: Path = typer.Option(..., "--manifest"), out: Path = typer.Option(..., "--out"), report: Path = typer.Option(..., "--report"), permission_manifest: Path | None = typer.Option(None, "--permission-manifest"), include_noncommercial: bool = typer.Option(False, "--include-noncommercial"), max_documents: int | None = typer.Option(None, "--max-documents")):
    _refresh_dynamic_agents()
    payload = acquire_modern_corpus(manifest, out, report, permission_manifest=permission_manifest, include_noncommercial=include_noncommercial, max_documents=max_documents)
    if not payload.get('ok', False):
        raise typer.Exit(code=1)

@app.command("audit-corpus-readiness")
def audit_corpus_readiness_cmd(input: Path = typer.Option(..., "--input"), out: Path = typer.Option(..., "--out"), fail_below: str | None = typer.Option(None, "--fail-below")):
    _refresh_dynamic_agents()
    payload = audit_corpus_readiness(input, out)
    if not payload.get("ok", False):
        raise typer.Exit(code=1)
    levels = ["insufficient", "smoke_test", "pilot_lora_candidate", "production_lora_candidate"]
    level = payload.get("readiness_level", "insufficient")
    effective_level = "insufficient" if level == "production_blocked" else level
    if fail_below and levels.index(effective_level if effective_level in levels else "insufficient") < levels.index(fail_below):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
