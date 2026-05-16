from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from qfr_pipeline.corpus_sources import ingest_corpus_sources, validate_corpus_source_manifest
from qfr_pipeline.corpus_readiness import audit_corpus_readiness
from qfr_pipeline.curation_policy import curate_ingested_corpus, validate_curation_policy_manifest
from qfr_pipeline.curated_split import split_curated_corpus, validate_split_policy_manifest
from qfr_pipeline.diagnostics import load_eval_rows, load_taxonomies, run_diagnostics, write_diagnostics_json, write_diagnostics_markdown
from qfr_pipeline.io import write_json
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, load_context_manifest, write_jsonl
from qfr_pipeline.modern_corpus import acquire_modern_corpus, validate_modern_corpus_manifest
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT, repo_relative_path
from qfr_pipeline.release_report import ReleaseReport, evaluate_release
from qfr_pipeline.training_export import export_training_dataset, validate_training_export_manifest
from qfr_pipeline.validation import validate_error_taxonomy_manifest, validate_repository


def _build_minimal_pair_report(*, context: Path, context_manifest, quality, gen_report=None) -> dict:
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


@dataclass
class ReleaseCandidateStage:
    name: str
    ok: bool
    artifacts: list[str]
    blocking: bool = True
    details: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ReleaseCandidateReport:
    ok: bool
    deterministic_timestamp: str
    stages: list[ReleaseCandidateStage]
    artifacts: dict[str, str]
    release_report_summary: dict[str, Any] | None = None
    diagnostics_summary: dict[str, Any] | None = None
    minimal_pair_summary: dict[str, Any] | None = None
    curation_summary: dict[str, Any] | None = None
    split_summary: dict[str, Any] | None = None
    training_export_summary: dict[str, Any] | None = None
    training_export_summary: dict[str, Any] | None = None
    blocking_failures: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "deterministic_timestamp": self.deterministic_timestamp,
            "stages": [asdict(s) for s in self.stages],
            "artifacts": self.artifacts,
            "release_report_summary": self.release_report_summary,
            "diagnostics_summary": self.diagnostics_summary,
            "minimal_pair_summary": self.minimal_pair_summary,
            "curation_summary": self.curation_summary,
            "split_summary": self.split_summary,
            "training_export_summary": self.training_export_summary,
            "blocking_failures": self.blocking_failures or [],
        }

    def to_markdown(self) -> str:
        lines = [f"# Release Candidate: {'PASS' if self.ok else 'FAIL'}", "", f"- Deterministic timestamp: `{self.deterministic_timestamp}`", "", "## Stages"]
        for stage in self.stages:
            lines.append(f"- {stage.name}: {'PASS' if stage.ok else 'FAIL'}")
            if stage.artifacts:
                lines.append(f"  - Artifacts: {stage.artifacts}")
            if stage.error:
                lines.append(f"  - Error: {stage.error}")
        if self.release_report_summary is not None:
            lines.extend(["", "## Release report summary", f"- Passed: `{self.release_report_summary.get('passed')}`"])
        if self.diagnostics_summary is not None:
            lines.extend(["", "## Diagnostics summary", f"- OK: `{self.diagnostics_summary.get('ok')}`", f"- Phenomena: `{self.diagnostics_summary.get('phenomena')}`"])
        if self.minimal_pair_summary is not None:
            lines.extend(["", "## Minimal-pair summary", f"- LP9 records: `{self.minimal_pair_summary.get('lp9_records')}`", f"- LP20 records: `{self.minimal_pair_summary.get('lp20_records')}`"])
        if self.curation_summary is not None:
            lines.extend(["", "## Corpus curation summary", f"- Accepted: `{self.curation_summary.get('accepted')}`", f"- Review required: `{self.curation_summary.get('review_required')}`", f"- Quarantine: `{self.curation_summary.get('quarantine')}`", f"- Rejected: `{self.curation_summary.get('rejected')}`"])
        if self.split_summary is not None:
            lines.extend(["", "## Curated split summary", f"- Records total: `{self.split_summary.get('records_total')}`", f"- Train: `{self.split_summary.get('train')}`", f"- Dev: `{self.split_summary.get('dev')}`", f"- Test: `{self.split_summary.get('test')}`"])
        if self.training_export_summary is not None:
            lines.extend(["", "## Training export summary", f"- Dataset: `{self.training_export_summary.get('dataset_name')}`", f"- Version: `{self.training_export_summary.get('dataset_version')}`", f"- Total: `{self.training_export_summary.get('total')}`", f"- Train/Dev/Test: `{self.training_export_summary.get('train')}`/`{self.training_export_summary.get('dev')}`/`{self.training_export_summary.get('test')}`", f"- Aggregate SHA-256: `{self.training_export_summary.get('aggregate_sha256')}`"])
        if self.blocking_failures:
            lines.extend(["", "## Blocking failures", *[f"- {x}" for x in self.blocking_failures]])
        return "\n".join(lines) + "\n"


def run_release_candidate(*, metrics: Path, diagnostics_input: Path, out_json: Path, out_md: Path, allow_missing_diagnostics_phenomena: bool = False, lp9_context: Path = ROOT / "rules/lp9_lexical_semantics.contexts.yaml", lp20_context: Path = ROOT / "rules/lp20_orphaned_preposition.contexts.yaml") -> ReleaseCandidateReport:
    artifacts = {
        "diagnostics_json": "reports/diagnostics.lp9_lp20.json",
        "diagnostics_md": "reports/diagnostics.lp9_lp20.md",
        "lp9_pairs": "data/generated/minimal_pairs.lp9.jsonl",
        "lp9_quality": "reports/minimal_pair_quality.lp9.json",
        "lp20_pairs": "data/generated/minimal_pairs.lp20.jsonl",
        "lp20_quality": "reports/minimal_pair_quality.lp20.json",
        "release_report_json": "reports/release_report.json",
        "release_report_md": "reports/release_report.md",
        "corpus_ingestion_jsonl": "reports/corpus_ingestion/harvest.jsonl",
        "corpus_ingestion_report": "reports/corpus_ingestion/report.json",
        "corpus_curation_report": "reports/corpus_curation/report.json",
        "curated_split_train": "reports/curated_splits/train.jsonl",
        "curated_split_dev": "reports/curated_splits/dev.jsonl",
        "curated_split_test": "reports/curated_splits/test.jsonl",
        "curated_split_report": "reports/curated_splits/split_report.json",
        "training_export_manifest": "manifests/training_export_manifest.template.yaml",
        "training_export_report": "reports/training_export/export_report.json",
        "training_manifest_json": "reports/training_export/training_manifest.json",
        "training_manifest_yaml": "reports/training_export/training_manifest.yaml",
        "training_dataset_card": "reports/training_export/dataset_card.md",
        "modern_corpus_dry_run_report": "reports/modern_corpus/dry_run_report.json",
        "donnees_quebec_fixture_report": "reports/modern_corpus/donnees_quebec_fixture_report.json",
        "assnat_fixture_report": "reports/modern_corpus/assnat_fixture_report.json",
        "corpus_readiness_report": "reports/corpus_readiness/report.json",
        "release_candidate_json": repo_relative_path(out_json),
        "release_candidate_md": repo_relative_path(out_md),
    }
    stages: list[ReleaseCandidateStage] = []
    blocking_failures: list[str] = []
    release_summary: dict[str, Any] | None = None
    diag_summary: dict[str, Any] | None = None
    mp_summary: dict[str, Any] | None = None
    curation_summary: dict[str, Any] | None = None
    split_summary: dict[str, Any] | None = None
    training_export_summary: dict[str, Any] | None = None
    det_ts = "project-status-last-updated:" + __import__("json").loads((ROOT / "project/status.json").read_text(encoding="utf-8")).get("last_updated", "unknown")
    try:
        repo_report = validate_repository(ROOT)
        ok = repo_report.ok
        stages.append(ReleaseCandidateStage(name="repository_validation", ok=ok, artifacts=[], details={"issues": [asdict(i) for i in repo_report.issues]}))
        if not ok:
            blocking_failures.append("repository_validation")

        validate_error_taxonomy_manifest(ROOT / "eval/lp9_error_taxonomy.yaml")
        validate_error_taxonomy_manifest(ROOT / "eval/lp20_error_taxonomy.yaml")
        stages.append(ReleaseCandidateStage(name="taxonomy_validation", ok=True, artifacts=[]))
        modern_manifest = ROOT / "manifests/modern_corpus_acquisition_manifest.template.yaml"
        validate_modern_corpus_manifest(modern_manifest)
        stages.append(ReleaseCandidateStage(name="modern_corpus_manifest_validation", ok=True, artifacts=[repo_relative_path(modern_manifest)]))
        modern_dry_run = acquire_modern_corpus(modern_manifest, ROOT / "reports/modern_corpus/harvest.jsonl", ROOT / artifacts["modern_corpus_dry_run_report"], max_documents=0)
        stages.append(ReleaseCandidateStage(name="modern_corpus_acquisition_dry_run", ok=modern_dry_run.get("ok", False), artifacts=[artifacts["modern_corpus_dry_run_report"]], details=modern_dry_run))
        if not modern_dry_run.get("ok", False):
            blocking_failures.append("modern_corpus_acquisition_dry_run")
        readiness: dict[str, Any] = {"ok": False, "readiness_level": "insufficient", "blocking_reasons": ["not_run"]}

        corpus_manifest = ROOT / "manifests/corpus_source_manifest.template.yaml"
        validate_corpus_source_manifest(corpus_manifest)
        stages.append(ReleaseCandidateStage(name="corpus_source_validation", ok=True, artifacts=[]))

        ingestion = ingest_corpus_sources(corpus_manifest, ROOT / artifacts["corpus_ingestion_jsonl"], min_chars=20)
        from qfr_pipeline.corpus_sources import write_ingestion_report
        write_ingestion_report(ingestion, ROOT / artifacts["corpus_ingestion_report"])
        stages.append(ReleaseCandidateStage(name="corpus_ingestion", ok=ingestion.ok, artifacts=[artifacts["corpus_ingestion_jsonl"], artifacts["corpus_ingestion_report"]]))
        if not ingestion.ok:
            blocking_failures.append("corpus_ingestion")

        curation_policy = ROOT / "manifests/curation_policy_manifest.template.yaml"
        validate_curation_policy_manifest(curation_policy)
        stages.append(ReleaseCandidateStage(name="curation_policy_validation", ok=True, artifacts=[repo_relative_path(curation_policy)]))

        curation = curate_ingested_corpus(ROOT / artifacts["corpus_ingestion_jsonl"], curation_policy, ROOT / "reports/corpus_curation")
        stages.append(ReleaseCandidateStage(name="corpus_curation", ok=curation.ok, artifacts=list(curation.outputs.values()), details={"accepted": curation.accepted, "review_required": curation.review_required, "quarantine": curation.quarantine, "rejected": curation.rejected}))
        if not curation.ok:
            blocking_failures.append("corpus_curation")
        readiness_input = ROOT / "reports/corpus_curation/accepted.jsonl"
        readiness = audit_corpus_readiness(readiness_input, ROOT / artifacts["corpus_readiness_report"])
        readiness_ok = readiness.get("ok", False) and not readiness.get("blocking_reasons")
        stages.append(ReleaseCandidateStage(name="corpus_readiness_audit", ok=readiness_ok, artifacts=[artifacts["corpus_readiness_report"]], details=readiness))
        if not readiness_ok:
            blocking_failures.append("corpus_readiness_audit")
        split_policy = ROOT / "manifests/split_policy_manifest.template.yaml"
        validate_split_policy_manifest(split_policy)
        stages.append(ReleaseCandidateStage(name="split_policy_validation", ok=True, artifacts=[repo_relative_path(split_policy)]))
        split = split_curated_corpus(ROOT / "reports/corpus_curation/accepted.jsonl", split_policy, ROOT / "reports/curated_splits")
        stages.append(ReleaseCandidateStage(name="curated_corpus_split", ok=split.ok, artifacts=list(split.outputs.values()), details={"train": split.train, "dev": split.dev, "test": split.test, "records_total": split.records_total}))
        if not split.ok:
            blocking_failures.append("curated_corpus_split")
        split_summary = {"records_total": split.records_total, "train": split.train, "dev": split.dev, "test": split.test, "forbidden_records_seen": split.forbidden_records_seen}

        training_manifest = ROOT / "manifests/training_export_manifest.template.yaml"
        validate_training_export_manifest(training_manifest)
        stages.append(ReleaseCandidateStage(name="training_export_manifest_validation", ok=True, artifacts=[repo_relative_path(training_manifest)]))
        tx_report = export_training_dataset(training_manifest, ROOT / "reports/training_export")
        stages.append(ReleaseCandidateStage(name="training_export_generation", ok=tx_report.ok, artifacts=list(tx_report.outputs.values()), details={"records": tx_report.records, "hashes": tx_report.hashes}))
        if not tx_report.ok:
            blocking_failures.append("training_export_generation")
        training_export_summary = {"dataset_name": tx_report.dataset_name, "dataset_version": tx_report.dataset_version, "train": tx_report.records["train"], "dev": tx_report.records["dev"], "test": tx_report.records["test"], "total": tx_report.records["total"], "aggregate_sha256": tx_report.hashes["aggregate_sha256"]}
        training_export_summary["modern_corpus_sources_total"] = modern_dry_run.get("sources_total", 0)
        training_export_summary["modern_corpus_active_sources"] = modern_dry_run.get("sources_active", 0)
        training_export_summary["modern_corpus_blocked_sources"] = len(modern_dry_run.get("skipped_sources", []))
        training_export_summary["corpus_readiness_level"] = readiness.get("readiness_level", "insufficient")
        training_export_summary["estimated_tokens"] = readiness.get("estimated_tokens", 0)
        training_export_summary["production_blocking_reasons"] = readiness.get("blocking_reasons", [])
        training_export_summary["modern_source_ratio"] = readiness.get("modern_source_ratio", 0.0)
        training_export_summary["institutional_source_ratio"] = readiness.get(
            "institutional_source_ratio", 0.0
        )
        training_export_summary["donnees_quebec_fixture_records"] = 0
        training_export_summary["assnat_fixture_records"] = 0
        fixture_report_paths = {
            "donnees_quebec_fixture_records": ROOT / artifacts["donnees_quebec_fixture_report"],
            "assnat_fixture_records": ROOT / artifacts["assnat_fixture_report"],
        }
        for field, fixture_path in fixture_report_paths.items():
            if fixture_path.exists():
                fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
                training_export_summary[field] = int(fixture_payload.get("records_written", 0))

        tax = load_taxonomies([ROOT / "eval/lp9_error_taxonomy.yaml", ROOT / "eval/lp20_error_taxonomy.yaml"])
        rows = load_eval_rows(diagnostics_input)
        diag = run_diagnostics(rows, tax, allow_missing_phenomena=allow_missing_diagnostics_phenomena)
        write_diagnostics_json(diag, ROOT / artifacts["diagnostics_json"])
        write_diagnostics_markdown(diag, ROOT / artifacts["diagnostics_md"])
        stages.append(ReleaseCandidateStage(name="diagnostics_generation", ok=diag.ok, artifacts=[artifacts["diagnostics_json"], artifacts["diagnostics_md"]], details=diag.to_json()))
        if not diag.ok:
            blocking_failures.append("diagnostics_generation")
        diag_summary = {"ok": diag.ok, "phenomena": sorted(diag.phenomena.keys())}

        lp9_recs, lp9_gen = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", lp9_context)
        write_jsonl(lp9_recs, ROOT / artifacts["lp9_pairs"])
        lp9_manifest = load_context_manifest(lp9_context)
        lp9_quality = validate_minimal_pairs_against_context([r.__dict__ for r in lp9_recs], lp9_manifest, str(lp9_context))
        write_json(ROOT / artifacts["lp9_quality"], _build_minimal_pair_report(context=lp9_context, context_manifest=lp9_manifest, quality=lp9_quality, gen_report=lp9_gen))
        lp9_ok = lp9_gen.ok and lp9_quality.ok
        stages.append(ReleaseCandidateStage(name="lp9_minimal_pairs", ok=lp9_ok, artifacts=[artifacts["lp9_pairs"], artifacts["lp9_quality"]]))
        if not lp9_ok:
            blocking_failures.append("lp9_minimal_pairs")

        lp20_recs, lp20_gen = generate_minimal_pairs(ROOT / "rules/lp_rule_manifest.template.yaml", lp20_context)
        write_jsonl(lp20_recs, ROOT / artifacts["lp20_pairs"])
        lp20_manifest = load_context_manifest(lp20_context)
        lp20_quality = validate_minimal_pairs_against_context([r.__dict__ for r in lp20_recs], lp20_manifest, str(lp20_context))
        write_json(ROOT / artifacts["lp20_quality"], _build_minimal_pair_report(context=lp20_context, context_manifest=lp20_manifest, quality=lp20_quality, gen_report=lp20_gen))
        lp20_ok = lp20_gen.ok and lp20_quality.ok
        stages.append(ReleaseCandidateStage(name="lp20_minimal_pairs", ok=lp20_ok, artifacts=[artifacts["lp20_pairs"], artifacts["lp20_quality"]]))
        if not lp20_ok:
            blocking_failures.append("lp20_minimal_pairs")
        mp_summary = {"lp9_records": len(lp9_recs), "lp20_records": len(lp20_recs), "lp9_artifact": artifacts["lp9_pairs"], "lp20_artifact": artifacts["lp20_pairs"]}

        rr = evaluate_release(metrics, RELEASE_GATES_PATH)
        rr = ReleaseReport(passed=rr.passed and diag.ok, per_gate=rr.per_gate, missing_metrics=rr.missing_metrics, diagnostics=diag.to_json())
        write_json(ROOT / artifacts["release_report_json"], rr.to_json())
        (ROOT / artifacts["release_report_md"]).write_text(rr.to_markdown(), encoding="utf-8")
        stages.append(ReleaseCandidateStage(name="release_report", ok=rr.passed, artifacts=[artifacts["release_report_json"], artifacts["release_report_md"]]))
        if not rr.passed:
            blocking_failures.append("release_report")
        release_summary = {"passed": rr.passed, "missing_metrics": rr.missing_metrics}
        curation_summary = {"accepted": curation.accepted, "review_required": curation.review_required, "quarantine": curation.quarantine, "rejected": curation.rejected}
    except Exception as exc:
        stages.append(ReleaseCandidateStage(name="runtime_exception", ok=False, artifacts=[], error=str(exc)))
        blocking_failures.append("runtime_exception")

    report = ReleaseCandidateReport(
        ok=len(blocking_failures) == 0,
        deterministic_timestamp=det_ts,
        stages=stages,
        artifacts=artifacts,
        release_report_summary=release_summary,
        diagnostics_summary=diag_summary,
        minimal_pair_summary=mp_summary,
        curation_summary=curation_summary,
        split_summary=split_summary,
        training_export_summary=training_export_summary,
        blocking_failures=blocking_failures,
    )
    write_json(out_json, report.to_json())
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    return report
