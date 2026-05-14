from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qfr_pipeline.diagnostics import load_eval_rows, load_taxonomies, run_diagnostics, write_diagnostics_json, write_diagnostics_markdown
from qfr_pipeline.io import write_json
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs_against_context
from qfr_pipeline.minimal_pairs import generate_minimal_pairs, load_context_manifest, write_jsonl
from qfr_pipeline.paths import RELEASE_GATES_PATH, ROOT
from qfr_pipeline.release_report import ReleaseReport, evaluate_release
from qfr_pipeline.validation import validate_error_taxonomy_manifest, validate_repository


def _build_minimal_pair_report(*, context: Path, context_manifest, quality, gen_report=None) -> dict:
    payload = {
        "ok": quality.ok if gen_report is None else (gen_report.ok and quality.ok),
        "total_records": quality.total_records,
        "issues": [asdict(i) for i in quality.issues],
        "context_manifest": str(context),
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
        "release_candidate_json": str(out_json),
        "release_candidate_md": str(out_md),
    }
    stages: list[ReleaseCandidateStage] = []
    blocking_failures: list[str] = []
    release_summary: dict[str, Any] | None = None
    diag_summary: dict[str, Any] | None = None
    mp_summary: dict[str, Any] | None = None
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
        blocking_failures=blocking_failures,
    )
    write_json(out_json, report.to_json())
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.to_markdown(), encoding="utf-8")
    return report
