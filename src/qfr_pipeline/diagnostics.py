from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.schemas import ErrorTaxonomyManifest


@dataclass
class DiagnosticIssue:
    code: str
    message: str
    blocking: bool = True


@dataclass
class PhenomenonSummary:
    lp_id: int
    phenomenon: str
    n: int
    correct: int
    binary_accuracy: float
    mean_semantic_similarity: float | None
    error_distribution: dict[str, int]
    blocking_error_count: int
    non_blocking_error_count: int
    top_error_codes: list[dict[str, int]]


@dataclass
class DiagnosticsReport:
    ok: bool
    phenomena: dict[str, PhenomenonSummary]
    global_summary: dict[str, Any]
    issues: list[DiagnosticIssue] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "phenomena": {k: asdict(v) for k, v in self.phenomena.items()},
            "global_summary": self.global_summary,
            "issues": [asdict(i) for i in self.issues],
        }


def _parse_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split()
    elif isinstance(value, list):
        items = value
    else:
        return None
    try:
        return [float(x) for x in items]
    except (TypeError, ValueError):
        return None


def _cosine(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def load_taxonomies(paths: list[Path]) -> dict[int, ErrorTaxonomyManifest]:
    out: dict[int, ErrorTaxonomyManifest] = {}
    for path in paths:
        manifest = ErrorTaxonomyManifest.model_validate(load_yaml(path))
        out[manifest.lp_id] = manifest
    return out


def run_diagnostics(rows: list[dict[str, Any]], taxonomies: dict[int, ErrorTaxonomyManifest], allow_missing_phenomena: bool = False) -> DiagnosticsReport:
    issues: list[DiagnosticIssue] = []
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    malformed_rows = 0
    unknown_codes = 0
    global_errors = Counter()

    for idx, row in enumerate(rows):
        try:
            lp_id = int(row["lp_id"])
            phenomenon = str(row["phenomenon"]).strip()
            is_correct = row["is_correct"]
            if isinstance(is_correct, int):
                is_correct = bool(is_correct)
            if not isinstance(is_correct, bool):
                raise ValueError("is_correct must be bool or 0/1")
            semantic = row.get("semantic_similarity")
            if semantic is not None:
                semantic = float(semantic)
                if semantic < 0 or semantic > 1:
                    raise ValueError("semantic_similarity must be in [0,1]")
            elif row.get("embedding_ref") is not None and row.get("embedding_pred") is not None:
                emb_ref = _parse_embedding(row.get("embedding_ref"))
                emb_pred = _parse_embedding(row.get("embedding_pred"))
                semantic = _cosine(emb_ref or [], emb_pred or [])
            key = (lp_id, phenomenon)
            bucket = by_key.setdefault(key, {"n": 0, "correct": 0, "semantic": [], "errors": Counter(), "blocking": 0, "non_blocking": 0})
            bucket["n"] += 1
            if is_correct:
                bucket["correct"] += 1
            if semantic is not None:
                bucket["semantic"].append(semantic)
            if not is_correct:
                error_code = row.get("error_code")
                if not error_code:
                    bucket["blocking"] += 1
                    issues.append(DiagnosticIssue(code="missing_error_code", message=f"row {idx} missing error_code", blocking=True))
                    continue
                manifest = taxonomies.get(lp_id)
                allowed = {i.error_code: i for i in manifest.taxonomy} if manifest else {}
                if error_code not in allowed:
                    bucket["blocking"] += 1
                    unknown_codes += 1
                    global_errors[error_code] += 1
                    bucket["errors"][error_code] += 1
                    issues.append(DiagnosticIssue(code="unknown_error_code", message=f"row {idx} unknown error_code {error_code} for lp_id {lp_id}", blocking=True))
                else:
                    sev = allowed[error_code].severity
                    bucket["errors"][error_code] += 1
                    global_errors[error_code] += 1
                    if sev == "blocking":
                        bucket["blocking"] += 1
                    else:
                        bucket["non_blocking"] += 1
        except Exception as exc:
            malformed_rows += 1
            issues.append(DiagnosticIssue(code="malformed_row", message=f"row {idx}: {exc}", blocking=True))

    present_lps = {lp for lp, _ in by_key}
    if not allow_missing_phenomena and not ({9, 20} <= present_lps):
        issues.append(DiagnosticIssue(code="missing_required_phenomena", message="Both LP9 and LP20 must be present", blocking=True))

    phenomena: dict[str, PhenomenonSummary] = {}
    total_block = 0
    total_non = 0
    for (lp_id, phenomenon), agg in sorted(by_key.items()):
        n = agg["n"]
        mean_sem = sum(agg["semantic"]) / len(agg["semantic"]) if agg["semantic"] else None
        total_block += agg["blocking"]
        total_non += agg["non_blocking"]
        key = f"LP{lp_id}:{phenomenon}"
        phenomena[key] = PhenomenonSummary(lp_id=lp_id, phenomenon=phenomenon, n=n, correct=agg["correct"], binary_accuracy=(agg["correct"] / n if n else 0.0), mean_semantic_similarity=mean_sem, error_distribution=dict(sorted(agg["errors"].items())), blocking_error_count=agg["blocking"], non_blocking_error_count=agg["non_blocking"], top_error_codes=[{"error_code": c, "count": k} for c, k in agg["errors"].most_common(3)])

    global_summary = {
        "total_records": len(rows),
        "malformed_rows": malformed_rows,
        "unknown_error_codes": unknown_codes,
        "blocking_error_count": total_block,
        "non_blocking_error_count": total_non,
        "top_error_codes": [{"error_code": c, "count": k} for c, k in global_errors.most_common(5)],
    }
    ok = all(not i.blocking for i in issues)
    return DiagnosticsReport(ok=ok, phenomena=phenomena, global_summary=global_summary, issues=issues)


def write_diagnostics_markdown(report: DiagnosticsReport, out: Path) -> None:
    lines = ["# LP9/LP20 Diagnostics", "", f"- OK: `{report.ok}`", f"- Total records: `{report.global_summary['total_records']}`", "", "## Per-phenomenon"]
    for key, p in report.phenomena.items():
        lines.extend([
            f"### {key}",
            f"- Binary accuracy: `{p.binary_accuracy:.4f}` ({p.correct}/{p.n})",
            f"- Mean semantic similarity: `{(p.mean_semantic_similarity if p.mean_semantic_similarity is not None else 'n/a')}`",
            f"- Blocking/non-blocking: `{p.blocking_error_count}/{p.non_blocking_error_count}`",
            f"- Top error codes: `{p.top_error_codes}`",
        ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostics_json(report: DiagnosticsReport, out: Path) -> None:
    write_json(out, report.to_json())
