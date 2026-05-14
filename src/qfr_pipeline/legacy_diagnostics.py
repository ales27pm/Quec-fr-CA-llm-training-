from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from qfr_pipeline.diagnostics import (
    DiagnosticIssue,
    _parse_embedding,
    parse_is_correct,
    load_taxonomies,
    run_diagnostics,
    write_diagnostics_json,
)
from qfr_pipeline.paths import EVAL_DIR

LEGACY_REQUIRED_COLUMNS = {
    "phenomenon",
    "is_correct",
    "embedding_ref",
    "embedding_pred",
    "error_label",
}

_PHENOMENON_ALIAS_TO_TARGET: dict[str, tuple[int, str]] = {
    "lp9": (9, "lexical_semantics"),
    "lp9:lexical_semantics": (9, "lexical_semantics"),
    "lexical_semantics": (9, "lexical_semantics"),
    "lp20": (20, "orphaned_preposition"),
    "lp20:orphaned_preposition": (20, "orphaned_preposition"),
    "orphaned_preposition": (20, "orphaned_preposition"),
}

DEFAULT_LEGACY_TAXONOMY_PATHS = [
    EVAL_DIR / "lp9_error_taxonomy.yaml",
    EVAL_DIR / "lp20_error_taxonomy.yaml",
]


def load_legacy_semantic_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        missing = sorted(LEGACY_REQUIRED_COLUMNS - header)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError("No diagnostic rows found.")
    return rows


def _convert_legacy_row(
    row: dict[str, str],
    idx: int,
    issues: list[DiagnosticIssue],
) -> dict[str, Any]:
    converted: dict[str, Any] = {"id": f"legacy-{idx}"}

    raw_phenomenon = (row.get("phenomenon") or "").strip()
    mapped = _PHENOMENON_ALIAS_TO_TARGET.get(raw_phenomenon.lower())
    if mapped is None:
        converted["lp_id"] = f"invalid_phenomenon:{raw_phenomenon or 'empty'}"
        converted["phenomenon"] = raw_phenomenon or "unknown"
    else:
        converted["lp_id"], converted["phenomenon"] = mapped

    # Preserve CSV values as strings so maintained diagnostics parsing rules apply.
    converted["is_correct"] = row.get("is_correct")
    converted["embedding_ref"] = row.get("embedding_ref")
    converted["embedding_pred"] = row.get("embedding_pred")

    try:
        is_correct = parse_is_correct(converted["is_correct"])
    except ValueError:
        return converted

    emb_ref_raw = converted["embedding_ref"]
    emb_pred_raw = converted["embedding_pred"]
    emb_ref = _parse_embedding(emb_ref_raw)
    emb_pred = _parse_embedding(emb_pred_raw)
    if emb_ref_raw is not None and emb_ref_raw != "" and not emb_ref:
        issues.append(
            DiagnosticIssue(
                code="malformed_embedding",
                message=f"legacy row {idx} malformed embedding_ref",
                blocking=False,
            )
        )
    if emb_pred_raw is not None and emb_pred_raw != "" and not emb_pred:
        issues.append(
            DiagnosticIssue(
                code="malformed_embedding",
                message=f"legacy row {idx} malformed embedding_pred",
                blocking=False,
            )
        )

    if not is_correct:
        error_label = (row.get("error_label") or "").strip()
        if error_label:
            converted["error_code"] = error_label

    return converted


def run_legacy_semantic_diagnostics(
    in_csv: Path,
    out_json: Path,
    *,
    taxonomy_paths: list[Path] | None = None,
    allow_missing_phenomena: bool = False,
) -> dict[str, Any]:
    raw_rows = load_legacy_semantic_csv(in_csv)
    pre_issues: list[DiagnosticIssue] = []
    rows = [_convert_legacy_row(row, idx, pre_issues) for idx, row in enumerate(raw_rows)]
    taxonomies = load_taxonomies(taxonomy_paths or DEFAULT_LEGACY_TAXONOMY_PATHS)
    report = run_diagnostics(
        rows,
        taxonomies,
        allow_missing_phenomena=allow_missing_phenomena,
    )
    report.issues = [*pre_issues, *report.issues]
    report.ok = (
        report.ok
        and all(not issue.blocking for issue in pre_issues)
    )
    write_diagnostics_json(report, out_json)
    return report.to_json()
