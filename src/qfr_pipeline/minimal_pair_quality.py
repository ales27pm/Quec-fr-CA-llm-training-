from __future__ import annotations

import unicodedata
from dataclasses import dataclass

FORBIDDEN_NGRAMS = {"le fin de semaine", "un fin de semaine", "du fin de semaine", "de le"}
NEUTRALIZATION_MARKERS = {"week-end", "email"}


@dataclass
class QualityIssue:
    code: str
    message: str
    record_id: str | None = None
    blocking: bool = True


@dataclass
class QualityReport:
    ok: bool
    issues: list[QualityIssue]
    total_records: int


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def validate_minimal_pairs(records: list[dict], min_len: int = 8, max_len: int = 220) -> QualityReport:
    issues: list[QualityIssue] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        rid = record.get("id")
        good = str(record.get("good", ""))
        bad = str(record.get("bad", ""))
        try:
            lp_id = int(record.get("lp_id", -1))
        except (ValueError, TypeError) as exc:
            issues.append(QualityIssue("invalid_lp_id", f"Invalid lp_id: {exc}", rid))
            lp_id = -1
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            issues.append(QualityIssue("invalid_metadata", "metadata must be an object/dict", rid))
            metadata = {}
        contrast_pos = str(metadata.get("positive_pattern", ""))
        contrast_neg = str(metadata.get("negative_pattern", ""))

        if not good.strip() or not bad.strip():
            issues.append(QualityIssue("empty_side", "Good/bad cannot be empty", rid))

        if unicodedata.normalize("NFC", good) != good or unicodedata.normalize("NFC", bad) != bad:
            issues.append(QualityIssue("non_nfc", "Text must be NFC-normalized", rid, blocking=False))

        ngood, nbad = normalize_text(good), normalize_text(bad)
        if ngood == nbad:
            issues.append(QualityIssue("identical_pair", "Good and bad are identical after normalization", rid))

        for side, text in (("good", ngood), ("bad", nbad)):
            if len(text) < min_len:
                issues.append(QualityIssue("too_short", f"{side} side shorter than {min_len}", rid, blocking=False))
            if len(text) > max_len:
                issues.append(QualityIssue("too_long", f"{side} side longer than {max_len}", rid, blocking=False))
            for forbidden in FORBIDDEN_NGRAMS:
                if forbidden in text:
                    issues.append(QualityIssue("forbidden_ngram", f"Forbidden n-gram detected: {forbidden}", rid))

        if lp_id == 9:
            if any(marker in ngood for marker in NEUTRALIZATION_MARKERS):
                issues.append(QualityIssue("neutralization_good", "LP9 good side contains neutralization marker", rid))
            n_contrast_pos = normalize_text(contrast_pos) if contrast_pos else ""
            n_contrast_neg = normalize_text(contrast_neg) if contrast_neg else ""
            if n_contrast_pos and n_contrast_pos not in ngood and f"« {n_contrast_pos} »" not in ngood:
                issues.append(QualityIssue("missing_positive_pattern", "LP9 good side missing positive pattern", rid))
            if n_contrast_neg and n_contrast_neg not in nbad and f"« {n_contrast_neg} »" not in nbad:
                issues.append(QualityIssue("missing_negative_pattern", "LP9 bad side missing negative pattern", rid))

        pair_key = (ngood, nbad)
        if pair_key in seen_pairs:
            issues.append(QualityIssue("duplicate_pair", "Duplicate normalized good/bad pair", rid))
        seen_pairs.add(pair_key)

    return QualityReport(ok=not any(i.blocking for i in issues), issues=issues, total_records=len(records))
