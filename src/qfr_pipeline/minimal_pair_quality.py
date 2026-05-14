from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from qfr_pipeline.minimal_pairs import build_authorized_pair_index, stable_minimal_pair_id
from qfr_pipeline.schemas import LPContextManifest

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


def _canonicalize_context_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    try:
        return str(path.resolve(strict=False))
    except OSError:
        return str(path)


def _strip_non_alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _wordish_contains(text: str, pattern: str) -> bool:
    if not pattern:
        return False
    if pattern in text:
        return True
    if re.search(r"\W", pattern):
        return False
    return re.search(rf"\b{re.escape(pattern)}\b", text) is not None


def _check_substring_constraints(
    issues: list[QualityIssue], text: str, substrings: list[str], rid: str | None, code: str, missing: bool
) -> None:
    for raw in substrings:
        present = _wordish_contains(text, raw)
        if missing and not present:
            issues.append(QualityIssue(code, f"missing required substring: {raw}", rid))
        if (not missing) and present:
            issues.append(QualityIssue(code, f"contains forbidden substring: {raw}", rid))


def _validate_preposition_focus(
    issues: list[QualityIssue], ngood: str, nbad: str, pos: str, neg: str, rid: str | None
) -> None:
    if len(ngood) < 16 or len(nbad) < 16:
        issues.append(QualityIssue("lp20_too_short", "LP20 minimal pair is too short for meaningful contrast", rid))
    if pos and pos not in ngood:
        issues.append(QualityIssue("lp20_missing_preposition_focus", "LP20 good side missing expected preposition pattern", rid))
    if neg and neg not in nbad:
        issues.append(QualityIssue("lp20_missing_negative_focus", "LP20 bad side missing expected malformed/negative pattern", rid))


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
        if _strip_non_alnum(ngood) == _strip_non_alnum(nbad) and ngood != nbad:
            issues.append(QualityIssue("punctuation_only_change", "Contrast differs only by punctuation/spacing", rid))

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


def validate_minimal_pairs_against_context(
    records: list[dict], context_manifest: LPContextManifest, source_context: str | None = None
) -> QualityReport:
    quality = validate_minimal_pairs(records)
    issues = list(quality.issues)
    contrast_index = {c.contrast_id: c for c in context_manifest.contrasts}
    authorized_pairs = build_authorized_pair_index(context_manifest)

    for record in records:
        rid = str(record.get("id", "")) or None
        contrast_id = str(record.get("contrast_id", ""))
        contrast = contrast_index.get(contrast_id)
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        if record.get("lp_id") != context_manifest.lp_id:
            issues.append(QualityIssue("context_lp_id_mismatch", "record lp_id does not match context manifest", rid))
        if record.get("phenomenon") != context_manifest.name:
            issues.append(QualityIssue("context_phenomenon_mismatch", "record phenomenon does not match context manifest", rid))
        if contrast is None:
            issues.append(QualityIssue("context_unknown_contrast", f"contrast_id not found: {contrast_id}", rid))
            continue

        if record.get("register") != contrast.register_value:
            issues.append(QualityIssue("context_register_mismatch", "record register does not match context contrast", rid))
        if record.get("term_type") != contrast.term_type:
            issues.append(QualityIssue("context_term_type_mismatch", "record term_type does not match context contrast", rid))
        if metadata.get("positive_pattern") != contrast.positive_pattern:
            issues.append(QualityIssue("context_positive_pattern_mismatch", "metadata positive_pattern does not match context", rid))
        if metadata.get("negative_pattern") != contrast.negative_pattern:
            issues.append(QualityIssue("context_negative_pattern_mismatch", "metadata negative_pattern does not match context", rid))
        if metadata.get("allowed_contexts") != contrast.allowed_contexts:
            issues.append(QualityIssue("context_allowed_contexts_mismatch", "metadata allowed_contexts does not match context", rid))
        if metadata.get("blocked_contexts") != contrast.blocked_contexts:
            issues.append(QualityIssue("context_blocked_contexts_mismatch", "metadata blocked_contexts does not match context", rid))
        if source_context is not None:
            expected_source = _canonicalize_context_path(source_context)
            record_source = _canonicalize_context_path(str(record.get("source_context", "")))
            if record_source != expected_source:
                issues.append(QualityIssue("context_source_path_mismatch", "record source_context does not match provided context path", rid))

        good = str(record.get("good", ""))
        bad = str(record.get("bad", ""))
        if (good, bad) not in authorized_pairs.get(contrast_id, set()):
            issues.append(QualityIssue("context_unauthorized_pair", "good/bad pair is not authorized by context templates", rid))

        expected_id = stable_minimal_pair_id(context_manifest.lp_id, contrast_id, good, bad)
        if record.get("id") != expected_id:
            issues.append(QualityIssue("context_stable_id_mismatch", f"record id must be deterministic stable id: {expected_id}", rid))

        ngood, nbad = normalize_text(good), normalize_text(bad)
        pos, neg = normalize_text(contrast.positive_pattern), normalize_text(contrast.negative_pattern)
        if pos and pos not in ngood and f"« {pos} »" not in ngood:
            issues.append(QualityIssue("context_missing_positive_pattern", "good side missing context positive pattern", rid))
        if neg and neg not in nbad and f"« {neg} »" not in nbad:
            issues.append(QualityIssue("context_missing_negative_pattern", "bad side missing context negative pattern", rid))
        _check_substring_constraints(issues, ngood, contrast.normalized_required_good_substrings, rid, "context_missing_required_good_substring", missing=True)
        _check_substring_constraints(issues, nbad, contrast.normalized_required_bad_substrings, rid, "context_missing_required_bad_substring", missing=True)
        _check_substring_constraints(issues, ngood, contrast.normalized_forbidden_good_substrings, rid, "context_forbidden_good_substring", missing=False)
        _check_substring_constraints(issues, nbad, contrast.normalized_forbidden_bad_substrings, rid, "context_forbidden_bad_substring", missing=False)

        focus_validators = {"preposition_attachment": _validate_preposition_focus, "required_preposition_retention": _validate_preposition_focus, "stranded_preposition": _validate_preposition_focus}
        validator = focus_validators.get(contrast.minimal_contrast_focus or "")
        if validator is not None:
            validator(issues, ngood, nbad, pos, neg, rid)

    return QualityReport(ok=not any(i.blocking for i in issues), issues=issues, total_records=len(records))
