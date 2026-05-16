from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import CorpusSourceEntry, CorpusSourceManifest

GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*start of(?: the)? project gutenberg", re.IGNORECASE
)
GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*end of(?: the)? project gutenberg", re.IGNORECASE
)
GUTENBERG_BOILERPLATE_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"project gutenberg",
        r"gutenberg-?tm",
        r"produced by",
        r"distributed proofreading team",
        r"this ebook",
        r"full project gutenberg license",
        r"www\.gutenberg\.org",
        r"donation",
        r"public domain",
        r"electronic works",
        r"start: full license",
        r"end: full license",
    ]
]


@dataclass
class CorpusIngestionReport:
    ok: bool
    manifest: str
    sources_total: int
    sources_ingested: int
    records_written: int
    skipped_sources: list[dict[str, str]]
    issues: list[str]
    output: str


def load_corpus_source_manifest(path: Path) -> CorpusSourceManifest:
    return CorpusSourceManifest.model_validate(load_yaml(path))


def validate_corpus_source_manifest(path: Path) -> CorpusSourceManifest:
    manifest = load_corpus_source_manifest(path)
    for source in manifest.sources:
        source_path = ROOT / source.path
        if source.source_type in {
            "local_text",
            "local_jsonl",
            "local_csv",
            "manual_fixture",
        } and not source_path.exists():
            raise ValueError(f"source path does not exist: {source.path}")
    return manifest


def resolve_source_paths(manifest: CorpusSourceManifest) -> list[Path]:
    return [
        ROOT / source.path
        for source in manifest.sources
        if source.source_type in {
            "local_text",
            "local_jsonl",
            "local_csv",
            "manual_fixture",
        }
    ]


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _normalized_text(text: str) -> str:
    return " ".join(_clean_text(text).casefold().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_gutenberg_bookends(text: str) -> str:
    start = GUTENBERG_START_RE.search(text)
    end = GUTENBERG_END_RE.search(text)
    if start is None or end is None or start.end() >= end.start():
        return text
    return text[start.end() : end.start()]


def _is_gutenberg_boilerplate_line(line: str) -> bool:
    candidate = line.strip()
    if not candidate:
        return False
    if candidate.startswith("***") and "gutenberg" in candidate.casefold():
        return True
    return any(regex.search(candidate) for regex in GUTENBERG_BOILERPLATE_RE)


def _segment_plain_text(raw_text: str, min_chars: int) -> tuple[list[str], str]:
    text = _strip_gutenberg_bookends(_normalize_newlines(raw_text))
    lines = text.split("\n")
    non_empty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if len(non_empty_indices) >= 2:
        first_non_empty = non_empty_indices[0]
        last_non_empty = non_empty_indices[-1]
        has_blank_lines = any(
            (not lines[index].strip())
            for index in range(first_non_empty + 1, last_non_empty)
        )
    else:
        has_blank_lines = False

    if not has_blank_lines:
        rows = []
        for line in lines:
            cleaned_line = _clean_text(line)
            if not cleaned_line or _is_gutenberg_boilerplate_line(cleaned_line):
                continue
            if len(cleaned_line) >= min_chars:
                rows.append(cleaned_line)
        return rows, "line_fallback"

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        paragraph = _clean_text(" ".join(buffer))
        buffer.clear()
        if paragraph and len(paragraph) >= min_chars:
            paragraphs.append(paragraph)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_buffer()
            continue
        if _is_gutenberg_boilerplate_line(stripped):
            continue
        buffer.append(stripped)
    flush_buffer()

    return paragraphs, "paragraph"


def _extract_text_from_json_row(row: dict[str, object]) -> str:
    for key in ("text", "content", "body"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _iter_source_segments(
    source: CorpusSourceEntry,
    min_chars: int,
    issues: list[str],
) -> list[tuple[str, str]]:
    path = ROOT / source.path

    if source.source_type in {"local_text", "manual_fixture"}:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        segments, mode = _segment_plain_text(raw_text, min_chars=min_chars)
        return [(segment, mode) for segment in segments]

    if source.source_type == "local_jsonl":
        segments: list[tuple[str, str]] = []
        for line_index, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines()
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                issues.append(
                    f"invalid_jsonl_line:{source.source_id}:{source.path}:{line_index}"
                )
                continue
            if not isinstance(payload, dict):
                continue
            text = _clean_text(_extract_text_from_json_row(payload))
            if text and len(text) >= min_chars:
                segments.append((text, "jsonl"))
        return segments

    if source.source_type == "local_csv":
        segments = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            for row_index, row in enumerate(reader):
                text = _clean_text(_extract_text_from_json_row(row))
                if not text:
                    for key in reader.fieldnames:
                        value = row.get(key)
                        if isinstance(value, str) and value.strip():
                            text = _clean_text(value)
                            break
                if text and len(text) >= min_chars:
                    segments.append((text, "csv"))
                elif text:
                    issues.append(
                        f"csv_text_too_short:{source.source_id}:{source.path}:{row_index}"
                    )
        return segments

    return []


def _build_record(
    *,
    source: CorpusSourceEntry,
    source_path: str,
    index: int,
    text: str,
    segment_mode: str,
) -> dict[str, object]:
    text_sha = _sha256(text)
    normalized = _normalized_text(text)
    normalized_sha = _sha256(normalized)
    record_id = f"{source.source_id}:{index:06d}:{normalized_sha[:12]}"
    domain = source.domain if source.domain else "unknown"

    quality_flags = [
        f"quality_tier:{source.quality_tier}",
        f"segmentation:{segment_mode}",
    ]

    return {
        "id": record_id,
        "record_id": record_id,
        "source_id": source.source_id,
        "source_name": source.name,
        "source_path": source_path,
        "text": text,
        "text_sha256": text_sha,
        "normalized_text_sha256": normalized_sha,
        "language": "fr-CA",
        "dialect_region": source.dialect_region,
        "register": source.language_register,
        "domain": domain,
        "source_type": source.source_type,
        "license": source.license,
        "license_url": source.license_url,
        "provenance": source.provenance,
        "quality_tier": source.quality_tier,
        "allowed_for_training": source.allowed_for_training,
        "allowed_for_evaluation": source.allowed_for_evaluation,
        "contains_holdout_material": source.contains_holdout_material,
        "holdout_only": source.holdout_only,
        "requires_review": source.requires_review,
        "quality_flags": quality_flags,
    }


def ingest_corpus_sources(
    manifest_path: Path,
    out_jsonl: Path,
    *,
    min_chars: int = 20,
    include_review_required: bool = False,
) -> CorpusIngestionReport:
    manifest = validate_corpus_source_manifest(manifest_path)

    skipped: list[dict[str, str]] = []
    issues: list[str] = []
    ingest_sources: list[CorpusSourceEntry] = []

    for source in manifest.sources:
        if source.source_type == "future_remote":
            skipped.append({"source_id": source.source_id, "reason": "future_remote"})
            continue
        if not source.allowed_for_training:
            skipped.append(
                {"source_id": source.source_id, "reason": "not_allowed_for_training"}
            )
            continue
        if source.requires_review and not include_review_required:
            skipped.append({"source_id": source.source_id, "reason": "requires_review"})
            continue
        if source.contains_holdout_material:
            skipped.append(
                {"source_id": source.source_id, "reason": "contains_holdout_material"}
            )
            continue
        if source.quality_tier == "quarantine":
            skipped.append(
                {"source_id": source.source_id, "reason": "quality_tier_quarantine"}
            )
            continue
        ingest_sources.append(source)

    seen_normalized_hashes: set[str] = set()
    records: list[dict[str, object]] = []

    for source in ingest_sources:
        source_path = repo_relative_path(ROOT / source.path)
        source_records = _iter_source_segments(source, min_chars=min_chars, issues=issues)
        for index, (text, mode) in enumerate(source_records):
            normalized_sha = _sha256(_normalized_text(text))
            if normalized_sha in seen_normalized_hashes:
                continue
            seen_normalized_hashes.add(normalized_sha)
            records.append(
                _build_record(
                    source=source,
                    source_path=source_path,
                    index=index,
                    text=text,
                    segment_mode=mode,
                )
            )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    out_jsonl.write_text(payload, encoding="utf-8")

    report = CorpusIngestionReport(
        ok=len(issues) == 0,
        manifest=repo_relative_path(manifest_path),
        sources_total=len(manifest.sources),
        sources_ingested=len(ingest_sources),
        records_written=len(records),
        skipped_sources=skipped,
        issues=issues,
        output=repo_relative_path(out_jsonl),
    )
    return report


def write_ingestion_report(report: CorpusIngestionReport, out_report: Path) -> None:
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_report, asdict(report))
