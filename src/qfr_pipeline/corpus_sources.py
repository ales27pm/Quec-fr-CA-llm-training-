from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from qfr_pipeline.data_pipeline import harvest
from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import CorpusSourceManifest


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
        if source.source_type in {"local_text", "local_jsonl", "local_csv", "manual_fixture"} and not source_path.exists():
            raise ValueError(f"source path does not exist: {source.path}")
    return manifest


def resolve_source_paths(manifest: CorpusSourceManifest) -> list[Path]:
    return [ROOT / s.path for s in manifest.sources if s.source_type in {"local_text", "local_jsonl", "local_csv", "manual_fixture"}]


def ingest_corpus_sources(manifest_path: Path, out_jsonl: Path, *, min_chars: int = 20, include_review_required: bool = False) -> CorpusIngestionReport:
    manifest = validate_corpus_source_manifest(manifest_path)
    skipped: list[dict[str, str]] = []
    issues: list[str] = []
    ingest_paths: list[Path] = []
    for source in manifest.sources:
        if source.source_type == "future_remote":
            skipped.append({"source_id": source.source_id, "reason": "future_remote"})
            continue
        if not source.allowed_for_training:
            skipped.append({"source_id": source.source_id, "reason": "not_allowed_for_training"})
            continue
        if source.requires_review and not include_review_required:
            skipped.append({"source_id": source.source_id, "reason": "requires_review"})
            continue
        if source.contains_holdout_material:
            skipped.append({"source_id": source.source_id, "reason": "contains_holdout_material"})
            continue
        if source.quality_tier == "quarantine":
            skipped.append({"source_id": source.source_id, "reason": "quality_tier_quarantine"})
            continue
        ingest_paths.append(ROOT / source.path)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = harvest(ingest_paths, out_jsonl, min_chars) if ingest_paths else 0
    report = CorpusIngestionReport(
        ok=len(issues) == 0,
        manifest=repo_relative_path(manifest_path),
        sources_total=len(manifest.sources),
        sources_ingested=len(ingest_paths),
        records_written=records,
        skipped_sources=skipped,
        issues=issues,
        output=repo_relative_path(out_jsonl),
    )
    return report


def write_ingestion_report(report: CorpusIngestionReport, out_report: Path) -> None:
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_report, asdict(report))
