from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import ModernCorpusAcquisitionManifest, ModernCorpusSource

MIN_TEXT_LEN = 40


class HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def text(self) -> str:
        return "\n".join(self._chunks)


@dataclass
class AcquisitionContext:
    include_noncommercial: bool
    permission_sources: set[str]
    max_documents: int | None
    timeout: int
    seen_hashes: set[str] = field(default_factory=set)


@dataclass
class AcquisitionResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped_sources: list[dict[str, str]] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)
    holdout_registry: list[str] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _segment_paragraphs(text: str) -> list[str]:
    segments = []
    for part in re.split(r"\n{2,}|[•●]\s*", text):
        clean = _clean_text(part)
        if len(clean) >= MIN_TEXT_LEN:
            segments.append(clean)
    return segments


def load_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return ModernCorpusAcquisitionManifest.model_validate(load_yaml(path))


def validate_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return load_modern_corpus_manifest(path)


def _record(source: ModernCorpusSource, text: str, index: int, *, source_url: str | None = None, source_path: str | None = None, register: str = "formal", domain: str = "general", license_name: str | None = None, license_url: str | None = None) -> dict[str, Any]:
    text_hash = _sha256(text)
    return {
        "record_id": f"{source.source_id}:{index}:{text_hash[:12]}",
        "source_id": source.source_id,
        "source_name": source.name,
        "text": text,
        "text_sha256": text_hash,
        "language": "fr-CA",
        "dialect_region": "Quebec",
        "register": register,
        "domain": domain,
        "source_type": source.source_type,
        "acquisition_status": source.acquisition_status,
        "license_status": source.license_status,
        "license_name": license_name if license_name is not None else source.license_name,
        "license_url": license_url if license_url is not None else source.license_url,
        "commercial_use": source.commercial_use,
        "allowed_for_training": source.allowed_for_training,
        "allowed_for_evaluation": source.allowed_for_evaluation,
        "holdout_only": source.holdout_only,
        "pii_risk": source.pii_risk,
        "source_url": source_url,
        "source_path": source_path,
        "collected_via_adapter": source.adapter.name,
        "date_min": source.date_min,
        "date_max": source.date_max,
        "quality_flags": [],
        "requires_review": False,
    }


def _catalog_skip(result: AcquisitionResult, source_id: str, reason: str) -> None:
    result.skipped_sources.append({"source_id": source_id, "reason": reason})


def _register_holdout(result: AcquisitionResult, source: ModernCorpusSource) -> None:
    result.holdout_registry.append(source.source_id)
    _catalog_skip(result, source.source_id, "holdout_only")
    result.per_source[source.source_id] = 0


def _append_record(result: AcquisitionResult, ctx: AcquisitionContext, source: ModernCorpusSource, record: dict[str, Any]) -> None:
    text_hash = record["text_sha256"]
    if text_hash in ctx.seen_hashes:
        return
    ctx.seen_hashes.add(text_hash)
    result.records.append(record)
    result.per_source[source.source_id] = result.per_source.get(source.source_id, 0) + 1


def acquire_donnees_quebec_ckan(source: ModernCorpusSource, result: AcquisitionResult, ctx: AcquisitionContext) -> None:
    if ctx.max_documents == 0:
        result.per_source[source.source_id] = 0
        _catalog_skip(result, source.source_id, "dry_run_max_documents_zero")
        return
    base_url = (source.adapter.base_url or "").rstrip("/")
    query = urlencode({"rows": source.adapter.rows, "q": source.adapter.query})
    try:
        with urlopen(f"{base_url}/package_search?{query}", timeout=ctx.timeout) as response:
            payload = json.loads(response.read())
    except (URLError, TimeoutError, ValueError) as exc:
        result.issues.append({"source_id": source.source_id, "error": f"ckan_fetch_failed: {exc}"})
        return

    entries = payload.get("result", {}).get("results", [])
    entries = sorted(entries, key=lambda x: (x.get("name", ""), x.get("id", "")))
    limit = len(entries) if ctx.max_documents is None else min(ctx.max_documents, len(entries))

    for idx, item in enumerate(entries[:limit]):
        license_name = item.get("license_title") or item.get("license_id") or source.license_name
        license_url = item.get("license_url") or source.license_url
        if source.license_status == "open_compatible" and not license_name:
            continue
        tags = ", ".join(sorted(tag.get("display_name", "") for tag in item.get("tags", [])))
        org = item.get("organization", {}) if isinstance(item.get("organization"), dict) else {}
        resources_text = "; ".join(_clean_text(f"{r.get('name', '')} {r.get('description', '')}") for r in item.get("resources", []) if isinstance(r, dict))
        text = _clean_text(
            " ".join(
                [
                    item.get("title", ""),
                    item.get("notes", ""),
                    org.get("title") or org.get("name") or "",
                    tags,
                    resources_text,
                ]
            )
        )
        if len(text) < MIN_TEXT_LEN:
            continue
        rec = _record(source, text, idx, source_url=item.get("url"), register=(source.registers[0] if source.registers else "administrative"), domain=(source.domains[0] if source.domains else "public_data"), license_name=license_name, license_url=license_url)
        _append_record(result, ctx, source, rec)


def acquire_assnat_seed_html(source: ModernCorpusSource, result: AcquisitionResult, ctx: AcquisitionContext) -> None:
    if not ctx.include_noncommercial:
        _catalog_skip(result, source.source_id, "noncommercial_requires_explicit_flag")
        result.per_source[source.source_id] = 0
        return
    if ctx.max_documents == 0:
        _catalog_skip(result, source.source_id, "dry_run_max_documents_zero")
        result.per_source[source.source_id] = 0
        return
    urls = sorted(source.adapter.seed_urls)
    limit = len(urls) if ctx.max_documents is None else min(ctx.max_documents, len(urls))
    for index, url in enumerate(urls[:limit]):
        try:
            if url.startswith("file://"):
                html = Path(url[7:]).read_text(encoding="utf-8")
            else:
                with urlopen(url, timeout=ctx.timeout) as response:
                    html = response.read().decode("utf-8", errors="ignore")
        except (OSError, URLError, TimeoutError) as exc:
            result.issues.append({"source_id": source.source_id, "error": f"assnat_fetch_failed:{url}: {exc}"})
            continue
        extractor = HTMLTextExtractor()
        extractor.feed(html)
        for paragraph in _segment_paragraphs(extractor.text()):
            rec = _record(source, paragraph, index, source_url=url, register=(source.registers[0] if source.registers else "parliamentary"), domain=(source.domains[0] if source.domains else "politics"))
            _append_record(result, ctx, source, rec)


def acquire_local_text_bundle(source: ModernCorpusSource, result: AcquisitionResult, ctx: AcquisitionContext) -> None:
    if source.source_id not in ctx.permission_sources:
        _catalog_skip(result, source.source_id, "permission_required")
        result.per_source[source.source_id] = 0
        return
    globs = sorted(source.adapter.local_globs)
    index = 0
    for pattern in globs:
        for file_path in sorted(ROOT.glob(pattern)):
            if not file_path.is_file():
                continue
            text = file_path.read_text(encoding="utf-8")
            for paragraph in _segment_paragraphs(text):
                rec = _record(source, paragraph, index, source_path=repo_relative_path(file_path), register=(source.registers[0] if source.registers else "mixed"), domain=(source.domains[0] if source.domains else "general"))
                _append_record(result, ctx, source, rec)
                index += 1
                if ctx.max_documents is not None and result.per_source.get(source.source_id, 0) >= ctx.max_documents:
                    return


def acquire_modern_corpus(manifest_path: Path, out_jsonl: Path, report_path: Path, permission_manifest: Path | None = None, include_noncommercial: bool = False, max_documents: int | None = None, timeout: int = 30) -> dict[str, Any]:
    manifest = load_modern_corpus_manifest(manifest_path)
    permission_data = load_yaml(permission_manifest) if permission_manifest and permission_manifest.exists() else {}
    permission_sources = set((permission_data or {}).get("sources", {}).keys())
    ctx = AcquisitionContext(include_noncommercial=include_noncommercial, permission_sources=permission_sources, max_documents=max_documents, timeout=timeout)
    result = AcquisitionResult()

    for source in sorted(manifest.sources, key=lambda s: s.source_id):
        if source.acquisition_status in {"catalog_only", "blocked_license"}:
            _catalog_skip(result, source.source_id, source.acquisition_status)
            result.per_source[source.source_id] = 0
            continue
        if source.source_type == "evaluation_holdout" or source.acquisition_status == "holdout_only":
            _register_holdout(result, source)
            continue
        if source.source_type == "permission_required":
            _catalog_skip(result, source.source_id, "permission_required")
            result.per_source[source.source_id] = 0
            continue

        if source.adapter.name == "donnees_quebec_ckan":
            acquire_donnees_quebec_ckan(source, result, ctx)
        elif source.adapter.name == "assnat_journal_debats":
            acquire_assnat_seed_html(source, result, ctx)
        elif source.adapter.name == "local_text_bundle":
            acquire_local_text_bundle(source, result, ctx)
        elif source.source_type == "github_dataset":
            _catalog_skip(result, source.source_id, "catalog_only")
            result.per_source[source.source_id] = 0
        else:
            _catalog_skip(result, source.source_id, "unsupported_adapter")
            result.per_source[source.source_id] = 0

    result.records.sort(key=lambda x: (x["source_id"], x["record_id"]))
    result.skipped_sources.sort(key=lambda x: (x["source_id"], x["reason"]))
    result.issues.sort(key=lambda x: (x["source_id"], x["error"]))
    result.holdout_registry.sort()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.records), encoding="utf-8")

    license_summary = Counter(r["license_status"] for r in result.records)
    domain_summary = Counter(r["domain"] for r in result.records)
    register_summary = Counter(r["register"] for r in result.records)

    report = {
        "ok": len(result.issues) == 0,
        "manifest": repo_relative_path(manifest_path),
        "sources_total": len(manifest.sources),
        "sources_active": sum(1 for s in manifest.sources if s.acquisition_status == "active"),
        "sources_acquired": sum(1 for count in result.per_source.values() if count > 0),
        "records_written": len(result.records),
        "skipped_sources": result.skipped_sources,
        "issues": result.issues,
        "per_source": dict(sorted(result.per_source.items())),
        "license_summary": dict(sorted(license_summary.items())),
        "domain_summary": dict(sorted(domain_summary.items())),
        "register_summary": dict(sorted(register_summary.items())),
        "holdout_registry": result.holdout_registry,
        "output": repo_relative_path(out_jsonl),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report
