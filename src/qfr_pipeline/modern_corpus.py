from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import unicodedata
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import ModernCorpusAcquisitionManifest, ModernCorpusSource

DEFAULT_MIN_TEXT_LEN = 40
_NAVIGATION_PATTERNS = (
    "accueil",
    "menu",
    "navigation",
    "fil d'ariane",
    "recherche",
    "nous joindre",
    "retour en haut",
    "imprimer",
    "partager",
    "plan du site",
    "english",
)
_NONCOMMERCIAL_REQUIRE_PERMISSION = "noncommercial_requires_explicit_flag"


class HTMLContentExtractor(HTMLParser):
    """Extract title and block-like text segments from HTML with stdlib only."""

    BLOCK_TAGS = {
        "p",
        "div",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "article",
        "section",
        "main",
        "td",
        "th",
        "blockquote",
        "dd",
        "dt",
        "span",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0
        self.title: str = ""
        self._inside_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag_lower = tag.casefold()
        if tag_lower in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._inside_title = True
        if tag_lower in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.casefold()
        if tag_lower in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag_lower == "title":
            self._inside_title = False
        if tag_lower in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._inside_title:
            self.title = _clean_text(" ".join([self.title, data]))
            return
        self._chunks.append(data)

    def lines(self) -> list[str]:
        return [line for line in _flatten_lines("".join(self._chunks)) if line]


@dataclass
class AcquisitionContext:
    include_noncommercial: bool
    permission_sources: set[str]
    max_documents: int | None
    timeout: int
    fixture_mode: bool
    seen_text_hashes: set[str] = field(default_factory=set)
    seen_normalized_hashes: set[str] = field(default_factory=set)


@dataclass
class AcquisitionResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped_sources: list[dict[str, str]] = field(default_factory=list)
    issues: list[dict[str, str]] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)
    holdout_registry: list[str] = field(default_factory=list)
    skipped_seed_urls: list[dict[str, str]] = field(default_factory=list)
    skipped_packages: list[dict[str, str]] = field(default_factory=list)
    source_reports: dict[str, dict[str, Any]] = field(default_factory=dict)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", _clean_text(text))
    return " ".join(normalized.casefold().split())


def _flatten_lines(text: str) -> list[str]:
    return [_clean_text(line) for line in text.replace("\r", "\n").split("\n")]


def _segment_paragraphs(text: str, min_chars: int = DEFAULT_MIN_TEXT_LEN) -> list[str]:
    segments: list[str] = []
    for part in re.split(r"\n{2,}|[•●]\s*", text):
        clean = _clean_text(part)
        if len(clean) >= min_chars:
            segments.append(clean)
    return segments


def _parse_file_url(url: str) -> Path:
    raw = url.removeprefix("file://")
    if raw.startswith("/"):
        return Path(raw)
    return ROOT / raw


def load_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return ModernCorpusAcquisitionManifest.model_validate(load_yaml(path))


def validate_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return load_modern_corpus_manifest(path)


def _record(
    source: ModernCorpusSource,
    text: str,
    index: int,
    *,
    source_url: str | None = None,
    source_path: str | None = None,
    register: str = "formal",
    domain: str = "general",
    license_name: str | None = None,
    license_url: str | None = None,
    license_id: str | None = None,
    adapter: str | None = None,
    quality_flags: list[str] | None = None,
    requires_review: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _normalize_text(text)
    text_hash = _sha256(text)
    normalized_hash = _sha256(normalized)
    payload: dict[str, Any] = {
        "record_id": f"{source.source_id}:{index:06d}:{normalized_hash[:12]}",
        "source_id": source.source_id,
        "source_name": source.name,
        "text": text,
        "text_sha256": text_hash,
        "normalized_text_sha256": normalized_hash,
        "language": "fr-CA",
        "dialect_region": "Quebec",
        "register": register,
        "domain": domain,
        "source_type": source.source_type,
        "adapter": adapter if adapter is not None else source.adapter.name,
        "acquisition_status": source.acquisition_status,
        "license_status": source.license_status,
        "license_id": license_id,
        "license_name": license_name if license_name is not None else source.license_name,
        "license_url": license_url if license_url is not None else source.license_url,
        "commercial_use": source.commercial_use,
        "allowed_for_training": source.allowed_for_training,
        "allowed_for_evaluation": source.allowed_for_evaluation,
        "holdout_only": source.holdout_only,
        "pii_risk": source.pii_risk,
        "source_url": source_url,
        "source_path": source_path,
        "date_min": source.date_min,
        "date_max": source.date_max,
        "quality_flags": quality_flags or [],
        "requires_review": source.requires_review if requires_review is None else requires_review,
    }
    if extra:
        payload.update(extra)
    return payload


def _catalog_skip(result: AcquisitionResult, source_id: str, reason: str) -> None:
    result.skipped_sources.append({"source_id": source_id, "reason": reason})


def _register_holdout(result: AcquisitionResult, source: ModernCorpusSource) -> None:
    result.holdout_registry.append(source.source_id)
    _catalog_skip(result, source.source_id, "holdout_only")
    result.per_source[source.source_id] = 0


def _append_record(
    result: AcquisitionResult,
    ctx: AcquisitionContext,
    source: ModernCorpusSource,
    record: dict[str, Any],
) -> bool:
    text_hash = str(record["text_sha256"])
    normalized_hash = str(record.get("normalized_text_sha256") or "")
    if text_hash in ctx.seen_text_hashes or normalized_hash in ctx.seen_normalized_hashes:
        return False
    ctx.seen_text_hashes.add(text_hash)
    if normalized_hash:
        ctx.seen_normalized_hashes.add(normalized_hash)
    result.records.append(record)
    result.per_source[source.source_id] = result.per_source.get(source.source_id, 0) + 1
    return True


def _effective_limit(source: ModernCorpusSource, ctx: AcquisitionContext, available: int) -> int:
    cap = available
    if ctx.max_documents is not None:
        cap = min(cap, ctx.max_documents)
    if source.max_documents is not None:
        cap = min(cap, source.max_documents)
    return cap


def _query_strings(source: ModernCorpusSource) -> list[str]:
    terms = [term.strip() for term in source.adapter.query_terms if term.strip()]
    if terms:
        return terms
    query = source.adapter.query.strip()
    return [query] if query else []


def _load_ckan_payload(source: ModernCorpusSource, ctx: AcquisitionContext) -> dict[str, Any]:
    fixture_path = source.adapter.fixture_response_path
    if fixture_path:
        fixture = ROOT / fixture_path if not Path(fixture_path).is_absolute() else Path(fixture_path)
        return json.loads(fixture.read_text(encoding="utf-8"))

    if ctx.fixture_mode:
        raise RuntimeError("fixture_mode_requires_fixture_response_path")

    base_url = (source.adapter.base_url or "").rstrip("/")
    if not base_url:
        raise RuntimeError("ckan_missing_base_url")

    all_results: dict[str, dict[str, Any]] = {}
    for query in _query_strings(source):
        params = {"rows": source.adapter.rows, "q": query}
        endpoint = f"{base_url}/package_search?{urlencode(params)}"
        with urlopen(endpoint, timeout=ctx.timeout) as response:
            payload = json.loads(response.read())
        for item in payload.get("result", {}).get("results", []):
            if not isinstance(item, dict):
                continue
            package_id = str(item.get("id") or "")
            key = package_id if package_id else str(item.get("name") or "")
            if not key:
                continue
            all_results[key] = item

    return {"result": {"results": list(all_results.values())}}


def _organization_fields(item: dict[str, Any]) -> tuple[str, str]:
    org = item.get("organization", {})
    if not isinstance(org, dict):
        return "", ""
    return str(org.get("name") or ""), str(org.get("title") or "")


def _tag_list(item: dict[str, Any]) -> list[str]:
    tags = []
    for tag in item.get("tags", []):
        if not isinstance(tag, dict):
            continue
        value = _clean_text(str(tag.get("display_name") or tag.get("name") or ""))
        if value:
            tags.append(value)
    return sorted(set(tags))


def _group_list(item: dict[str, Any]) -> list[str]:
    groups = []
    for group in item.get("groups", []):
        if not isinstance(group, dict):
            continue
        value = _clean_text(str(group.get("display_name") or group.get("title") or group.get("name") or ""))
        if value:
            groups.append(value)
    return sorted(set(groups))


def _resource_metadata(item: dict[str, Any]) -> list[dict[str, str | None]]:
    resources: list[dict[str, str | None]] = []
    for resource in item.get("resources", []):
        if not isinstance(resource, dict):
            continue
        resources.append(
            {
                "resource_id": str(resource.get("id") or "") or None,
                "name": _clean_text(str(resource.get("name") or "")) or None,
                "description": _clean_text(str(resource.get("description") or "")) or None,
                "format": _clean_text(str(resource.get("format") or "")) or None,
                "url": str(resource.get("url") or "") or None,
            }
        )
    return resources


def _ckan_text(item: dict[str, Any], resource_metadata: list[dict[str, str | None]]) -> str:
    organization_name, organization_title = _organization_fields(item)
    tags = ", ".join(_tag_list(item))
    groups = ", ".join(_group_list(item))

    paragraphs: list[str] = []
    if title := _clean_text(str(item.get("title") or "")):
        paragraphs.append(f"Titre: {title}")
    if notes := _clean_text(str(item.get("notes") or "")):
        paragraphs.append(f"Description: {notes}")
    org_parts = [part for part in [organization_title, organization_name] if part]
    if org_parts:
        paragraphs.append(f"Organisation: {' / '.join(org_parts)}")
    if tags:
        paragraphs.append(f"Mots-clés: {tags}")
    if groups:
        paragraphs.append(f"Groupes: {groups}")
    resource_lines = []
    for resource in resource_metadata:
        name = resource.get("name") or "Ressource sans nom"
        description = resource.get("description") or ""
        merged = _clean_text(f"{name}: {description}" if description else str(name))
        if merged:
            resource_lines.append(merged)
    if resource_lines:
        paragraphs.append("Ressources: " + " | ".join(resource_lines))
    return "\n\n".join(paragraphs)


def _classify_license(license_id: str, license_name: str, license_url: str) -> tuple[str, str]:
    marker = " ".join([license_id, license_name, license_url]).casefold()
    if not marker.strip():
        return "unknown", "unknown"

    if any(token in marker for token in ["all rights reserved", "copyright", "propriétaire"]):
        return "blocked", "prohibited"

    if any(
        token in marker
        for token in [
            "cc-by-nc",
            "cc by-nc",
            "cc-by-nc-sa",
            "cc-by-nc-nd",
            "noncommercial",
            "non-commercial",
        ]
    ):
        return "noncommercial_only", "permission_required"

    if any(
        token in marker
        for token in [
            "cc-by",
            "cc by",
            "cc0",
            "odc",
            "open government",
            "licence ouverte",
            "open licence",
            "gouv",
        ]
    ):
        return "open_compatible", "allowed"

    if any(token in marker for token in ["unknown", "inconnue", "not specified", "n/a"]):
        return "unknown", "unknown"

    return "unknown", "unknown"


def _ckan_package_allowed(source: ModernCorpusSource, package_license_status: str) -> bool:
    if package_license_status in {"blocked", "unknown"}:
        return source.license_status in {"permission_required", "unclear"}
    if package_license_status == "noncommercial_only":
        return source.license_status in {
            "open_compatible",
            "permission_required",
            "unclear",
            "noncommercial_only",
        }
    return True


def acquire_donnees_quebec_ckan(
    source: ModernCorpusSource,
    result: AcquisitionResult,
    ctx: AcquisitionContext,
) -> None:
    package_count_seen = 0
    package_count_selected = 0
    package_count_skipped = 0
    license_summary: Counter[str] = Counter()
    organization_summary: Counter[str] = Counter()
    domain_summary: Counter[str] = Counter()
    source_issues: list[dict[str, str]] = []
    source_skipped_packages: list[dict[str, str]] = []

    if ctx.max_documents == 0:
        result.per_source[source.source_id] = 0
        _catalog_skip(result, source.source_id, "dry_run_max_documents_zero")
        result.source_reports[source.source_id] = {
            "source_id": source.source_id,
            "adapter": source.adapter.name,
            "package_count_seen": 0,
            "package_count_selected": 0,
            "package_count_skipped": 0,
            "records_written": 0,
            "license_summary": {},
            "organization_summary": {},
            "domain_summary": {},
            "skipped_packages": [],
            "issues": [],
        }
        return

    try:
        payload = _load_ckan_payload(source, ctx)
    except (RuntimeError, OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        issue = {"source_id": source.source_id, "error": f"ckan_fetch_failed: {exc}"}
        result.issues.append(issue)
        source_issues.append(issue)
        result.source_reports[source.source_id] = {
            "source_id": source.source_id,
            "adapter": source.adapter.name,
            "package_count_seen": 0,
            "package_count_selected": 0,
            "package_count_skipped": 1,
            "records_written": 0,
            "license_summary": {},
            "organization_summary": {},
            "domain_summary": {},
            "skipped_packages": [
                {
                    "source_id": source.source_id,
                    "package_id": "",
                    "package_name": "",
                    "reason": "ckan_fetch_failed",
                }
            ],
            "issues": source_issues,
        }
        return

    entries = payload.get("result", {}).get("results", [])
    if not isinstance(entries, list):
        entries = []
    filtered_entries = [entry for entry in entries if isinstance(entry, dict)]
    ordered = sorted(
        filtered_entries,
        key=lambda item: (
            str(item.get("name") or ""),
            str(item.get("id") or ""),
            str(item.get("title") or ""),
        ),
    )
    limit = _effective_limit(source, ctx, len(ordered))

    for index, item in enumerate(ordered[:limit]):
        package_count_seen += 1
        package_id = str(item.get("id") or "")
        package_name = str(item.get("name") or "")
        package_title = str(item.get("title") or "")
        license_id = str(item.get("license_id") or "")
        license_name = str(item.get("license_title") or license_id)
        license_url = str(item.get("license_url") or source.license_url or "")
        package_license_status, package_commercial_use = _classify_license(
            license_id, license_name, license_url
        )

        if not _ckan_package_allowed(source, package_license_status):
            package_count_skipped += 1
            skipped = {
                "source_id": source.source_id,
                "package_id": package_id,
                "package_name": package_name,
                "reason": f"license_not_allowed:{package_license_status}",
            }
            source_skipped_packages.append(skipped)
            result.skipped_packages.append(skipped)
            continue

        resource_metadata = _resource_metadata(item)
        text = _ckan_text(item, resource_metadata)
        min_text_chars = source.min_text_chars if source.min_text_chars > 0 else DEFAULT_MIN_TEXT_LEN
        if len(_clean_text(text)) < min_text_chars:
            package_count_skipped += 1
            skipped = {
                "source_id": source.source_id,
                "package_id": package_id,
                "package_name": package_name,
                "reason": "text_too_short",
            }
            source_skipped_packages.append(skipped)
            result.skipped_packages.append(skipped)
            continue

        organization_name, organization_title = _organization_fields(item)
        tags = _tag_list(item)
        groups = _group_list(item)
        source_url = str(item.get("url") or "") or None
        metadata_created = str(item.get("metadata_created") or "") or None
        metadata_modified = str(item.get("metadata_modified") or "") or None

        quality_flags: list[str] = []
        if package_license_status in {"unknown", "blocked"}:
            quality_flags.append("license_review_required")

        allowed_for_training = source.allowed_for_training and (
            package_license_status == "open_compatible"
            or (package_license_status == "noncommercial_only" and ctx.include_noncommercial)
        )

        extra = {
            "package_id": package_id,
            "package_name": package_name,
            "package_title": package_title,
            "organization_name": organization_name,
            "organization_title": organization_title,
            "tags": tags,
            "groups": groups,
            "resource_count": len(resource_metadata),
            "resource_metadata": resource_metadata,
            "source_url": source_url,
            "metadata_created": metadata_created,
            "metadata_modified": metadata_modified,
            "license_status": package_license_status,
            "commercial_use": package_commercial_use,
            "allowed_for_training": allowed_for_training,
        }
        rec = _record(
            source,
            text,
            index,
            source_url=source_url,
            register=(source.registers[0] if source.registers else "administrative"),
            domain=(source.domains[0] if source.domains else "public_data"),
            license_name=license_name,
            license_url=license_url or None,
            license_id=license_id or None,
            adapter="donnees_quebec_ckan",
            quality_flags=quality_flags,
            requires_review=source.requires_review or bool(quality_flags),
            extra=extra,
        )
        if _append_record(result, ctx, source, rec):
            package_count_selected += 1
            license_summary[package_license_status] += 1
            org_bucket = organization_title or organization_name or "unknown"
            organization_summary[org_bucket] += 1
            domain_summary[rec.get("domain") or "unknown"] += 1
        else:
            package_count_skipped += 1
            skipped = {
                "source_id": source.source_id,
                "package_id": package_id,
                "package_name": package_name,
                "reason": "duplicate_text",
            }
            source_skipped_packages.append(skipped)
            result.skipped_packages.append(skipped)

    result.source_reports[source.source_id] = {
        "source_id": source.source_id,
        "adapter": source.adapter.name,
        "package_count_seen": package_count_seen,
        "package_count_selected": package_count_selected,
        "package_count_skipped": package_count_skipped,
        "records_written": result.per_source.get(source.source_id, 0),
        "license_summary": dict(sorted(license_summary.items())),
        "organization_summary": dict(sorted(organization_summary.items())),
        "domain_summary": dict(sorted(domain_summary.items())),
        "skipped_packages": source_skipped_packages,
        "issues": source_issues,
    }


def _looks_like_navigation(text: str) -> bool:
    clean = _clean_text(text)
    lowered = clean.casefold()
    if len(clean) < 8:
        return True
    if lowered in _NAVIGATION_PATTERNS:
        return True
    # Treat terse menu-style prefixes as navigation while preserving full content lines
    # that merely mention words like "recherche" in actual parliamentary text.
    return any(
        lowered.startswith(f"{pattern} ")
        or lowered.startswith(f"{pattern} :")
        or lowered.startswith(f"{pattern} -")
        for pattern in _NAVIGATION_PATTERNS
    )


def _detect_speaker(line: str) -> str | None:
    speaker_pattern = re.compile(
        r"^(M\.|Mme|Mmes|MM\.|Le président|La présidente|Le député|La députée|[A-ZÉÈÀÂÊÎÔÛÇ][^:]{2,80})\s*:"
    )
    match = speaker_pattern.match(line)
    if not match:
        return None
    speaker = _clean_text(match.group(1))
    return speaker or None


def _detect_date(text: str) -> str | None:
    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if iso_match:
        return iso_match.group(1)
    fr_match = re.search(
        r"\b([0-3]?\d\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|"
        r"octobre|novembre|décembre)\s+20\d{2})\b",
        text.casefold(),
    )
    if fr_match:
        return fr_match.group(1)
    return None


def _segment_assnat_lines(lines: list[str], min_chars: int) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        clean = _clean_text(line)
        if not clean:
            continue
        if _looks_like_navigation(clean):
            continue
        if len(clean) < min_chars:
            continue
        filtered.append(clean)
    return filtered


def acquire_assnat_seed_html(
    source: ModernCorpusSource,
    result: AcquisitionResult,
    ctx: AcquisitionContext,
) -> None:
    if not ctx.include_noncommercial:
        _catalog_skip(result, source.source_id, _NONCOMMERCIAL_REQUIRE_PERMISSION)
        result.per_source[source.source_id] = 0
        result.source_reports[source.source_id] = {
            "source_id": source.source_id,
            "adapter": source.adapter.name,
            "seed_urls_total": len(source.adapter.seed_urls),
            "seed_urls_fetched": 0,
            "records_written": 0,
            "skipped_sources": [{"source_id": source.source_id, "reason": _NONCOMMERCIAL_REQUIRE_PERMISSION}],
            "skipped_seed_urls": [{"seed_url": url, "reason": _NONCOMMERCIAL_REQUIRE_PERMISSION} for url in source.adapter.seed_urls],
            "issues": [],
        }
        for url in source.adapter.seed_urls:
            result.skipped_seed_urls.append(
                {
                    "source_id": source.source_id,
                    "seed_url": url,
                    "reason": _NONCOMMERCIAL_REQUIRE_PERMISSION,
                }
            )
        return

    if ctx.max_documents == 0:
        _catalog_skip(result, source.source_id, "dry_run_max_documents_zero")
        result.per_source[source.source_id] = 0
        result.source_reports[source.source_id] = {
            "source_id": source.source_id,
            "adapter": source.adapter.name,
            "seed_urls_total": len(source.adapter.seed_urls),
            "seed_urls_fetched": 0,
            "records_written": 0,
            "skipped_sources": [{"source_id": source.source_id, "reason": "dry_run_max_documents_zero"}],
            "skipped_seed_urls": [
                {"seed_url": url, "reason": "dry_run_max_documents_zero"}
                for url in source.adapter.seed_urls
            ],
            "issues": [],
        }
        for url in source.adapter.seed_urls:
            result.skipped_seed_urls.append(
                {
                    "source_id": source.source_id,
                    "seed_url": url,
                    "reason": "dry_run_max_documents_zero",
                }
            )
        return

    urls = sorted(source.adapter.seed_urls)
    limit = _effective_limit(source, ctx, len(urls))
    source_issues: list[dict[str, str]] = []
    source_skipped_urls: list[dict[str, str]] = []
    seed_urls_fetched = 0
    record_index = 0

    min_chars = source.min_text_chars if source.min_text_chars > 0 else DEFAULT_MIN_TEXT_LEN

    for index, url in enumerate(urls[:limit]):
        try:
            if url.startswith("file://"):
                html = _parse_file_url(url).read_text(encoding="utf-8")
            else:
                if ctx.fixture_mode:
                    raise RuntimeError("fixture_mode_disallows_live_fetch")
                with urlopen(url, timeout=ctx.timeout) as response:
                    html = response.read().decode("utf-8", errors="ignore")
        except (OSError, URLError, TimeoutError, RuntimeError) as exc:
            issue = {"source_id": source.source_id, "error": f"assnat_fetch_failed:{url}: {exc}"}
            result.issues.append(issue)
            source_issues.append(issue)
            source_skipped_urls.append({"seed_url": url, "reason": "fetch_failed"})
            result.skipped_seed_urls.append(
                {
                    "source_id": source.source_id,
                    "seed_url": url,
                    "reason": "fetch_failed",
                }
            )
            continue

        seed_urls_fetched += 1
        extractor = HTMLContentExtractor()
        extractor.feed(html)
        title = extractor.title or None
        raw_lines = extractor.lines()
        if title is None:
            candidate = next((line for line in raw_lines if len(line) >= min_chars), "")
            title = candidate or None
        date = _detect_date("\n".join(raw_lines))

        paragraphs = _segment_assnat_lines(raw_lines, min_chars)
        for paragraph in paragraphs:
            speaker = _detect_speaker(paragraph)
            cleaned_paragraph = _clean_text(paragraph)
            rec = _record(
                source,
                cleaned_paragraph,
                record_index,
                source_url=url,
                register=(source.registers[0] if source.registers else "parliamentary"),
                domain=(source.domains[0] if source.domains else "politics"),
                license_name=source.license_name,
                license_url=source.license_url,
                license_id=None,
                adapter="assnat_journal_debats",
                requires_review=source.requires_review,
                extra={
                    "speaker": speaker,
                    "date": date,
                    "title": title,
                    "license_status": "noncommercial_only",
                    "commercial_use": "permission_required",
                    "allowed_for_training": source.allowed_for_training
                    and ctx.include_noncommercial,
                    "allowed_for_evaluation": False,
                    "holdout_only": False,
                    "pii_risk": source.pii_risk,
                },
            )
            _append_record(result, ctx, source, rec)
            record_index += 1

        is_file_fixture = url.startswith("file://")
        should_sleep = (
            source.min_delay_seconds > 0
            and (index + 1) < limit
            and not is_file_fixture
        )
        if should_sleep:
            time.sleep(source.min_delay_seconds)

    result.source_reports[source.source_id] = {
        "source_id": source.source_id,
        "adapter": source.adapter.name,
        "seed_urls_total": len(source.adapter.seed_urls),
        "seed_urls_fetched": seed_urls_fetched,
        "records_written": result.per_source.get(source.source_id, 0),
        "skipped_sources": [
            item for item in result.skipped_sources if item.get("source_id") == source.source_id
        ],
        "skipped_seed_urls": source_skipped_urls,
        "issues": source_issues,
    }


def acquire_local_text_bundle(
    source: ModernCorpusSource,
    result: AcquisitionResult,
    ctx: AcquisitionContext,
) -> None:
    if source.source_id not in ctx.permission_sources:
        _catalog_skip(result, source.source_id, "permission_required")
        result.per_source[source.source_id] = 0
        return
    globs = sorted(source.adapter.local_globs)
    index = 0
    source_limit = _effective_limit(source, ctx, 10**9)
    min_chars = source.min_text_chars if source.min_text_chars > 0 else DEFAULT_MIN_TEXT_LEN
    for pattern in globs:
        for file_path in sorted(ROOT.glob(pattern)):
            if not file_path.is_file():
                continue
            text = file_path.read_text(encoding="utf-8")
            for paragraph in _segment_paragraphs(text, min_chars=min_chars):
                rec = _record(
                    source,
                    paragraph,
                    index,
                    source_path=repo_relative_path(file_path),
                    register=(source.registers[0] if source.registers else "mixed"),
                    domain=(source.domains[0] if source.domains else "general"),
                )
                _append_record(result, ctx, source, rec)
                index += 1
                if result.per_source.get(source.source_id, 0) >= source_limit:
                    return


def _flatten_source_reports(source_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    package_seen = 0
    package_selected = 0
    package_skipped = 0
    seed_total = 0
    seed_fetched = 0
    license_summary: Counter[str] = Counter()
    organization_summary: Counter[str] = Counter()
    domain_summary: Counter[str] = Counter()
    skipped_packages: list[dict[str, str]] = []
    skipped_seed_urls: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    for source_id in sorted(source_reports):
        report = source_reports[source_id]
        package_seen += int(report.get("package_count_seen", 0) or 0)
        package_selected += int(report.get("package_count_selected", 0) or 0)
        package_skipped += int(report.get("package_count_skipped", 0) or 0)
        seed_total += int(report.get("seed_urls_total", 0) or 0)
        seed_fetched += int(report.get("seed_urls_fetched", 0) or 0)
        license_summary.update(report.get("license_summary", {}))
        organization_summary.update(report.get("organization_summary", {}))
        domain_summary.update(report.get("domain_summary", {}))
        skipped_packages.extend(report.get("skipped_packages", []))
        skipped_seed_urls.extend(report.get("skipped_seed_urls", []))
        issues.extend(report.get("issues", []))

    return {
        "package_count_seen": package_seen,
        "package_count_selected": package_selected,
        "package_count_skipped": package_skipped,
        "license_summary": dict(sorted(license_summary.items())),
        "organization_summary": dict(sorted(organization_summary.items())),
        "domain_summary": dict(sorted(domain_summary.items())),
        "seed_urls_total": seed_total,
        "seed_urls_fetched": seed_fetched,
        "skipped_packages": skipped_packages,
        "skipped_seed_urls": skipped_seed_urls,
        "issues": issues,
    }


def acquire_modern_corpus(
    manifest_path: Path,
    out_jsonl: Path,
    report_path: Path,
    permission_manifest: Path | None = None,
    include_noncommercial: bool = False,
    max_documents: int | None = None,
    timeout: int = 30,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    manifest = load_modern_corpus_manifest(manifest_path)
    permission_data = (
        load_yaml(permission_manifest)
        if permission_manifest and permission_manifest.exists()
        else {}
    )
    permission_sources = set((permission_data or {}).get("sources", {}).keys())
    ctx = AcquisitionContext(
        include_noncommercial=include_noncommercial,
        permission_sources=permission_sources,
        max_documents=max_documents,
        timeout=timeout,
        fixture_mode=fixture_mode,
    )
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
    result.skipped_seed_urls.sort(
        key=lambda x: (x.get("source_id", ""), x.get("seed_url", ""), x.get("reason", ""))
    )
    result.skipped_packages.sort(
        key=lambda x: (x.get("source_id", ""), x.get("package_name", ""), x.get("reason", ""))
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.records),
        encoding="utf-8",
    )

    license_summary = Counter(str(r.get("license_status") or "unknown") for r in result.records)
    domain_summary = Counter(str(r.get("domain") or "unknown") for r in result.records)
    register_summary = Counter(str(r.get("register") or "unknown") for r in result.records)
    source_rollup = _flatten_source_reports(result.source_reports)

    report = {
        "ok": len(result.issues) == 0,
        "manifest": repo_relative_path(manifest_path),
        "sources_total": len(manifest.sources),
        "sources_active": sum(
            1 for source in manifest.sources if source.acquisition_status == "active"
        ),
        "sources_acquired": sum(1 for count in result.per_source.values() if count > 0),
        "records_written": len(result.records),
        "skipped_sources": result.skipped_sources,
        "issues": result.issues,
        "per_source": dict(sorted(result.per_source.items())),
        "source_reports": result.source_reports,
        "license_summary": dict(sorted(license_summary.items())),
        "domain_summary": dict(sorted(domain_summary.items())),
        "register_summary": dict(sorted(register_summary.items())),
        "organization_summary": source_rollup["organization_summary"],
        "package_count_seen": source_rollup["package_count_seen"],
        "package_count_selected": source_rollup["package_count_selected"],
        "package_count_skipped": source_rollup["package_count_skipped"],
        "skipped_packages": result.skipped_packages,
        "seed_urls_total": source_rollup["seed_urls_total"],
        "seed_urls_fetched": source_rollup["seed_urls_fetched"],
        "skipped_seed_urls": result.skipped_seed_urls,
        "holdout_registry": result.holdout_registry,
        "output": repo_relative_path(out_jsonl),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report
