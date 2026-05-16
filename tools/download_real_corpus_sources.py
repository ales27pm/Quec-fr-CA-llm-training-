#!/usr/bin/env python3
"""Download approved remote corpus sources into local files.

This script intentionally lives outside the core deterministic validation path.
It converts a manifest containing `future_remote` entries plus `download_url`
metadata into a local manifest that existing `qfr ingest-corpus-sources` can use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
ALLOWED_SCHEMES = {"http", "https"}


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest path must be repo-relative: {value}")
    resolved = (ROOT / path).resolve()
    if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
        raise ValueError(f"Manifest path escapes repository root: {value}")
    return resolved


def display_path(path: Path) -> str:
    if path.is_absolute() and path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    return path.as_posix()


def strip_project_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end and start.end() < end.start():
        return text[start.end() : end.start()].strip() + "\n"
    return text


def validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Blocked URL scheme: {scheme or '<missing>'}")
    if not parsed.netloc:
        raise ValueError(f"Malformed download URL: {url}")


def download_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "qfr-pipeline-corpus-downloader/0.2"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="replace")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_local_manifest(remote_manifest: dict[str, Any], downloaded_sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "corpus_source_manifest",
        "schema_version": remote_manifest.get("schema_version", "1.0"),
        "primary_language": remote_manifest.get("primary_language", "fr-CA"),
        "sources": downloaded_sources,
    }


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code <= 599
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return True
        if isinstance(reason, OSError) and reason.errno in {101, 104, 110}:
            return True
        message = str(reason).lower()
        return any(
            token in message
            for token in (
                "connection reset by peer",
                "network is unreachable",
                "timed out",
                "timeout",
            )
        )
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        if exc.errno in {101, 104, 110}:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "connection reset by peer",
                "network is unreachable",
                "timed out",
                "timeout",
            )
        )
    return False


def to_local_source(source: dict[str, Any], url: str, digest: str) -> dict[str, Any]:
    local_source = dict(source)
    local_source["source_type"] = "local_text"
    local_source["collection_method"] = "remote_text_download_materialized_locally"
    notes = str(source.get("notes", "")).strip()
    suffix = f"Downloaded from {url}. SHA-256={digest}"
    local_source["notes"] = f"{notes} {suffix}".strip()
    return local_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download approved remote corpus sources and emit a local ingestion manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/corpus_source_manifest.real_downloads.yaml")
    parser.add_argument("--out-manifest", type=Path, default=ROOT / "manifests/corpus_source_manifest.real_downloaded.local.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/corpus_ingestion/downloads.real.json")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even when the target path already exists.")
    parser.add_argument("--resume", action="store_true", help="Treat existing files as cached and continue remaining downloads.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not download sources with existing output files.")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for transient failures (in addition to the first attempt).")
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial success when minimum downloaded/cached sources are available.")
    parser.add_argument("--fail-under-downloaded", type=int, default=None, help="Minimum downloaded+cached source count required when --allow-partial is set.")
    parser.add_argument("--keep-gutenberg-boilerplate", action="store_true")
    args = parser.parse_args()

    if args.retries < 0:
        parser.error("--retries must be >= 0")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.fail_under_downloaded is not None and args.fail_under_downloaded < 0:
        parser.error("--fail-under-downloaded must be >= 0")
    return args


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    out_manifest_path = args.out_manifest if args.out_manifest.is_absolute() else ROOT / args.out_manifest
    report_path = args.report if args.report.is_absolute() else ROOT / args.report

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("Manifest sources must be a list")

    downloadable_total = sum(1 for source in sources if isinstance(source, dict) and source.get("download_url"))
    minimum_required_downloaded = downloadable_total
    if args.allow_partial and args.fail_under_downloaded is not None:
        minimum_required_downloaded = args.fail_under_downloaded

    downloaded_sources: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "ok": True,
        "partial_ok": False,
        "manifest": display_path(manifest_path),
        "out_manifest": display_path(out_manifest_path),
        "sources_total": len(sources),
        "downloadable_sources_total": downloadable_total,
        "downloaded_count": 0,
        "cached_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "minimum_required_downloaded": minimum_required_downloaded,
        "failed_sources": [],
        "downloaded": [],
        "skipped": [],
        "issues": [],
        "sources": [],
    }

    for source in sources:
        if not isinstance(source, dict):
            report["skipped_count"] += 1
            row = {
                "source_id": "<non_mapping_source>",
                "status": "skipped",
                "attempts": 0,
                "last_error": None,
                "reason": "invalid_source_entry",
            }
            report["skipped"].append({"source_id": row["source_id"], "reason": row["reason"]})
            report["sources"].append(row)
            continue

        source_id = str(source.get("source_id", "<missing>"))
        url = source.get("download_url")

        if not url:
            report["skipped_count"] += 1
            row = {
                "source_id": source_id,
                "status": "skipped",
                "attempts": 0,
                "last_error": None,
                "reason": "missing_download_url",
            }
            report["skipped"].append({"source_id": source_id, "reason": "missing_download_url"})
            report["sources"].append(row)
            continue

        try:
            target = repo_path(str(source["path"]))
        except Exception as exc:
            report["failed_count"] += 1
            report["failed_sources"].append(source_id)
            row = {
                "source_id": source_id,
                "status": "failed",
                "attempts": 1,
                "last_error": str(exc),
                "download_url": str(url),
            }
            report["issues"].append({"source_id": source_id, "error": str(exc)})
            report["sources"].append(row)
            continue

        should_use_cache = target.exists() and (
            args.resume or args.skip_existing or not args.overwrite
        )
        if should_use_cache:
            try:
                text = target.read_text(encoding="utf-8")
            except OSError as exc:
                report["failed_count"] += 1
                report["failed_sources"].append(source_id)
                row = {
                    "source_id": source_id,
                    "status": "failed",
                    "attempts": 1,
                    "last_error": str(exc),
                    "path": display_path(target),
                    "download_url": str(url),
                }
                report["issues"].append({"source_id": source_id, "error": str(exc)})
                report["sources"].append(row)
                continue

            digest = sha256_text(text)
            report["cached_count"] += 1
            downloaded_sources.append(to_local_source(source, str(url), digest))
            row = {
                "source_id": source_id,
                "status": "cached",
                "attempts": 0,
                "last_error": None,
                "path": display_path(target),
                "download_url": str(url),
                "characters": len(text),
                "sha256": digest,
            }
            report["downloaded"].append(row)
            report["sources"].append(row)
            continue

        last_error: str | None = None
        text: str | None = None
        attempts = 0
        max_attempts = max(1, int(args.retries) + 1)

        try:
            validate_download_url(str(url))
            for attempts in range(1, max_attempts + 1):
                try:
                    text = download_text(str(url), timeout=float(args.timeout))
                    break
                except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
                    last_error = str(exc)
                    if is_transient_error(exc) and attempts < max_attempts:
                        time.sleep(float(args.retry_delay_seconds))
                        continue
                    break
        except ValueError as exc:
            attempts = 1
            last_error = str(exc)

        if text is None:
            report["failed_count"] += 1
            report["failed_sources"].append(source_id)
            row = {
                "source_id": source_id,
                "status": "failed",
                "attempts": attempts,
                "last_error": last_error,
                "path": display_path(target),
                "download_url": str(url),
            }
            report["issues"].append({"source_id": source_id, "error": last_error})
            report["sources"].append(row)
            continue

        if not args.keep_gutenberg_boilerplate and "gutenberg.org" in str(url):
            text = strip_project_gutenberg_boilerplate(text)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

        digest = sha256_text(text)
        report["downloaded_count"] += 1
        downloaded_sources.append(to_local_source(source, str(url), digest))
        row = {
            "source_id": source_id,
            "status": "downloaded",
            "attempts": attempts,
            "last_error": None,
            "path": display_path(target),
            "download_url": str(url),
            "characters": len(text),
            "sha256": digest,
        }
        report["downloaded"].append(row)
        report["sources"].append(row)

    available_sources = report["downloaded_count"] + report["cached_count"]
    all_required_succeeded = report["failed_count"] == 0

    if all_required_succeeded:
        report["ok"] = True
        report["partial_ok"] = False
    else:
        threshold_met = available_sources >= report["minimum_required_downloaded"]
        report["partial_ok"] = bool(args.allow_partial and threshold_met)
        report["ok"] = bool(report["partial_ok"])

    if downloaded_sources:
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        out_manifest_path.write_text(
            yaml.safe_dump(build_local_manifest(manifest, downloaded_sources), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        report["out_manifest_written"] = True
    else:
        report["out_manifest_written"] = False

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
