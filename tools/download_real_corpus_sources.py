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
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest path must be repo-relative: {value}")
    resolved = (ROOT / path).resolve()
    if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
        raise ValueError(f"Manifest path escapes repository root: {value}")
    return resolved


def strip_project_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end and start.end() < end.start():
        return text[start.end() : end.start()].strip() + "\n"
    return text


def download_text(url: str, *, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "qfr-pipeline-corpus-downloader/0.1"})
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download approved remote corpus sources and emit a local ingestion manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/corpus_source_manifest.real_downloads.yaml")
    parser.add_argument("--out-manifest", type=Path, default=ROOT / "manifests/corpus_source_manifest.real_downloaded.local.yaml")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/corpus_ingestion/downloads.real.json")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even when the target path already exists.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--keep-gutenberg-boilerplate", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    downloaded_sources: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "ok": True,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "out_manifest": str(args.out_manifest.relative_to(ROOT) if args.out_manifest.is_absolute() and ROOT in args.out_manifest.parents else args.out_manifest),
        "sources_total": len(manifest.get("sources", [])),
        "downloaded": [],
        "skipped": [],
        "issues": [],
    }

    for source in manifest.get("sources", []):
        source_id = source.get("source_id", "<missing>")
        url = source.get("download_url")
        if not url:
            report["skipped"].append({"source_id": source_id, "reason": "missing_download_url"})
            continue
        target = repo_path(source["path"])
        try:
            if target.exists() and not args.overwrite:
                text = target.read_text(encoding="utf-8")
                status = "already_exists"
            else:
                text = download_text(url, timeout=args.timeout)
                if not args.keep_gutenberg_boilerplate and "gutenberg.org" in url:
                    text = strip_project_gutenberg_boilerplate(text)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                status = "downloaded"
            local_source = dict(source)
            local_source["source_type"] = "local_text"
            local_source["collection_method"] = "remote_text_download_materialized_locally"
            local_source["notes"] = f"{source.get('notes', '').strip()} Downloaded from {url}. SHA-256={sha256_text(text)}".strip()
            downloaded_sources.append(local_source)
            report["downloaded"].append(
                {
                    "source_id": source_id,
                    "status": status,
                    "path": str(target.relative_to(ROOT)),
                    "download_url": url,
                    "characters": len(text),
                    "sha256": sha256_text(text),
                }
            )
        except (OSError, urllib.error.URLError, ValueError) as exc:
            report["ok"] = False
            report["issues"].append({"source_id": source_id, "error": str(exc)})

    out_manifest_path = args.out_manifest if args.out_manifest.is_absolute() else ROOT / args.out_manifest
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(yaml.safe_dump(build_local_manifest(manifest, downloaded_sources), allow_unicode=True, sort_keys=False), encoding="utf-8")

    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
