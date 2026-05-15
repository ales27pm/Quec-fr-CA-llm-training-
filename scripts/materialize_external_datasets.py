#!/usr/bin/env python3
"""Materialize approved external datasets into normalized JSONL.

This script is intentionally policy-first:
- benchmarks stay holdout-only unless explicitly materialized for evaluation;
- review-required sources are skipped unless explicitly included;
- training materialization is blocked unless the catalog marks a source as training-allowed;
- catalog-only, speech, model, code, and web/forum sources are reported but not scraped.

Supported adapters:
- remote_text: direct UTF-8 text download to normalized JSONL;
- huggingface_dataset: Hugging Face Datasets rows to normalized JSONL;
- catalog_only: report-only placeholder for sources requiring a future adapter or legal review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
DEFAULT_REPORT = ROOT / "reports" / "external_datasets" / "materialization_report.json"


@dataclass(frozen=True)
class SourceDecision:
    allowed: bool
    reason: str


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        raise ValueError(f"Path must be repo-relative: {path}")
    resolved = (ROOT / p).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes repository root: {path}")
    return resolved


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start and end and start.end() < end.start():
        return text[start.end() : end.start()].strip() + "\n"
    return text


def download_text(url: str, *, timeout: int) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "qfr-external-dataset-materializer/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace")


def normalize_ws(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def paragraph_records(text: str, *, min_chars: int) -> list[str]:
    chunks = [normalize_ws(p) for p in re.split(r"\n\s*\n+", text)]
    return [p for p in chunks if len(p) >= min_chars]


def recursive_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(recursive_strings(v))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(recursive_strings(v))
        return out
    return []


def extract_text(row: dict[str, Any], fields: list[str], *, min_chars: int) -> str | None:
    values: list[str] = []
    if fields:
        for field in fields:
            value = row.get(field)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, (list, dict)):
                values.extend(recursive_strings(value))
    else:
        # Conservative default: concatenate textual leaves, longest first.
        values = recursive_strings(row)
    cleaned = [normalize_ws(v) for v in values if isinstance(v, str) and len(normalize_ws(v)) >= min_chars]
    if not cleaned:
        return None
    cleaned = sorted(set(cleaned), key=lambda s: (-len(s), s))
    return "\n".join(cleaned[:4]).strip()


def jsonl_record(source: dict[str, Any], text: str, index: int, *, split: str | None = None) -> dict[str, Any]:
    return {
        "record_id": f"{source['source_id']}:{index:08d}",
        "source_id": source["source_id"],
        "source_name": source.get("name"),
        "text": text,
        "text_sha256": sha256_text(text),
        "language": "fr-CA",
        "dialect_region": source.get("dialect_region", "unknown"),
        "register": source.get("register", "unknown"),
        "modality": source.get("modality", "text"),
        "category": source.get("category"),
        "adapter": source.get("adapter"),
        "split": split,
        "allowed_for_training": bool(source.get("allowed_for_training", False)),
        "allowed_for_evaluation": bool(source.get("allowed_for_evaluation", False)),
        "holdout_only": bool(source.get("holdout_only", False)),
        "requires_review": bool(source.get("requires_review", True)),
        "license_reviewed": bool(source.get("license_reviewed", False)),
        "license_status": source.get("license_status"),
        "license_url": source.get("license_url"),
        "notes": source.get("notes"),
    }


def source_decision(
    source: dict[str, Any],
    *,
    mode: str,
    include_review_required: bool,
    include_unreviewed_license: bool,
) -> SourceDecision:
    adapter = source.get("adapter")
    modality = source.get("modality", "text")
    if adapter == "catalog_only":
        return SourceDecision(False, "catalog_only_requires_future_adapter_or_review")
    if modality != "text":
        return SourceDecision(False, f"unsupported_modality_{modality}")
    if source.get("requires_review") and not include_review_required:
        return SourceDecision(False, "requires_review")
    if (not source.get("license_reviewed", False)) and (not include_unreviewed_license):
        return SourceDecision(False, "license_not_reviewed")
    if mode == "training":
        if not source.get("allowed_for_training", False):
            return SourceDecision(False, "not_allowed_for_training")
        if source.get("holdout_only") or source.get("allowed_for_evaluation") or "benchmark" in str(source.get("category", "")):
            return SourceDecision(False, "benchmark_or_holdout_blocked_from_training")
    elif mode == "evaluation":
        if not source.get("allowed_for_evaluation", False):
            return SourceDecision(False, "not_allowed_for_evaluation")
    elif mode == "all-safe":
        if source.get("holdout_only"):
            return SourceDecision(False, "holdout_excluded_from_all_safe")
        if not source.get("allowed_for_training", False):
            return SourceDecision(False, "all_safe_only_materializes_training_allowed_sources")
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return SourceDecision(True, "selected")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    tmp.replace(path)
    return count


def materialize_remote_text(source: dict[str, Any], *, min_chars: int, timeout: int, keep_boilerplate: bool, max_records: int | None) -> dict[str, Any]:
    url = source.get("url") or source.get("download_url")
    if not url:
        raise ValueError("remote_text source missing url/download_url")
    text = download_text(url, timeout=timeout)
    if "gutenberg.org" in url and not keep_boilerplate:
        text = strip_gutenberg_boilerplate(text)
    paragraphs = paragraph_records(text, min_chars=min_chars)
    if max_records is not None:
        paragraphs = paragraphs[:max_records]
    out_path = repo_path(source["output_path"])
    records = [jsonl_record(source, p, i) for i, p in enumerate(paragraphs, start=1)]
    written = atomic_write_jsonl(out_path, records)
    return {"records_written": written, "output_path": str(out_path.relative_to(ROOT)), "source_sha256": sha256_text(text)}


def materialize_huggingface(source: dict[str, Any], *, min_chars: int, max_records: int | None) -> dict[str, Any]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError("Install optional requirements first: pip install -r requirements/dataset-adapters.txt") from exc

    dataset_id = source.get("hf_dataset_id")
    if not dataset_id:
        raise ValueError("huggingface_dataset source missing hf_dataset_id")
    config = source.get("config")
    split = source.get("split") or "train"
    if not config:
        try:
            configs = get_dataset_config_names(dataset_id)
            if len(configs) == 1:
                config = configs[0]
        except Exception:
            config = None
    dataset = load_dataset(dataset_id, config, split=split, streaming=False) if config else load_dataset(dataset_id, split=split, streaming=False)
    fields = source.get("text_fields") or []
    out_path = repo_path(source["output_path"])

    def iter_records() -> Iterable[dict[str, Any]]:
        emitted = 0
        for row in dataset:
            if max_records is not None and emitted >= max_records:
                break
            if not isinstance(row, dict):
                continue
            text = extract_text(row, list(fields), min_chars=min_chars)
            if not text:
                continue
            emitted += 1
            yield jsonl_record(source, text, emitted, split=split)

    written = atomic_write_jsonl(out_path, iter_records())
    return {"records_written": written, "output_path": str(out_path.relative_to(ROOT)), "split": split, "config": config}


def load_catalog(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("kind") != "external_dataset_catalog":
        raise ValueError("Catalog must be kind: external_dataset_catalog")
    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Catalog must include a non-empty sources list")
    ids = [s.get("source_id") for s in sources if isinstance(s, dict)]
    if len(ids) != len(set(ids)):
        raise ValueError("source_id entries must be unique")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be a mapping")
        for key in ("source_id", "name", "category", "modality", "adapter", "output_path"):
            if not str(source.get(key, "")).strip():
                raise ValueError(f"Source missing required key {key}: {source}")
        if source.get("allowed_for_training") and source.get("holdout_only"):
            raise ValueError(f"Training source cannot be holdout_only: {source['source_id']}")
        if source.get("allowed_for_training") and source.get("allowed_for_evaluation"):
            raise ValueError(f"Source cannot be both training and evaluation by default: {source['source_id']}")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize policy-approved external datasets into normalized JSONL.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "project/external_dataset_catalog.yaml")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mode", choices=["training", "evaluation", "all-safe"], default="all-safe")
    parser.add_argument("--source-id", action="append", default=[], help="Restrict to one or more source IDs.")
    parser.add_argument("--include-review-required", action="store_true")
    parser.add_argument("--include-unreviewed-license", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-gutenberg-boilerplate", action="store_true")
    args = parser.parse_args()

    catalog_path = args.catalog if args.catalog.is_absolute() else ROOT / args.catalog
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    started = time.time()
    report: dict[str, Any] = {
        "ok": True,
        "catalog": str(catalog_path.relative_to(ROOT)),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "sources_total": 0,
        "sources_selected": 0,
        "sources_materialized": 0,
        "records_written": 0,
        "materialized": [],
        "skipped": [],
        "issues": [],
    }

    try:
        catalog = load_catalog(catalog_path)
        sources = catalog["sources"]
        if args.source_id:
            wanted = set(args.source_id)
            sources = [s for s in sources if s.get("source_id") in wanted]
            missing = wanted - {s.get("source_id") for s in sources}
            if missing:
                raise ValueError(f"Unknown source IDs: {sorted(missing)}")
        report["sources_total"] = len(sources)
        for source in sources:
            decision = source_decision(
                source,
                mode=args.mode,
                include_review_required=args.include_review_required,
                include_unreviewed_license=args.include_unreviewed_license,
            )
            if not decision.allowed:
                report["skipped"].append({"source_id": source["source_id"], "reason": decision.reason})
                continue
            report["sources_selected"] += 1
            if args.dry_run:
                report["materialized"].append({"source_id": source["source_id"], "status": "dry_run_selected"})
                continue
            try:
                adapter = source["adapter"]
                if adapter == "remote_text":
                    result = materialize_remote_text(
                        source,
                        min_chars=args.min_chars,
                        timeout=args.timeout,
                        keep_boilerplate=args.keep_gutenberg_boilerplate,
                        max_records=args.max_records,
                    )
                elif adapter == "huggingface_dataset":
                    result = materialize_huggingface(source, min_chars=args.min_chars, max_records=args.max_records)
                else:
                    raise ValueError(f"Unsupported materializing adapter: {adapter}")
                report["sources_materialized"] += 1
                report["records_written"] += int(result.get("records_written", 0))
                report["materialized"].append({"source_id": source["source_id"], **result})
            except Exception as exc:
                report["ok"] = False
                report["issues"].append({"source_id": source["source_id"], "error": str(exc)})
    except Exception as exc:
        report["ok"] = False
        report["issues"].append({"source_id": None, "error": str(exc)})

    report["duration_seconds"] = round(time.time() - started, 3)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
