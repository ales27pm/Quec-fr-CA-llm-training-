from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unicodedata

from qfr_pipeline.io import write_json
from qfr_pipeline.paths import repo_relative_path

LEVEL_ORDER = ["insufficient", "smoke_test", "pilot_lora_candidate", "production_lora_candidate"]


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_report(input_jsonl: Path, rows: list[dict]) -> dict:
    records_total = len(rows)
    chars = sum(len(r.get("text", "")) for r in rows)
    words = sum(len(r.get("text", "").split()) for r in rows)
    estimated_tokens = int(max(chars / 4, words * 1.35))

    source_balance = Counter(str(r.get("source_id", "unknown")) for r in rows)
    domain_balance = Counter(str(r.get("domain", "unknown")) for r in rows)
    register_balance = Counter(str(r.get("register", "unknown")) for r in rows)
    max_single_source_share = (max(source_balance.values()) / records_total) if records_total else 0.0

    exact_hashes = [r.get("text_sha256") or r.get("text", "") for r in rows]
    duplicates_exact = len(exact_hashes) - len(set(exact_hashes))
    normalized_texts = [_normalize_text(r.get("text", "")) for r in rows]
    duplicates_normalized = len(normalized_texts) - len(set(normalized_texts))

    instruction_like_count = sum(1 for r in rows if r.get("task_type") == "instruction" or any(k in r for k in ("messages", "user", "assistant")))
    dialog_like_count = sum(1 for r in rows if r.get("task_type") == "dialogue")
    holdout_risk = sum(1 for r in rows if bool(r.get("holdout_only")))

    legal_ok = sum(1 for r in rows if r.get("license_status") in {"open_compatible", "noncommercial_only"})
    commercial_ok = sum(1 for r in rows if r.get("commercial_use") == "allowed")

    level = "insufficient"
    if estimated_tokens >= 500_000:
        level = "smoke_test"
    if estimated_tokens >= 20_000_000:
        level = "pilot_lora_candidate"
    if estimated_tokens >= 150_000_000:
        level = "production_lora_candidate"

    blockers: list[str] = []
    if records_total == 0:
        blockers.append("no_records")
    if holdout_risk > 0:
        blockers.append("holdout_contamination_risk")
    if level == "production_lora_candidate":
        if instruction_like_count < 200_000:
            blockers.append("insufficient_instruction_turns")
        if max_single_source_share > 0.35:
            blockers.append("single_source_dominance")
        if len(domain_balance) < 5:
            blockers.append("low_domain_diversity")
        if len(register_balance) < 4:
            blockers.append("low_register_diversity")

    if level == "production_lora_candidate" and blockers:
        level = "production_blocked"

    return {
        "ok": True,
        "input": repo_relative_path(input_jsonl),
        "records_total": records_total,
        "estimated_tokens": estimated_tokens,
        "source_count": len(source_balance),
        "domain_balance": dict(sorted(domain_balance.items())),
        "register_balance": dict(sorted(register_balance.items())),
        "source_balance": dict(sorted(source_balance.items())),
        "max_single_source_share": round(max_single_source_share, 6),
        "instruction_like_count": instruction_like_count,
        "instruction_like_ratio": (instruction_like_count / records_total) if records_total else 0.0,
        "dialog_like_count": dialog_like_count,
        "dialog_like_ratio": (dialog_like_count / records_total) if records_total else 0.0,
        "legal_license_ok_ratio": (legal_ok / records_total) if records_total else 0.0,
        "commercial_ready_ratio": (commercial_ok / records_total) if records_total else 0.0,
        "holdout_contamination_risk_count": holdout_risk,
        "duplicates_exact": duplicates_exact,
        "duplicates_normalized": duplicates_normalized,
        "readiness_level": level,
        "blocking_reasons": sorted(blockers),
        "recommendations": [
            "Increase modern (2020-2026) institutional, educational, and instruction/dialogue coverage.",
            "Reduce source concentration and improve register/domain diversity for production readiness.",
        ],
    }


def audit_corpus_readiness(input_jsonl: Path, out_report: Path, policy_manifest: Path | None = None) -> dict:
    del policy_manifest
    if not input_jsonl.exists():
        report = {
            "ok": False,
            "input": repo_relative_path(input_jsonl),
            "records_total": 0,
            "estimated_tokens": 0,
            "source_count": 0,
            "domain_balance": {},
            "register_balance": {},
            "source_balance": {},
            "max_single_source_share": 0.0,
            "instruction_like_count": 0,
            "instruction_like_ratio": 0.0,
            "dialog_like_count": 0,
            "dialog_like_ratio": 0.0,
            "legal_license_ok_ratio": 0.0,
            "commercial_ready_ratio": 0.0,
            "holdout_contamination_risk_count": 0,
            "duplicates_exact": 0,
            "duplicates_normalized": 0,
            "readiness_level": "insufficient",
            "blocking_reasons": ["missing_input"],
            "recommendations": ["Provide an input JSONL corpus before readiness auditing."],
        }
        out_report.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_report, report)
        return report

    rows = _read_jsonl(input_jsonl)
    report = _build_report(input_jsonl, rows)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_report, report)
    return report
