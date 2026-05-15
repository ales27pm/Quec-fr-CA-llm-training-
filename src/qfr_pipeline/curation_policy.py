from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re

from qfr_pipeline.io import load_yaml
from qfr_pipeline.paths import repo_relative_path
from qfr_pipeline.schemas import CurationPolicyManifest


@dataclass
class CurationAssessment:
    score: float
    label: str
    reasons: list[str]


@dataclass
class CurationRunReport:
    ok: bool
    policy: str
    input: str
    records_total: int
    accepted: int
    review_required: int
    quarantine: int
    rejected: int
    reason_counts: dict[str, int]
    outputs: dict[str, str]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""), encoding="utf-8")


def _norm_text(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", text.casefold()).split())


def load_curation_policy_manifest(path: Path) -> CurationPolicyManifest:
    return CurationPolicyManifest.model_validate(load_yaml(path))


def validate_curation_policy_manifest(path: Path) -> CurationPolicyManifest:
    return load_curation_policy_manifest(path)


def score_record(record: dict, policy: CurationPolicyManifest, seen_texts: set[str] | None = None) -> CurationAssessment:
    text = str(record.get("text", ""))
    normalized = _norm_text(text)
    score = float(policy.scoring.base_score)
    reasons: list[str] = []
    marker_hits = 0
    for marker, weight in policy.scoring.marker_weights.items():
        if marker.casefold() in normalized:
            marker_hits += 1
            score += float(weight)
            reasons.append(f"marker:{marker}")
    for penalty_term, penalty_val in policy.scoring.penalties.items():
        if penalty_term in {"excessive_uppercase_ratio", "too_short", "duplicate_text", "low_fr_ca_marker_score"}:
            continue
        if penalty_term.casefold() in normalized:
            score -= float(penalty_val)
            reasons.append(f"penalty:{penalty_term}")

    if len(text.strip()) < 20:
        score -= float(policy.scoring.penalties.get("too_short", 0.0))
        reasons.append("penalty:too_short")
    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    if upper_ratio > 0.45:
        score -= float(policy.scoring.penalties.get("excessive_uppercase_ratio", 0.0))
        reasons.append("penalty:excessive_uppercase_ratio")
    if marker_hits == 0:
        score -= float(policy.scoring.penalties.get("low_fr_ca_marker_score", 0.0))
        reasons.append("penalty:low_fr_ca_marker_score")
    if seen_texts is not None:
        if normalized in seen_texts:
            score -= float(policy.scoring.penalties.get("duplicate_text", 0.0))
            reasons.append("penalty:duplicate_text")
        else:
            seen_texts.add(normalized)

    for rule in policy.scoring.blocking_rules:
        if rule.pattern.casefold() in normalized:
            reasons.append(f"blocking:{rule.rule_id}")
    for rule in policy.scoring.review_rules:
        if rule.pattern.casefold() in normalized:
            reasons.append(f"review:{rule.rule_id}")
    for rule in policy.scoring.quarantine_rules:
        if rule.pattern.casefold() in normalized:
            reasons.append(f"quarantine:{rule.rule_id}")

    t = policy.scoring.thresholds
    label = "rejected"
    if any(x.startswith("blocking:") for x in reasons) or score < t.reject_below_score:
        label = "rejected"
    elif any(x.startswith("quarantine:") for x in reasons) or score < t.quarantine_below_score:
        label = "quarantine"
    elif any(x.startswith("review:") for x in reasons) or score < t.review_min_score:
        label = "review_required"
    elif score >= t.accept_min_score:
        label = "accepted"
    else:
        label = "review_required"

    if label != "accepted" and not reasons:
        reasons.append("policy:non_accepted_without_specific_reason")
    return CurationAssessment(score=round(score, 6), label=label, reasons=reasons)


def curate_ingested_corpus(input_jsonl: Path, policy_path: Path, out_dir: Path) -> CurationRunReport:
    policy = validate_curation_policy_manifest(policy_path)
    records = _read_jsonl(input_jsonl)
    seen: set[str] = set()
    buckets: dict[str, list[dict]] = {"accepted": [], "review_required": [], "quarantine": [], "rejected": []}
    reasons_counter: Counter[str] = Counter()
    for rec in records:
        assessment = score_record(rec, policy, seen)
        enriched = dict(rec)
        enriched["curation_score"] = assessment.score
        enriched["curation_label"] = assessment.label
        enriched["curation_reasons"] = assessment.reasons
        enriched["policy_id"] = policy.policy_id
        buckets[assessment.label].append(enriched)
        reasons_counter.update(assessment.reasons)

    outputs = {k: out_dir / f"{k}.jsonl" for k in buckets}
    for k, path in outputs.items():
        _write_jsonl(path, buckets[k])
    report_path = out_dir / "report.json"
    report = CurationRunReport(
        ok=sum(len(v) for v in buckets.values()) == len(records),
        policy=repo_relative_path(policy_path),
        input=repo_relative_path(input_jsonl),
        records_total=len(records),
        accepted=len(buckets["accepted"]),
        review_required=len(buckets["review_required"]),
        quarantine=len(buckets["quarantine"]),
        rejected=len(buckets["rejected"]),
        reason_counts=dict(sorted(reasons_counter.items())),
        outputs={k: repo_relative_path(v) for k, v in outputs.items()} | {"report": repo_relative_path(report_path)},
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
