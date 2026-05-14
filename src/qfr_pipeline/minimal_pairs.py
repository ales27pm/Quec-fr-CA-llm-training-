from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from qfr_pipeline.io import load_yaml
from qfr_pipeline.minimal_pair_quality import validate_minimal_pairs
from qfr_pipeline.schemas import LPContextManifest


@dataclass
class MinimalPairRecord:
    id: str
    lp_id: int
    phenomenon: str
    contrast_id: str
    good: str
    bad: str
    expected: str
    register: str
    term_type: str
    source_rule: str
    source_context: str
    metadata: dict


@dataclass
class MinimalPairGenerationIssue:
    code: str
    message: str
    record_id: str | None = None
    blocking: bool = True


@dataclass
class MinimalPairGenerationReport:
    ok: bool
    records_generated: int
    issues: list[MinimalPairGenerationIssue]


def load_context_manifest(path: Path) -> LPContextManifest:
    return LPContextManifest.model_validate(load_yaml(path))


def _stable_id(lp_id: int, contrast_id: str, good: str, bad: str) -> str:
    digest = hashlib.sha1(f"{lp_id}|{contrast_id}|{good}|{bad}".encode("utf-8")).hexdigest()[:10]
    return f"lp{lp_id}-{digest}"


def generate_minimal_pairs(rule_path: Path, context_path: Path) -> tuple[list[MinimalPairRecord], MinimalPairGenerationReport]:
    load_yaml(rule_path)
    contexts = load_context_manifest(context_path)
    records: list[MinimalPairRecord] = []
    for contrast in contexts.contrasts:
        for g_t, b_t in zip(contrast.good_templates, contrast.bad_templates):
            good = g_t.format(positive_pattern=contrast.positive_pattern, negative_pattern=contrast.negative_pattern)
            bad = b_t.format(positive_pattern=contrast.positive_pattern, negative_pattern=contrast.negative_pattern)
            rid = _stable_id(contexts.lp_id, contrast.contrast_id, good, bad)
            records.append(
                MinimalPairRecord(
                    id=rid,
                    lp_id=contexts.lp_id,
                    phenomenon=contexts.name,
                    contrast_id=contrast.contrast_id,
                    good=good,
                    bad=bad,
                    expected="good",
                    register=contrast.register,
                    term_type=contrast.term_type,
                    source_rule=str(rule_path),
                    source_context=str(context_path),
                    metadata={
                        "dialect": "fr-CA",
                        "normative": contrast.register == "formal",
                        "quality_checks": [],
                        "positive_pattern": contrast.positive_pattern,
                        "negative_pattern": contrast.negative_pattern,
                    },
                )
            )

    quality = validate_minimal_pairs([asdict(r) for r in records])
    issues = [MinimalPairGenerationIssue(code=i.code, message=i.message, record_id=i.record_id, blocking=i.blocking) for i in quality.issues]
    for rec in records:
        rec.metadata["quality_checks"] = [i.code for i in issues if i.record_id == rec.id]
    return records, MinimalPairGenerationReport(ok=quality.ok, records_generated=len(records), issues=issues)


def write_jsonl(records: list[MinimalPairRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
