from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from qfr_pipeline.io import load_yaml
from qfr_pipeline.paths import repo_relative_path
from qfr_pipeline.schemas import LPContextContrast, LPContextManifest


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


def stable_minimal_pair_id(lp_id: int, contrast_id: str, good: str, bad: str) -> str:
    digest = hashlib.sha1(f"{lp_id}|{contrast_id}|{good}|{bad}".encode("utf-8")).hexdigest()[:10]
    return f"lp{lp_id}-{digest}"


def expand_contrast_templates(contrast: LPContextContrast) -> list[tuple[str, str]]:
    if len(contrast.good_templates) != len(contrast.bad_templates):
        raise ValueError(
            f"Mismatched template counts for {contrast.contrast_id}: "
            f"{len(contrast.good_templates)} good vs {len(contrast.bad_templates)} bad"
        )
    return [
        (
            g_t.format(positive_pattern=contrast.positive_pattern, negative_pattern=contrast.negative_pattern),
            b_t.format(positive_pattern=contrast.positive_pattern, negative_pattern=contrast.negative_pattern),
        )
        for g_t, b_t in zip(contrast.good_templates, contrast.bad_templates)
    ]


def build_authorized_pair_index(context_manifest: LPContextManifest) -> dict[str, set[tuple[str, str]]]:
    return {contrast.contrast_id: set(expand_contrast_templates(contrast)) for contrast in context_manifest.contrasts}


def generate_minimal_pairs(rule_path: Path, context_path: Path) -> tuple[list[MinimalPairRecord], MinimalPairGenerationReport]:
    load_yaml(rule_path)
    contexts = load_context_manifest(context_path)
    records: list[MinimalPairRecord] = []
    for contrast in contexts.contrasts:
        for good, bad in expand_contrast_templates(contrast):
            rid = stable_minimal_pair_id(contexts.lp_id, contrast.contrast_id, good, bad)
            records.append(MinimalPairRecord(id=rid, lp_id=contexts.lp_id, phenomenon=contexts.name, contrast_id=contrast.contrast_id, good=good, bad=bad, expected="good", register=contrast.register_value, term_type=contrast.term_type, source_rule=repo_relative_path(rule_path), source_context=repo_relative_path(context_path), metadata={"dialect": "fr-CA", "normative": contrast.register_value == "formal", "allowed_contexts": contrast.allowed_contexts, "blocked_contexts": contrast.blocked_contexts, "quality_checks": [], "positive_pattern": contrast.positive_pattern, "negative_pattern": contrast.negative_pattern}))

    return records, MinimalPairGenerationReport(ok=True, records_generated=len(records), issues=[])


def write_jsonl(records: list[MinimalPairRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
