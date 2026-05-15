from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qfr_pipeline.io import load_yaml
from qfr_pipeline.paths import repo_relative_path
from qfr_pipeline.schemas import SplitPolicyManifest


@dataclass
class CuratedSplitReport:
    ok: bool
    policy: str
    input: str
    records_total: int
    train: int
    dev: int
    test: int
    seed: int
    forbidden_records_seen: int
    issues: list[str]
    outputs: dict[str, str]


def load_split_policy_manifest(path: Path) -> SplitPolicyManifest:
    return SplitPolicyManifest.model_validate(load_yaml(path))


def validate_split_policy_manifest(path: Path) -> SplitPolicyManifest:
    return load_split_policy_manifest(path)


def _validate_input_records(records: list[dict[str, Any]], policy: SplitPolicyManifest) -> tuple[list[str], int]:
    issues: list[str] = []
    forbidden = set(policy.source_requirements.forbid_labels)
    forbidden_seen = 0
    for idx, record in enumerate(records):
        label = record.get("curation_label")
        if label in forbidden:
            forbidden_seen += 1
        if label != policy.input_label_required:
            issues.append(f"record[{idx}] has curation_label={label!r}; expected {policy.input_label_required!r}")
        if policy.source_requirements.require_policy_id and not record.get("policy_id"):
            issues.append(f"record[{idx}] missing policy_id")
        if policy.source_requirements.require_curation_score and (record.get("curation_score") is None):
            issues.append(f"record[{idx}] missing curation_score")
        if policy.source_requirements.require_curation_reasons and not record.get("curation_reasons"):
            issues.append(f"record[{idx}] missing curation_reasons")
    return issues, forbidden_seen


def split_curated_corpus(input_jsonl: Path, policy_path: Path, out_dir: Path) -> CuratedSplitReport:
    policy = validate_split_policy_manifest(policy_path)
    records = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    issues, forbidden_seen = _validate_input_records(records, policy)

    total = len(records)
    if total < policy.min_records:
        issues.append(f"records_total {total} below min_records {policy.min_records}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    if not issues:
        shuffled = list(records)
        random.Random(policy.seed).shuffle(shuffled)
        train_n = int(math.floor(total * policy.split_ratios.train))
        dev_n = int(math.floor(total * policy.split_ratios.dev))
        test_n = total - train_n - dev_n
        if total > 0 and train_n == 0:
            train_n = 1
            if dev_n > 0:
                dev_n -= 1
            elif test_n > 0:
                test_n -= 1
        if (dev_n == 0 or test_n == 0) and total >= 3 and not policy.allow_empty_dev_test_for_small_fixture:
            issues.append("dev/test split cannot be empty when allow_empty_dev_test_for_small_fixture=false")

        train_records = shuffled[:train_n]
        dev_records = shuffled[train_n:train_n + dev_n]
        test_records = shuffled[train_n + dev_n:]

    outputs = {
        "train": repo_relative_path(out_dir / "train.jsonl"),
        "dev": repo_relative_path(out_dir / "dev.jsonl"),
        "test": repo_relative_path(out_dir / "test.jsonl"),
        "report": repo_relative_path(out_dir / "split_report.json"),
    }

    for path, payload in ((out_dir / "train.jsonl", train_records), (out_dir / "dev.jsonl", dev_records), (out_dir / "test.jsonl", test_records)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload), encoding="utf-8")

    report = CuratedSplitReport(
        ok=(len(issues) == 0 and forbidden_seen == 0),
        policy=repo_relative_path(policy_path),
        input=repo_relative_path(input_jsonl),
        records_total=total,
        train=len(train_records),
        dev=len(dev_records),
        test=len(test_records),
        seed=policy.seed,
        forbidden_records_seen=forbidden_seen,
        issues=issues,
        outputs=outputs,
    )
    write_curated_split_report(report, out_dir / "split_report.json")
    return report


def write_curated_split_report(report: CuratedSplitReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
