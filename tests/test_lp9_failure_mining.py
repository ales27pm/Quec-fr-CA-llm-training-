from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml
import pytest

from qfr_pipeline.lp9_failure_mining import generate_lp9_failure_pack, validate_lp9_failure_mining_policy
from qfr_pipeline.paths import ROOT


def _write_policy(tmp_root: Path, source_report_path: str, source_generations_path: str) -> Path:
    policy = {
        "kind": "lp9_failure_mining_policy",
        "schema_version": "1.0.0",
        "primary_language": "fr-CA",
        "source_eval_report": source_report_path,
        "source_generations": source_generations_path,
        "output_dir": "reports/lp9_failure_pack",
        "random_seed": 42,
        "repetitions_per_failure": 2,
        "min_failures_per_pair": 1,
        "include_adapter_under_base": True,
        "include_missing_preferred": True,
        "include_contains_forbidden": True,
        "task_type_weights": {
            "rewrite": 1.0,
            "direct_preference": 1.0,
            "choose_best_term": 1.0,
            "correction": 1.0,
            "open_generation": 1.0,
        },
        "lexical_pair_overrides": {},
        "max_examples": 0,
        "train_ratio": 0.8,
    }
    policy_path = tmp_root / "lp9_failure_mining_policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return policy_path


def test_generate_lp9_failure_pack_outputs_unique_training_examples() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        source_eval_path = tmp_root / "lp9_eval_report.json"
        source_generations_path = tmp_root / "lp9_generations.jsonl"
        report = {
            "failures": [
                {
                    "prompt_id": "test-prompt-1",
                    "lexical_pair_id": "01:email",
                    "task_type": "choose_best_term",
                    "base_score": 1,
                    "adapter_score": -1,
                    "reasons": [
                        "adapter_under_base",
                        "adapter_contains_forbidden",
                        "adapter_missing_preferred",
                    ],
                },
                {
                    "prompt_id": "test-prompt-2",
                    "lexical_pair_id": "02:parking",
                    "task_type": "rewrite",
                    "base_score": 0,
                    "adapter_score": 0,
                    "reasons": ["adapter_missing_preferred"],
                },
            ]
        }
        source_eval_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        generation_rows = [
            {
                "prompt_id": "test-prompt-1",
                "forbidden_terms": ["email"],
                "expected_terms": ["courriel"],
            },
            {
                "prompt_id": "test-prompt-2",
                "forbidden_terms": ["parking"],
                "expected_terms": ["stationnement"],
            },
        ]
        source_generations_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generation_rows),
            encoding="utf-8",
        )
        policy_path = _write_policy(
            tmp_root,
            source_report_path=source_eval_path.relative_to(ROOT).as_posix(),
            source_generations_path=source_generations_path.relative_to(ROOT).as_posix(),
        )
        out_dir = tmp_root / "lp9_failure_pack"
        payload = generate_lp9_failure_pack(policy_path, out_dir=out_dir)

        assert payload["ok"]
        assert payload["examples_generated"] == payload["train_count"] + payload["dev_count"]
        train_rows = [
            json.loads(line)
            for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dev_rows = [
            json.loads(line)
            for line in (out_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        all_rows = train_rows + dev_rows
        assert all_rows
        assert len({row["text"] for row in all_rows}) == len(all_rows)
        assert (out_dir / "report.json").exists()


def test_validate_lp9_failure_mining_policy_requires_repo_relative_paths() -> None:
    policy = {
        "kind": "lp9_failure_mining_policy",
        "schema_version": "1.0.0",
        "primary_language": "fr-CA",
        "source_eval_report": "/absolute/path/report.json",
        "source_generations": "./relative/path/generations.jsonl",
        "output_dir": "reports/lp9_failure_pack",
        "random_seed": 42,
        "repetitions_per_failure": 1,
        "min_failures_per_pair": 1,
        "include_adapter_under_base": True,
        "include_missing_preferred": True,
        "include_contains_forbidden": True,
        "task_type_weights": {
            "rewrite": 1.0,
            "direct_preference": 1.0,
            "choose_best_term": 1.0,
            "correction": 1.0,
            "open_generation": 1.0,
        },
        "lexical_pair_overrides": {},
        "max_examples": 0,
        "train_ratio": 0.8,
    }
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        policy_path = Path(tmp) / "bad_policy.yaml"
        policy_path.write_text(yaml.safe_dump(policy, sort_keys=False, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_lp9_failure_mining_policy(policy_path)
