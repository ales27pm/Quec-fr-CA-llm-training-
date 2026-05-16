from __future__ import annotations

from collections.abc import Generator
import json
from pathlib import Path
import tempfile

import pytest
import yaml
from typer.testing import CliRunner

from qfr_pipeline.cli import app
from qfr_pipeline.paths import ROOT
from qfr_pipeline.training_pack import (
    audit_training_pack,
    build_training_pack,
    validate_training_pack_policy,
)


@pytest.fixture()
def repo_tmp_dir() -> Generator[Path, None, None]:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        yield Path(tmp)


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _base_row(text: str, source_id: str = "src_a", **overrides) -> dict:
    record_suffix = (
        json.dumps(text, ensure_ascii=False).encode("utf-8").hex()[:16]
    )
    payload = {
        "record_id": f"{source_id}:{record_suffix}",
        "source_id": source_id,
        "source_name": source_id,
        "text": text,
        "domain": "general",
        "register": "literary",
        "dialect_region": "Quebec",
        "license_status": "open_compatible",
        "commercial_use": "allowed",
        "allowed_for_training": True,
        "holdout_only": False,
        "requires_review": False,
        "quality_flags": [],
    }
    payload.update(overrides)
    return payload


def _policy_template(repo_tmp_dir: Path, input_rel_paths: list[str]) -> dict:
    policy = yaml.safe_load(
        (ROOT / "manifests/training_pack_policy.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    output_dir = _repo_rel(repo_tmp_dir / "pack")
    policy["output_dir"] = output_dir
    policy["instructionization"]["preserve_raw_text_probability"] = 1.0
    policy["instructionization"]["min_text_chars"] = 1
    policy["instructionization"]["max_text_chars"] = 4000
    policy["instructionization"]["max_examples_per_record"] = 10
    policy["balancing"]["max_single_source_share"] = 1.0
    policy["balancing"]["max_single_source_family_share"] = 1.0
    policy["balancing"]["min_domain_count_for_pilot"] = 1
    policy["balancing"]["min_register_count_for_pilot"] = 1
    policy["source_inputs"] = []
    for idx, input_rel in enumerate(input_rel_paths):
        policy["source_inputs"].append(
            {
                "path": input_rel,
                "source_family": f"family_{idx}",
                "source_priority": idx,
                "allowed_for_training_required": True,
                "max_source_share": 1.0,
                "include_if_exists": True,
                "required": True,
            }
        )
    return policy


def _write_policy(
    repo_tmp_dir: Path, payload: dict, name: str = "policy.yaml"
) -> Path:
    repo_tmp_dir.mkdir(parents=True, exist_ok=True)
    path = repo_tmp_dir / name
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _build_with_rows(
    repo_tmp_dir: Path,
    rows: list[dict],
    *,
    policy_overrides: dict | None = None,
) -> tuple[Path, dict]:
    input_path = repo_tmp_dir / "input.jsonl"
    _write_jsonl(input_path, rows)
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    if policy_overrides:
        for key, value in policy_overrides.items():
            policy[key] = value
    policy_path = _write_policy(repo_tmp_dir, policy)
    report = build_training_pack(policy_path)
    return policy_path, report


def test_valid_training_pack_policy_passes(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(
        input_path, [_base_row("Courriel et fin de semaine pour test.")]
    )
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy_path = _write_policy(repo_tmp_dir, policy)
    validate_training_pack_policy(policy_path)


def test_invalid_split_ratios_fail(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte valide pour ratios.")])
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy["split_ratios"] = {"train": 0.8, "dev": 0.2, "test": 0.2}
    policy_path = _write_policy(repo_tmp_dir, policy)
    with pytest.raises(Exception):
        validate_training_pack_policy(policy_path)


def test_absolute_source_path_fails(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte valide.")])
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy["source_inputs"][0]["path"] = str(input_path.resolve())
    policy_path = _write_policy(repo_tmp_dir, policy)
    with pytest.raises(Exception):
        validate_training_pack_policy(policy_path)


def test_holdout_record_is_rejected(repo_tmp_dir: Path) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir,
        [_base_row("Texte holdout.", holdout_only=True)],
    )
    assert report["holdout_rejected"] >= 1


def test_allowed_for_training_false_is_rejected(repo_tmp_dir: Path) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir,
        [_base_row("Texte non entraînable.", allowed_for_training=False)],
    )
    assert (
        report["rejection_reasons"].get("allowed_for_training_false", 0) >= 1
    )


def test_requires_review_true_without_permission_is_rejected(
    repo_tmp_dir: Path,
) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir,
        [
            _base_row(
                "Texte sous revue.",
                requires_review=True,
                commercial_use="permission_required",
            )
        ],
    )
    assert report["permission_rejected"] >= 1


def test_duplicate_exact_text_rejected(repo_tmp_dir: Path) -> None:
    rows = [
        _base_row("Même texte en double."),
        _base_row("Même texte en double.", source_id="src_b"),
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    assert report["duplicate_exact_rejected"] >= 1


def test_duplicate_normalized_text_rejected(repo_tmp_dir: Path) -> None:
    rows = [
        _base_row("Bonjour   Québec"),
        _base_row("bonjour québec", source_id="src_b"),
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    assert report["duplicate_normalized_rejected"] >= 1


def test_source_dominance_downsampling_works(repo_tmp_dir: Path) -> None:
    rows = [
        _base_row(f"Texte dominant {idx} avec courriel.", source_id="dominant")
        for idx in range(10)
    ] + [_base_row("Texte minoritaire avec courriel.", source_id="minor")]
    policy_path, report = _build_with_rows(
        repo_tmp_dir,
        rows,
        policy_overrides={
            "balancing": {
                "max_single_source_share": 0.5,
                "max_single_source_family_share": 1.0,
                "min_domain_count_for_pilot": 1,
                "min_register_count_for_pilot": 1,
                "target_register_mix": {},
                "target_domain_mix": {},
            }
        },
    )
    assert report["ok"]
    pack = json.loads(
        (ROOT / report["artifacts"]["report"]).read_text(encoding="utf-8")
    )
    assert pack["max_single_source_share"] <= 0.6
    validate_training_pack_policy(policy_path)


def test_instructionization_creates_qwen_chatml_messages(
    repo_tmp_dir: Path,
) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir, [_base_row("Courriel et fin de semaine en contexte.")]
    )
    train = _read_jsonl(ROOT / report["artifacts"]["train"])
    assert train
    assert "<|im_start|>" in train[0]["text"]
    assert "<|im_end|>" in train[0]["text"]


def test_preserve_raw_example_format_is_correct(repo_tmp_dir: Path) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir, [_base_row("Passage source à conserver tel quel.")]
    )
    rows = (
        _read_jsonl(ROOT / report["artifacts"]["train"])
        + _read_jsonl(ROOT / report["artifacts"]["dev"])
        + _read_jsonl(ROOT / report["artifacts"]["test"])
    )
    raw = next(
        (row for row in rows if row["task_type"] == "preserve_raw"), None
    )
    assert raw is not None
    assert raw["messages"][0]["role"] == "system"
    assert raw["messages"][1]["role"] == "user"
    assert raw["messages"][2]["role"] == "assistant"


def test_summarize_uses_deterministic_extractive_summary(
    repo_tmp_dir: Path,
) -> None:
    text = (
        "Première phrase claire. Deuxième phrase utile. "
        "Troisième phrase complémentaire."
    )
    _, report = _build_with_rows(repo_tmp_dir, [_base_row(text)])
    rows = (
        _read_jsonl(ROOT / report["artifacts"]["train"])
        + _read_jsonl(ROOT / report["artifacts"]["dev"])
        + _read_jsonl(ROOT / report["artifacts"]["test"])
    )
    summary = next(
        (row for row in rows if row["task_type"] == "summarize"), None
    )
    assert summary is not None
    assert "Première phrase claire." in summary["messages"][2]["content"]


def test_explain_term_only_emits_when_marker_exists(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row(
            "Le courriel est prêt pour la fin de semaine.", source_id="marker"
        ),
        _base_row(
            "Texte neutre sans marqueur explicite.", source_id="neutral"
        ),
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    output_rows = (
        _read_jsonl(ROOT / report["artifacts"]["train"])
        + _read_jsonl(ROOT / report["artifacts"]["dev"])
        + _read_jsonl(ROOT / report["artifacts"]["test"])
    )
    explain_rows = [
        row for row in output_rows if row["task_type"] == "explain_term"
    ]
    assert explain_rows
    assert all(
        "courriel" in row["messages"][1]["content"].casefold()
        for row in explain_rows
    )


def test_normalize_to_quebec_fr_applies_known_replacements_only(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row("Mon email parle de shopping et de parking."),
        _base_row("Texte sans termes à remplacer.", source_id="plain"),
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    output_rows = (
        _read_jsonl(ROOT / report["artifacts"]["train"])
        + _read_jsonl(ROOT / report["artifacts"]["dev"])
        + _read_jsonl(ROOT / report["artifacts"]["test"])
    )
    norm_rows = [
        row
        for row in output_rows
        if row["task_type"] == "normalize_to_quebec_fr"
    ]
    assert norm_rows
    assert any(
        "courriel" in row["messages"][2]["content"].casefold()
        for row in norm_rows
    )


def test_examples_split_deterministically(repo_tmp_dir: Path) -> None:
    rows = [
        _base_row(f"Texte split {idx} avec courriel.", source_id=f"s{idx % 2}")
        for idx in range(12)
    ]
    _, report_one = _build_with_rows(repo_tmp_dir / "a", rows)
    _, report_two = _build_with_rows(repo_tmp_dir / "b", rows)
    train_one = _read_jsonl(ROOT / report_one["artifacts"]["train"])
    train_two = _read_jsonl(ROOT / report_two["artifacts"]["train"])
    assert [row["example_id"] for row in train_one] == [
        row["example_id"] for row in train_two
    ]


def test_small_pack_populates_train_dev_test_when_total_ge_three(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row(
            "Texte court. Deuxième phrase. Troisième phrase.", source_id="solo"
        )
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    train_rows = _read_jsonl(ROOT / report["artifacts"]["train"])
    dev_rows = _read_jsonl(ROOT / report["artifacts"]["dev"])
    test_rows = _read_jsonl(ROOT / report["artifacts"]["test"])
    total = len(train_rows) + len(dev_rows) + len(test_rows)
    if total >= 3:
        assert train_rows
        assert dev_rows
        assert test_rows


def test_no_source_record_id_across_multiple_splits_when_enough_examples(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row(f"Texte multi-split {idx}.", source_id=f"src_{idx}")
        for idx in range(20)
    ]
    _, report = _build_with_rows(repo_tmp_dir, rows)
    train_ids = {
        row["source_record_id"]
        for row in _read_jsonl(ROOT / report["artifacts"]["train"])
    }
    dev_ids = {
        row["source_record_id"]
        for row in _read_jsonl(ROOT / report["artifacts"]["dev"])
    }
    test_ids = {
        row["source_record_id"]
        for row in _read_jsonl(ROOT / report["artifacts"]["test"])
    }
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)


def test_report_paths_are_repo_relative(repo_tmp_dir: Path) -> None:
    _, report = _build_with_rows(repo_tmp_dir, [_base_row("Texte relatif.")])
    assert not Path(report["output_dir"]).is_absolute()
    assert all(
        not Path(path).is_absolute() for path in report["artifacts"].values()
    )


def test_dataset_card_generated(repo_tmp_dir: Path) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir, [_base_row("Texte dataset card.")]
    )
    card_path = ROOT / report["artifacts"]["dataset_card"]
    assert card_path.exists()
    assert "Dataset Card" in card_path.read_text(encoding="utf-8")


def test_cli_validate_training_pack_policy_works(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte CLI validate.")])
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy_path = _write_policy(repo_tmp_dir, policy)
    runner = CliRunner()
    result = runner.invoke(
        app, ["validate-training-pack-policy", "--policy", str(policy_path)]
    )
    assert result.exit_code == 0


def test_local_real_training_pack_policy_validates() -> None:
    validate_training_pack_policy(
        ROOT / "manifests/training_pack_policy.local_real.template.yaml"
    )


def test_local_research_mode_includes_noncommercial_but_not_commercial_ready(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row(
            "Texte institutionnel noncommercial pour validation.",
            source_id="assnat_seed",
            license_status="noncommercial_only",
            commercial_use="permission_required",
            requires_review=False,
        )
    ]
    _, report = _build_with_rows(
        repo_tmp_dir,
        rows,
        policy_overrides={"pack_mode": "local_research"},
    )
    assert report["records_accepted"] >= 1
    assert report["noncommercial_records_count"] >= 1
    assert report["commercial_release_ready"] is False
    assert (
        "pack_mode_local_research_blocks_commercial_release"
        in report["commercial_blocking_reasons"]
    )


def test_production_commercial_mode_rejects_noncommercial(
    repo_tmp_dir: Path,
) -> None:
    rows = [
        _base_row(
            "Texte institutionnel noncommercial à exclure.",
            source_id="assnat_seed",
            license_status="noncommercial_only",
            commercial_use="permission_required",
            requires_review=False,
        )
    ]
    _, report = _build_with_rows(
        repo_tmp_dir,
        rows,
        policy_overrides={"pack_mode": "production_commercial"},
    )
    assert report["records_accepted"] == 0
    assert report["rejection_reasons"].get("commercial_mode_reject", 0) >= 1


def test_optional_missing_inputs_are_reported(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte principal.")])
    missing_path = _repo_rel(repo_tmp_dir / "missing.jsonl")
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy["source_inputs"].append(
        {
            "path": missing_path,
            "source_family": "missing_optional",
            "source_priority": 999,
            "allowed_for_training_required": True,
            "max_source_share": 1.0,
            "include_if_exists": True,
            "required": False,
        }
    )
    policy_path = _write_policy(repo_tmp_dir, policy)
    report = build_training_pack(policy_path)
    assert report["ok"] is True
    assert missing_path in report["input_files_missing_optional"]


def test_cli_validate_training_pack_policy_local_real_works() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate-training-pack-policy",
            "--policy",
            "manifests/training_pack_policy.local_real.template.yaml",
        ],
    )
    assert result.exit_code == 0


def test_cli_build_training_pack_with_live_like_input_has_examples(
    repo_tmp_dir: Path,
) -> None:
    input_path = repo_tmp_dir / "live_like.jsonl"
    _write_jsonl(
        input_path,
        [
            _base_row(
                "Texte institutionnel pour génération d'exemples en mode local research."
            )
        ],
    )
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy["pack_mode"] = "local_research"
    policy_path = _write_policy(repo_tmp_dir, policy, name="policy_local.yaml")
    out_dir = repo_tmp_dir / "pack_live_like"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-training-pack",
            "--policy",
            str(policy_path),
            "--out-dir",
            _repo_rel(out_dir),
        ],
    )
    assert result.exit_code == 0
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["examples_generated"] > 0


def test_cli_build_training_pack_works(repo_tmp_dir: Path) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte CLI build.")])
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy_path = _write_policy(repo_tmp_dir, policy)
    out_dir = repo_tmp_dir / "pack_cli"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-training-pack",
            "--policy",
            str(policy_path),
            "--out-dir",
            _repo_rel(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "report.json").exists()


def test_cli_audit_training_pack_fail_below_production_on_fixture_scale(
    repo_tmp_dir: Path,
) -> None:
    input_path = repo_tmp_dir / "in.jsonl"
    _write_jsonl(input_path, [_base_row("Texte audit CLI.")])
    policy = _policy_template(repo_tmp_dir, [_repo_rel(input_path)])
    policy_path = _write_policy(repo_tmp_dir, policy)
    out_dir = repo_tmp_dir / "pack_cli"
    build_training_pack(policy_path, out_dir=out_dir)
    audit_out = repo_tmp_dir / "audit.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "audit-training-pack",
            "--pack-dir",
            _repo_rel(out_dir),
            "--out",
            str(audit_out),
            "--fail-below",
            "production_lora_candidate",
        ],
    )
    assert result.exit_code != 0


def test_release_candidate_includes_training_pack_fields(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    out_json = tmp_path / "rc.json"
    out_md = tmp_path / "rc.md"
    result = runner.invoke(
        app,
        [
            "release-candidate",
            "--metrics",
            "fixtures/valid_metrics.json",
            "--diagnostics-input",
            "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["training_export_summary"]
    for key in [
        "training_pack_readiness_level",
        "training_pack_examples_total",
        "training_pack_train_count",
        "training_pack_dev_count",
        "training_pack_test_count",
        "training_pack_estimated_tokens",
        "training_pack_blocking_reasons",
    ]:
        assert key in summary


def test_no_workspace_or_runner_leakage_in_generated_reports(
    repo_tmp_dir: Path,
) -> None:
    _, report = _build_with_rows(
        repo_tmp_dir, [_base_row("Texte fuite path.")]
    )
    audit_out = repo_tmp_dir / "audit.json"
    audit_training_pack(ROOT / report["output_dir"], audit_out)
    combined = (
        (ROOT / report["artifacts"]["report"]).read_text(encoding="utf-8")
        + (ROOT / report["artifacts"]["dataset_card"]).read_text(
            encoding="utf-8"
        )
        + audit_out.read_text(encoding="utf-8")
    )
    assert "/workspace" not in combined
    assert "/home/runner" not in combined
