from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import yaml

from qfr_pipeline.lp9_micro_pack import (
    generate_lp9_micro_pack,
    load_lp9_micro_pack_rows,
    validate_lp9_micro_pack_manifest,
)
from qfr_pipeline.paths import ROOT

TRAINING_SCRIPT_PATH = ROOT / "scripts" / "train_qfr_dolphin3_unsloth_lora.py"
spec = importlib.util.spec_from_file_location("qfr_training_inputs_lp9", TRAINING_SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load script module from {TRAINING_SCRIPT_PATH}")
training_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_inputs)


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_manifest_with_output_dir(tmp_root: Path, output_dir_rel: str) -> Path:
    manifest = yaml.safe_load(
        (ROOT / "manifests/lp9_lexical_preference_pack.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest["output_dir"] = output_dir_rel
    manifest_path = tmp_root / "lp9_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest_path


def test_lp9_manifest_validates() -> None:
    validate_lp9_micro_pack_manifest(
        ROOT / "manifests/lp9_lexical_preference_pack.template.yaml"
    )


def test_lp9_micro_pack_generation_no_duplicates_and_disjoint_splits() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        out_dir = tmp_root / "lp9_pack"
        manifest_path = _write_manifest_with_output_dir(tmp_root, _repo_rel(out_dir))

        report = generate_lp9_micro_pack(manifest_path)
        train_rows, dev_rows = load_lp9_micro_pack_rows(out_dir)

        assert report["ok"]
        assert report["duplicate_text_hashes"] == 0
        assert train_rows
        assert dev_rows

        train_ids = {row["example_id"] for row in train_rows}
        dev_ids = {row["example_id"] for row in dev_rows}
        assert train_ids.isdisjoint(dev_ids)

        train_hashes = {row["text_hash"] for row in train_rows}
        dev_hashes = {row["text_hash"] for row in dev_rows}
        assert train_hashes.isdisjoint(dev_hashes)

        all_hashes = [row["text_hash"] for row in [*train_rows, *dev_rows]]
        assert len(all_hashes) == len(set(all_hashes))


def test_lp9_micro_pack_pair_task_coverage_and_safety_flags() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        out_dir = tmp_root / "lp9_pack"
        manifest_path = _write_manifest_with_output_dir(tmp_root, _repo_rel(out_dir))

        generate_lp9_micro_pack(manifest_path)
        train_rows, dev_rows = load_lp9_micro_pack_rows(out_dir)
        rows = [*train_rows, *dev_rows]

        required_tasks = {
            "rewrite_to_quebec_fr",
            "direct_preference",
            "explain_preference",
            "negative_contrast",
        }
        tasks_by_pair: dict[str, set[str]] = {}
        for row in rows:
            tasks_by_pair.setdefault(str(row["lexical_pair_id"]), set()).add(
                str(row["task_type"])
            )
            assert row["source_id"] == "lp9_micro_pack"
            assert row["source_family"] == "manual_targeted"
            assert row["language"] == "fr-CA"
            assert row["dialect_region"] == "Quebec"
            assert row["register"] == "standard"
            assert row["allowed_for_training"] is True
            assert row["holdout_only"] is False
            assert row["requires_review"] is False
            assert row["commercial_use"] == "allowed"
            assert row["license_status"] == "internal_manual"
            assert row["quality_flags"] == []
            assert "<|im_start|>" in row["text"]
            assert "<|im_end|>" in row["text"]

        assert tasks_by_pair
        assert all(required_tasks.issubset(tasks) for tasks in tasks_by_pair.values())


def test_lp9_micro_pack_rows_are_accepted_by_training_pack_loader() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        out_dir = tmp_root / "lp9_pack"
        manifest_path = _write_manifest_with_output_dir(tmp_root, _repo_rel(out_dir))

        generate_lp9_micro_pack(manifest_path)
        train_path = out_dir / "train.jsonl"
        dev_path = out_dir / "dev.jsonl"

        train_rows, train_format, train_warnings = training_inputs.load_input_rows(
            train_path,
            requested_input_format="training-pack",
        )
        dev_rows, dev_format, dev_warnings = training_inputs.load_input_rows(
            dev_path,
            requested_input_format="training-pack",
        )

        assert train_format == "training-pack"
        assert dev_format == "training-pack"
        assert train_rows
        assert dev_rows
        assert train_warnings == 0
        assert dev_warnings == 0


def test_lp9_micro_pack_report_has_no_path_leakage() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        tmp_root = Path(tmp)
        out_dir = tmp_root / "lp9_pack"
        manifest_path = _write_manifest_with_output_dir(tmp_root, _repo_rel(out_dir))

        generate_lp9_micro_pack(manifest_path)

        for path in [
            out_dir / "train.jsonl",
            out_dir / "dev.jsonl",
            out_dir / "report.json",
        ]:
            text = path.read_text(encoding="utf-8")
            assert "/workspace" not in text
            assert "/home/runner" not in text

            if path.suffix == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        payload = json.loads(line)
                        assert payload["holdout_only"] is False
