from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from qfr_pipeline.training_pack import validate_training_pack_policy

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_qfr_dolphin3_unsloth_lora.py"

spec = importlib.util.spec_from_file_location("qfr_unsloth_training_inputs", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load script module from {SCRIPT_PATH}")
training_inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_inputs)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _curated_row(text: str = "Texte accepté.") -> dict:
    return {
        "curation_label": "accepted",
        "text": text,
        "curation_score": 0.92,
        "curation_reasons": ["fixture"],
        "policy_id": "curation_policy_manifest.template",
    }


def _training_pack_messages_row() -> dict:
    return {
        "example_id": "ex-1",
        "task_type": "summarize",
        "source_id": "gutenberg_real",
        "messages": [
            {"role": "system", "content": "Tu réponds en fr-CA."},
            {"role": "user", "content": "Résume ce passage."},
            {"role": "assistant", "content": "Résumé concis."},
        ],
        "allowed_for_training": True,
        "holdout_only": False,
        "contains_holdout_material": False,
        "requires_review": False,
        "commercial_use": "allowed",
    }


def test_curated_format_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "curated.jsonl"
    _write_jsonl(path, [_curated_row()])

    rows, detected_format, warnings = training_inputs.load_input_rows(
        path,
        requested_input_format="curated",
    )

    assert detected_format == "curated"
    assert len(rows) == 1
    assert warnings == 0


def test_auto_detects_curated_input(tmp_path: Path) -> None:
    path = tmp_path / "curated_auto.jsonl"
    _write_jsonl(path, [_curated_row()])

    _, detected_format, _ = training_inputs.load_input_rows(
        path,
        requested_input_format="auto",
    )

    assert detected_format == "curated"


def test_auto_detects_training_pack_messages_input(tmp_path: Path) -> None:
    path = tmp_path / "pack_auto.jsonl"
    _write_jsonl(path, [_training_pack_messages_row()])

    rows, detected_format, warnings = training_inputs.load_input_rows(
        path,
        requested_input_format="auto",
    )

    assert detected_format == "training-pack"
    assert len(rows) == 1
    assert warnings == 0


def test_curated_input_rejects_missing_curation_label(tmp_path: Path) -> None:
    path = tmp_path / "curated_missing_label.jsonl"
    row = _curated_row()
    row.pop("curation_label")
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="Forbidden non-accepted record"):
        training_inputs.load_input_rows(path, requested_input_format="curated")


def test_training_pack_input_accepts_missing_curation_label(tmp_path: Path) -> None:
    path = tmp_path / "pack_no_label.jsonl"
    _write_jsonl(path, [_training_pack_messages_row()])

    rows, detected_format, _ = training_inputs.load_input_rows(
        path,
        requested_input_format="training-pack",
    )

    assert detected_format == "training-pack"
    assert len(rows) == 1
    assert "curation_label" not in rows[0]


def test_training_pack_row_with_messages_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "pack_messages.jsonl"
    _write_jsonl(path, [_training_pack_messages_row()])

    rows, _, _ = training_inputs.load_input_rows(
        path,
        requested_input_format="training-pack",
    )

    assert rows[0]["messages"][0]["role"] == "system"


def test_training_pack_rejects_holdout_rows(tmp_path: Path) -> None:
    path = tmp_path / "pack_holdout.jsonl"
    row = _training_pack_messages_row()
    row["holdout_only"] = True
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="Forbidden holdout record"):
        training_inputs.load_input_rows(path, requested_input_format="training-pack")


def test_training_pack_rejects_allowed_for_training_false(tmp_path: Path) -> None:
    path = tmp_path / "pack_disallowed.jsonl"
    row = _training_pack_messages_row()
    row["allowed_for_training"] = False
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="allowed_for_training=false"):
        training_inputs.load_input_rows(path, requested_input_format="training-pack")


def test_training_pack_messages_render_to_qwen_chatml() -> None:
    rendered = training_inputs.render_training_pack_row(
        tokenizer=object(),
        row=_training_pack_messages_row(),
    )
    assert "<|im_start|>system" in rendered
    assert "<|im_start|>assistant" in rendered
    assert "<|im_end|>" in rendered


def test_local_smoke_policy_validates() -> None:
    validate_training_pack_policy(
        ROOT / "manifests/training_pack_policy.local_smoke.template.yaml"
    )


def test_build_local_smoke_pack_script_contains_no_training_command() -> None:
    script = (ROOT / "scripts/build_local_smoke_pack.sh").read_text(
        encoding="utf-8"
    )
    lowered = script.casefold()
    assert "train_qfr_dolphin3_unsloth_lora.py" not in lowered
    assert "--base-model" not in lowered


def test_no_model_or_gguf_outputs_are_tracked() -> None:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    blocked = [
        path
        for path in tracked
        if path.startswith("models/") or path.endswith(".gguf")
    ]
    assert blocked == []
