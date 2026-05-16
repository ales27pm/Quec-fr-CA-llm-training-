#!/usr/bin/env python3
"""Continue Dolphin3.0-Qwen2.5-3b with Québec-French data using Unsloth QLoRA.

This script intentionally trains from the non-GGUF Dolphin checkpoint:
    dphn/Dolphin3.0-Qwen2.5-3b

It does not train from bartowski's GGUF artifact. GGUF is exported after training.

Input formats:
  - curated accepted rows (`reports/curated_splits/*.jsonl`)
  - generated training-pack rows (`reports/training_pack*/train.jsonl`)

Smoke test:
  python scripts/train_qfr_dolphin3_unsloth_lora.py \
    --train reports/curated_splits/train.jsonl \
    --eval reports/curated_splits/test.jsonl \
    --output-dir models/qfr-dolphin3-qwen25-3b-lora-smoke \
    --max-steps 10

Full local run starts from the same command with more data and max-steps -1.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Literal

SYSTEM_PROMPT = (
    "Tu es un assistant IA québécois précis, utile et naturel. "
    "Tu préserves le français québécois authentique, tu évites de neutraliser les marqueurs fr-CA, "
    "et tu respectes les formes normatives lorsqu'elles sont pertinentes."
)

DEFAULT_USER_PROMPT = (
    "Rédige une réponse naturelle en français québécois en respectant le registre et le contenu attendu."
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
INPUT_FORMAT_CHOICES = ("auto", "curated", "training-pack")
CURATED_REQUIRED_FIELDS = ("text", "curation_score", "curation_reasons", "policy_id")
LOCAL_RESEARCH_ONLY_COMMERCIAL_USE = {"permission_required", "unknown", "prohibited"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL object at {path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _first_non_empty_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if any(_is_present(value) for value in row.values()):
            return row
    return None


def _looks_curated_row(row: dict[str, Any]) -> bool:
    return all(field in row for field in ("curation_label", "text", "curation_score"))


def _looks_training_pack_row(row: dict[str, Any]) -> bool:
    if "messages" in row:
        return True
    if "example_id" in row or "task_type" in row:
        return True
    if "assistant_text" in row or "prompt" in row:
        return True
    text = str(row.get("text") or "").strip()
    return bool(text and "<|im_start|>" in text and "<|im_end|>" in text)


def detect_input_format(
    rows: list[dict[str, Any]],
    *,
    requested_format: str,
    path: Path,
) -> Literal["curated", "training-pack"]:
    if requested_format == "curated":
        return "curated"
    if requested_format == "training-pack":
        return "training-pack"
    probe = _first_non_empty_row(rows)
    if probe is None:
        raise ValueError(f"No non-empty rows found in {path}")
    if _looks_curated_row(probe):
        return "curated"
    if _looks_training_pack_row(probe):
        return "training-pack"
    keys = sorted(probe.keys())
    raise ValueError(
        f"Unable to auto-detect input format for {path}. "
        f"First non-empty row keys: {keys}. "
        "Use --input-format curated or --input-format training-pack."
    )


def accepted_curated_rows(rows: list[dict[str, Any]], *, path: Path) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        label = row.get("curation_label")
        if label != "accepted":
            raise ValueError(f"Forbidden non-accepted record in {path} at row {idx}: {label!r}")
        missing = [k for k in CURATED_REQUIRED_FIELDS if k not in row]
        if missing:
            raise ValueError(f"Missing required curation metadata in {path} at row {idx}: {missing}")
        text = str(row["text"]).strip()
        if text:
            accepted.append(row)
    if not accepted:
        raise ValueError(f"No accepted training text found in {path}")
    return accepted


def _normalize_messages(messages: Any, *, path: Path, row_idx: int) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError(f"Invalid messages field in {path} at row {row_idx}: expected list")
    normalized: list[dict[str, str]] = []
    for message_idx, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(
                f"Invalid message in {path} at row {row_idx}, message {message_idx}: expected object"
            )
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            raise ValueError(
                f"Invalid message in {path} at row {row_idx}, message {message_idx}: role/content required"
            )
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError(f"Empty messages/text in {path} at row {row_idx}")
    return normalized


def accepted_training_pack_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path,
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    local_research_warnings = 0

    for idx, row in enumerate(rows, start=1):
        holdout_only = _parse_bool(row.get("holdout_only"), default=False)
        contains_holdout = _parse_bool(row.get("contains_holdout_material"), default=False)
        if holdout_only or contains_holdout:
            raise ValueError(f"Forbidden holdout record in {path} at row {idx}")

        allowed_for_training = _parse_bool(row.get("allowed_for_training"), default=True)
        if not allowed_for_training:
            raise ValueError(f"Forbidden allowed_for_training=false record in {path} at row {idx}")

        requires_review = _parse_bool(row.get("requires_review"), default=False)
        permission_granted = _parse_bool(row.get("permission_granted"), default=False)
        if requires_review and not permission_granted:
            raise ValueError(
                f"Forbidden requires_review=true without permission_granted in {path} at row {idx}"
            )

        normalized = dict(row)
        text = str(row.get("text") or "").strip()
        messages: list[dict[str, str]] | None = None

        if row.get("messages") is not None:
            messages = _normalize_messages(row.get("messages"), path=path, row_idx=idx)
        elif row.get("assistant_text") is not None or row.get("prompt") is not None:
            assistant_text = str(row.get("assistant_text") or "").strip()
            prompt = str(row.get("prompt") or DEFAULT_USER_PROMPT).strip() or DEFAULT_USER_PROMPT
            system_prompt = str(row.get("system_prompt") or SYSTEM_PROMPT).strip() or SYSTEM_PROMPT
            if not assistant_text:
                raise ValueError(f"Empty assistant_text in {path} at row {idx}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_text},
            ]

        if not text and not messages:
            raise ValueError(f"Empty messages/text in {path} at row {idx}")

        if text:
            normalized["text"] = text
        if messages is not None:
            normalized["messages"] = messages

        commercial_use = str(row.get("commercial_use") or "").strip().casefold()
        if commercial_use in LOCAL_RESEARCH_ONLY_COMMERCIAL_USE:
            local_research_warnings += 1
            normalized["_local_research_only_commercial_use"] = commercial_use

        accepted.append(normalized)

    if not accepted:
        raise ValueError(f"No accepted training-pack records found in {path}")
    return accepted, local_research_warnings


def load_input_rows(
    path: Path,
    *,
    requested_input_format: str,
) -> tuple[list[dict[str, Any]], Literal["curated", "training-pack"], int]:
    rows = read_jsonl(path)
    detected_format = detect_input_format(rows, requested_format=requested_input_format, path=path)
    if detected_format == "curated":
        return accepted_curated_rows(rows, path=path), "curated", 0
    accepted, warnings_count = accepted_training_pack_rows(rows, path=path)
    return accepted, "training-pack", warnings_count


def fallback_qwen_chatml(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            continue
        lines.append(f"<|im_start|>{role}")
        lines.append(content)
        lines.append("<|im_end|>")
    rendered = "\n".join(lines).strip()
    return rendered + "\n" if rendered else ""


def fallback_qwen_chat(system: str, user: str, assistant: str) -> str:
    return fallback_qwen_chatml(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return fallback_qwen_chatml(messages)


def render_chat(tokenizer: Any, assistant_text: str, *, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    return render_messages(tokenizer, messages)


def render_training_pack_row(tokenizer: Any, row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    if text:
        return text
    if row.get("messages") is not None:
        messages = _normalize_messages(row.get("messages"), path=Path("<in-memory>"), row_idx=0)
        return render_messages(tokenizer, messages)
    assistant_text = str(row.get("assistant_text") or "").strip()
    if assistant_text:
        prompt = str(row.get("prompt") or DEFAULT_USER_PROMPT).strip() or DEFAULT_USER_PROMPT
        system_prompt = str(row.get("system_prompt") or SYSTEM_PROMPT).strip() or SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text},
        ]
        return render_messages(tokenizer, messages)
    raise ValueError("Training-pack row does not include renderable text/messages")


def rows_to_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    input_format: Literal["curated", "training-pack"],
    format_mode: str,
    system_prompt: str,
    user_prompt: str,
):
    from datasets import Dataset

    records: list[dict[str, str]] = []
    for row in rows:
        if input_format == "curated":
            text = str(row["text"]).strip()
            if format_mode == "chat":
                rendered = render_chat(tokenizer, text, system_prompt=system_prompt, user_prompt=user_prompt)
            elif format_mode == "completion":
                rendered = text
            else:
                raise ValueError(f"Unsupported format mode: {format_mode}")
        elif input_format == "training-pack":
            rendered = render_training_pack_row(tokenizer, row)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        if not str(rendered).strip():
            continue

        record: dict[str, str] = {"text": rendered}
        for field in ("example_id", "source_id", "task_type", "source_record_id", "source_family"):
            value = row.get(field)
            if _is_present(value):
                record[field] = str(value)
        records.append(record)

    if not records:
        raise ValueError("No training examples remained after rendering input rows.")
    return Dataset.from_list(records)


def summarize_input_rows(
    *,
    input_format: Literal["curated", "training-pack"],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    local_research_warning_count: int,
) -> None:
    task_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    for row in [*train_rows, *eval_rows]:
        if _is_present(row.get("task_type")):
            task_counts[str(row["task_type"])] += 1
        if _is_present(row.get("source_id")):
            source_ids.add(str(row["source_id"]))

    print("Training input summary:")
    print(f"  input_format: {input_format}")
    print(f"  train_rows: {len(train_rows)}")
    print(f"  eval_rows: {len(eval_rows)}")
    print(f"  task_type_counts: {dict(sorted(task_counts.items())) if task_counts else {}}")
    print(f"  source_count: {len(source_ids)}")
    print(f"  local_research_license_warnings: {local_research_warning_count}")
    if local_research_warning_count:
        print(
            "  warning: rows with commercial_use in "
            "{permission_required, unknown, prohibited} are local-research-only."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue Dolphin3.0-Qwen2.5-3b with Unsloth QLoRA on accepted Québec-French data.")
    parser.add_argument("--base-model", default="dphn/Dolphin3.0-Qwen2.5-3b", help="Non-GGUF Dolphin safetensors checkpoint.")
    parser.add_argument("--train", type=Path, default=Path("reports/curated_splits/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("reports/curated_splits/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/qfr-dolphin3-qwen25-3b-lora"))
    parser.add_argument("--merged-16bit-dir", type=Path, default=None, help="Optional: save merged 16-bit HF model here after training.")
    parser.add_argument("--gguf-dir", type=Path, default=None, help="Optional: export GGUF directory directly through Unsloth after training.")
    parser.add_argument("--gguf-quantization", default="q4_k_m", help="Unsloth GGUF quantization method, e.g. q4_k_m, q5_k_m, q8_0.")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use small values like 10 for smoke tests. -1 means train by epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing remote model code for trusted model repositories only.",
    )
    parser.add_argument("--format", choices=["chat", "completion"], default="chat")
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMAT_CHOICES,
        default="auto",
        help="Input record format: auto-detect, curated accepted rows, or generated training-pack rows.",
    )
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    train_rows, input_format, warning_count = load_input_rows(
        args.train,
        requested_input_format=args.input_format,
    )
    eval_rows: list[dict[str, Any]] = []
    if args.eval.exists() and args.eval.stat().st_size > 0:
        eval_rows, eval_input_format, eval_warnings = load_input_rows(
            args.eval,
            requested_input_format=args.input_format,
        )
        if eval_input_format != input_format:
            raise ValueError(
                f"Train/eval input-format mismatch: train={input_format}, eval={eval_input_format}. "
                "Use matching file formats."
            )
        warning_count += eval_warnings

    summarize_input_rows(
        input_format=input_format,
        train_rows=train_rows,
        eval_rows=eval_rows,
        local_research_warning_count=warning_count,
    )

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = rows_to_dataset(
        train_rows,
        tokenizer,
        input_format=input_format,
        format_mode=args.format,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )
    eval_dataset = (
        rows_to_dataset(
            eval_rows,
            tokenizer,
            input_format=input_format,
            format_mode=args.format,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
        )
        if eval_rows
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=25 if eval_dataset is not None else None,
        warmup_steps=5,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=1,
        packing=args.packing,
        args=training_args,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.merged_16bit_dir is not None:
        args.merged_16bit_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(args.merged_16bit_dir), tokenizer, save_method="merged_16bit")

    if args.gguf_dir is not None:
        args.gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(str(args.gguf_dir), tokenizer, quantization_method=args.gguf_quantization)

    manifest = {
        "kind": "qfr_unsloth_lora_training_run",
        "base_model": args.base_model,
        "training_framework": "unsloth",
        "method": "qlora" if args.load_in_4bit else "lora",
        "train": str(args.train),
        "eval": str(args.eval),
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "input_format": input_format,
        "format": args.format,
        "local_research_license_warnings": warning_count,
        "output_dir": str(args.output_dir),
        "merged_16bit_dir": str(args.merged_16bit_dir) if args.merged_16bit_dir else None,
        "gguf_dir": str(args.gguf_dir) if args.gguf_dir else None,
        "gguf_quantization": args.gguf_quantization if args.gguf_dir else None,
        "notes": "Continues Dolphin3.0-Qwen2.5-3b safetensors. GGUF is exported after training, not used as the training source.",
    }
    (args.output_dir / "qfr_unsloth_training_run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
