#!/usr/bin/env python3
"""Continue Dolphin3.0-Qwen2.5-3b with Québec-French accepted data using Unsloth QLoRA.

This script intentionally trains from the non-GGUF Dolphin checkpoint:
    dphn/Dolphin3.0-Qwen2.5-3b

It does not train from bartowski's GGUF artifact. GGUF is exported after training.

Input records must come from qfr-pipeline curated accepted splits and must contain:
  - text
  - curation_label == "accepted"
  - curation_score
  - curation_reasons
  - policy_id

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
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "Tu es un assistant IA québécois précis, utile et naturel. "
    "Tu préserves le français québécois authentique, tu évites de neutraliser les marqueurs fr-CA, "
    "et tu respectes les formes normatives lorsqu'elles sont pertinentes."
)

DEFAULT_USER_PROMPT = (
    "Rédige une réponse naturelle en français québécois en respectant le registre et le contenu attendu."
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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
            rows.append(row)
    return rows


def accepted_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    accepted: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        label = row.get("curation_label")
        if label != "accepted":
            raise ValueError(f"Forbidden non-accepted record in {path} at row {idx}: {label!r}")
        missing = [k for k in ("text", "curation_score", "curation_reasons", "policy_id") if k not in row]
        if missing:
            raise ValueError(f"Missing required curation metadata in {path} at row {idx}: {missing}")
        text = str(row["text"]).strip()
        if text:
            accepted.append(row)
    if not accepted:
        raise ValueError(f"No accepted training text found in {path}")
    return accepted


def fallback_qwen_chat(system: str, user: str, assistant: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant}<|im_end|>\n"
    )


def render_chat(tokenizer: Any, assistant_text: str, *, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return fallback_qwen_chat(system_prompt, user_prompt, assistant_text)


def rows_to_dataset(rows: list[dict[str, Any]], tokenizer: Any, *, format_mode: str, system_prompt: str, user_prompt: str):
    from datasets import Dataset

    records: list[dict[str, str]] = []
    for row in rows:
        text = str(row["text"]).strip()
        if format_mode == "chat":
            rendered = render_chat(tokenizer, text, system_prompt=system_prompt, user_prompt=user_prompt)
        elif format_mode == "completion":
            rendered = text
        else:
            raise ValueError(f"Unsupported format mode: {format_mode}")
        records.append({"text": rendered})
    return Dataset.from_list(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue Dolphin3.0-Qwen2.5-3b with Unsloth QLoRA on accepted Québec-French data.")
    parser.add_argument("--base-model", default="dphn/Dolphin3.0-Qwen2.5-3b", help="Non-GGUF Dolphin safetensors checkpoint.")
    parser.add_argument("--train", type=Path, default=Path("reports/curated_splits/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("reports/curated_splits/test.jsonl"))
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
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--format", choices=["chat", "completion"], default="chat")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    train_rows = accepted_rows(args.train)
    eval_rows = accepted_rows(args.eval) if args.eval.exists() and args.eval.stat().st_size > 0 else []

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
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

    train_dataset = rows_to_dataset(train_rows, tokenizer, format_mode=args.format, system_prompt=args.system_prompt, user_prompt=args.user_prompt)
    eval_dataset = rows_to_dataset(eval_rows, tokenizer, format_mode=args.format, system_prompt=args.system_prompt, user_prompt=args.user_prompt) if eval_rows else None

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
        "format": args.format,
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
