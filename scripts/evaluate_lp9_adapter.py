#!/usr/bin/env python3
"""Evaluate LP9 lexical preference behavior for base vs LoRA adapter models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from qfr_pipeline.io import write_json
from qfr_pipeline.lp9_eval import load_eval_prompts, score_response

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond en français québécois standard et privilégie "
    "les termes recommandés au Québec."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _fallback_qwen_chatml(messages: list[dict[str, str]]) -> str:
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


def _render_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return _fallback_qwen_chatml(messages)


def generate_text(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate a deterministic greedy response for one eval prompt."""
    import torch

    rendered_prompt = _render_prompt(tokenizer, prompt)
    inputs = tokenizer([rendered_prompt], return_tensors="pt")
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "cpu"
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    continuation = generated[0][input_len:]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def evaluate_prompt_pair(
    *,
    prompt_row: dict[str, Any],
    base_response: str,
    adapter_response: str,
) -> dict[str, Any]:
    scoring = prompt_row["scoring"]
    base_eval = score_response(base_response, scoring)
    adapter_eval = score_response(adapter_response, scoring)
    return {
        "prompt_id": prompt_row["prompt_id"],
        "lexical_pair_id": prompt_row["lexical_pair_id"],
        "task_type": prompt_row["task_type"],
        "expected_terms": list(prompt_row.get("expected_terms", [])),
        "forbidden_terms": list(prompt_row.get("forbidden_terms", [])),
        "base_response": base_response,
        "adapter_response": adapter_response,
        "base": base_eval,
        "adapter": adapter_eval,
    }


def _derive_generations_path(report_out: Path) -> Path:
    stem = report_out.stem
    if stem.endswith("_report"):
        generation_stem = stem[: -len("_report")] + "_generations"
    else:
        generation_stem = stem + "_generations"
    return report_out.with_name(generation_stem + ".jsonl")


def _summary_bucket() -> dict[str, int]:
    return {
        "count": 0,
        "base_score": 0,
        "adapter_score": 0,
        "delta": 0,
    }


def build_report(
    *,
    prompt_path: Path,
    report_out: Path,
    generations: list[dict[str, Any]],
) -> dict[str, Any]:
    base_score = sum(int(item["base"]["score"]) for item in generations)
    adapter_score = sum(int(item["adapter"]["score"]) for item in generations)

    per_pair: dict[str, dict[str, int]] = defaultdict(_summary_bucket)
    per_task_type: dict[str, dict[str, int]] = defaultdict(_summary_bucket)
    failures: list[dict[str, Any]] = []

    for item in generations:
        pair_id = str(item["lexical_pair_id"])
        task_type = str(item["task_type"])
        base_val = int(item["base"]["score"])
        adapter_val = int(item["adapter"]["score"])
        delta = adapter_val - base_val

        pair_bucket = per_pair[pair_id]
        pair_bucket["count"] += 1
        pair_bucket["base_score"] += base_val
        pair_bucket["adapter_score"] += adapter_val
        pair_bucket["delta"] += delta

        task_bucket = per_task_type[task_type]
        task_bucket["count"] += 1
        task_bucket["base_score"] += base_val
        task_bucket["adapter_score"] += adapter_val
        task_bucket["delta"] += delta

        reasons: list[str] = []
        if adapter_val < base_val:
            reasons.append("adapter_under_base")
        if bool(item["adapter"].get("forbidden_hit", False)):
            reasons.append("adapter_contains_forbidden")
        if not bool(item["adapter"].get("preferred_hit", False)):
            reasons.append("adapter_missing_preferred")
        if reasons:
            failures.append(
                {
                    "prompt_id": item["prompt_id"],
                    "lexical_pair_id": pair_id,
                    "task_type": task_type,
                    "base_score": base_val,
                    "adapter_score": adapter_val,
                    "reasons": reasons,
                }
            )

    generations_path = _derive_generations_path(report_out)
    report = {
        "ok": True,
        "kind": "lp9_base_vs_adapter_report",
        "prompts": str(prompt_path),
        "total_prompts": len(generations),
        "base_score": base_score,
        "adapter_score": adapter_score,
        "delta": adapter_score - base_score,
        "per_pair": dict(per_pair),
        "per_task_type": dict(per_task_type),
        "failures": failures,
        "generations_jsonl": str(generations_path),
    }
    return report


def run_evaluation(
    *,
    prompts_path: Path,
    report_out: Path,
    generate_base: Any,
    generate_adapter: Any,
) -> dict[str, Any]:
    prompts = load_eval_prompts(prompts_path)
    generations: list[dict[str, Any]] = []

    for prompt_row in prompts:
        prompt = str(prompt_row["prompt"])
        base_response = str(generate_base(prompt_row, prompt))
        adapter_response = str(generate_adapter(prompt_row, prompt))
        generations.append(
            evaluate_prompt_pair(
                prompt_row=prompt_row,
                base_response=base_response,
                adapter_response=adapter_response,
            )
        )

    report = build_report(
        prompt_path=prompts_path,
        report_out=report_out,
        generations=generations,
    )
    generations_path = Path(report["generations_jsonl"])
    _write_jsonl(generations_path, generations)
    write_json(report_out, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LP9 lexical preference behavior for base vs adapter model.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Keep heavy model dependencies local to runtime entrypoint.
    from unsloth import FastLanguageModel

    if args.temperature != 0.0:
        print(
            "Warning: --temperature is ignored because greedy decoding "
            "uses do_sample=false."
        )

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    FastLanguageModel.for_inference(base_model)

    adapter_model, adapter_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    if adapter_tokenizer.pad_token is None:
        adapter_tokenizer.pad_token = adapter_tokenizer.eos_token
    FastLanguageModel.for_inference(adapter_model)

    report = run_evaluation(
        prompts_path=args.prompts,
        report_out=args.out,
        generate_base=lambda _row, prompt: generate_text(
            model=base_model,
            tokenizer=base_tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        ),
        generate_adapter=lambda _row, prompt: generate_text(
            model=adapter_model,
            tokenizer=adapter_tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    print(
        {
            "total_prompts": report["total_prompts"],
            "base_score": report["base_score"],
            "adapter_score": report["adapter_score"],
            "delta": report["delta"],
            "failures": len(report["failures"]),
            "out": str(args.out),
            "generations": report["generations_jsonl"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
