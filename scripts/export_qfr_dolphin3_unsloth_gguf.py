#!/usr/bin/env python3
"""Merge a Dolphin3 LoRA adapter and export GGUF through Unsloth.

This path keeps the base as dphn/Dolphin3.0-Qwen2.5-3b and uses Unsloth's
`save_pretrained_gguf` API instead of a separate llama.cpp shell conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Québec-French Dolphin3 LoRA to merged 16-bit and/or GGUF using Unsloth.")
    parser.add_argument("--base-model", default="dphn/Dolphin3.0-Qwen2.5-3b")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged-16bit-dir", type=Path, default=Path("models/qfr-dolphin3-qwen25-3b-merged"))
    parser.add_argument("--gguf-dir", type=Path, default=Path("models/qfr-dolphin3-qwen25-3b-gguf"))
    parser.add_argument("--quantization", default="q4_k_m", help="Recommended for RTX 2070. Alternatives: q5_k_m, q8_0.")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing remote model code for trusted model repositories only.",
    )
    args = parser.parse_args()

    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))

    args.merged_16bit_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(args.merged_16bit_dir), tokenizer, save_method="merged_16bit")

    args.gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(str(args.gguf_dir), tokenizer, quantization_method=args.quantization)

    print(f"Merged 16-bit model: {args.merged_16bit_dir}")
    print(f"GGUF export directory: {args.gguf_dir}")
    print(f"Quantization: {args.quantization}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
