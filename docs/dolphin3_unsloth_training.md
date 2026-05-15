# Dolphin3 + Unsloth Québec-French Training Path

Use this path when the goal is to **continue Dolphin's existing instruction/personality tuning** instead of starting from clean Qwen.

## Correct lineage

```text
Qwen/Qwen2.5-3B
  -> dphn/Dolphin3.0-Qwen2.5-3b
      -> Québec-French LoRA with Unsloth
          -> merged safetensors and/or GGUF runtime export
```

Do **not** train from `bartowski/Dolphin3.0-Qwen2.5-3b-GGUF`. That model is the runtime GGUF quantization. Train from `dphn/Dolphin3.0-Qwen2.5-3b`.

Use `Qwen/Qwen2.5-3B` only for clean-base ablations where Dolphin's existing assistant style/personality should be removed.

## Install

```bash
python3 -m venv .venv-unsloth
source .venv-unsloth/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/training-dolphin3-unsloth.txt
```

## Smoke-test training

This uses the deterministic accepted curated split. With the committed fixture data this only verifies that the training path works; it is not enough data for model quality.

```bash
python scripts/train_qfr_dolphin3_unsloth_lora.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --train reports/curated_splits/train.jsonl \
  --eval reports/curated_splits/test.jsonl \
  --output-dir models/qfr-dolphin3-qwen25-3b-lora-smoke \
  --max-steps 10
```

## RTX 2070 8 GB defaults

```text
max_seq_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
lora_r: 16
lora_alpha: 32
load_in_4bit: true
format: chat
quantization: q4_k_m
```

If VRAM fails, reduce `--max-seq-length` to `768` or `512` before lowering LoRA rank.

## Full local training after real corpus ingestion

After running the real corpus download, ingestion, curation, accepted-only split, and training export pipeline, point the trainer at the larger accepted split artifacts.

```bash
python scripts/train_qfr_dolphin3_unsloth_lora.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --train reports/curated_splits/train.jsonl \
  --eval reports/curated_splits/test.jsonl \
  --output-dir models/qfr-dolphin3-qwen25-3b-lora \
  --epochs 1 \
  --max-steps -1
```

## Export merged model and GGUF

```bash
python scripts/export_qfr_dolphin3_unsloth_gguf.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --adapter models/qfr-dolphin3-qwen25-3b-lora \
  --merged-16bit-dir models/qfr-dolphin3-qwen25-3b-merged \
  --gguf-dir models/qfr-dolphin3-qwen25-3b-gguf \
  --quantization q4_k_m
```

## Why chat format is default

The script defaults to `--format chat` because this project is continuing Dolphin's instruction-tuned assistant behaviour. Each accepted Québec-French record is wrapped in a Qwen/Dolphin-style chat example with a system prompt that preserves Québec-French markers and avoids dialect neutralization.

Use `--format completion` only for ablations.

## Safety and release notes

- The trainer refuses records whose `curation_label` is not `accepted`.
- Required curation metadata is enforced before training.
- Evaluation and holdout datasets remain excluded from training.
- Real model quality should not be judged from the tiny committed fixture dataset.
- Check Dolphin/Qwen/Unsloth licenses and model cards before commercial release.
