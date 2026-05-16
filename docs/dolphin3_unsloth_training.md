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

## Input formats

`scripts/train_qfr_dolphin3_unsloth_lora.py` now supports:

- `--input-format auto` (default)
- `--input-format curated`
- `--input-format training-pack`

`curated` requires accepted curation rows (`curation_label`, `text`, `curation_score`, `curation_reasons`, `policy_id`).
`training-pack` accepts generated rows with `messages` or pre-rendered `text` (and compatible `prompt`/`assistant_text` rows).

## Governance-balanced local real pack

`reports/training_pack_real/` is governance-balanced and can remain very small when source-dominance and diversity constraints trigger.
Treat it as a strict governance indicator, not a guaranteed high-volume training pack.

## Local smoke pack (technical only)

Use `manifests/training_pack_policy.local_smoke.template.yaml` for local technical smoke/pilot runs.
This policy intentionally allows source dominance for research while still keeping release readiness blocked.

```bash
qfr build-training-pack \
  --policy manifests/training_pack_policy.local_smoke.template.yaml \
  --out-dir reports/training_pack_smoke
```

or run:

```bash
bash scripts/build_local_smoke_pack.sh
```

`reports/training_pack_smoke/` is local-research-only and not production-ready. Commercial release remains blocked until modern/diverse commercially safe sources are materially expanded.

## Smoke-test training

This command validates the Unsloth path with a small step budget and training-pack input.

```bash
python scripts/train_qfr_dolphin3_unsloth_lora.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --train reports/training_pack_smoke/train.jsonl \
  --eval reports/training_pack_smoke/dev.jsonl \
  --output-dir models/qfr-dolphin3-qwen25-3b-lora-smoke \
  --max-steps 10 \
  --input-format training-pack
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

After running real corpus download/ingestion/curation and building a training pack, point the trainer at the pack artifacts.

```bash
python scripts/train_qfr_dolphin3_unsloth_lora.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --train reports/training_pack_smoke/train.jsonl \
  --eval reports/training_pack_smoke/dev.jsonl \
  --output-dir models/qfr-dolphin3-qwen25-3b-lora \
  --epochs 1 \
  --max-steps -1 \
  --input-format training-pack
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

- Curated mode still enforces `curation_label == accepted` plus required curation metadata.
- Training-pack mode rejects holdout material, `allowed_for_training=false`, and review-gated rows without `permission_granted=true`.
- Training-pack rows marked `commercial_use` as `permission_required`/`unknown`/`prohibited` are surfaced as local-research warnings and must not be treated as commercial-release-safe.
- Evaluation and holdout datasets remain excluded from training.
- Real model quality should not be judged from tiny fixture/governance-balanced packs.
- Check Dolphin/Qwen/Unsloth licenses and model cards before commercial release.
