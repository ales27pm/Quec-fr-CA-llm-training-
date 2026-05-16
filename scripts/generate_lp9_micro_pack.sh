#!/usr/bin/env bash
set -euo pipefail

qfr generate-lp9-micro-pack \
  --manifest manifests/lp9_lexical_preference_pack.template.yaml \
  --out-dir reports/lp9_micro_pack

qfr generate-lp9-eval \
  --manifest manifests/lp9_lexical_preference_pack.template.yaml \
  --out reports/lp9_eval/eval_prompts.jsonl
