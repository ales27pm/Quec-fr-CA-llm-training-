#!/usr/bin/env bash
set -euo pipefail

qfr validate-training-pack-policy \
  --policy manifests/training_pack_policy.local_smoke.template.yaml

qfr build-training-pack \
  --policy manifests/training_pack_policy.local_smoke.template.yaml \
  --out-dir reports/training_pack_smoke

qfr audit-training-pack \
  --pack-dir reports/training_pack_smoke \
  --out reports/training_pack_smoke/audit.json || true

wc -l reports/training_pack_smoke/train.jsonl
wc -l reports/training_pack_smoke/dev.jsonl
wc -l reports/training_pack_smoke/test.jsonl
