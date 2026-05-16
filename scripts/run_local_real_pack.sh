#!/usr/bin/env bash
set -euo pipefail

echo "==> Bootstrap dev environment"
bash scripts/bootstrap_dev_env.sh

echo "==> Download real approved corpus sources"
python tools/download_real_corpus_sources.py \
  --manifest manifests/corpus_source_manifest.real_downloads.yaml \
  --out-manifest manifests/corpus_source_manifest.real_downloaded.local.yaml \
  --report reports/corpus_ingestion/downloads.real.json \
  --overwrite

echo "==> Validate and ingest downloaded corpus"
qfr validate-corpus-sources \
  --manifest manifests/corpus_source_manifest.real_downloaded.local.yaml
qfr ingest-corpus-sources \
  --manifest manifests/corpus_source_manifest.real_downloaded.local.yaml \
  --out reports/corpus_ingestion/harvest.real.jsonl \
  --report reports/corpus_ingestion/report.real.json \
  --min-chars 80

echo "==> Curate real harvest"
qfr curate-corpus \
  --input reports/corpus_ingestion/harvest.real.jsonl \
  --policy manifests/curation_policy_manifest.template.yaml \
  --out-dir reports/corpus_curation_real

echo "==> Prepare temporary live modern manifests"
TMP_DQ_MANIFEST="$(mktemp /tmp/qfr_dq_live.XXXXXX.yaml)"
TMP_ASSNAT_MANIFEST="$(mktemp /tmp/qfr_assnat_live.XXXXXX.yaml)"
trap 'rm -f "$TMP_DQ_MANIFEST" "$TMP_ASSNAT_MANIFEST"' EXIT

python - "$TMP_DQ_MANIFEST" "$TMP_ASSNAT_MANIFEST" <<'PY'
from pathlib import Path
import sys
import yaml

base = yaml.safe_load(
    Path("manifests/modern_corpus_acquisition_manifest.template.yaml").read_text(
        encoding="utf-8"
    )
)
sources = {item["source_id"]: item for item in base["sources"]}

common = {
    "kind": base["kind"],
    "schema_version": base["schema_version"],
    "primary_language": base["primary_language"],
}

dq_manifest = dict(common)
dq_manifest["sources"] = [sources["donnees_quebec_ckan_textual"]]

assnat_manifest = dict(common)
assnat_manifest["sources"] = [sources["assnat_journal_debats_modern"]]

Path(sys.argv[1]).write_text(
    yaml.safe_dump(dq_manifest, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
Path(sys.argv[2]).write_text(
    yaml.safe_dump(assnat_manifest, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

echo "==> Acquire live Données Québec metadata"
qfr validate-modern-corpus --manifest "$TMP_DQ_MANIFEST"
qfr acquire-modern-corpus \
  --manifest "$TMP_DQ_MANIFEST" \
  --out reports/modern_corpus/donnees_quebec_live_harvest.jsonl \
  --report reports/modern_corpus/donnees_quebec_live_report.json \
  --max-documents 25

echo "==> Acquire live AssNat seed content"
qfr validate-modern-corpus --manifest "$TMP_ASSNAT_MANIFEST"
qfr acquire-modern-corpus \
  --manifest "$TMP_ASSNAT_MANIFEST" \
  --out reports/modern_corpus/assnat_live_harvest.jsonl \
  --report reports/modern_corpus/assnat_live_report.json \
  --include-noncommercial \
  --max-documents 10

echo "==> Build local real training pack"
qfr validate-training-pack-policy \
  --policy manifests/training_pack_policy.local_real.template.yaml
qfr build-training-pack \
  --policy manifests/training_pack_policy.local_real.template.yaml \
  --out-dir reports/training_pack_real
qfr audit-training-pack \
  --pack-dir reports/training_pack_real \
  --out reports/training_pack_real/audit.json || true

echo "==> Train/dev/test example counts"
wc -l \
  reports/training_pack_real/train.jsonl \
  reports/training_pack_real/dev.jsonl \
  reports/training_pack_real/test.jsonl

echo "==> Training pack summary"
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("reports/training_pack_real/report.json").read_text(encoding="utf-8")
)
summary = {
    "ok": report.get("ok"),
    "pack_mode": report.get("pack_mode"),
    "records_seen": report.get("records_seen"),
    "records_accepted": report.get("records_accepted"),
    "examples_generated": report.get("examples_generated"),
    "readiness_level": report.get("readiness_level"),
    "commercial_release_ready": report.get("commercial_release_ready"),
    "commercial_blocking_reasons": report.get("commercial_blocking_reasons", []),
}
print(summary)
PY

echo "NOTE: This script generates live/local artifacts under reports/ and data/corpus/raw/."
echo "NOTE: Review and commit only small deterministic artifacts; do not commit large live harvests."
