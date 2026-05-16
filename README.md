# Quec-fr-CA-llm-training

## Strategic planning
This repository includes a sovereignty-first roadmap and centralized release thresholds for the Québécois (fr-CA) training program.
- Roadmap: [docs/strategic_roadmap.md](docs/strategic_roadmap.md)
- Threshold source of truth: [project/release_gates.yaml](project/release_gates.yaml)

## Executable Pipeline
- Install: `pip install -e .[dev]`
- Validate repo: `qfr validate`
- Generate quality-gated LP9 minimal pairs: `qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp9_lexical_semantics.contexts.yaml --out data/generated/minimal_pairs.lp9.jsonl --report reports/minimal_pair_quality.lp9.json`
- Validate generated LP9 minimal pairs: `qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp9.jsonl --context rules/lp9_lexical_semantics.contexts.yaml --report reports/minimal_pair_quality.lp9.json`
- Generate quality-gated LP20 minimal pairs: `qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp20_orphaned_preposition.contexts.yaml --out data/generated/minimal_pairs.lp20.jsonl --report reports/minimal_pair_quality.lp20.json`
- Validate generated LP20 minimal pairs: `qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp20.jsonl --context rules/lp20_orphaned_preposition.contexts.yaml --report reports/minimal_pair_quality.lp20.json`
- Contamination guardrail: `qfr contamination-check --train <train-data.json> --holdout <holdout-data.json> --threshold 0.92 --out reports/contamination_report.json` (detects train/holdout overlap risk).
- Release metrics reporting: `qfr release-report --metrics <metrics-data.json> --out-json <report.json> --out-md <report.md>` (evaluates release readiness against gates).
- Generated records are blocked when grammar, normalization, duplicate, lexical-quality, or context-binding authorization checks fail.

- Validate LP taxonomy: `qfr validate-taxonomy --taxonomy eval/lp9_error_taxonomy.yaml` (repeat for LP20).
- Generate LP9/LP20 diagnostics: `qfr diagnose-eval --input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --taxonomy eval/lp9_error_taxonomy.yaml --taxonomy eval/lp20_error_taxonomy.yaml --out-json reports/diagnostics.lp9_lp20.json --out-md reports/diagnostics.lp9_lp20.md`.
- Legacy semantic CSV compatibility (maintained diagnostics engine): `python tools/pipeline_ops.py diagnose-semantic --in-csv fixtures/diagnostics/lp9_lp20_legacy_semantic.csv --out-json reports/diagnostics.legacy_semantic.json`.
- Explicit maintained command for legacy CSV: `qfr diagnose-legacy-csv --in-csv fixtures/diagnostics/lp9_lp20_legacy_semantic.csv --out-json reports/diagnostics.legacy_semantic.json`.
- Optional release diagnostics integration: `qfr release-report --metrics <metrics-data.json> --diagnostics reports/diagnostics.lp9_lp20.json --out-json <report.json> --out-md <report.md>`.

- GitHub Actions CI is kept in `.github/workflows/ci.yml`; for local execution use: `bash scripts/run_local_validation.sh`.

- Unified release-candidate orchestration: `qfr release-candidate --metrics <metrics-data.json> --diagnostics-input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --out-json reports/release_candidate.json --out-md reports/release_candidate.md`.

## Data pipeline demo
Use legacy + package commands to generate deterministic artifacts under `reports/data_pipeline_demo/`.

- Validate corpus source contract: `qfr validate-corpus-sources --manifest manifests/corpus_source_manifest.template.yaml`.
- Ingest contract-approved local corpus fixtures: `qfr ingest-corpus-sources --manifest manifests/corpus_source_manifest.template.yaml --out reports/corpus_ingestion/harvest.jsonl --report reports/corpus_ingestion/report.json --min-chars 20`.

- Validate curation policy: `qfr validate-curation-policy --policy manifests/curation_policy_manifest.template.yaml`.
- Curate ingested corpus deterministically: `qfr curate-corpus --input reports/corpus_ingestion/harvest.jsonl --policy manifests/curation_policy_manifest.template.yaml --out-dir reports/corpus_curation`.

## Real corpus download workflow
The committed template corpus is intentionally tiny. To materialize real remote text sources, use the separate real-download manifest and downloader:

```bash
bash scripts/bootstrap_dev_env.sh
python tools/download_real_corpus_sources.py \
  --manifest manifests/corpus_source_manifest.real_downloads.yaml \
  --out-manifest manifests/corpus_source_manifest.real_downloaded.local.yaml \
  --report reports/corpus_ingestion/downloads.real.json \
  --resume \
  --skip-existing \
  --retries 3 \
  --retry-delay-seconds 2 \
  --timeout 30 \
  --allow-partial \
  --fail-under-downloaded 5

qfr validate-corpus-sources --manifest manifests/corpus_source_manifest.real_downloaded.local.yaml
qfr ingest-corpus-sources \
  --manifest manifests/corpus_source_manifest.real_downloaded.local.yaml \
  --out reports/corpus_ingestion/harvest.real.jsonl \
  --report reports/corpus_ingestion/report.real.json \
  --min-chars 80
qfr curate-corpus \
  --input reports/corpus_ingestion/harvest.real.jsonl \
  --policy manifests/curation_policy_manifest.template.yaml \
  --out-dir reports/corpus_curation_real
```

`manifests/corpus_source_manifest.real_downloads.yaml` uses `future_remote` entries plus `download_url` metadata so normal deterministic validation does not download network content. The downloader creates local UTF-8 text files under `data/corpus/raw/` and emits a generated local manifest that the existing ingestion command can validate and ingest.

Retry-only resume command (useful after transient Gutenberg errors):

```bash
python tools/download_real_corpus_sources.py \
  --manifest manifests/corpus_source_manifest.real_downloads.yaml \
  --out-manifest manifests/corpus_source_manifest.real_downloaded.local.yaml \
  --report reports/corpus_ingestion/downloads.real.json \
  --resume \
  --skip-existing \
  --retries 5 \
  --allow-partial \
  --fail-under-downloaded 5
```

Review licensing/jurisdiction before using downloaded text for an actual training release. The default real-download manifest uses Project Gutenberg links, which are public-domain-oriented but still require checking local copyright and Project Gutenberg terms.

## Local real pack build (T26)
The previous local real run produced `0` ingested records because Gutenberg files were read line by line, and hard-wrapped lines were shorter than `--min-chars 80`. Ingestion now performs paragraph-level reconstruction for plain text sources and filters common Gutenberg boilerplate.

Use fixture-scale pack build in CI for deterministic checks, and use the local real workflow only for local experimentation:

```bash
bash scripts/run_local_real_pack.sh
```

This script runs real local ingestion/curation plus live modern acquisition and then builds `reports/training_pack_real/`.
If transient remote failures happen, the downloader resumes from cached files, accepts partial local runs once at least five downloads are available, and prints a warning to rerun later for missing sources.

Mode guidance:
- `manifests/training_pack_policy.template.yaml` is `fixture_ci` and intentionally tiny.
- `manifests/training_pack_policy.local_real.template.yaml` is `local_research` and can include noncommercial sources for research training experiments.
- `production_commercial` mode is stricter and rejects noncommercial/permission-required/unknown commercial-use records.

Commercial caveat:
- Assemblée nationale content remains noncommercial/permission-required unless explicit commercial permission is obtained.
- A pack can be technically trainable for research while still blocked for commercial production release.

## T21 Curated Split
Use `qfr validate-split-policy` and `qfr split-curated-corpus` to produce `reports/curated_splits/*` from accepted curated corpus only.

## Dolphin3 + Unsloth local training path

Use this path when you want to continue Dolphin's existing instruction/personality tuning. The correct non-GGUF training base is `dphn/Dolphin3.0-Qwen2.5-3b`. The clean upstream base is `Qwen/Qwen2.5-3B`, useful for ablations only. The GGUF model `bartowski/Dolphin3.0-Qwen2.5-3b-GGUF` remains runtime-only and must not be used as the training source.

See:

- `project/model_selection.dolphin3_unsloth.yaml`
- `docs/dolphin3_unsloth_training.md`
- `requirements/training-dolphin3-unsloth.txt`
- `scripts/train_qfr_dolphin3_unsloth_lora.py`
- `scripts/export_qfr_dolphin3_unsloth_gguf.py`

Smoke-test commands:

```bash
python3 -m venv .venv-unsloth
source .venv-unsloth/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements/training-dolphin3-unsloth.txt

python scripts/train_qfr_dolphin3_unsloth_lora.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --train reports/curated_splits/train.jsonl \
  --eval reports/curated_splits/test.jsonl \
  --output-dir models/qfr-dolphin3-qwen25-3b-lora-smoke \
  --max-steps 10
```

Direct Unsloth GGUF export after training:

```bash
python scripts/export_qfr_dolphin3_unsloth_gguf.py \
  --base-model dphn/Dolphin3.0-Qwen2.5-3b \
  --adapter models/qfr-dolphin3-qwen25-3b-lora-smoke \
  --merged-16bit-dir models/qfr-dolphin3-qwen25-3b-merged \
  --gguf-dir models/qfr-dolphin3-qwen25-3b-gguf \
  --quantization q4_k_m
```

## Fresh checkout
Install dependencies before any `qfr` or governance commands:

```bash
python3 -m pip install -e ".[dev]"
# or:
bash scripts/bootstrap_dev_env.sh

python3 tools/update_agents.py --ensure-nested
python3 tools/update_agents.py --validate
qfr validate
bash scripts/run_local_validation.sh
```

`qfr` is installed by the editable package install. `tools/update_agents.py` imports PyYAML, so dependencies must be installed first. No top-level `qfr` wrapper is intentionally provided.


## Modern corpus acquisition (T23/T24)
- `qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition_manifest.template.yaml`
- `qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition.donnees_quebec.template.yaml`
- `qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition.assnat_seed.template.yaml`
- `qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition_manifest.template.yaml --out reports/modern_corpus/harvest.jsonl --report reports/modern_corpus/dry_run_report.json --max-documents 0`
- `qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition.donnees_quebec.template.yaml --out reports/modern_corpus/donnees_quebec_fixture_harvest.jsonl --report reports/modern_corpus/donnees_quebec_fixture_report.json --max-documents 25`
- `qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition.assnat_seed.template.yaml --out reports/modern_corpus/assnat_fixture_harvest.jsonl --report reports/modern_corpus/assnat_fixture_report.json --include-noncommercial --max-documents 10`
- `qfr audit-corpus-readiness --input reports/modern_corpus/donnees_quebec_fixture_harvest.jsonl --out reports/corpus_readiness/modern_fixture_report.json`

`manifests/modern_corpus_acquisition.donnees_quebec.template.yaml` runs deterministic fixture-mode CKAN extraction in CI/local validation and only extracts dataset/resource metadata text (no arbitrary resource file download in T24).

`manifests/modern_corpus_acquisition.assnat_seed.template.yaml` is explicit-seed only (no crawling). Assemblée nationale records are noncommercial unless explicit commercial permission is obtained.

Use `--fixture-mode` to hard-require adapter fixtures for networked adapters, `--timeout` to tune request limits, and `--fail-on-empty` when empty harvests should fail the command.

Holdouts (`QFrCoLA`, `QFrBLiMP`, `QFrCoRE/QFrCoRT`, `COLE`) are evaluation-only by default and must not be used for training.

## Training pack builder (T25)
- `qfr validate-training-pack-policy --policy manifests/training_pack_policy.template.yaml`
- `qfr validate-training-pack-policy --policy manifests/training_pack_policy.local_real.template.yaml`
- `qfr build-training-pack --policy manifests/training_pack_policy.template.yaml --out-dir reports/training_pack`
- `qfr audit-training-pack --pack-dir reports/training_pack --out reports/training_pack/audit.json`
- Optional gate: `qfr audit-training-pack --pack-dir reports/training_pack --out reports/training_pack/audit.json --fail-below pilot_lora_candidate`

`manifests/training_pack_policy.template.yaml` defines deterministic source merging, global deduplication, source/share balancing, and instructionization into Qwen ChatML-compatible examples for Dolphin3/Unsloth continuation training inputs.

Safety behavior is strict by default:
- holdout and holdout-like records are rejected;
- non-trainable or review-gated records are rejected unless explicitly permission-granted;
- permission-required/noncommercial material is excluded unless a local permission manifest is provided;
- duplicate exact and normalized text is rejected.

Readiness thresholds are enforced in policy:
- smoke test: `500000` tokens
- pilot LoRA candidate: `20000000` tokens
- production LoRA candidate: `150000000` tokens and `200000` instruction examples

Current deterministic fixture-scale outputs are expected to be smoke/pilot-oriented only and not production-ready; production requires much larger volume, broader source diversity, and commercially safe licensing coverage.
