#!/usr/bin/env bash
set -euo pipefail

echo "==> Install"
python -m pip install --upgrade pip
pip install -e .[dev]

echo "==> Lint"
ruff check .

echo "==> Tests"
python -m pytest

echo "==> Governance validation"
python3 tools/update_agents.py --validate

echo "==> Repository validation"
qfr validate

echo "==> Modern corpus acquisition contracts"
qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition_manifest.template.yaml
qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition.donnees_quebec.template.yaml
qfr validate-modern-corpus --manifest manifests/modern_corpus_acquisition.assnat_seed.template.yaml
qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition_manifest.template.yaml --out reports/modern_corpus/harvest.jsonl --report reports/modern_corpus/dry_run_report.json --max-documents 0
qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition.donnees_quebec.template.yaml --out reports/modern_corpus/donnees_quebec_fixture_harvest.jsonl --report reports/modern_corpus/donnees_quebec_fixture_report.json --max-documents 25
qfr acquire-modern-corpus --manifest manifests/modern_corpus_acquisition.assnat_seed.template.yaml --out reports/modern_corpus/assnat_fixture_harvest.jsonl --report reports/modern_corpus/assnat_fixture_report.json --include-noncommercial --max-documents 10
qfr audit-corpus-readiness --input reports/modern_corpus/donnees_quebec_fixture_harvest.jsonl --out reports/corpus_readiness/modern_fixture_report.json
qfr validate-training-pack-policy --policy manifests/training_pack_policy.template.yaml
qfr build-training-pack --policy manifests/training_pack_policy.template.yaml --out-dir reports/training_pack
qfr audit-training-pack --pack-dir reports/training_pack --out reports/training_pack/audit.json


echo "==> Corpus source contract"
qfr validate-corpus-sources --manifest manifests/corpus_source_manifest.template.yaml
qfr ingest-corpus-sources --manifest manifests/corpus_source_manifest.template.yaml --out reports/corpus_ingestion/harvest.jsonl --report reports/corpus_ingestion/report.json --min-chars 20
qfr validate-curation-policy --policy manifests/curation_policy_manifest.template.yaml
qfr curate-corpus --input reports/corpus_ingestion/harvest.jsonl --policy manifests/curation_policy_manifest.template.yaml --out-dir reports/corpus_curation
qfr validate-split-policy --policy manifests/split_policy_manifest.template.yaml
qfr split-curated-corpus --input reports/corpus_curation/accepted.jsonl --policy manifests/split_policy_manifest.template.yaml --out-dir reports/curated_splits
qfr validate-training-export --manifest manifests/training_export_manifest.template.yaml
qfr export-training-dataset --manifest manifests/training_export_manifest.template.yaml --out-dir reports/training_export

echo "==> Taxonomy validation"
qfr validate-taxonomy --taxonomy eval/lp9_error_taxonomy.yaml
qfr validate-taxonomy --taxonomy eval/lp20_error_taxonomy.yaml

echo "==> Diagnostics generation"
qfr diagnose-eval --input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --taxonomy eval/lp9_error_taxonomy.yaml --taxonomy eval/lp20_error_taxonomy.yaml --out-json reports/diagnostics.lp9_lp20.json --out-md reports/diagnostics.lp9_lp20.md

echo "==> Legacy diagnostics compatibility"
# Intentionally sequential with a shared output path: the CLI run overwrites the pipeline_ops run so we can
# verify both execution paths succeed and deterministically converge on reports/diagnostics.legacy_semantic.json.
python tools/pipeline_ops.py diagnose-semantic --in-csv fixtures/diagnostics/lp9_lp20_legacy_semantic.csv --out-json reports/diagnostics.legacy_semantic.json
qfr diagnose-legacy-csv --in-csv fixtures/diagnostics/lp9_lp20_legacy_semantic.csv --out-json reports/diagnostics.legacy_semantic.json

echo "==> LP9 minimal pairs"
qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp9_lexical_semantics.contexts.yaml --out data/generated/minimal_pairs.lp9.jsonl --report reports/minimal_pair_quality.lp9.json
qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp9.jsonl --context rules/lp9_lexical_semantics.contexts.yaml --report reports/minimal_pair_quality.lp9.json

echo "==> LP20 minimal pairs"
qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp20_orphaned_preposition.contexts.yaml --out data/generated/minimal_pairs.lp20.jsonl --report reports/minimal_pair_quality.lp20.json
qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp20.jsonl --context rules/lp20_orphaned_preposition.contexts.yaml --report reports/minimal_pair_quality.lp20.json

echo "==> Release report"
qfr release-report --metrics fixtures/valid_metrics.json --diagnostics reports/diagnostics.lp9_lp20.json --out-json reports/release_report.json --out-md reports/release_report.md

echo "==> Release candidate"
qfr release-candidate --metrics fixtures/valid_metrics.json --diagnostics-input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --out-json reports/release_candidate.json --out-md reports/release_candidate.md

echo "==> Data pipeline demo"

python tools/pipeline_ops.py harvest --inputs fixtures/data_pipeline/raw_sample_1.txt fixtures/data_pipeline/raw_sample_2.txt --out reports/data_pipeline_demo/harvest.jsonl --min-chars 20
qfr curate --in reports/data_pipeline_demo/harvest.jsonl --out reports/data_pipeline_demo/curated.jsonl --min-fr-ca-score 0.05
qfr edit-normative --in reports/data_pipeline_demo/curated.jsonl --out reports/data_pipeline_demo/edited.jsonl
qfr split --in reports/data_pipeline_demo/edited.jsonl --out-dir reports/data_pipeline_demo/splits --seed 42
qfr training-recipe --data-dir reports/data_pipeline_demo/splits --out reports/data_pipeline_demo/training_recipe.yaml
qfr monitor-lp7 --pre 0.95 --post 0.94 --release-gates project/release_gates.yaml --out reports/data_pipeline_demo/lp7_monitor.json

echo "==> Deterministic artifact guard"
git diff --exit-code -- data/generated/minimal_pairs.lp9.jsonl reports/minimal_pair_quality.lp9.json data/generated/minimal_pairs.lp20.jsonl reports/minimal_pair_quality.lp20.json reports/diagnostics.lp9_lp20.json reports/diagnostics.lp9_lp20.md reports/diagnostics.legacy_semantic.json reports/release_report.json reports/release_report.md reports/release_candidate.json reports/release_candidate.md reports/data_pipeline_demo/harvest.jsonl reports/data_pipeline_demo/curated.jsonl reports/data_pipeline_demo/edited.jsonl reports/data_pipeline_demo/splits/train.jsonl reports/data_pipeline_demo/splits/dev.jsonl reports/data_pipeline_demo/splits/test.jsonl reports/data_pipeline_demo/training_recipe.yaml reports/data_pipeline_demo/lp7_monitor.json reports/corpus_ingestion/harvest.jsonl reports/corpus_ingestion/report.json reports/corpus_curation/accepted.jsonl reports/corpus_curation/review_required.jsonl reports/corpus_curation/quarantine.jsonl reports/corpus_curation/rejected.jsonl reports/corpus_curation/report.json reports/curated_splits/train.jsonl reports/curated_splits/dev.jsonl reports/curated_splits/test.jsonl reports/curated_splits/split_report.json reports/training_export/training_manifest.json reports/training_export/training_manifest.yaml reports/training_export/dataset_card.md reports/training_export/export_report.json reports/modern_corpus/harvest.jsonl reports/modern_corpus/dry_run_report.json reports/modern_corpus/donnees_quebec_fixture_harvest.jsonl reports/modern_corpus/donnees_quebec_fixture_report.json reports/modern_corpus/assnat_fixture_harvest.jsonl reports/modern_corpus/assnat_fixture_report.json reports/corpus_readiness/modern_fixture_report.json reports/training_pack/train.jsonl reports/training_pack/dev.jsonl reports/training_pack/test.jsonl reports/training_pack/report.json reports/training_pack/dataset_card.md reports/training_pack/audit.json
