# Release Candidate: PASS

- Deterministic timestamp: `project-status-last-updated:2026-05-16`

## Stages
- repository_validation: PASS
- taxonomy_validation: PASS
- modern_corpus_manifest_validation: PASS
  - Artifacts: ['manifests/modern_corpus_acquisition_manifest.template.yaml']
- modern_corpus_acquisition_dry_run: PASS
  - Artifacts: ['reports/modern_corpus/dry_run_report.json']
- corpus_source_validation: PASS
- corpus_ingestion: PASS
  - Artifacts: ['reports/corpus_ingestion/harvest.jsonl', 'reports/corpus_ingestion/report.json']
- curation_policy_validation: PASS
  - Artifacts: ['manifests/curation_policy_manifest.template.yaml']
- corpus_curation: PASS
  - Artifacts: ['reports/corpus_curation/accepted.jsonl', 'reports/corpus_curation/review_required.jsonl', 'reports/corpus_curation/quarantine.jsonl', 'reports/corpus_curation/rejected.jsonl', 'reports/corpus_curation/report.json']
- corpus_readiness_audit: PASS
  - Artifacts: ['reports/corpus_readiness/report.json']
- split_policy_validation: PASS
  - Artifacts: ['manifests/split_policy_manifest.template.yaml']
- curated_corpus_split: PASS
  - Artifacts: ['reports/curated_splits/train.jsonl', 'reports/curated_splits/dev.jsonl', 'reports/curated_splits/test.jsonl', 'reports/curated_splits/split_report.json']
- training_export_manifest_validation: PASS
  - Artifacts: ['manifests/training_export_manifest.template.yaml']
- training_export_generation: PASS
  - Artifacts: ['reports/training_export/training_manifest.json', 'reports/training_export/training_manifest.yaml', 'reports/training_export/dataset_card.md', 'reports/training_export/export_report.json']
- training_pack_policy_validation: PASS
  - Artifacts: ['manifests/training_pack_policy.template.yaml']
- training_pack_build: PASS
  - Artifacts: ['reports/training_pack/train.jsonl', 'reports/training_pack/dev.jsonl', 'reports/training_pack/test.jsonl', 'reports/training_pack/report.json', 'reports/training_pack/dataset_card.md']
- training_pack_readiness: PASS
  - Artifacts: ['reports/training_pack/audit.json']
- diagnostics_generation: PASS
  - Artifacts: ['reports/diagnostics.lp9_lp20.json', 'reports/diagnostics.lp9_lp20.md']
- lp9_minimal_pairs: PASS
  - Artifacts: ['data/generated/minimal_pairs.lp9.jsonl', 'reports/minimal_pair_quality.lp9.json']
- lp20_minimal_pairs: PASS
  - Artifacts: ['data/generated/minimal_pairs.lp20.jsonl', 'reports/minimal_pair_quality.lp20.json']
- release_report: PASS
  - Artifacts: ['reports/release_report.json', 'reports/release_report.md']

## Release report summary
- Passed: `True`

## Diagnostics summary
- OK: `True`
- Phenomena: `['LP20:orphaned_preposition', 'LP9:lexical_semantics']`

## Minimal-pair summary
- LP9 records: `3`
- LP20 records: `8`

## Corpus curation summary
- Accepted: `2`
- Review required: `0`
- Quarantine: `1`
- Rejected: `0`

## Curated split summary
- Records total: `2`
- Train: `1`
- Dev: `0`
- Test: `1`

## Training export summary
- Dataset: `quebec-fr-curated-training`
- Version: `0.1.0`
- Total: `2`
- Train/Dev/Test: `1`/`0`/`1`
- Aggregate SHA-256: `7eebddfcab2d35fd854d06fdcb99c7e8e678b3d76d085ba7bcef8c641c69647a`

## Training pack summary
- Readiness level: `insufficient`
- Examples total: `3`
- Train/Dev/Test examples: `1`/`1`/`1`
- Estimated tokens: `406`
- Blocking reasons: `['single_source_dominance', 'single_source_family_dominance']`
- Pack mode: `fixture_ci`
- Commercial release ready: `False`
