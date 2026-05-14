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
- Optional release diagnostics integration: `qfr release-report --metrics <metrics-data.json> --diagnostics reports/diagnostics.lp9_lp20.json --out-json <report.json> --out-md <report.md>`.

- GitHub Actions CI is kept in `.github/workflows/ci.yml`; for local execution use: `bash scripts/run_local_validation.sh`.
