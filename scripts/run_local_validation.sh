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

echo "==> Taxonomy validation"
qfr validate-taxonomy --taxonomy eval/lp9_error_taxonomy.yaml
qfr validate-taxonomy --taxonomy eval/lp20_error_taxonomy.yaml

echo "==> Diagnostics generation"
qfr diagnose-eval --input fixtures/diagnostics/lp9_lp20_eval_sample.jsonl --taxonomy eval/lp9_error_taxonomy.yaml --taxonomy eval/lp20_error_taxonomy.yaml --out-json reports/diagnostics.lp9_lp20.json --out-md reports/diagnostics.lp9_lp20.md

echo "==> LP9 minimal pairs"
qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp9_lexical_semantics.contexts.yaml --out data/generated/minimal_pairs.lp9.jsonl --report reports/minimal_pair_quality.lp9.json
qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp9.jsonl --context rules/lp9_lexical_semantics.contexts.yaml --report reports/minimal_pair_quality.lp9.json

echo "==> LP20 minimal pairs"
qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --context rules/lp20_orphaned_preposition.contexts.yaml --out data/generated/minimal_pairs.lp20.jsonl --report reports/minimal_pair_quality.lp20.json
qfr validate-minimal-pairs --input data/generated/minimal_pairs.lp20.jsonl --context rules/lp20_orphaned_preposition.contexts.yaml --report reports/minimal_pair_quality.lp20.json

echo "==> Release report"
qfr release-report --metrics fixtures/valid_metrics.json --diagnostics reports/diagnostics.lp9_lp20.json --out-json reports/release_report.json --out-md reports/release_report.md

echo "==> Deterministic artifact guard"
git diff --exit-code -- data/generated/minimal_pairs.lp9.jsonl reports/minimal_pair_quality.lp9.json data/generated/minimal_pairs.lp20.jsonl reports/minimal_pair_quality.lp20.json reports/diagnostics.lp9_lp20.json reports/diagnostics.lp9_lp20.md reports/release_report.json reports/release_report.md
