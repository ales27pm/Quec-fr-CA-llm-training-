# Quec-fr-CA-llm-training

## Strategic planning
This repository includes a sovereignty-first roadmap and centralized release thresholds for the Québécois (fr-CA) training program.
- Roadmap: [docs/strategic_roadmap.md](docs/strategic_roadmap.md)
- Threshold source of truth: [project/release_gates.yaml](project/release_gates.yaml)


## Executable Pipeline
- Install: `pip install -e .[dev]`
- Validate repo: `qfr validate`
- Generate LP9 minimal pairs: `qfr generate-minimal-pairs --rule rules/lp_rule_manifest.template.yaml --out data/generated/minimal_pairs.lp9.jsonl`
- Contamination check: `qfr contamination-check --train train.json --holdout holdout.json --threshold 0.92 --out reports/contamination_report.json`
- Release report: `qfr release-report --metrics metrics.json --out-json reports/release_report.json --out-md reports/release_report.md`
