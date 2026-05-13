# Québec French Training Pipeline Assets

This repository now includes starter assets to bootstrap a top-tier fr-CA pipeline:

- `manifests/dataset_manifest.template.yaml`: data governance manifest template.
- `rules/lp_rule_manifest.template.yaml`: per-linguistic-phenomenon rule manifest.
- `eval/evaluation_manifest.template.yaml`: evaluation and release-gate manifest.
- `tools/minimal_pair_generator_spec.md`: synthetic minimal pair generation spec.

Use these files as canonical templates and version them alongside training runs.

- `docs/iterative_training_loop.md`: operator runbook for threshold-driven iterative alignment loops.
