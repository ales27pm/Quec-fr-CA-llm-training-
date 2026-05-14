# Release Candidate: PASS

- Deterministic timestamp: `project-status-last-updated:2026-05-14`

## Stages
- repository_validation: PASS
- taxonomy_validation: PASS
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
