# Changelog

## 2026-05-14 — T14 LP20 orphaned-preposition gold pairs
- Added manifest-driven LP20 orphaned-preposition context set (`rules/lp20_orphaned_preposition.contexts.yaml`) and deterministic generated artifacts (`data/generated/minimal_pairs.lp20.jsonl`, `reports/minimal_pair_quality.lp20.json`).
- Generalized minimal-pair quality/context validation for LP-specific rules, including LP20-focused checks for preposition retention, unauthorized/forged pairs, stable IDs, punctuation-only diffs, and minimum meaningful length.
- Extended CLI and CI to generate/validate LP9 and LP20 artifacts and fail on stale generated outputs.
- Added LP20 success/failure test coverage while keeping LP9 regression/context-binding guarantees intact.

## 2026-05-13
- Added governance and manifest scaffolding for Québec French LLM pipeline.
- Added dynamic AGENTS updater and nested AGENTS scaffolding.
- Added validation command support for contamination guardrails and schema presence checks.

- Added `docs/strategic_roadmap.md` with data, benchmarking, alignment, and deployment gates for a Québec-first LLM pipeline.
- Updated `README.md` with a direct roadmap pointer.

- Centralized roadmap numeric thresholds in `project/release_gates.yaml` and updated strategic docs to reference this single source of truth.
- Enhanced `README.md` strategic planning section with a short summary and direct links to roadmap and threshold config.
- Corrected LP8 label in the roadmap from "Determinants" to "Determiners" for standard terminology.

- Corrected README H1 to remove the trailing hyphen in the project title.
- Standardized roadmap gate units on 0–1 ratios and renamed LP7 alignment threshold key to `alignment.lp7_standard_negation_max_post_alignment_drop_ratio`.

- Added `docs/iterative_training_loop.md` to operationalize the strategic loop with centralized threshold enforcement, LP7 post-alignment rollback triggers, LP9/LP20 semantic diagnostics, and Float16 CTranslate2 release packaging guidance.

- Hardened `tools/update_agents.py --validate` to enforce evaluation-gate synchronization against `project/release_gates.yaml` (removing hardcoded threshold checks in validation logic).

- Added `tools/pipeline_ops.py` with concrete algorithms for data harvesting, fr-CA marker curation, normative lexical editing, train/dev/test splitting, and Float16/CTranslate2 training recipe generation.
- Extended `tools/pipeline_ops.py` with centralized-gate LP7 post-alignment monitoring (`monitor-lp7`) and LP9/LP20 semantic diagnostics + error taxonomy export (`diagnose-semantic`) for iterative alignment governance.
- Updated `docs/iterative_training_loop.md` with standardized LP7 monitoring and semantic diagnostics CLI workflows tied to `project/release_gates.yaml`.

## 2026-05-14
- Implemented executable `qfr-pipeline` package under `src/qfr_pipeline` with strict schemas, repository validation, CLI commands, release reporting, contamination checks, and LP9 minimal-pair generation.
- Added deterministic YAML/JSON IO helpers with atomic writes and UTF-8 handling.
- Added pytest coverage for schema validation, gate sync, contamination detection, minimal-pair generation, release-report pass/fail behavior, and CLI smoke validation.
- Added GitHub Actions CI workflow running `ruff`, `pytest`, `python3 tools/update_agents.py --validate`, and `qfr validate`.
- Extended `tools/update_agents.py` task mapping and validation checks for executable-phase tasks T7–T11.

- Implemented T12 gold minimal-pair quality gate with grammar-aware LP9 context manifest and deterministic validation checks.
- Removed subprocess usage from CLI AGENTS governance refresh path in favor of direct safe module invocation.

- Implemented T13 context-bound gold-data validation: minimal-pair validation now binds each JSONL record to manifest-authorized contrasts/templates, stable IDs, and context metadata; CLI/CI now enforce context authorization on generation and validation.
