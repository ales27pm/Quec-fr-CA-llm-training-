# Changelog

## 2026-05-15 — T20 corpus curation policy and deterministic quality scoring contract
- Added `manifests/curation_policy_manifest.template.yaml` and strict curation schema validation models.
- Added deterministic corpus curation module and CLI commands: `qfr validate-curation-policy` and `qfr curate-corpus`.
- Integrated curation policy validation and curation artifacts into repository validation, CI/local validation, and release-candidate orchestration reports.
- Resolved Pydantic field shadowing by aliasing `register` to internal `language_register` in corpus source schemas.
- Added curation and regression tests including duplicate detection, output accounting, and no field-shadow warning checks.

## 2026-05-15 — T19 corpus source manifest ingestion contract
- Added `manifests/corpus_source_manifest.template.yaml` with strict source-policy fields for licensing, provenance, holdout safety, review gating, and fixture-only local paths.
- Added `CorpusSourceManifest`/`CorpusSourceEntry` strict schema models and repository validation wiring (`qfr validate` now checks corpus manifests).
- Added `qfr_pipeline.corpus_sources` deterministic ingestion contract logic and new CLI commands: `qfr validate-corpus-sources` and `qfr ingest-corpus-sources`.
- Added deterministic ingestion artifacts under `reports/corpus_ingestion/` and policy-focused pytest coverage for allowed/blocked source handling.
- Extended CI/local validation and deterministic-artifact guard to include corpus source validation/ingestion outputs.


## 2026-05-14 — T17 legacy diagnostics compatibility wrapper
- Added `src/qfr_pipeline/legacy_diagnostics.py` with legacy CSV ingestion and LP9/LP20 alias mapping, then delegated diagnostics execution to maintained `qfr_pipeline.diagnostics` taxonomy-aware logic.
- Refactored `tools/pipeline_ops.py diagnose-semantic` into a compatibility wrapper over maintained diagnostics logic (kept `--in-csv`/`--out-json`; added optional `--taxonomy` and `--allow-missing-phenomena`).
- Added `qfr diagnose-legacy-csv` and deterministic legacy diagnostics fixture/artifact coverage.
- Added legacy compatibility tests (adapter mapping, malformed behavior, blocking/non-blocking outcomes, CLI parity) and removed standalone semantic diagnostics engine duplication from `tools/pipeline_ops.py`.
- Extended CI/local validation plus deterministic stale-artifact guard to include `reports/diagnostics.legacy_semantic.json`.

## 2026-05-14 — T16 release-candidate orchestration
- Added deterministic `qfr release-candidate` command orchestrating repository validation, taxonomy validation, LP9/LP20 diagnostics, LP9/LP20 minimal-pair generation+validation, and release report generation in one entry point.
- Added consolidated `reports/release_candidate.json` and `reports/release_candidate.md` artifacts, plus CI/local-runner integration and stale-artifact guard coverage.
- Added focused tests for success/failure/partial-report behavior and pipeline wiring guarantees without subprocess usage.


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

## 2026-05-14 — T15 LP9/LP20 diagnostic taxonomy integration
- Added first-class LP9/LP20 error taxonomy manifests and schema validation.
- Added `qfr diagnose-eval` deterministic JSON/Markdown diagnostics with semantic similarity aggregation and taxonomy-aware blocking logic.
- Extended release-report JSON/Markdown to optionally embed diagnostics and fail on blocking diagnostics issues.
- Added CI checks and deterministic diagnostics artifacts stale-guard coverage.

## 2026-05-14 — CI restoration and local validator
- Restored `.github/workflows/ci.yml` with valid LP9/LP20 diagnostics + deterministic artifact checks (including release report artifacts).
- Added `scripts/run_local_validation.sh` to run the same validation sequence locally when GitHub Actions is unreliable.
- Preserved governance status for T11/T15 without downgrade after AGENTS/status refresh.

- T18: Data-pipeline logic moved into package modules with legacy `tools/pipeline_ops.py` compatibility wrappers and qfr-native commands.

- Added T21 curated split contract and commands.

## 2026-05-15 — T22 training export manifest + dataset card
- Added deterministic training export manifest template, schema/model validation, CLI commands, and export artifacts.
- Added release-candidate integration with training export validation/generation summary.
- Added bootstrap script and fresh-checkout docs clarifying dependency/install order.
