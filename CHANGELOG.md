# Changelog

## 2026-05-16 — T30 LP9 lexical micro-pack + evaluation harness

- Added `manifests/lp9_lexical_preference_pack.template.yaml` and strict manifest validation model support for deterministic LP9 lexical preference steering packs.
- Added `src/qfr_pipeline/lp9_micro_pack.py` to generate deterministic training-pack-style LP9 train/dev JSONL artifacts with Qwen ChatML rendering, task-type coverage, split disjointness, and duplicate-hash guardrails.
- Added `src/qfr_pipeline/lp9_eval.py` to generate deterministic LP9 lexical eval prompts and reusable preferred/forbidden lexical scoring helpers.
- Added `scripts/evaluate_lp9_adapter.py` to run manual local base-vs-adapter LP9 evaluation with greedy decoding, per-prompt scoring, per-pair/per-task aggregation, and generations/report artifacts.
- Added `qfr generate-lp9-micro-pack` and `qfr generate-lp9-eval` CLI commands (model-free) plus `scripts/generate_lp9_micro_pack.sh`.
- Added `qfr generate-lp9-failure-pack` CLI command and `manifests/lp9_failure_mining_policy.template.yaml` for failure-driven LP9 hard-negative pack generation from eval failures.
- Added LP9-focused tests covering manifest validation, micro-pack split/duplicate/task guarantees, failure-mining report-driven generation, training-pack loader compatibility, eval prompt schema/scoring behavior, fake base-vs-adapter report scoring, and path-leakage safeguards.
- Added deterministic artifacts under `reports/lp9_micro_pack/` and `reports/lp9_eval/`, plus governance wiring for T30 in `project/status.json` and `tools/update_agents.py`.

## 2026-05-16 — T29 Unsloth training-pack compatibility and local smoke policy

- Updated `scripts/train_qfr_dolphin3_unsloth_lora.py` to support `--input-format {auto,curated,training-pack}` with deterministic auto-detection and format-specific validation.
- Preserved strict curated-mode enforcement (`curation_label == accepted` + required curation metadata) while adding training-pack-mode support for `messages`, pre-rendered `text`, and compatible `prompt`/`assistant_text` rows.
- Added training-pack safety checks in the trainer input loader: holdout rejection, `allowed_for_training=false` rejection, review-gated rejection without explicit `permission_granted=true`, and local-research warnings for `commercial_use` values requiring non-commercial handling.
- Added pre-training input summary output (input format, train/eval row counts, task-type counts, source counts, local-research warning count).
- Added `manifests/training_pack_policy.local_smoke.template.yaml` and `scripts/build_local_smoke_pack.sh` for local technical smoke/pilot pack generation without model training.
- Added `tests/test_dolphin3_unsloth_training_inputs.py` covering curated/training-pack loaders, auto-detection, safety rejections, ChatML rendering fallback, smoke-policy validation, no-training smoke script guard, and no tracked model/GGUF output guard.
- Added local-research blocking reasons in training-pack reporting for smoke-style packs (`source_family_dominance`, `insufficient_modern_diversity`) so local high-volume packs remain explicitly non-production.

## 2026-05-16 — T27 resilient real corpus downloader

- Added resilient retry-aware behavior to `tools/download_real_corpus_sources.py` with transient network retry handling (timeouts, reset/unreachable errors, HTTP 429, HTTP 5xx), permanent-failure no-retry behavior (blocked schemes, malformed URLs, HTTP 404), and per-source attempt/error/status accounting.
- Added resumable cache reuse controls (`--resume`, `--skip-existing`) so existing downloaded files are treated as cached with recomputed character/SHA-256 metadata instead of being discarded.
- Added partial-success controls (`--allow-partial`, `--fail-under-downloaded`) and richer report fields (`downloaded_count`, `cached_count`, `failed_count`, `skipped_count`, `failed_sources`, `minimum_required_downloaded`, `partial_ok`) while keeping failed sources out of generated local manifests.
- Updated `scripts/run_local_real_pack.sh` to use retries/resume/partial-threshold download flow and to print an explicit warning when partial local corpus download is accepted.
- Added deterministic unit tests for cache reuse, retry success/failure paths, partial acceptance gating, out-manifest filtering, skipped-holdout handling, report ordering, and path-leak prevention.

## 2026-05-16 — T26 real ingestion and local real pack workflow

- Fixed corpus ingestion for real downloaded plain-text sources: `qfr ingest-corpus-sources` now performs paragraph-level reconstruction for hard-wrapped text, filters common Project Gutenberg boilerplate, supports line-level fallback for no-blank-line files, and keeps deterministic normalized-text deduplication.
- Preserved corpus-source metadata in ingestion records (`source_id`, source provenance/licensing flags, register/dialect, training/eval gates) while emitting curation/training-pack-compatible fields and repo-relative paths.
- Added provenance-aware curation scoring bonus (`source_trust:quebec_fr_validated_provenance`) so trusted Québec/French-Canadian sources are not systematically dropped when lexical marker density is low, without overriding holdout or review/permission gating.
- Added `manifests/training_pack_policy.local_real.template.yaml` with `pack_mode: local_research` and live-input-aware source list (real curated corpus + optional live modern harvest fallbacks).
- Extended training-pack builder reporting and gating with `pack_mode`, optional-input accounting, commercial-readiness status, and explicit commercial blocking reasons; local research builds remain non-commercial-release-ready by design.
- Added `scripts/run_local_real_pack.sh` to run the full local real ingestion/curation/live-modern-acquisition/training-pack pipeline without running model training.
- Added/updated tests for real plain-text ingestion, provenance curation behavior, and local-research vs production-commercial training-pack behavior.

## 2026-05-16 — T25 deterministic training-pack builder

- Added `manifests/training_pack_policy.template.yaml` and strict schema validation for production-oriented training pack policy controls (split ratios, source input contracts, safety gates, balancing, instructionization, and readiness thresholds).
- Added `src/qfr_pipeline/training_pack.py` with deterministic source merge, global duplicate suppression (exact + normalized), holdout/permission/commercial safety filtering, source and source-family dominance caps, instructionization strategies, and train/dev/test JSONL emission.
- Added Qwen ChatML-compatible example rendering and metadata-preserving provenance so output artifacts can be consumed by Dolphin3/Qwen continuation training workflows (including Unsloth pipelines).
- Added `qfr` CLI commands: `validate-training-pack-policy`, `build-training-pack`, and `audit-training-pack` with readiness-level gating support.
- Integrated training-pack validation/build/readiness stages into `qfr release-candidate` and surfaced training-pack readiness/count/token/blocker fields in release-candidate JSON/Markdown summaries.
- Added deterministic fixture-scale artifacts under `reports/training_pack/` (`train.jsonl`, `dev.jsonl`, `test.jsonl`, `report.json`, `dataset_card.md`, `audit.json`) and a full T25 test suite in `tests/test_training_pack.py`.
- Updated CI and local validation workflows to build/audit the training pack without requiring live external network acquisition.

## 2026-05-16 — T24 modern Québec source adapters (fixture-safe)

- Implemented production-grade `donnees_quebec_ckan` metadata extraction: deterministic per-package records, package/resource metadata text normalization, package-level license gating, duplicate suppression via normalized text hash, fixture-response mode, and detailed package/license/organization/domain reporting.
- Implemented safer `assnat_journal_debats` acquisition: explicit seed-only fetches, noncommercial explicit-flag gating, fixture `file://` support, deterministic paragraph segmentation, navigation junk filtering, and no-crawl behavior.
- Added new manifests and fixtures: `manifests/modern_corpus_acquisition.donnees_quebec.template.yaml`, `manifests/modern_corpus_acquisition.assnat_seed.template.yaml`, and `fixtures/modern_corpus/*`.
- Extended CLI acquisition controls with `--timeout`, `--fixture-mode`, and `--fail-on-empty`, plus concise acquisition summary output.
- Extended readiness reporting with acquisition source-type summary, license/commercial summaries, modern/institutional ratios, and targeted corpus-mix recommendations.
- Integrated T24 fixture-aware fields into release-candidate summary without making live network acquisition mandatory.
- Updated CI/local validation to run fixture-only T24 acquisition/audit commands and guard deterministic modern-corpus artifacts.

## 2026-05-15 — Dolphin3 + Unsloth continuation path

- Added `project/model_selection.dolphin3_unsloth.yaml` to make `dphn/Dolphin3.0-Qwen2.5-3b` the preferred base when preserving Dolphin instruction/personality tuning.
- Added `requirements/training-dolphin3-unsloth.txt` and Unsloth-based scripts for QLoRA training plus merged/GGUF export.
- Added `docs/dolphin3_unsloth_training.md` with RTX 2070-friendly smoke-test and `q4_k_m` export commands.
- Documented that `bartowski/Dolphin3.0-Qwen2.5-3b-GGUF` is runtime-only and must not be used as a training source.

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

## T23

- Added modern corpus acquisition and readiness auditing scaffolding with legal gating and holdout separation.

- T23 cleanup/finalization: hardened deterministic modern-corpus acquisition and readiness auditing; removed backup artifact; integrated release-candidate dry-run/readiness stages.
