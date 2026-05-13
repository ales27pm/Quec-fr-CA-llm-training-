# Changelog

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
