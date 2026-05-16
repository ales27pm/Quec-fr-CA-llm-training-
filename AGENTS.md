# AGENTS.md — Dynamic Project Operating Contract

## Scope
This file governs the entire repository unless a deeper nested `AGENTS.md` overrides subtree-local process.

## Mission
Build and maintain a top-tier Québécois (fr-CA) LLM training pipeline aligned with OQLF/BDL standards, reproducible evaluation, and explicit governance.

## Dynamic State (auto-generated)
> Last auto-update: **2026-05-16 UTC**  
> Source: `project/status.json` (managed by `python3 tools/update_agents.py --write`)

| Task ID | Goal | Status | Evidence |
|---|---|---|---|
| T1 | Dataset manifest template | ✅ fully_implemented | `Provided` |
| T2 | LP rule manifest template | ✅ fully_implemented | `Provided` |
| T3 | Evaluation/release manifest | ✅ fully_implemented | `Provided` |
| T4 | Minimal-pair generation spec | ✅ fully_implemented | `Provided` |
| T5 | Auto-updating project contract | ✅ fully_implemented | `Provided` |
| T6 | Nested AGENTS propagation tooling | ✅ fully_implemented | `Provided` |
| T7 | Executable Python validation package | ✅ fully_implemented | `Provided` |
| T8 | Contamination detection tooling | ✅ fully_implemented | `Provided` |
| T9 | Minimal-pair generator | ✅ fully_implemented | `Provided` |
| T10 | Release-readiness reporter | ✅ fully_implemented | `Provided` |
| T11 | CI validation workflow | ✅ fully_implemented | `Provided` |
| T12 | Gold minimal-pair quality gate | ✅ fully_implemented | `Provided` |
| T13 | Context-bound gold-data validation | ✅ fully_implemented | `Provided` |
| T14 | LP20 orphaned-preposition gold-pair support | ✅ fully_implemented | `Provided` |
| T15 | LP9/LP20 diagnostic taxonomy and release integration | ✅ fully_implemented | `Provided` |
| T16 | Release-candidate orchestration command | ✅ fully_implemented | `Provided` |
| T17 | Legacy diagnostics compatibility wrapper | ✅ fully_implemented | `Provided` |
| T18 | Package data-pipeline compatibility wrappers | ✅ fully_implemented | `Provided` |
| T19 | Corpus source manifest and ingestion contract | ✅ fully_implemented | `Provided` |
| T20 | Corpus curation policy and quality scoring contract | ✅ fully_implemented | `Provided` |
| T21 | Curated corpus train/dev/test split contract | ✅ fully_implemented | `Provided` |
| T22 | Training export manifest and dataset card | ✅ fully_implemented | `Provided` |
| T23 | Modern corpus acquisition contracts, adapters, and readiness gate | ✅ fully_implemented | `Provided` |
| T24 | Real modern Québec source adapters with deterministic fixture acquisition | ✅ fully_implemented | `Provided` |

## Non-Negotiable Rules (Drift Prevention)
1. **No benchmark contamination**: do not train/evaluate on holdouts in `eval/evaluation_manifest.template.yaml`.
2. **No dialect neutralization**: reject transformations to fr_FR or dialect erasure.
3. **No silent schema edits**: document all manifest/schema changes in `CHANGELOG.md`.
4. **Refresh dynamic state** after scope/task edits:
   - `python3 tools/update_agents.py --write`
5. **Validation required** before commit:
   - `python3 tools/update_agents.py --validate`

## Required Workflow
1. Read nearest in-scope `AGENTS.md`.
2. Implement changes following manifest/rule schemas.
3. Run validation checks (`--validate`).
4. Update dynamic state (`--write`).
5. Commit with concise message referencing relevant Task IDs (T1–T24).

## Creation/Modification of Root and Nested AGENTS
```bash
# refresh root dynamic section
python3 tools/update_agents.py --write

# initialize nested AGENTS.md in any subtree
python3 tools/update_agents.py --init-nested manifests
python3 tools/update_agents.py --init-nested rules
```
Nested AGENTS must include `Scope`, `Mission`, `Local Rules`, `Overrides`, and a parent-contract link.

## Québec Linguistic Enforcement Highlights
- Preserve Québec lexical markers (LP9): prefer `courriel`, `fin de semaine` in formal contexts.
- Track hard phenomena with release floors:
  - LP9 ≥ 0.80
  - LP20 ≥ 0.70
- Overall release gate: LP accuracy ≥ 0.88.
