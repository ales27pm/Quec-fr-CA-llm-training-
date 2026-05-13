# AGENTS.md — Dynamic Project Operating Contract

## Scope
This file governs the entire repository (`/workspace/Quec-fr-CA-llm-training-`) unless a deeper nested `AGENTS.md` overrides part of it.

## Mission
Build and maintain a top-of-the-line Québécois (fr-CA) LLM training pipeline with:
1. OQLF/BDL-aligned linguistic quality,
2. reproducible evaluation,
3. explicit governance and anti-drift controls.

## Dynamic State (auto-generated)
> Last auto-update: **2026-05-13 UTC**  
> Source: `project/status.json` (managed by `python3 tools/update_agents.py`)

| Task ID | Goal | Status | Evidence |
|---|---|---|---|
| T1 | Dataset manifest template | ✅ fully_implemented | `manifests/dataset_manifest.template.yaml` |
| T2 | LP rule manifest template | ✅ fully_implemented | `rules/lp_rule_manifest.template.yaml` |
| T3 | Evaluation/release manifest | ✅ fully_implemented | `eval/evaluation_manifest.template.yaml` |
| T4 | Minimal-pair generation spec | ✅ fully_implemented | `tools/minimal_pair_generator_spec.md` |
| T5 | Auto-updating project contract | 🟡 partially_implemented | `AGENTS.md, tools/update_agents.py` |
| T6 | Nested AGENTS propagation tooling | ❌ not_implemented | `Pending` |
## Non-Negotiable Rules (Drift Prevention)
1. **Do not train/evaluate on benchmark holdouts** listed in evaluation manifests.
2. **No silent schema edits**: manifest schema changes require changelog notes.
3. **Status must be refreshed** after file edits that affect scope/tasks by running:
   - `python3 tools/update_agents.py --write`
4. **Commit hygiene**:
   - include task IDs in commit body when relevant.
5. **Nested precedence**:
   - deeper `AGENTS.md` overrides style/process for its subtree only.

## Required Workflow
1. Read nearest in-scope `AGENTS.md` before editing.
2. Implement change.
3. Run validations/checks.
4. Refresh agent state with updater script.
5. Commit with a concise message.

## Creating/Modifying AGENTS.md (Root + Nested)
Use the updater utility to safely create or refresh contracts:

```bash
# refresh root AGENTS dynamic section
python3 tools/update_agents.py --write

# create nested AGENTS.md for a subtree (if missing)
python3 tools/update_agents.py --init-nested manifests
python3 tools/update_agents.py --init-nested rules
```

Rules for nested files:
- Must include: `Scope`, `Mission`, `Local Rules`, and `Overrides`.
- Must reference parent contract path.
- Must not weaken root safety constraints (data leakage, legal, benchmark contamination).

## Project Roadmap Snapshot
- **Fully implemented:** T1, T2, T3, T4
- **Partially implemented:** T5 (dynamic state exists; richer telemetry pending)
- **Not implemented:** T6 (automatic nested propagation + drift checks)

## Ownership
- Primary owner: AI training pipeline maintainers
- Escalation: linguistics + ML platform reviewers
