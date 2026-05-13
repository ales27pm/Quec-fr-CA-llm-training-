# Iterative Training Loop Runbook (fr-CA Alignment)

This runbook operationalizes `docs/strategic_roadmap.md` for release-candidate execution while enforcing the centralized gates in `project/release_gates.yaml`.

## 1) Threshold resolution (single source of truth)
- Load all numeric gates from `project/release_gates.yaml`.
- Never hardcode values in scripts, notebooks, or release notes.
- Required gates:
  - `asr.wer_max`
  - `linguistic_phenomena.overall_accuracy_min`
  - `linguistic_phenomena.lp9_lexical_semantics_min`
  - `linguistic_phenomena.lp20_orphaned_preposition_min`
  - `alignment.lp7_standard_negation_max_post_alignment_drop_ratio`

## 2) Iterative loop steps
1. Ingest Québec-native corpora and refresh dataset manifests.
   - Reproducible data-prep CLI (standardized):

     ```bash
     python3 tools/pipeline_ops.py harvest \
       --inputs <raw_input_1.txt> <raw_input_2.txt> \
       --out <artifacts/harvest.jsonl> \
       --min-chars 20

     python3 tools/pipeline_ops.py curate \
       --in <artifacts/harvest.jsonl> \
       --out <artifacts/curated.jsonl> \
       --min-fr-ca-score 0.34

     python3 tools/pipeline_ops.py edit \
       --in <artifacts/curated.jsonl> \
       --out <artifacts/edited.jsonl>

     python3 tools/pipeline_ops.py split \
       --in <artifacts/edited.jsonl> \
       --out-dir <artifacts/splits> \
       --seed 42

     python3 tools/pipeline_ops.py recipe \
       --data-dir <artifacts/splits> \
       --out <artifacts/recipe.yaml>
     ```
2. Fine-tune with dialect-preserving objectives.
3. Evaluate LP1–LP20 and ASR metrics against centralized gates.
4. Run alignment with human-validated QFrBLiMP minimal-pair safeguards.
5. Compute post-alignment LP7 drop ratio and compare to centralized max drop.
6. Run LP9/LP20 semantic diagnostics and generate qualitative error taxonomy.
7. Package release candidates as Float16 weights for CTranslate2.

## 3) Instruction-tuning tax mitigation controls
- Include human-validated QFrBLiMP minimal pairs in SFT/RLHF preference sets.
- Block promotion when LP7 post-alignment drop exceeds `alignment.lp7_standard_negation_max_post_alignment_drop_ratio`.
- When blocked, run rollback or corrective fine-tuning before re-evaluation.
- Standardized LP7 monitor CLI:

  ```bash
  python3 tools/pipeline_ops.py monitor-lp7 \
    --pre <lp7_pre_alignment_score> \
    --post <lp7_post_alignment_score> \
    --release-gates project/release_gates.yaml
  ```

## 4) Semantic diagnostics requirements (LP9/LP20)
For each release candidate:
- Record binary accuracy for LP9 and LP20.
- Record semantic similarity diagnostics (for example sentence-embedding cosine similarity).
- Produce an error taxonomy with at least these labels:
  - `regional_idiomatic_fluency`
  - `rare_syntax_orphaned_preposition`
  - `register_or_normative_grounding`
- Standardized diagnostics CLI:

  ```bash
  python3 tools/pipeline_ops.py diagnose-semantic \
    --in-csv <lp9_lp20_diagnostics.csv> \
    --out-json <artifacts/semantic_diagnostics.json>
  ```

## 5) Pre-commit and pre-release governance checks
Run before any commit or checkpoint promotion:

```bash
python3 tools/update_agents.py --validate
```

If scope/task metadata changed, refresh dynamic AGENTS state:

```bash
python3 tools/update_agents.py --write
```

## 6) Traceability requirements
- Record roadmap adjustments and threshold-related updates in `CHANGELOG.md`.
- Reference relevant Task IDs from root AGENTS (T1–T6) in commit messages.
