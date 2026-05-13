# Synthetic Minimal Pair Generator Spec (fr-CA / OQLF-aligned)

## Goal
Generate `(good_sentence, bad_sentence)` pairs where only one targeted linguistic phenomenon differs.

## Input Contract
- `lp_id`: integer in `[1..20]`
- `source_sentence`: grammatical fr-CA sentence
- `rule_manifest`: LP rule object
- `safety_mode`: `strict | permissive`
- `benchmark_holdouts`: object containing blocked `pair_ids`, `lp_ids`, and source sentence hashes/text spans

## Output Contract
```json
{
  "pair_id": "lp09_000123",
  "lp_id": 9,
  "good_sentence": "...",
  "bad_sentence": "...",
  "transform": "anglicism_substitution",
  "rationale": "Replaced OQLF-preferred term with non-standard anglicism.",
  "validations": {
    "single_edit_distance": true,
    "semantic_preservation": true,
    "grammar_targeted": true
  }
}
```

## Pre-Generation Exclusion (Mandatory)
1. Filter out any source whose `lp_id` is in `benchmark_holdouts.lp_ids`.
2. Filter out any source whose sentence hash/text appears in `benchmark_holdouts` before transformation.
3. Reject any generated pair whose `pair_id` collides with a holdout identifier.

## Rule-Safe Transformations
1. Single targeted edit only (token-level or phrase-level).
2. Preserve sentence length window (±20%).
3. Preserve named entities and numbers.
4. No multi-error injections.
5. Reject if transformation introduces ambiguity outside target LP.

## Validation Pipeline
1. **Holdout exclusion checks**: enforce `benchmark_holdouts` filtering before generation/acceptance.
2. **Static checks**: schema, lp_id coverage, single-edit heuristic.
3. **Linguistic checks**: grammar parser + lexicon guardrails.
4. **Semantic checks**: entailment/paraphrase model threshold.
5. **Human sampling**: linguist review on at least 10% per LP batch.

## Pseudocode
```text
for sentence in source_pool:
  if is_holdout(sentence, lp_id, benchmark_holdouts): continue
  candidate = apply_lp_transform(sentence, lp_id)
  if not single_target_edit(sentence, candidate): continue
  if not semantic_close(sentence, candidate): continue
  if violates_non_target_rules(candidate): continue
  if is_holdout_pair(sentence, candidate, benchmark_holdouts): continue
  emit_pair(good=sentence, bad=candidate)
```

## Suggested Batch Policy
- Generate 2x target volume.
- Auto-filter to top confidence decile.
- Stratify final sample evenly across LPs and domains.
