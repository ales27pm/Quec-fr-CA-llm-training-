# Synthetic Minimal Pair Generator Spec (fr-CA / OQLF-aligned)

## Goal
Generate `(good_sentence, bad_sentence)` pairs where only one targeted linguistic phenomenon differs.

## Input Contract
- `lp_id`: integer in `[1..20]`
- `source_sentence`: grammatical fr-CA sentence
- `rule_manifest`: LP rule object
- `safety_mode`: `strict | permissive`

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

## Rule-Safe Transformations
1. Single targeted edit only (token-level or phrase-level).
2. Preserve sentence length window (±20%).
3. Preserve named entities and numbers.
4. No multi-error injections.
5. Reject if transformation introduces ambiguity outside target LP.

## Validation Pipeline
1. **Static checks**: schema, lp_id coverage, single-edit heuristic.
2. **Linguistic checks**: grammar parser + lexicon guardrails.
3. **Semantic checks**: entailment/paraphrase model threshold.
4. **Human sampling**: linguist review on at least 10% per LP batch.

## Pseudocode
```text
for sentence in source_pool:
  candidate = apply_lp_transform(sentence, lp_id)
  if not single_target_edit(sentence, candidate): continue
  if not semantic_close(sentence, candidate): continue
  if violates_non_target_rules(candidate): continue
  emit_pair(good=sentence, bad=candidate)
```

## Suggested Batch Policy
- Generate 2x target volume.
- Auto-filter to top confidence decile.
- Stratify final sample evenly across LPs and domains.
