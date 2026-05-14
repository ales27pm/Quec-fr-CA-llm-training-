# Release Readiness: PASS

## Gate Results
- asr_wer: PASS (0.05 vs 0.0665)
- overall_lp_accuracy: PASS (0.9 vs 0.88)
- lp9_lexical_semantics: PASS (0.9 vs 0.8)
- lp20_orphaned_preposition: PASS (0.8 vs 0.7)
- lp7_standard_negation_post_alignment_drop_ratio: PASS (0.02 vs 0.03)

## LP9/LP20 Diagnostics
### LP20:orphaned_preposition
- Binary accuracy: 0.5000
- Mean semantic similarity: 0.7093089146662548
- Top recurring failure classes: [{'error_code': 'relative_clause_attachment_error', 'count': 1}]
- Blocking/non-blocking error counts: 0/1
### LP9:lexical_semantics
- Binary accuracy: 0.5000
- Mean semantic similarity: 0.9869418673368094
- Top recurring failure classes: [{'error_code': 'normative_register_mismatch', 'count': 1}]
- Blocking/non-blocking error counts: 0/1
