# Strategic Roadmap: Engineering the Ultimate Québécois LLM Training Pipeline

## Purpose
This document operationalizes a sovereignty-first roadmap for a Québécois (fr-CA) LLM training pipeline that preserves dialectal authenticity, enforces OQLF/BDL normative grounding, and supports production deployment.

## 1) Data Foundation: Multimodal Corpus Acquisition and Curation

### Objectives
- Prioritize Québec-native linguistic signal over generic Metropolitan French corpora.
- Capture formal and informal registers, accents, and sociolinguistic variation.

### ASR integration baseline
- Fine-tune Whisper Large V3 Turbo using:
  - **Assemblée nationale du Québec** corpus (formal/political register).
  - **Common Voice, Québec subset** (informal, accent-diverse register).
- Production readiness WER threshold is defined in `project/release_gates.yaml` (`asr.wer_max`).

### Regional lexical corpora
- Integrate:
  - **QFrCoRE** (4,633 idiomatic expressions).
  - **QFrCoRT** (171 regional idiomatic words).
- Use these assets to reduce cultural mismatch and improve idiomatic fluency.

### Normative grounding
- Treat **Banque de dépannage linguistique (BDL / OQLF)** as authoritative for normative observations.

## 2) Architectural Benchmarking: QFrBLiMP Validation Framework

### Why QFrBLiMP
- Generic NLU benchmarks are insufficient for dialectal grammatical competence.
- Use minimal pairs to isolate rule-level linguistic knowledge.

### 20-phenomena stress test
All candidates must be evaluated across LP1–LP20:
1. Past Participle Agreement
2. Verb Inflection
3. Ne... que
4. Functional Morphology
5. Clitics in Infinitive Negation
6. Rising Clitics
7. Standard Negation
8. Determiners
9. Lexical Semantics
10. Idiomatic Agreement
11. Adjective Agreement
12. -é / -er
13. Lexical Selection
14. Infinitive Negation
15. Subject Island
16. Addition Island
17. Qu- Island
18. SN Island
19. Parasitic Dependence
20. Orphaned Preposition

### Scaling-law caution
- Expect grammar accuracy growth with log-parameters and a plateau range defined in `project/release_gates.yaml` (`benchmarking.scaling_plateau_expected_min/max`).
- Do not rely on parameter scaling alone to pass release gates.

### Dialectal robustness check
- Compare QFrBLiMP versus MultiBLiMP performance.
- Require statistical significance testing (e.g., Z-test) to detect Parisian-centric overfitting.

## 3) Optimization Strategy: Mitigating the Instruction-Tuning Tax

### Risk
Instruction tuning can degrade formal grammatical competence.

### Mitigation protocol (HITL + RLHF)
- Integrate native-speaker validated gold minimal pairs into RLHF and alignment data.
- Annotation protocol:
  - 12 annotators (target panel size).
  - Majority-vote aggregation.
  - Prioritize high-confidence examples (target WAWA ~86.31%).

### Governance trigger
- The maximum allowable LP7 post-alignment drop is defined in `project/release_gates.yaml` (`alignment.lp7_standard_negation_max_post_alignment_drop_pp`).
- Exceeding that threshold triggers rollback or corrective fine-tuning.

## 4) Frontier: Unsolved Phenomena (Semantics + Rare Syntax)

### Priority failure modes
- **LP9 Lexical Semantics**.
- **LP20 Orphaned Prepositions**.

### Additional diagnostics
- Add targeted probing suites for semantic disambiguation.
- Track semantic similarity metrics in addition to binary accuracy.
- Require qualitative error taxonomy for LP9/LP20 failures each release.

## 5) Technical Implementation: Optimization and Inference Readiness

### Inference stack
- ASR: **WhisperX** or **Faster-Whisper**.
- Transformer runtime: **CTranslate2**.

### Weight/runtime standardization
- Production default: **Float16** weights.
- Preferred optimized ASR artifact: `ele-sage/whisper-large-v3-turbo-fr-quebecois-ct2`.

## Release Gates (Normative + Technical)

A model is release-candidate ready only if all criteria pass:
- Thresholds in `project/release_gates.yaml` for overall LP accuracy, LP9, LP20, and ASR WER.
- No significant QFrBLiMP regression versus prior release.
- No contamination of evaluation holdouts.

This document intentionally references the centralized threshold file to avoid drift across docs and governance artifacts.

## Execution Plan (Iterative Loop)
1. Ingest and normalize multimodal Québec corpora.
2. Fine-tune base models with dialect-preserving objectives.
3. Evaluate with QFrBLiMP LP1–LP20 and robustness deltas.
4. Apply alignment with gold-pair safeguards.
5. Re-evaluate, diagnose LP9/LP20, and iterate.
6. Package optimized Float16/CTranslate2 artifacts for deployment.

## Ownership and Traceability
- Track roadmap decisions and threshold changes in `CHANGELOG.md`.
- Keep this roadmap aligned with repository AGENTS governance and validation workflows.
