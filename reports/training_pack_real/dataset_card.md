# Dataset Card — qfr_dolphin3_unsloth_pack_local_real

- Version: `0.1.0`
- Pack mode: `local_research`
- Readiness level: `insufficient`
- Commercial release ready: `False`
- Train / Dev / Test: `6` / `3` / `3`
- Estimated tokens: `1267`

## Source summary
- assnat_journal_debats_seed_modern: `6`
- donnees_quebec_ckan_metadata_modern: `6`

## Intended use
- Continuation training pack for Québec-French adaptation with Dolphin3/Qwen-compatible chat format.

## Forbidden uses
- Do not use evaluation holdouts (QFrCoLA/QFrBLiMP/QFrCoRE/QFrCoRT/COLE) for training.
- Do not use permission-required sources without explicit local grant.

## Licensing caveats
- Commercial and license status are preserved per example metadata and must be revalidated before production release.

## Holdout contamination policy
- Records flagged as holdout-only or containing holdout material are rejected during pack build.

## Known limitations
- No blocking limitations detected by policy checks.
