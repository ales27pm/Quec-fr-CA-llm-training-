# Dataset Card — qfr_dolphin3_unsloth_pack

- Version: `0.1.0`
- Readiness level: `insufficient`
- Train / Dev / Test: `1` / `1` / `1`
- Estimated tokens: `406`

## Source summary
- donnees_quebec_ckan_metadata_modern: `3`

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
- single_source_dominance
- single_source_family_dominance
