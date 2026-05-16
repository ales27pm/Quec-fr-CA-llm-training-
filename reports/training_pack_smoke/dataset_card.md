# Dataset Card — qfr_dolphin3_unsloth_pack_local_smoke

- Version: `0.1.0`
- Pack mode: `local_research`
- Readiness level: `smoke_test`
- Commercial release ready: `False`
- Train / Dev / Test: `13321` / `741` / `741`
- Estimated tokens: `1571335`

## Source summary
- assnat_journal_debats_seed_modern: `18`
- donnees_quebec_ckan_metadata_modern: `6`
- donnees_quebec_ckan_textual: `48`
- gutenberg_angeline_de_montbrun_fr_17267: `3159`
- gutenberg_au_large_de_l_ecueil_fr_17791: `2392`
- gutenberg_ce_que_disait_la_flamme_fr_14399: `3249`
- gutenberg_l_influence_d_un_livre_fr_15305: `859`
- gutenberg_le_cap_au_diable_fr_13059: `500`
- gutenberg_le_debutant_fr_19497: `1849`
- gutenberg_maria_chapdelaine_fr_13525: `2233`
- gutenberg_un_amour_vrai_fr_14537: `490`

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
- insufficient_modern_diversity
- source_family_dominance
