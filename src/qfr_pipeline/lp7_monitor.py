from __future__ import annotations

from pathlib import Path

import yaml


def monitor_lp7(pre_alignment_score: float, post_alignment_score: float, release_gates_path: Path) -> dict[str, float | bool]:
    try:
        gates = yaml.safe_load(release_gates_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to read/parse release gates YAML at {release_gates_path}: {exc}") from exc
    if not isinstance(gates, dict):
        raise ValueError(f"Malformed release gates config at {release_gates_path}: expected top-level mapping.")
    if "alignment" not in gates or not isinstance(gates["alignment"], dict):
        raise ValueError(f"Malformed release gates config at {release_gates_path}: missing mapping key `alignment`.")
    key = "lp7_standard_negation_max_post_alignment_drop_ratio"
    if key not in gates["alignment"]:
        raise ValueError(f"Malformed release gates config at {release_gates_path}: missing key `alignment.{key}`.")
    try:
        max_drop = float(gates["alignment"][key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed release gates config at {release_gates_path}: `alignment.{key}` must be numeric.") from exc
    if not 0.0 <= max_drop <= 1.0:
        raise ValueError(f"Malformed release gates config at {release_gates_path}: `alignment.{key}` must be in [0.0, 1.0].")
    pre = float(pre_alignment_score)
    post = float(post_alignment_score)
    if not 0.0 < pre <= 1.0:
        raise ValueError("LP7 pre-alignment score must be in (0.0, 1.0].")
    if not 0.0 <= post <= 1.0:
        raise ValueError("LP7 post-alignment score must be in [0.0, 1.0].")
    drop_ratio = (pre - post) / pre
    return {
        "pre_alignment_score": round(pre, 6),
        "post_alignment_score": round(post, 6),
        "drop_ratio": round(drop_ratio, 6),
        "max_allowed_drop_ratio": max_drop,
        "rollback_required": drop_ratio > max_drop,
    }
