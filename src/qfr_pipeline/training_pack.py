from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import TrainingPackPolicyManifest

MARKERS = [
    "courriel",
    "dépanneur",
    "stationnement",
    "magasinage",
    "cellulaire",
    "clavarder",
    "fin de semaine",
    "char",
    "icitte",
    "ben",
    "pis",
    "fait que",
    "tuque",
    "patente",
]
NORMALIZATION_REPLACEMENTS = [
    (r"\bemail\b", "courriel"),
    (r"\bparking\b", "stationnement"),
    (r"\bshopping\b", "magasinage"),
    (r"\btéléphone portable\b", "cellulaire"),
    (r"\bportable\b", "cellulaire"),
]
FR_CA_CONTRASTS = [
    ("week-end", "fin de semaine"),
    ("email", "courriel"),
    ("parking", "stationnement"),
    ("shopping", "magasinage"),
    ("téléphone portable", "cellulaire"),
    ("portable", "cellulaire"),
]
READINESS_LEVELS = [
    "insufficient",
    "smoke_test",
    "pilot_lora_candidate",
    "production_lora_candidate",
]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    return " ".join(_clean_text(text).casefold().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_permission_sources(permission_manifest: Path | None) -> set[str]:
    if permission_manifest is None or not permission_manifest.exists():
        return set()
    payload = load_yaml(permission_manifest)
    permission_sources: set[str] = set()
    for key in ("sources", "approved_sources"):
        value = payload.get(key)
        if isinstance(value, dict):
            permission_sources.update(str(item) for item in value.keys())
        elif isinstance(value, list):
            permission_sources.update(str(item) for item in value)
    return permission_sources


def _estimate_tokens(text: str) -> int:
    chars = len(text)
    words = len(text.split())
    return int(max(chars / 4, words * 1.35))


def load_training_pack_policy(path: Path) -> TrainingPackPolicyManifest:
    return TrainingPackPolicyManifest.model_validate(load_yaml(path))


def validate_training_pack_policy(path: Path) -> TrainingPackPolicyManifest:
    return load_training_pack_policy(path)


def _extract_sentences(text: str) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _extract_summary(text: str, max_chars: int) -> str:
    sentences = _extract_sentences(text)
    if not sentences:
        return ""
    summary_parts: list[str] = []
    for sentence in sentences[:2]:
        candidate = _clean_text(" ".join(summary_parts + [sentence]))
        if len(candidate) <= max_chars:
            summary_parts.append(sentence)
        elif not summary_parts:
            summary_parts.append(sentence[:max_chars].rstrip())
        else:
            break
    summary = _clean_text(" ".join(summary_parts))
    return summary[:max_chars].rstrip()


def _first_sentence(text: str, max_chars: int) -> str:
    sentences = _extract_sentences(text)
    if not sentences:
        return _clean_text(text)[:max_chars].rstrip()
    return sentences[0][:max_chars].rstrip()


def _detect_marker(text: str) -> str | None:
    lowered = text.casefold()
    positions: list[tuple[int, str]] = []
    for marker in MARKERS:
        idx = lowered.find(marker.casefold())
        if idx >= 0:
            positions.append((idx, marker))
    if not positions:
        return None
    positions.sort()
    return positions[0][1]


def _apply_normalization_replacements(text: str) -> str:
    rewritten = text
    for pattern, replacement in NORMALIZATION_REPLACEMENTS:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    return _clean_text(rewritten)


def _detect_contrast_pair(text: str) -> tuple[str, str] | None:
    lowered = text.casefold()
    for fr_fr, fr_ca in FR_CA_CONTRASTS:
        if fr_fr.casefold() in lowered or fr_ca.casefold() in lowered:
            return (fr_fr, fr_ca)
    return None


def _render_chat(messages: list[dict[str, str]], chat_format: str) -> str:
    if chat_format == "qwen_chatml":
        lines: list[str] = []
        for message in messages:
            lines.append(f"<|im_start|>{message['role']}")
            lines.append(message["content"])
            lines.append("<|im_end|>")
        return "\n".join(lines)
    if chat_format == "alpaca":
        user_msg = next((item["content"] for item in messages if item["role"] == "user"), "")
        assistant_msg = next(
            (item["content"] for item in messages if item["role"] == "assistant"), ""
        )
        return (
            "### Instruction:\n"
            f"{user_msg}\n\n"
            "### Response:\n"
            f"{assistant_msg}"
        )
    if chat_format == "raw_completion":
        return next((item["content"] for item in messages if item["role"] == "assistant"), "")
    raise ValueError(f"Unsupported chat format: {chat_format}")


def _probabilistic_keep(record: dict[str, Any], probability: float) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    basis = f"{record['normalized_text_sha256']}:{record['source_record_id']}"
    score = int(_sha256(basis)[:8], 16) / 0xFFFFFFFF
    return score <= probability


def _build_example(
    *,
    record: dict[str, Any],
    task_type: str,
    messages: list[dict[str, str]],
    chat_format: str,
    permission_sources: set[str],
) -> dict[str, Any]:
    rendered = _render_chat(messages, chat_format)
    rendered_hash = _sha256(rendered)
    example_id = f"{record['source_record_id']}:{task_type}:{rendered_hash[:12]}"
    requires_review = bool(record.get("requires_review", False)) and (
        record.get("source_id") in permission_sources
    )
    return {
        "example_id": example_id,
        "messages": messages,
        "text": rendered,
        "task_type": task_type,
        "source_record_id": record["source_record_id"],
        "source_id": record["source_id"],
        "source_family": record["source_family"],
        "source_name": record["source_name"],
        "text_sha256": record["text_sha256"],
        "normalized_text_sha256": record["normalized_text_sha256"],
        "language": "fr-CA",
        "dialect_region": record.get("dialect_region", "Quebec"),
        "register": record["register"],
        "domain": record["domain"],
        "license_status": record["license_status"],
        "commercial_use": record["commercial_use"],
        "allowed_for_training": True,
        "holdout_only": False,
        "requires_review": requires_review,
        "quality_flags": list(record.get("quality_flags", [])),
        "provenance": {
            "policy": record["policy"],
            "source_input": record["source_input"],
            "source_priority": record["source_priority"],
            "source_record_id": record["source_record_id"],
            "strategy": task_type,
        },
    }


def _apply_share_cap(
    records: list[dict[str, Any]], key_name: str, max_share: float
) -> list[dict[str, Any]]:
    if max_share >= 1.0 or not records:
        return records

    rows = list(records)
    while rows:
        total = len(rows)
        counts = Counter(str(row.get(key_name, "unknown")) for row in rows)
        overfull = [
            key for key, count in counts.items() if (count / total) > (max_share + 1e-9) and count > 1
        ]
        if not overfull:
            break
        removed_any = False
        for key in sorted(overfull):
            candidates = [
                (idx, row)
                for idx, row in enumerate(rows)
                if str(row.get(key_name, "unknown")) == key
            ]
            if len(candidates) <= 1:
                continue
            drop_idx = max(
                candidates,
                key=lambda item: (
                    int(item[1].get("drop_score", "0")[:12], 16),
                    item[1].get("normalized_text_sha256", ""),
                    item[1].get("source_record_id", ""),
                ),
            )[0]
            rows.pop(drop_idx)
            removed_any = True
        if not removed_any:
            break
    return rows


def _apply_source_input_caps(
    records: list[dict[str, Any]],
    source_inputs_by_path: dict[str, Any],
) -> list[dict[str, Any]]:
    if not records:
        return records
    rows = list(records)
    while rows:
        total = len(rows)
        counts = Counter(str(row.get("source_input", "")) for row in rows)
        overfull: list[str] = []
        for source_input_path, count in counts.items():
            source_input = source_inputs_by_path.get(source_input_path)
            max_share = float(source_input.max_source_share) if source_input is not None else 1.0
            if (count / total) > (max_share + 1e-9) and count > 1:
                overfull.append(source_input_path)
        if not overfull:
            break
        removed_any = False
        for source_input_path in sorted(overfull):
            candidates = [
                (idx, row)
                for idx, row in enumerate(rows)
                if str(row.get("source_input", "")) == source_input_path
            ]
            if len(candidates) <= 1:
                continue
            drop_idx = max(
                candidates,
                key=lambda item: (
                    int(item[1].get("drop_score", "0")[:12], 16),
                    item[1].get("normalized_text_sha256", ""),
                    item[1].get("source_record_id", ""),
                ),
            )[0]
            rows.pop(drop_idx)
            removed_any = True
        if not removed_any:
            break
    return rows


def _compute_split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = int(total * ratios["train"])
    dev = int(total * ratios["dev"])
    test = total - train - dev

    if total >= 3:
        if dev == 0:
            dev = 1
        if test == 0:
            test = 1
        train = max(total - dev - test, 1)

    while train + dev + test > total:
        if train >= dev and train >= test and train > 0:
            train -= 1
        elif dev >= test and dev > 0:
            dev -= 1
        elif test > 0:
            test -= 1
    while train + dev + test < total:
        train += 1
    return {"train": train, "dev": dev, "test": test}


def _split_examples(
    examples: list[dict[str, Any]],
    seed: int,
    split_ratios: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    if not examples:
        return {"train": [], "dev": [], "test": []}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example["source_record_id"])].append(example)

    rng = random.Random(seed)
    group_keys = sorted(grouped)
    rng.shuffle(group_keys)

    targets = _compute_split_counts(len(examples), split_ratios)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    for group_key in group_keys:
        group_examples = sorted(grouped[group_key], key=lambda row: row["example_id"])
        choices = sorted(
            ["train", "dev", "test"],
            key=lambda split_name: (
                -(targets[split_name] - len(splits[split_name])),
                len(splits[split_name]),
                split_name,
            ),
        )
        destination = choices[0]
        splits[destination].extend(group_examples)

    if len(examples) >= 3:
        # Prefer moving whole source_record groups to avoid cross-split leakage for the same record.
        for split_name in ["dev", "test"]:
            if splits[split_name]:
                continue
            donor = max(["train", "dev", "test"], key=lambda name: len(splits[name]))
            donor_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in splits[donor]:
                donor_groups[str(row["source_record_id"])].append(row)
            if len(donor_groups) <= 1:
                continue
            move_group_id = sorted(
                donor_groups,
                key=lambda group_id: (len(donor_groups[group_id]), group_id),
            )[0]
            move_rows = donor_groups[move_group_id]
            splits[donor] = [
                row for row in splits[donor] if str(row["source_record_id"]) != move_group_id
            ]
            splits[split_name].extend(move_rows)

        # Fallback: if a split is still empty, move a single example deterministically.
        # This only happens when there are not enough independent source_record groups.
        for split_name in ["train", "dev", "test"]:
            if splits[split_name]:
                continue
            donor = max(["train", "dev", "test"], key=lambda name: len(splits[name]))
            if len(splits[donor]) <= 1:
                continue
            move_row = sorted(splits[donor], key=lambda row: row["example_id"])[-1]
            splits[donor] = [
                row for row in splits[donor] if row["example_id"] != move_row["example_id"]
            ]
            splits[split_name].append(move_row)

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda row: row["example_id"])

    return splits


def _build_dataset_card(
    *,
    path: Path,
    policy: TrainingPackPolicyManifest,
    report: dict[str, Any],
) -> None:
    lines = [
        f"# Dataset Card — {policy.pack_id}",
        "",
        f"- Version: `{policy.pack_version}`",
        f"- Readiness level: `{report['readiness_level']}`",
        f"- Train / Dev / Test: `{report['train_count']}` / `{report['dev_count']}` / `{report['test_count']}`",
        f"- Estimated tokens: `{report['estimated_tokens_total']}`",
        "",
        "## Source summary",
    ]
    for key, value in sorted(report["source_summary"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Intended use",
            "- Continuation training pack for Québec-French adaptation with Dolphin3/Qwen-compatible chat format.",
            "",
            "## Forbidden uses",
            "- Do not use evaluation holdouts (QFrCoLA/QFrBLiMP/QFrCoRE/QFrCoRT/COLE) for training.",
            "- Do not use permission-required sources without explicit local grant.",
            "",
            "## Licensing caveats",
            "- Commercial and license status are preserved per example metadata and must be revalidated before production release.",
            "",
            "## Holdout contamination policy",
            "- Records flagged as holdout-only or containing holdout material are rejected during pack build.",
            "",
            "## Known limitations",
        ]
    )
    if report["blocking_reasons"]:
        for reason in report["blocking_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- No blocking limitations detected by policy checks.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _readiness_level_from_tokens(tokens: int, thresholds: dict[str, int]) -> str:
    if tokens >= thresholds["production_lora_tokens"]:
        return "production_lora_candidate"
    if tokens >= thresholds["pilot_lora_tokens"]:
        return "pilot_lora_candidate"
    if tokens >= thresholds["smoke_test_tokens"]:
        return "smoke_test"
    return "insufficient"


def build_training_pack(
    policy_path: Path,
    out_dir: Path | None = None,
    permission_manifest: Path | None = None,
) -> dict[str, Any]:
    policy = validate_training_pack_policy(policy_path)
    permission_sources = _load_permission_sources(permission_manifest)

    output_dir = out_dir if out_dir is not None else Path(policy.output_dir)
    output_root = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    input_files: list[str] = []
    issues: list[str] = []
    rejection_reasons: Counter[str] = Counter()
    duplicate_exact_rejected = 0
    duplicate_normalized_rejected = 0
    holdout_rejected = 0
    permission_rejected = 0
    downsample_source_input_rejected = 0
    downsample_source_rejected = 0
    downsample_source_family_rejected = 0

    records_seen = 0
    accepted_records: list[dict[str, Any]] = []
    seen_text_hashes: set[str] = set()
    seen_normalized_hashes: set[str] = set()

    source_inputs_by_path = {item.path: item for item in policy.source_inputs}

    for source_input in sorted(
        policy.source_inputs,
        key=lambda item: (item.source_priority, item.source_family, item.path),
    ):
        path = ROOT / source_input.path
        exists = path.exists()
        if not exists:
            if source_input.required:
                issues.append(f"required_input_missing:{source_input.path}")
            continue
        if not source_input.include_if_exists:
            continue

        input_files.append(source_input.path)
        rows = _read_jsonl(path)
        for raw_row in rows:
            records_seen += 1
            text = _clean_text(str(raw_row.get("text") or ""))
            if policy.safety.reject_empty_text and not text:
                rejection_reasons["empty_text"] += 1
                continue
            if len(text) < policy.instructionization.min_text_chars:
                rejection_reasons["text_too_short"] += 1
                continue
            if len(text) > policy.instructionization.max_text_chars:
                rejection_reasons["text_too_long"] += 1
                continue

            source_id = str(
                raw_row.get("source_id")
                or raw_row.get("source")
                or f"{source_input.source_family}_{path.stem}"
            )
            source_name = str(raw_row.get("source_name") or source_id)
            source_record_id = str(raw_row.get("record_id") or raw_row.get("id") or "")
            if not source_record_id:
                source_record_id = _sha256(f"{source_id}:{text}")[:24]

            raw_allowed = raw_row.get("allowed_for_training")
            if raw_allowed is None:
                allowed_for_training = True
            else:
                allowed_for_training = bool(raw_allowed)

            holdout_only = bool(raw_row.get("holdout_only", False))
            contains_holdout = bool(raw_row.get("contains_holdout_material", False))
            requires_review = bool(raw_row.get("requires_review", False))
            source_type = str(raw_row.get("source_type") or "unknown")
            license_status = str(raw_row.get("license_status") or "unknown")
            commercial_use = str(raw_row.get("commercial_use") or "unknown")

            if policy.safety.reject_holdout_material and (holdout_only or contains_holdout):
                rejection_reasons["holdout_material"] += 1
                holdout_rejected += 1
                continue

            if not allowed_for_training:
                rejection_reasons["allowed_for_training_false"] += 1
                continue

            if source_input.allowed_for_training_required and raw_allowed is False:
                rejection_reasons["source_input_training_requirement_failed"] += 1
                continue

            permission_required = (
                commercial_use == "permission_required"
                or license_status == "permission_required"
                or source_type == "permission_required"
                or requires_review
            )
            if (
                policy.safety.reject_permission_required_without_grant
                and permission_required
                and source_id not in permission_sources
            ):
                rejection_reasons["permission_required_without_grant"] += 1
                permission_rejected += 1
                continue

            if (
                policy.safety.reject_commercial_unknown_for_production
                and commercial_use in {"unknown", "prohibited"}
            ):
                rejection_reasons["commercial_not_safe"] += 1
                continue

            text_sha = str(raw_row.get("text_sha256") or _sha256(text))
            normalized_text = _normalize_text(text)
            normalized_sha = str(raw_row.get("normalized_text_sha256") or _sha256(normalized_text))

            if text_sha in seen_text_hashes:
                rejection_reasons["duplicate_exact_text"] += 1
                duplicate_exact_rejected += 1
                continue
            if (
                policy.safety.reject_duplicate_normalized_text
                and normalized_sha in seen_normalized_hashes
            ):
                rejection_reasons["duplicate_normalized_text"] += 1
                duplicate_normalized_rejected += 1
                continue

            seen_text_hashes.add(text_sha)
            seen_normalized_hashes.add(normalized_sha)

            drop_score = _sha256(f"{policy.random_seed}:{source_id}:{normalized_sha}:{source_record_id}")
            accepted_records.append(
                {
                    "source_record_id": source_record_id,
                    "record_id": source_record_id,
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_family": source_input.source_family,
                    "source_priority": source_input.source_priority,
                    "source_input": source_input.path,
                    "text": text,
                    "text_sha256": text_sha,
                    "normalized_text_sha256": normalized_sha,
                    "language": str(raw_row.get("language") or "fr-CA"),
                    "dialect_region": str(raw_row.get("dialect_region") or "Quebec"),
                    "register": str(raw_row.get("register") or "unknown"),
                    "domain": str(raw_row.get("domain") or "unknown"),
                    "license_status": license_status,
                    "commercial_use": commercial_use,
                    "allowed_for_training": True,
                    "holdout_only": False,
                    "requires_review": requires_review,
                    "quality_flags": list(raw_row.get("quality_flags") or []),
                    "policy": repo_relative_path(policy_path),
                    "drop_score": drop_score,
                }
            )

    if issues:
        report = {
            "ok": False,
            "policy": repo_relative_path(policy_path),
            "output_dir": repo_relative_path(output_root),
            "input_files": sorted(input_files),
            "records_seen": records_seen,
            "records_accepted": len(accepted_records),
            "records_rejected": sum(rejection_reasons.values()),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "examples_generated": 0,
            "train_count": 0,
            "dev_count": 0,
            "test_count": 0,
            "estimated_tokens_total": 0,
            "estimated_tokens_train": 0,
            "source_summary": {},
            "source_family_summary": {},
            "domain_summary": {},
            "register_summary": {},
            "task_type_summary": {},
            "license_status_summary": {},
            "commercial_use_summary": {},
            "max_single_source_share": 0.0,
            "max_single_source_family_share": 0.0,
            "duplicate_exact_rejected": duplicate_exact_rejected,
            "duplicate_normalized_rejected": duplicate_normalized_rejected,
            "holdout_rejected": holdout_rejected,
            "permission_rejected": permission_rejected,
            "readiness_level": "insufficient",
            "blocking_reasons": sorted(set(issues)),
            "recommendations": ["Provide required source inputs to build the training pack."],
            "artifacts": {},
        }
        write_json(output_root / "report.json", report)
        return report

    accepted_records = sorted(
        accepted_records,
        key=lambda row: (
            row["source_family"],
            row["source_id"],
            row["normalized_text_sha256"],
            row["source_record_id"],
        ),
    )

    before_source_input_cap = len(accepted_records)
    accepted_records = _apply_source_input_caps(
        accepted_records,
        source_inputs_by_path=source_inputs_by_path,
    )
    downsample_source_input_rejected += before_source_input_cap - len(accepted_records)

    before_source_cap = len(accepted_records)
    accepted_records = _apply_share_cap(
        accepted_records,
        key_name="source_id",
        max_share=policy.balancing.max_single_source_share,
    )
    downsample_source_rejected += before_source_cap - len(accepted_records)

    before_source_family_cap = len(accepted_records)
    accepted_records = _apply_share_cap(
        accepted_records,
        key_name="source_family",
        max_share=policy.balancing.max_single_source_family_share,
    )
    downsample_source_family_rejected += before_source_family_cap - len(accepted_records)

    if downsample_source_input_rejected > 0:
        rejection_reasons["downsample_source_input_share"] += downsample_source_input_rejected
    if downsample_source_rejected > 0:
        rejection_reasons["downsample_source_share"] += downsample_source_rejected
    if downsample_source_family_rejected > 0:
        rejection_reasons["downsample_source_family_share"] += downsample_source_family_rejected

    examples: list[dict[str, Any]] = []
    seen_example_hashes: set[str] = set()
    seen_example_normalized_hashes: set[str] = set()

    strategies = list(policy.instructionization.strategies)
    if policy.instructionization.enabled and "preserve_raw" not in strategies:
        strategies.append("preserve_raw")

    for record in accepted_records:
        generated_for_record = 0
        for strategy in strategies:
            if generated_for_record >= policy.instructionization.max_examples_per_record:
                break

            messages: list[dict[str, str]] | None = None
            task_type = strategy
            text = record["text"]

            if strategy == "preserve_raw":
                if not _probabilistic_keep(
                    record, policy.instructionization.preserve_raw_text_probability
                ):
                    continue
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Québec-French language adaptation corpus. "
                            "Preserve natural fr-CA usage."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Continue ce passage en français québécois naturel.",
                    },
                    {"role": "assistant", "content": text},
                ]
            elif strategy == "summarize":
                summary = _extract_summary(text, policy.instructionization.max_text_chars)
                if not summary:
                    continue
                messages = [
                    {
                        "role": "system",
                        "content": "Tu réponds en français québécois clair et naturel.",
                    },
                    {
                        "role": "user",
                        "content": "Résume ce passage en français québécois clair et naturel.",
                    },
                    {"role": "assistant", "content": summary},
                ]
            elif strategy == "style_rewrite":
                task_type = "style_rewrite_identity"
                messages = [
                    {
                        "role": "system",
                        "content": "Tu réponds en français québécois clair et naturel.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Réécris ce passage dans un français québécois clair, naturel et "
                            "standard."
                        ),
                    },
                    {"role": "assistant", "content": text},
                ]
            elif strategy == "explain_term":
                marker = _detect_marker(text)
                if marker is None:
                    continue
                first_sentence = _first_sentence(text, policy.instructionization.max_text_chars)
                explanation = (
                    f"Dans ce passage, « {marker} » est utilisé dans un contexte québécois: "
                    f"{first_sentence}"
                )
                messages = [
                    {
                        "role": "system",
                        "content": "Tu expliques en français québécois clair et contextualisé.",
                    },
                    {
                        "role": "user",
                        "content": f"Explique le terme ou l’expression « {marker} » dans ce passage.",
                    },
                    {"role": "assistant", "content": explanation},
                ]
            elif strategy == "qa_from_context":
                answer = _first_sentence(text, policy.instructionization.max_text_chars)
                if not answer:
                    continue
                messages = [
                    {
                        "role": "system",
                        "content": "Tu réponds de façon concise en français québécois.",
                    },
                    {
                        "role": "user",
                        "content": "D’après ce passage, quel est le point principal?",
                    },
                    {"role": "assistant", "content": answer},
                ]
            elif strategy == "normalize_to_quebec_fr":
                normalized = _apply_normalization_replacements(text)
                if normalized == text:
                    continue
                messages = [
                    {
                        "role": "system",
                        "content": "Tu normalises vers un français québécois standard.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Réécris ce passage en privilégiant les formes usuelles du français "
                            "québécois."
                        ),
                    },
                    {"role": "assistant", "content": normalized},
                ]
            elif strategy == "contrast_fr_fr_vs_fr_ca":
                pair = _detect_contrast_pair(text)
                if pair is None:
                    continue
                fr_fr, fr_ca = pair
                messages = [
                    {
                        "role": "system",
                        "content": "Tu compares les usages fr-FR et fr-CA sans halluciner.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Quel terme est préférable en français québécois dans ce contexte: "
                            f"{fr_fr} ou {fr_ca}?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"En français québécois, on privilégie {fr_ca} dans ce contexte; "
                            f"{fr_fr} est moins usuel ici."
                        ),
                    },
                ]
            else:
                continue

            if messages is None:
                continue

            example = _build_example(
                record=record,
                task_type=task_type,
                messages=messages,
                chat_format=policy.instructionization.chat_format,
                permission_sources=permission_sources,
            )
            rendered_hash = _sha256(example["text"])
            normalized_example_hash = _sha256(_normalize_text(example["text"]))
            if rendered_hash in seen_example_hashes:
                continue
            if normalized_example_hash in seen_example_normalized_hashes:
                continue
            seen_example_hashes.add(rendered_hash)
            seen_example_normalized_hashes.add(normalized_example_hash)
            example["example_sha256"] = rendered_hash
            examples.append(example)
            generated_for_record += 1

    examples = sorted(examples, key=lambda row: row["example_id"])
    splits = _split_examples(
        examples,
        seed=policy.random_seed,
        split_ratios={
            "train": policy.split_ratios.train,
            "dev": policy.split_ratios.dev,
            "test": policy.split_ratios.test,
        },
    )

    train_rows = splits["train"]
    dev_rows = splits["dev"]
    test_rows = splits["test"]

    _write_jsonl(output_root / "train.jsonl", train_rows)
    _write_jsonl(output_root / "dev.jsonl", dev_rows)
    _write_jsonl(output_root / "test.jsonl", test_rows)

    source_summary = Counter(row["source_id"] for row in examples)
    source_family_summary = Counter(row["source_family"] for row in examples)
    domain_summary = Counter(row["domain"] for row in examples)
    register_summary = Counter(row["register"] for row in examples)
    task_type_summary = Counter(row["task_type"] for row in examples)
    license_status_summary = Counter(row["license_status"] for row in examples)
    commercial_use_summary = Counter(row["commercial_use"] for row in examples)

    estimated_tokens_total = sum(_estimate_tokens(row["text"]) for row in examples)
    estimated_tokens_train = sum(_estimate_tokens(row["text"]) for row in train_rows)
    raw_records_tokens_total = sum(_estimate_tokens(row["text"]) for row in accepted_records)
    instruction_like_examples = sum(
        1 for row in examples if row.get("task_type") != "preserve_raw"
    )
    instruction_like_ratio = (
        instruction_like_examples / len(examples)
        if examples
        else 0.0
    )

    max_source_share = (
        max(source_summary.values()) / len(examples) if examples else 0.0
    )
    max_source_family_share = (
        max(source_family_summary.values()) / len(examples) if examples else 0.0
    )

    thresholds = {
        "smoke_test_tokens": policy.readiness_thresholds.smoke_test_tokens,
        "pilot_lora_tokens": policy.readiness_thresholds.pilot_lora_tokens,
        "production_lora_tokens": policy.readiness_thresholds.production_lora_tokens,
        "production_instruction_examples": (
            policy.readiness_thresholds.production_instruction_examples
        ),
    }
    readiness_level = _readiness_level_from_tokens(estimated_tokens_total, thresholds)

    blocking_reasons: list[str] = []
    if not accepted_records:
        blocking_reasons.append("no_records_accepted")
    if not examples:
        blocking_reasons.append("no_examples_generated")

    if readiness_level in {"pilot_lora_candidate", "production_lora_candidate"}:
        if len(domain_summary) < policy.balancing.min_domain_count_for_pilot:
            blocking_reasons.append("low_domain_diversity_for_pilot")
        if len(register_summary) < policy.balancing.min_register_count_for_pilot:
            blocking_reasons.append("low_register_diversity_for_pilot")

    if max_source_share > policy.balancing.max_single_source_share + 1e-9:
        blocking_reasons.append("single_source_dominance")
    if max_source_family_share > policy.balancing.max_single_source_family_share + 1e-9:
        blocking_reasons.append("single_source_family_dominance")

    if readiness_level == "production_lora_candidate":
        if len(examples) < thresholds["production_instruction_examples"]:
            blocking_reasons.append("insufficient_instruction_examples_for_production")
        if blocking_reasons:
            readiness_level = "production_blocked"

    recommendations = [
        "Increase instruction-style examples and diversified registers/domains for pilot readiness.",
        "Keep adding modern institutional and conversational sources to reduce dominance.",
    ]
    noncommercial_ratio = (
        commercial_use_summary.get("permission_required", 0) / len(examples)
        if examples
        else 0.0
    )
    if noncommercial_ratio > 0.3:
        recommendations.append(
            "High permission-required share detected; secure commercial-safe sources for production."
        )
    if task_type_summary.get("preserve_raw", 0) / max(len(examples), 1) > 0.8:
        recommendations.append(
            "Instruction variety is low; increase summarize/qa/explain-style examples."
        )
    if instruction_like_ratio < 0.5:
        recommendations.append(
            "Instruction-like ratio is low; increase non-preserve_raw strategies."
        )

    artifacts = {
        "train": repo_relative_path(output_root / "train.jsonl"),
        "dev": repo_relative_path(output_root / "dev.jsonl"),
        "test": repo_relative_path(output_root / "test.jsonl"),
        "report": repo_relative_path(output_root / "report.json"),
        "dataset_card": repo_relative_path(output_root / "dataset_card.md"),
    }

    report = {
        "ok": True,
        "policy": repo_relative_path(policy_path),
        "output_dir": repo_relative_path(output_root),
        "input_files": sorted(input_files),
        "records_seen": records_seen,
        "records_accepted": len(accepted_records),
        "records_rejected": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "examples_generated": len(examples),
        "raw_records_count": len(accepted_records),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "test_count": len(test_rows),
        "estimated_tokens_total": estimated_tokens_total,
        "estimated_tokens_train": estimated_tokens_train,
        "raw_records_tokens_total": raw_records_tokens_total,
        "instruction_examples_total": len(examples),
        "instruction_like_ratio": round(instruction_like_ratio, 6),
        "source_diversity_count": len(source_summary),
        "domain_diversity_count": len(domain_summary),
        "register_diversity_count": len(register_summary),
        "source_summary": dict(sorted(source_summary.items())),
        "source_family_summary": dict(sorted(source_family_summary.items())),
        "domain_summary": dict(sorted(domain_summary.items())),
        "register_summary": dict(sorted(register_summary.items())),
        "task_type_summary": dict(sorted(task_type_summary.items())),
        "license_status_summary": dict(sorted(license_status_summary.items())),
        "commercial_use_summary": dict(sorted(commercial_use_summary.items())),
        "max_single_source_share": round(max_source_share, 6),
        "max_single_source_family_share": round(max_source_family_share, 6),
        "duplicate_exact_rejected": duplicate_exact_rejected,
        "duplicate_normalized_rejected": duplicate_normalized_rejected,
        "holdout_rejected": holdout_rejected,
        "permission_rejected": permission_rejected,
        "readiness_level": readiness_level,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "recommendations": recommendations,
        "artifacts": artifacts,
    }

    _build_dataset_card(path=output_root / "dataset_card.md", policy=policy, report=report)
    write_json(output_root / "report.json", report)
    return report


def audit_training_pack(pack_dir: Path, out_report: Path) -> dict[str, Any]:
    pack_root = pack_dir if pack_dir.is_absolute() else ROOT / pack_dir
    report_path = pack_root / "report.json"

    if not report_path.exists():
        payload = {
            "ok": False,
            "pack_dir": repo_relative_path(pack_root),
            "report": repo_relative_path(report_path),
            "blocking_reasons": ["missing_training_pack_report"],
            "readiness_level": "insufficient",
            "examples_generated": 0,
            "train_count": 0,
            "dev_count": 0,
            "test_count": 0,
            "estimated_tokens_total": 0,
        }
        write_json(out_report, payload)
        return payload

    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = {
        "ok": bool(report.get("ok", False)),
        "pack_dir": repo_relative_path(pack_root),
        "report": repo_relative_path(report_path),
        "readiness_level": report.get("readiness_level", "insufficient"),
        "blocking_reasons": report.get("blocking_reasons", []),
        "examples_generated": int(report.get("examples_generated", 0)),
        "raw_records_count": int(report.get("raw_records_count", 0)),
        "train_count": int(report.get("train_count", 0)),
        "dev_count": int(report.get("dev_count", 0)),
        "test_count": int(report.get("test_count", 0)),
        "estimated_tokens_total": int(report.get("estimated_tokens_total", 0)),
        "raw_records_tokens_total": int(report.get("raw_records_tokens_total", 0)),
        "instruction_examples_total": int(report.get("instruction_examples_total", 0)),
        "instruction_like_ratio": float(report.get("instruction_like_ratio", 0.0)),
        "source_diversity_count": int(report.get("source_diversity_count", 0)),
        "domain_diversity_count": int(report.get("domain_diversity_count", 0)),
        "register_diversity_count": int(report.get("register_diversity_count", 0)),
        "source_summary": report.get("source_summary", {}),
        "domain_summary": report.get("domain_summary", {}),
        "register_summary": report.get("register_summary", {}),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_report, payload)
    return payload
