from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import LP9FailureMiningPolicyManifest

SYSTEM_PROMPT = (
    "Tu es un assistant linguistique qui répond en français québécois standard, "
    "avec un vocabulaire recommandé par les usages normatifs du Québec."
)

FAILURE_TASKS = [
    "rewrite",
    "direct_preference",
    "choose_best_term",
    "correction",
    "open_generation",
]

DIGITAL_CONTEXTS = [
    "dans un contexte numérique",
    "dans une application mobile",
    "sur un site web",
    "dans un service en ligne",
    "dans un message ou une interface digitale",
]

GENERAL_CONTEXTS = [
    "dans un message professionnel au Québec",
    "dans une communication de service québécois",
    "dans une phrase naturelle en français québécois",
    "pour une personne vivant au Québec",
    "dans un contexte courant québécois",
]

PAIR_OVERRIDE_KEYS = {
    "chat": "chat",
    "weekend": "weekend",
    "email": "email",
    "smartphone": "smartphone",
}


def _is_repo_relative_path(value: str) -> bool:
    return bool(value and not value.startswith(('/', './', '../')))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_lp9_failure_mining_policy(path: Path) -> LP9FailureMiningPolicyManifest:
    return LP9FailureMiningPolicyManifest.model_validate(load_yaml(path))


def validate_lp9_failure_mining_policy(path: Path) -> LP9FailureMiningPolicyManifest:
    return load_lp9_failure_mining_policy(path)


def _resolve_repo_path(path_str: str) -> Path:
    if Path(path_str).is_absolute():
        return Path(path_str)
    return ROOT / path_str


def _derive_pair_key(lexical_pair_id: str) -> str:
    if ":" in lexical_pair_id:
        return lexical_pair_id.split(":", 1)[1]
    return lexical_pair_id


def _select_override(pair_key: str, policy: LP9FailureMiningPolicyManifest) -> dict[str, Any] | None:
    if pair_key in policy.lexical_pair_overrides:
        return policy.lexical_pair_overrides[pair_key].model_dump()
    return None


def _build_context(pair_override: dict[str, Any] | None) -> str:
    if pair_override and pair_override.get("require_digital_context"):
        return random.choice(DIGITAL_CONTEXTS)
    return random.choice(GENERAL_CONTEXTS)


def _build_messages(
    task_type: str,
    source_term: str,
    preferred_term: str,
    forbidden_terms: list[str],
    context: str,
    pair_override: dict[str, Any] | None,
    variant: int,
) -> list[dict[str, str]]:
    forbidden = ", ".join(forbidden_terms)
    if pair_override and pair_override.get("preferred_terms"):
        preferred_term = pair_override["preferred_terms"][0]
    if pair_override and pair_override.get("forbidden_terms"):
        forbidden_terms = pair_override["forbidden_terms"]
        forbidden = ", ".join(forbidden_terms)

    if task_type == "rewrite":
        user = (
            f"Réécris cette phrase en français québécois standard en remplaçant « {source_term} » "
            f"par « {preferred_term} » : {context}."
        )
        assistant = f"{preferred_term} est préférable dans ce contexte."
    elif task_type == "direct_preference":
        if variant % 2 == 0:
            user = (
                f"Quel terme est préférable ici en français québécois: « {source_term} » ou « {preferred_term} »? "
                f"{context}."
            )
            assistant = (
                f"En français québécois standard, on privilégie « {preferred_term} » plutôt que « {source_term} ».")
        else:
            user = (
                f"Pourquoi choisirais-tu « {preferred_term} » plutôt que « {source_term} » en français québécois? "
                f"{context}."
            )
            assistant = (
                f"Parce que « {preferred_term} » est le terme recommandé au Québec, alors que « {source_term} » est moins naturel.")
    elif task_type == "choose_best_term":
        if variant % 2 == 0:
            user = (
                f"Choisis le meilleur terme pour cette phrase en français québécois: "
                f"« Pour ce contexte, on retient ______. » Options: {source_term} | {preferred_term}. "
                f"{context}."
            )
        else:
            user = (
                f"Parmi les options « {source_term} » et « {preferred_term} », quel terme est le plus approprié en français québécois? "
                f"{context}."
            )
        assistant = f"Le terme préféré est « {preferred_term} »." 
    elif task_type == "correction":
        user = (
            f"Corrige l'utilisation de « {source_term} » dans cette phrase et remplace-le par le terme recommandé: "
            f"{context}."
        )
        assistant = f"On devrait utiliser « {preferred_term} » à la place de « {source_term} » dans ce contexte."
    elif task_type == "open_generation":
        user = (
            f"Rédige une phrase naturelle en français québécois sur {context}, en utilisant « {preferred_term} » et en évitant {forbidden}."
        )
        assistant = f"Voici une phrase naturelle en français québécois qui emploie « {preferred_term} »." 
    else:
        raise ValueError(f"Unsupported LP9 failure mining task type: {task_type}")

    if variant == 2 and task_type == "rewrite":
        user = (
            f"Réécris sans utiliser « {source_term} » et privilégie « {preferred_term} » : {context}."
        )
        assistant = f"La formulation québécoise utilise « {preferred_term} » ici."

    if variant == 2 and task_type == "open_generation":
        user = (
            f"Rédige une phrase en français québécois en parlant de {context} et remplace « {source_term} » par « {preferred_term} »."
        )
        assistant = f"La phrase suivante utilise le terme recommandé : « {preferred_term} »."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _render_qwen_chatml(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"<|im_start|>{message['role']}")
        lines.append(message["content"].strip())
        lines.append("<|im_end|>")
    return "\n".join(lines)


def _build_example(
    *,
    failure: dict[str, Any],
    source_term: str,
    preferred_term: str,
    forbidden_terms: list[str],
    pair_override: dict[str, Any] | None,
    variant: int,
    ordinal: int,
) -> dict[str, Any]:
    context = _build_context(pair_override)
    messages = _build_messages(
        task_type=failure["task_type"],
        source_term=source_term,
        preferred_term=preferred_term,
        forbidden_terms=forbidden_terms,
        context=context,
        pair_override=pair_override,
        variant=variant,
    )
    rendered = _render_qwen_chatml(messages)
    rendered_hash = _sha256(rendered)
    pair_id = failure["lexical_pair_id"]
    example_id = f"lp9_failure_pack:{pair_id}:{failure['task_type']}:{ordinal:03d}:{rendered_hash[:10]}"
    return {
        "record_id": example_id,
        "example_id": example_id,
        "messages": messages,
        "text": rendered,
        "text_hash": rendered_hash,
        "task_type": failure["task_type"],
        "source_id": "lp9_failure_pack",
        "source_family": "manual_targeted",
        "language": "fr-CA",
        "dialect_region": "Quebec",
        "register": "standard",
        "domain": "digital" if pair_override and pair_override.get("require_digital_context") else "general",
        "allowed_for_training": True,
        "holdout_only": False,
        "requires_review": False,
        "commercial_use": "allowed",
        "license_status": "internal_manual",
        "quality_flags": [],
        "source_term": source_term,
        "preferred_term": preferred_term,
        "rejected_terms": forbidden_terms,
        "lexical_pair_id": pair_id,
        "failure_reasons": list(failure.get("reasons", [])),
        "context": context,
        "pair_override": pair_override or {},
    }


def _failure_included(failure: dict[str, Any], policy: LP9FailureMiningPolicyManifest) -> bool:
    reasons = set(failure.get("reasons", []))
    if not reasons:
        return False
    if "adapter_under_base" in reasons and policy.include_adapter_under_base:
        return True
    if "adapter_missing_preferred" in reasons and policy.include_missing_preferred:
        return True
    if "adapter_contains_forbidden" in reasons and policy.include_contains_forbidden:
        return True
    return False


def _failure_weight(failure: dict[str, Any], policy: LP9FailureMiningPolicyManifest) -> int:
    task_type = str(failure["task_type"])
    weight = float(policy.task_type_weights.model_dump().get(task_type, 1.0))
    count = max(1, int(round(policy.repetitions_per_failure * weight)))
    return count


def _derive_text_stats(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    by_pair = Counter(row["lexical_pair_id"] for row in rows)
    by_task = Counter(row["task_type"] for row in rows)
    by_reason = Counter(reason for row in rows for reason in row.get("failure_reasons", []))
    return dict(by_pair), dict(by_task), dict(by_reason)


def _split_examples(rows: list[dict[str, Any]], train_ratio: float, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["example_id"]):
        hash_val = int(_sha256(f"{random_seed}:{row['example_id']}")[:8], 16) / 0xFFFFFFFF
        if hash_val < train_ratio:
            train_rows.append(row)
        else:
            dev_rows.append(row)
    return train_rows, dev_rows


def _derive_next_eval_focus(
    rows: list[dict[str, Any]], policy: LP9FailureMiningPolicyManifest
) -> list[str]:
    counts = Counter(row["task_type"] for row in rows)
    focus: list[str] = []
    if counts["choose_best_term"]:
        focus.append("choose_best_term")
    if counts["open_generation"]:
        focus.append("open_generation")
    if any(_derive_pair_key(row["lexical_pair_id"]) == "chat" for row in rows):
        focus.append("chat digital")
    if any(_derive_pair_key(row["lexical_pair_id"]) == "smartphone" for row in rows):
        focus.append("smartphone/cellulaire")
    if any(_derive_pair_key(row["lexical_pair_id"]) == "email" for row in rows):
        focus.append("email/courriel")
    if not focus:
        focus.append("broad lexical steering")
    return focus


def generate_lp9_failure_pack(
    policy_path: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    policy = validate_lp9_failure_mining_policy(policy_path)
    output_dir = out_dir or (ROOT / policy.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    report = _read_json(_resolve_repo_path(policy.source_eval_report))
    failures = [failure for failure in report.get("failures", []) if _failure_included(failure, policy)]

    generation_rows = _read_jsonl(_resolve_repo_path(policy.source_generations))
    generation_map = {row["prompt_id"]: row for row in generation_rows}

    all_examples: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    examples_by_pair: dict[str, int] = defaultdict(int)
    failures_by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for failure in failures:
        failures_by_pair[failure["lexical_pair_id"]].append(failure)

    for failure in failures:
        prompt_row = generation_map.get(failure["prompt_id"], {})
        forbidden_terms = [str(term) for term in prompt_row.get("forbidden_terms", [])]
        preferred_terms = [str(term) for term in prompt_row.get("expected_terms", [])]
        source_term = forbidden_terms[0] if forbidden_terms else (preferred_terms[0] if preferred_terms else _derive_pair_key(failure["lexical_pair_id"]))
        preferred_term = preferred_terms[0] if preferred_terms else source_term
        pair_key = _derive_pair_key(failure["lexical_pair_id"])
        pair_override = _select_override(pair_key, policy)
        if pair_override and pair_override.get("preferred_terms"):
            preferred_term = pair_override["preferred_terms"][0]
        if pair_override and pair_override.get("forbidden_terms"):
            forbidden_terms = pair_override["forbidden_terms"]
        count = _failure_weight(failure, policy)
        ordinal = 1
        for variant in range(1, count + 1):
            example = _build_example(
                failure=failure,
                source_term=source_term,
                preferred_term=preferred_term,
                forbidden_terms=forbidden_terms,
                pair_override=pair_override,
                variant=variant,
                ordinal=ordinal,
            )
            ordinal += 1
            if example["text"] in seen_texts:
                continue
            seen_texts.add(example["text"])
            all_examples.append(example)
            examples_by_pair[failure["lexical_pair_id"]] += 1

    # Ensure minimum coverage by lexical pair.
    for pair_id, failures_for_pair in failures_by_pair.items():
        while examples_by_pair[pair_id] < policy.min_failures_per_pair and failures_for_pair:
            failure = failures_for_pair[examples_by_pair[pair_id] % len(failures_for_pair)]
            prompt_row = generation_map.get(failure["prompt_id"], {})
            forbidden_terms = [str(term) for term in prompt_row.get("forbidden_terms", [])]
            preferred_terms = [str(term) for term in prompt_row.get("expected_terms", [])]
            source_term = forbidden_terms[0] if forbidden_terms else (preferred_terms[0] if preferred_terms else _derive_pair_key(pair_id))
            preferred_term = preferred_terms[0] if preferred_terms else source_term
            pair_key = _derive_pair_key(pair_id)
            pair_override = _select_override(pair_key, policy)
            if pair_override and pair_override.get("preferred_terms"):
                preferred_term = pair_override["preferred_terms"][0]
            if pair_override and pair_override.get("forbidden_terms"):
                forbidden_terms = pair_override["forbidden_terms"]
            example = _build_example(
                failure=failure,
                source_term=source_term,
                preferred_term=preferred_term,
                forbidden_terms=forbidden_terms,
                pair_override=pair_override,
                variant=examples_by_pair[pair_id] + 1,
                ordinal=examples_by_pair[pair_id] + 1,
            )
            if example["text"] in seen_texts:
                break
            seen_texts.add(example["text"])
            all_examples.append(example)
            examples_by_pair[pair_id] += 1

    if policy.max_examples and len(all_examples) > policy.max_examples:
        all_examples = sorted(all_examples, key=lambda item: item["example_id"])[: policy.max_examples]

    train_rows, dev_rows = _split_examples(all_examples, policy.train_ratio, policy.random_seed)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    report_path = output_dir / "report.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)

    by_pair, by_task, by_reason = _derive_text_stats(all_examples)
    failures_seen = len(report.get("failures", [])) if isinstance(report, dict) else 0
    report_payload = {
        "ok": True,
        "kind": "lp9_failure_pack_report",
        "policy": repo_relative_path(policy_path),
        "source_eval_report": repo_relative_path(Path(policy.source_eval_report)),
        "source_generations": repo_relative_path(Path(policy.source_generations)),
        "output_dir": repo_relative_path(output_dir),
        "failures_seen": failures_seen,
        "failures_used": len(failures),
        "examples_generated": len(all_examples),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "by_pair": by_pair,
        "by_task_type": by_task,
        "by_reason": by_reason,
        "expected_next_eval_focus": _derive_next_eval_focus(all_examples, policy),
        "outputs": {
            "train": repo_relative_path(train_path),
            "dev": repo_relative_path(dev_path),
            "report": repo_relative_path(report_path),
        },
    }
    write_json(report_path, report_payload)
    return report_payload
