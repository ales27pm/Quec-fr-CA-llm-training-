from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from qfr_pipeline.io import write_json
from qfr_pipeline.lp9_micro_pack import load_lp9_micro_pack_manifest
from qfr_pipeline.paths import repo_relative_path

PROMPT_TASK_ORDER = [
    "rewrite",
    "direct_preference",
    "choose_best_term",
    "correction",
    "open_generation",
]

PROMPT_CONTEXTS = [
    "Dans un message professionnel à un client québécois.",
    "Dans une consigne destinée au personnel d'un service public du Québec.",
    "Dans un guide d'utilisation pour une application mobile au Québec.",
    "Dans un contexte scolaire standard en français québécois.",
    "Dans une communication courante entre collègues au Québec.",
]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _term_pattern(term: str) -> re.Pattern[str]:
    normalized = _normalize(term)
    expr = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){expr}(?!\w)", flags=re.IGNORECASE)


def contains_term(text: str, term: str) -> bool:
    if not term.strip():
        return False
    normalized_text = _normalize(text)
    return bool(_term_pattern(term).search(normalized_text))


def contains_any_term(text: str, terms: list[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


def score_response(response: str, scoring: dict[str, Any]) -> dict[str, Any]:
    required_any = [str(term) for term in scoring.get("required_any", [])]
    forbidden_any = [str(term) for term in scoring.get("forbidden_any", [])]
    bonus_term = str(scoring.get("exact_preferred_bonus", "")).strip()

    preferred_hit = contains_any_term(response, required_any)
    forbidden_hit = contains_any_term(response, forbidden_any)
    bonus_hit = bool(bonus_term) and contains_term(response, bonus_term)

    score = 0
    if preferred_hit:
        score += 1
    if forbidden_hit:
        score -= 1

    return {
        "score": score,
        "preferred_hit": preferred_hit,
        "forbidden_hit": forbidden_hit,
        "bonus_hit": bonus_hit,
    }


def _choose_rejected_term(rejected_terms: list[str], preferred_term: str) -> str:
    for term in rejected_terms:
        if _normalize(term) != _normalize(preferred_term):
            return term
    return rejected_terms[0] if rejected_terms else preferred_term


def _build_prompt(task_type: str, source_term: str, preferred_term: str, context: str) -> str:
    if task_type == "rewrite":
        return (
            "Réécris cette phrase en français québécois standard en conservant le sens: "
            f"« Le terme {source_term} est utilisé dans ce message. » {context}"
        )
    if task_type == "direct_preference":
        return (
            "Quel terme privilégierais-tu en français québécois standard: "
            f"« {source_term} » ou « {preferred_term} »? {context}"
        )
    if task_type == "choose_best_term":
        return (
            "Choisis le meilleur terme pour compléter cette phrase en français québécois standard: "
            f"« Pour ce contexte, nous retenons ______. » Options: {source_term} | {preferred_term}. "
            f"{context}"
        )
    if task_type == "correction":
        return (
            "Corrige le terme moins approprié dans cette phrase: "
            f"« Dans ce document, on conserve le mot {source_term}. » {context}"
        )
    if task_type == "open_generation":
        return (
            "Rédige une phrase naturelle en français québécois standard sur ce contexte et "
            f"emploie le terme recommandé pour remplacer « {source_term} ». {context}"
        )
    raise ValueError(f"Unsupported LP9 eval task_type: {task_type}")


def generate_lp9_eval_prompts(manifest_path: Path, out_path: Path) -> dict[str, Any]:
    manifest = load_lp9_micro_pack_manifest(manifest_path)
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    rows: list[dict[str, Any]] = []
    prompt_counter = 0

    for pair_idx, pair in enumerate(manifest.lexical_pairs, start=1):
        pair_id = f"{pair_idx:02d}:{_normalize(pair.source_term).replace(' ', '_')}"
        rejected = [str(term) for term in pair.rejected_terms]
        fallback_rejected = _choose_rejected_term(rejected, pair.preferred_term)

        for task_idx, task_type in enumerate(PROMPT_TASK_ORDER):
            context = PROMPT_CONTEXTS[task_idx % len(PROMPT_CONTEXTS)]
            prompt_counter += 1
            prompt_text = _build_prompt(task_type, pair.source_term, pair.preferred_term, context)
            expected_terms = [pair.preferred_term]
            forbidden_terms = list(dict.fromkeys([*rejected, fallback_rejected]))
            rows.append(
                {
                    "prompt_id": f"lp9-eval-{prompt_counter:04d}",
                    "prompt": prompt_text,
                    "expected_terms": expected_terms,
                    "forbidden_terms": forbidden_terms,
                    "lexical_pair_id": pair_id,
                    "task_type": task_type,
                    "domain": pair.domain,
                    "scoring": {
                        "required_any": expected_terms,
                        "forbidden_any": forbidden_terms,
                        "exact_preferred_bonus": pair.preferred_term,
                    },
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    out_path.write_text(payload, encoding="utf-8")

    summary = {
        "ok": True,
        "kind": "lp9_eval_prompt_report",
        "manifest": repo_relative_path(manifest_path),
        "output": repo_relative_path(out_path),
        "prompt_count": len(rows),
        "task_counts": dict(Counter(row["task_type"] for row in rows)),
        "lexical_pair_count": len(manifest.lexical_pairs),
    }
    return summary


def load_eval_prompts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_baseline_expected_report(path: Path, prompts_path: Path) -> dict[str, Any]:
    prompts = load_eval_prompts(prompts_path)
    strategy = "Réponse simulée contenant seulement le terme source (forbidden)."
    expected_score = -1 * len(prompts)
    report = {
        "kind": "lp9_eval_baseline_expected_report",
        "prompts": repo_relative_path(prompts_path),
        "strategy": strategy,
        "total_prompts": len(prompts),
        "expected_total_score": expected_score,
        "notes": (
            "Ce rapport sert de repère déterministe: si un système répond avec les termes "
            "forbidden sans termes préférés, le score total attendu est négatif."
        ),
    }
    write_json(path, report)
    return report
