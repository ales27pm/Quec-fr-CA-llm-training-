from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import LP9LexicalPair, LP9LexicalPreferencePackManifest

SYSTEM_PROMPT = (
    "Tu es un assistant linguistique qui répond en français québécois standard, "
    "avec un vocabulaire recommandé par les usages normatifs du Québec."
)

CONTEXT_GROUP_ORDER = [
    "question",
    "customer support",
    "workplace",
    "school",
    "family",
    "government/service",
    "commerce",
    "mobile/device",
    "email/message",
    "everyday casual",
]

DEV_CONTEXT_GROUPS = {"government/service", "everyday casual"}

CONTEXT_TEMPLATES: dict[str, list[str]] = {
    "question": [
        "Dans cette question fréquente, est-ce que le terme « {source_term} » convient?",
        "Je prépare une FAQ; dois-je garder « {source_term} » dans la réponse?",
        "Pour ce formulaire de questions, faut-il écrire « {source_term} »?",
        "Dans un questionnaire d'aide, le mot « {source_term} » est-il approprié?",
        "Question de vocabulaire: faut-il conserver « {source_term} » ici?",
    ],
    "customer support": [
        "Notre agent de soutien a écrit « {source_term} » dans le courriel au client.",
        "Au service à la clientèle, la note interne contient « {source_term} ».",
        "Le script de clavardage du soutien mentionne « {source_term} ».",
        "La réponse standard du centre d'aide utilise « {source_term} ».",
        "Le message de suivi client inclut le terme « {source_term} ».",
    ],
    "workplace": [
        "Dans un compte rendu d'équipe, on a écrit « {source_term} ».",
        "La communication interne du bureau contient « {source_term} ».",
        "Le mémo hebdomadaire de l'organisation mentionne « {source_term} ».",
        "Un collègue a proposé d'utiliser « {source_term} » dans la procédure.",
        "Dans l'intranet du travail, le terme « {source_term} » apparaît.",
    ],
    "school": [
        "Le guide scolaire affiche le mot « {source_term} » dans un exemple.",
        "Dans un devoir de français, l'élève a utilisé « {source_term} ».",
        "Le portail de classe emploie « {source_term} » dans ses consignes.",
        "Le message aux parents contient « {source_term} ».",
        "Une fiche pédagogique propose « {source_term} ».",
    ],
    "family": [
        "Dans le groupe familial, quelqu'un a écrit « {source_term} ».",
        "Le rappel pour la famille contient le terme « {source_term} ».",
        "Notre tableau de tâches à la maison affiche « {source_term} ».",
        "Pendant une discussion en famille, on a repris « {source_term} ».",
        "Le mémo du frigo mentionne « {source_term} ».",
    ],
    "government/service": [
        "Dans un avis de service municipal, on lit « {source_term} ».",
        "Le portail d'un organisme public contient « {source_term} ».",
        "Une lettre administrative utilise « {source_term} ».",
        "Le formulaire de service gouvernemental affiche « {source_term} ».",
        "Le guide d'un service citoyen inclut « {source_term} ».",
    ],
    "commerce": [
        "Une annonce commerciale locale utilise « {source_term} ».",
        "Le site d'un détaillant affiche « {source_term} » dans sa bannière.",
        "Le reçu numérique du commerce mentionne « {source_term} ».",
        "La description d'un produit emploie « {source_term} ».",
        "Le message promotionnel en ligne contient « {source_term} ».",
    ],
    "mobile/device": [
        "Dans les paramètres de l'appareil, on voit « {source_term} ».",
        "Un message d'application mobile contient « {source_term} ».",
        "Le guide de configuration du téléphone affiche « {source_term} ».",
        "La notification de l'app indique « {source_term} ».",
        "Le tutoriel de l'appareil mentionne « {source_term} ».",
    ],
    "email/message": [
        "Dans un message professionnel, on retrouve « {source_term} ».",
        "Le gabarit de message interne inclut « {source_term} ».",
        "L'objet du message contient le mot « {source_term} ».",
        "Un message d'équipe réutilise « {source_term} ».",
        "Le brouillon du message affiche « {source_term} ».",
    ],
    "everyday casual": [
        "Dans une conversation quotidienne, une personne dit « {source_term} ».",
        "En contexte courant, on entend souvent « {source_term} ».",
        "Dans un échange entre amis, le mot « {source_term} » apparaît.",
        "Une discussion de tous les jours contient « {source_term} ».",
        "Au quotidien, on lit parfois « {source_term} » dans les textos.",
    ],
}


TASK_ORDER = [
    "rewrite_to_quebec_fr",
    "direct_preference",
    "explain_preference",
    "negative_contrast",
]


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", _normalize_text(text))
    clean = clean.strip("-")
    return clean or "x"


def _render_qwen_chatml(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        lines.append(f"<|im_start|>{message['role']}")
        lines.append(message["content"].strip())
        lines.append("<|im_end|>")
    return "\n".join(lines)


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
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def load_lp9_micro_pack_manifest(path: Path) -> LP9LexicalPreferencePackManifest:
    return LP9LexicalPreferencePackManifest.model_validate(load_yaml(path))


def validate_lp9_micro_pack_manifest(path: Path) -> LP9LexicalPreferencePackManifest:
    return load_lp9_micro_pack_manifest(path)


def _enabled_task_types(manifest: LP9LexicalPreferencePackManifest) -> list[str]:
    enabled: list[str] = []
    if manifest.include_rewrite_tasks:
        enabled.append("rewrite_to_quebec_fr")
    if manifest.include_direct_preference_tasks:
        enabled.append("direct_preference")
    if manifest.include_explanation_tasks:
        enabled.append("explain_preference")
    if manifest.include_negative_contrast_tasks:
        enabled.append("negative_contrast")
    return [task for task in TASK_ORDER if task in enabled]


def _build_context_pool() -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    for group in CONTEXT_GROUP_ORDER:
        templates = CONTEXT_TEMPLATES[group]
        for idx, template in enumerate(templates, start=1):
            contexts.append(
                {
                    "context_group": group,
                    "template_id": f"{_slugify(group)}-{idx:02d}",
                    "template": template,
                }
            )
    return contexts


def _repeat_contexts(pool: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0:
        return []
    if not pool:
        raise ValueError("Context pool cannot be empty")
    rows: list[dict[str, str]] = []
    cycle = 0
    while len(rows) < count:
        for item in pool:
            if len(rows) >= count:
                break
            entry = dict(item)
            if cycle > 0:
                entry["template"] = (
                    f"{item['template']} Variante {cycle + 1} pour le même contexte.".strip()
                )
                entry["template_id"] = f"{item['template_id']}-v{cycle + 1}"
            rows.append(entry)
        cycle += 1
    return rows


def _context_splits(repetitions_per_pattern: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_contexts = _build_context_pool()
    dev_pool = [ctx for ctx in all_contexts if ctx["context_group"] in DEV_CONTEXT_GROUPS]
    train_pool = [ctx for ctx in all_contexts if ctx["context_group"] not in DEV_CONTEXT_GROUPS]

    target_dev = max(1, repetitions_per_pattern // 5)
    target_dev = min(target_dev, repetitions_per_pattern - 1)
    target_train = repetitions_per_pattern - target_dev

    train_contexts = _repeat_contexts(train_pool, target_train)
    dev_contexts = _repeat_contexts(dev_pool, target_dev)
    return train_contexts, dev_contexts


def _replace_source_term(sentence: str, pair: LP9LexicalPair) -> str:
    return sentence.replace(pair.source_term, pair.preferred_term)


def _build_messages(
    *,
    task_type: str,
    pair: LP9LexicalPair,
    sentence_with_source: str,
    sentence_with_preferred: str,
) -> list[dict[str, str]]:
    if task_type == "rewrite_to_quebec_fr":
        user = f"Réécris ceci en français québécois standard: {sentence_with_source}"
        assistant = sentence_with_preferred
    elif task_type == "direct_preference":
        user = (
            "Quel terme est préférable en français québécois dans ce contexte: "
            f"« {pair.source_term} » ou « {pair.preferred_term} »? "
            f"Contexte: {sentence_with_source}"
        )
        assistant = (
            "En français québécois standard, on privilégie "
            f"« {pair.preferred_term} » dans ce contexte."
        )
    elif task_type == "explain_preference":
        user = (
            f"Pourquoi privilégier « {pair.preferred_term} » plutôt que « {pair.source_term} » "
            f"en français québécois standard? Contexte: {sentence_with_source}"
        )
        assistant = (
            f"« {pair.preferred_term} » est le terme recommandé ou plus naturel en "
            "français québécois standard dans ce contexte."
        )
    elif task_type == "negative_contrast":
        user = f"Corrige le terme moins approprié dans cette phrase: {sentence_with_source}"
        assistant = sentence_with_preferred
    else:
        raise ValueError(f"Unsupported LP9 task type: {task_type}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _build_example(
    *,
    pair: LP9LexicalPair,
    pair_id: str,
    task_type: str,
    context_group: str,
    template_id: str,
    sentence_with_source: str,
    sentence_with_preferred: str,
    split: str,
    ordinal: int,
) -> dict[str, Any]:
    messages = _build_messages(
        task_type=task_type,
        pair=pair,
        sentence_with_source=sentence_with_source,
        sentence_with_preferred=sentence_with_preferred,
    )
    rendered = _render_qwen_chatml(messages)
    rendered_hash = _sha256(rendered)
    return {
        "example_id": (
            f"lp9:{pair_id}:{task_type}:{split}:{ordinal:03d}:{rendered_hash[:10]}"
        ),
        "messages": messages,
        "text": rendered,
        "text_hash": rendered_hash,
        "task_type": task_type,
        "source_id": "lp9_micro_pack",
        "source_family": "manual_targeted",
        "language": "fr-CA",
        "dialect_region": "Quebec",
        "register": "standard",
        "domain": pair.domain,
        "allowed_for_training": True,
        "holdout_only": False,
        "requires_review": False,
        "commercial_use": "allowed",
        "license_status": "internal_manual",
        "quality_flags": [],
        "source_term": pair.source_term,
        "preferred_term": pair.preferred_term,
        "rejected_terms": list(pair.rejected_terms),
        "lexical_pair_id": pair_id,
        "context_group": context_group,
        "template_id": template_id,
    }


def generate_lp9_micro_pack(
    manifest_path: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = validate_lp9_micro_pack_manifest(manifest_path)
    output_dir = out_dir or (ROOT / manifest.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    train_contexts, dev_contexts = _context_splits(manifest.repetitions_per_pattern)
    task_types = _enabled_task_types(manifest)

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    seen_text_hashes: set[str] = set()
    duplicates = 0

    for pair_idx, pair in enumerate(manifest.lexical_pairs, start=1):
        pair_id = f"{pair_idx:02d}-{_slugify(pair.source_term)}-to-{_slugify(pair.preferred_term)}"
        for task_type in task_types:
            for split_name, contexts, sink in (
                ("train", train_contexts, train_rows),
                ("dev", dev_contexts, dev_rows),
            ):
                for ordinal, context in enumerate(contexts, start=1):
                    sentence_with_source = context["template"].format(
                        source_term=pair.source_term,
                        preferred_term=pair.preferred_term,
                    )
                    sentence_with_preferred = _replace_source_term(
                        sentence_with_source,
                        pair,
                    )
                    row = _build_example(
                        pair=pair,
                        pair_id=pair_id,
                        task_type=task_type,
                        context_group=context["context_group"],
                        template_id=context["template_id"],
                        sentence_with_source=sentence_with_source,
                        sentence_with_preferred=sentence_with_preferred,
                        split=split_name,
                        ordinal=ordinal,
                    )
                    text_hash = row["text_hash"]
                    if text_hash in seen_text_hashes:
                        duplicates += 1
                        continue
                    seen_text_hashes.add(text_hash)
                    sink.append(row)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    report_path = output_dir / "report.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)

    by_pair = Counter(row["lexical_pair_id"] for row in train_rows + dev_rows)
    by_task = Counter(row["task_type"] for row in train_rows + dev_rows)
    split_task_counts: dict[str, dict[str, int]] = {
        "train": dict(Counter(row["task_type"] for row in train_rows)),
        "dev": dict(Counter(row["task_type"] for row in dev_rows)),
    }
    split_context_counts: dict[str, dict[str, int]] = {
        "train": dict(Counter(row["context_group"] for row in train_rows)),
        "dev": dict(Counter(row["context_group"] for row in dev_rows)),
    }

    pair_task_coverage: dict[str, list[str]] = {}
    observed_pair_tasks: dict[str, set[str]] = defaultdict(set)
    for row in train_rows + dev_rows:
        observed_pair_tasks[row["lexical_pair_id"]].add(row["task_type"])
    for pair_id, tasks in observed_pair_tasks.items():
        pair_task_coverage[pair_id] = sorted(tasks)

    report = {
        "ok": duplicates == 0,
        "kind": "lp9_micro_pack_report",
        "manifest": repo_relative_path(manifest_path),
        "pack_id": manifest.pack_id,
        "schema_version": manifest.schema_version,
        "primary_language": manifest.primary_language,
        "random_seed": manifest.random_seed,
        "repetitions_per_pattern": manifest.repetitions_per_pattern,
        "lexical_pair_count": len(manifest.lexical_pairs),
        "task_types": task_types,
        "records_total": len(train_rows) + len(dev_rows),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "duplicate_text_hashes": duplicates,
        "unique_text_hashes": len(seen_text_hashes),
        "by_pair": dict(by_pair),
        "by_task_type": dict(by_task),
        "split_task_counts": split_task_counts,
        "split_context_counts": split_context_counts,
        "pair_task_coverage": pair_task_coverage,
        "outputs": {
            "train": repo_relative_path(train_path),
            "dev": repo_relative_path(dev_path),
            "report": repo_relative_path(report_path),
        },
    }
    write_json(report_path, report)
    return report


def load_lp9_micro_pack_rows(pack_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = _read_jsonl(pack_dir / "train.jsonl")
    dev_rows = _read_jsonl(pack_dir / "dev.jsonl")
    return train_rows, dev_rows
