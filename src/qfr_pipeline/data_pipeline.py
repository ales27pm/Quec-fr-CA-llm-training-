from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import yaml

from qfr_pipeline.paths import RELEASE_GATES_PATH

FR_CA_MARKERS = {
    "courriel",
    "fin de semaine",
    "magasiner",
    "char",
    "dépanneur",
    "chu",
    "icitte",
    "pantoute",
}

FR_FR_TO_FR_CA_REPLACEMENTS = {
    "email": "courriel",
    "week-end": "fin de semaine",
    "voiture": "char",
}


def _norm_text(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def harvest(input_paths: list[Path], out_jsonl: Path, min_chars: int, dedupe_batch_size: int = 0) -> int:
    seen: set[str] = set()
    kept = 0
    kept_since_clear = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as out:
        for path in input_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = _norm_text(line)
                    if len(line) < min_chars:
                        continue
                    digest = _hash_text(line)
                    if digest in seen:
                        continue
                    seen.add(digest)
                    out.write(json.dumps({"id": digest, "source": path.name, "text": line}, ensure_ascii=False) + "\n")
                    kept += 1
                    kept_since_clear += 1
                    if dedupe_batch_size > 0 and kept_since_clear >= dedupe_batch_size:
                        seen.clear()
                        kept_since_clear = 0
    return kept


def curate(in_jsonl: Path, out_jsonl: Path, min_fr_ca_score: float) -> int:
    kept = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with in_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for raw in src:
            rec = json.loads(raw)
            text = rec["text"].lower()
            marker_hits = sum(1 for m in FR_CA_MARKERS if re.search(rf"\b{re.escape(m)}\b", text))
            score = marker_hits / max(1, len(text.split()))
            if score < min_fr_ca_score:
                continue
            rec["fr_ca_marker_hits"] = marker_hits
            rec["fr_ca_score"] = round(score, 4)
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def edit_normative(in_jsonl: Path, out_jsonl: Path) -> int:
    changed = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def _match_case(src: str, replacement: str) -> str:
        if src.isupper():
            return replacement.upper()
        if src[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    with in_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for raw in src:
            rec = json.loads(raw)
            text = rec["text"]
            new_text = text
            for bad, good in FR_FR_TO_FR_CA_REPLACEMENTS.items():
                pattern = re.compile(rf"\b{re.escape(bad)}\b", flags=re.IGNORECASE)
                new_text = pattern.sub(lambda m: _match_case(m.group(0), good), new_text)
            rec["text"] = new_text
            rec["normative_edits"] = int(new_text != text)
            changed += rec["normative_edits"]
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed


def split_train_dev_test(in_jsonl: Path, out_dir: Path, seed: int = 42) -> dict[str, int]:
    rows = [json.loads(line) for line in in_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)
    splits = {"train": rows[:n_train], "dev": rows[n_train : n_train + n_dev], "test": rows[n_train + n_dev :]}
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, data in splits.items():
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[name] = len(data)
    return counts


def write_training_recipe(data_dir: Path, out_yaml: Path) -> None:
    splits = {name: data_dir / f"{name}.jsonl" for name in ("train", "dev", "test")}
    missing = [name for name, path in splits.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing split files in {data_dir}: {missing}")
    gates = yaml.safe_load(RELEASE_GATES_PATH.read_text(encoding="utf-8")) or {}
    lp7_key = "alignment.lp7_standard_negation_max_post_alignment_drop_ratio"
    if not isinstance(gates, dict):
        raise SystemExit(f"Malformed release gates in {RELEASE_GATES_PATH}: expected top-level mapping")
    alignment = gates.get("alignment")
    if not isinstance(alignment, dict):
        raise SystemExit(f"Malformed release gates in {RELEASE_GATES_PATH}: `alignment` must be a mapping")
    gate_name = lp7_key.split(".")[-1]
    if gate_name not in alignment:
        raise SystemExit(f"Missing `{lp7_key}` in {RELEASE_GATES_PATH}")
    recipe = {
        "model": {"base": "mistral-7b-instruct", "dtype": "float16"},
        "runtime": {"target": "ctranslate2"},
        "data": {k: str(v) for k, v in splits.items()},
        "alignment": {
            "include_qfrblimp_gold_pairs": True,
            "lp7_post_alignment_max_drop_key": lp7_key,
            "lp7_post_alignment_max_drop_ratio": alignment["lp7_standard_negation_max_post_alignment_drop_ratio"],
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")
