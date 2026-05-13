#!/usr/bin/env python3
"""Core fr-CA data pipeline algorithms: harvest, curate, edit, and train-prep.

Designed to be dependency-light (stdlib only) for reproducible execution in CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


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

FR_FR_REPLACEMENTS = {
    "email": "courriel",
    "week-end": "fin de semaine",
    "voiture": "char",
}


@dataclass
class Record:
    source: str
    text: str
    register: str
    confidence: float


def _norm_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def harvest(input_paths: list[Path], out_jsonl: Path, min_chars: int) -> int:
    seen: set[str] = set()
    kept = 0
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
                    rec = {
                        "id": digest,
                        "source": path.name,
                        "text": line,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
    return kept


def curate(in_jsonl: Path, out_jsonl: Path, min_fr_ca_score: float) -> int:
    kept = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with in_jsonl.open("r", encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for raw in src:
            rec = json.loads(raw)
            text = rec["text"].lower()
            marker_hits = sum(1 for m in FR_CA_MARKERS if re.search(rf"\b{re.escape(m)}\b", text))
            score = marker_hits / max(1, len(FR_CA_MARKERS) ** 0.5)
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
            for bad, good in FR_FR_REPLACEMENTS.items():
                pattern = re.compile(rf"\b{re.escape(bad)}\b", flags=re.IGNORECASE)
                new_text = pattern.sub(lambda m: _match_case(m.group(0), good), new_text)
            rec["text"] = new_text
            rec["normative_edits"] = int(new_text != text)
            changed += rec["normative_edits"]
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed


def split_train_dev_test(in_jsonl: Path, out_dir: Path, seed: int = 42) -> dict[str, int]:
    rows = [json.loads(r) for r in in_jsonl.read_text(encoding="utf-8").splitlines() if r.strip()]
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.9)
    n_dev = int(n * 0.05)
    splits = {
        "train": rows[:n_train],
        "dev": rows[n_train:n_train + n_dev],
        "test": rows[n_train + n_dev:],
    }
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
    recipe = f"""model:\n  base: mistral-7b-instruct\n  dtype: float16\nruntime:\n  target: ctranslate2\ndata:\n  train: {data_dir / 'train.jsonl'}\n  dev: {data_dir / 'dev.jsonl'}\n  test: {data_dir / 'test.jsonl'}\nalignment:\n  include_qfrblimp_gold_pairs: true\n  lp7_post_alignment_max_drop_key: alignment.lp7_standard_negation_max_post_alignment_drop_ratio\n"""
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(recipe, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("harvest")
    p_h.add_argument("--inputs", nargs="+", type=Path, required=True)
    p_h.add_argument("--out", type=Path, required=True)
    p_h.add_argument("--min-chars", type=int, default=20)

    p_c = sub.add_parser("curate")
    p_c.add_argument("--in", dest="inp", type=Path, required=True)
    p_c.add_argument("--out", type=Path, required=True)
    p_c.add_argument("--min-fr-ca-score", type=float, default=0.34)

    p_e = sub.add_parser("edit")
    p_e.add_argument("--in", dest="inp", type=Path, required=True)
    p_e.add_argument("--out", type=Path, required=True)

    p_s = sub.add_parser("split")
    p_s.add_argument("--in", dest="inp", type=Path, required=True)
    p_s.add_argument("--out-dir", type=Path, required=True)
    p_s.add_argument("--seed", type=int, default=42)

    p_r = sub.add_parser("recipe")
    p_r.add_argument("--data-dir", type=Path, required=True)
    p_r.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "harvest":
        print(harvest(args.inputs, args.out, args.min_chars))
    elif args.cmd == "curate":
        print(curate(args.inp, args.out, args.min_fr_ca_score))
    elif args.cmd == "edit":
        print(edit_normative(args.inp, args.out))
    elif args.cmd == "split":
        print(json.dumps(split_train_dev_test(args.inp, args.out_dir, args.seed)))
    elif args.cmd == "recipe":
        write_training_recipe(args.data_dir, args.out)
        print(args.out)


if __name__ == "__main__":
    main()
