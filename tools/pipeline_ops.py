#!/usr/bin/env python3
"""Core fr-CA data pipeline algorithms: harvest, curate, edit, and train-prep.

Designed to be dependency-light (stdlib only) for reproducible execution in CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
import yaml


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
                    rec = {
                        "id": digest,
                        "source": path.name,
                        "text": line,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
    with in_jsonl.open("r", encoding="utf-8") as src:
        rows = [json.loads(line) for line in src if line.strip()]
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
    splits = {
        "train": data_dir / "train.jsonl",
        "dev": data_dir / "dev.jsonl",
        "test": data_dir / "test.jsonl",
    }
    missing = [name for name, path in splits.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing split files in {data_dir}: {missing}")

    release_gates_path = Path(__file__).resolve().parents[1] / "project" / "release_gates.yaml"
    gates = yaml.safe_load(release_gates_path.read_text(encoding="utf-8")) or {}
    lp7_key = "alignment.lp7_standard_negation_max_post_alignment_drop_ratio"
    if not isinstance(gates, dict) or "alignment" not in gates or "lp7_standard_negation_max_post_alignment_drop_ratio" not in gates["alignment"]:
        raise SystemExit(f"Missing `{lp7_key}` in {release_gates_path}")

    recipe = {
        "model": {"base": "mistral-7b-instruct", "dtype": "float16"},
        "runtime": {"target": "ctranslate2"},
        "data": {k: str(v) for k, v in splits.items()},
        "alignment": {
            "include_qfrblimp_gold_pairs": True,
            "lp7_post_alignment_max_drop_key": lp7_key,
            "lp7_post_alignment_max_drop_ratio": gates["alignment"]["lp7_standard_negation_max_post_alignment_drop_ratio"],
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensions must match.")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def monitor_lp7(pre_alignment_score: float, post_alignment_score: float, release_gates_path: Path) -> dict[str, float | bool]:
    try:
        gates = yaml.safe_load(release_gates_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Failed to read/parse release gates YAML at {release_gates_path}: {exc}")
    if not isinstance(gates, dict):
        raise SystemExit(f"Malformed release gates config at {release_gates_path}: expected top-level mapping.")
    if "alignment" not in gates or not isinstance(gates["alignment"], dict):
        raise SystemExit(f"Malformed release gates config at {release_gates_path}: missing mapping key `alignment`.")
    if "lp7_standard_negation_max_post_alignment_drop_ratio" not in gates["alignment"]:
        raise SystemExit(
            "Malformed release gates config at "
            f"{release_gates_path}: missing key `alignment.lp7_standard_negation_max_post_alignment_drop_ratio`."
        )
    try:
        max_drop = float(gates["alignment"]["lp7_standard_negation_max_post_alignment_drop_ratio"])
    except (TypeError, ValueError):
        raise SystemExit(
            "Malformed release gates config at "
            f"{release_gates_path}: `alignment.lp7_standard_negation_max_post_alignment_drop_ratio` must be numeric."
        )
    if not 0.0 <= max_drop <= 1.0:
        raise SystemExit(
            "Malformed release gates config at "
            f"{release_gates_path}: `alignment.lp7_standard_negation_max_post_alignment_drop_ratio` must be in [0.0, 1.0]."
        )
    try:
        pre = float(pre_alignment_score)
        post = float(post_alignment_score)
    except (TypeError, ValueError):
        raise SystemExit("LP7 scores must be numeric values.")
    if not 0.0 < pre <= 1.0:
        raise SystemExit("LP7 pre-alignment score must be in (0.0, 1.0].")
    if not 0.0 <= post <= 1.0:
        raise SystemExit("LP7 post-alignment score must be in [0.0, 1.0].")
    drop_ratio = (pre - post) / pre
    return {
        "pre_alignment_score": round(pre, 6),
        "post_alignment_score": round(post, 6),
        "drop_ratio": round(drop_ratio, 6),
        "max_allowed_drop_ratio": max_drop,
        "rollback_required": drop_ratio > max_drop,
    }


def lp_semantic_diagnostics(in_csv: Path, out_json: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    required = {"phenomenon", "is_correct", "embedding_ref", "embedding_pred", "error_label"}
    if not rows:
        raise SystemExit("No diagnostic rows found.")
    header = set(fieldnames or [])
    if not required.issubset(header):
        missing = sorted(required - header)
        raise SystemExit(f"Missing required columns: {missing}")

    by_lp: dict[str, dict[str, float | int]] = {"LP9": {"n": 0, "correct": 0, "semantic_sum": 0.0}, "LP20": {"n": 0, "correct": 0, "semantic_sum": 0.0}}
    taxonomy: dict[str, int] = {}
    malformed_rows = 0
    for row in rows:
        ph = (row.get("phenomenon") or "").strip().upper()
        if ph not in by_lp:
            malformed_rows += 1
            continue
        correct_raw = (row.get("is_correct") or "").strip()
        if correct_raw not in {"0", "1"}:
            malformed_rows += 1
            continue
        try:
            embedding_ref_raw = (row.get("embedding_ref") or "").strip()
            embedding_pred_raw = (row.get("embedding_pred") or "").strip()
            if not embedding_ref_raw or not embedding_pred_raw:
                raise ValueError("Missing embedding values.")
            ref = [float(x) for x in embedding_ref_raw.split()]
            pred = [float(x) for x in embedding_pred_raw.split()]
            if not ref or not pred:
                raise ValueError("Empty embedding vectors.")
            sim = _cosine_similarity(ref, pred)
        except (ValueError, TypeError):
            malformed_rows += 1
            continue
        correct = correct_raw == "1"
        by_lp[ph]["n"] += 1
        by_lp[ph]["correct"] += int(correct)
        by_lp[ph]["semantic_sum"] += sim
        if not correct:
            label = (row.get("error_label") or "").strip() or "unknown"
            taxonomy[label] = taxonomy.get(label, 0) + 1

    result = {"phenomena": {}, "error_taxonomy": taxonomy, "malformed_rows": malformed_rows}
    missing_phenomena: list[str] = []
    for ph, stats in by_lp.items():
        n = int(stats["n"])
        if n == 0:
            missing_phenomena.append(ph)
            continue
        result["phenomena"][ph] = {
            "n": n,
            "binary_accuracy": round(stats["correct"] / n, 6),
            "mean_semantic_similarity": round(stats["semantic_sum"] / n, 6),
        }
    if missing_phenomena:
        missing_str = ", ".join(missing_phenomena)
        raise SystemExit(f"Missing diagnostics rows for required phenomena: {missing_str}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("harvest")
    p_h.add_argument("--inputs", nargs="+", type=Path, required=True)
    p_h.add_argument("--out", type=Path, required=True)
    p_h.add_argument("--min-chars", type=int, default=20)
    p_h.add_argument("--dedupe-batch-size", type=int, default=0)

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

    p_m = sub.add_parser("monitor-lp7")
    p_m.add_argument("--pre", type=float, required=True)
    p_m.add_argument("--post", type=float, required=True)
    p_m.add_argument("--release-gates", type=Path, default=Path("project/release_gates.yaml"))

    p_d = sub.add_parser("diagnose-semantic")
    p_d.add_argument("--in-csv", type=Path, required=True)
    p_d.add_argument("--out-json", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "harvest":
        print(harvest(args.inputs, args.out, args.min_chars, args.dedupe_batch_size))
    elif args.cmd == "curate":
        print(curate(args.inp, args.out, args.min_fr_ca_score))
    elif args.cmd == "edit":
        print(edit_normative(args.inp, args.out))
    elif args.cmd == "split":
        print(json.dumps(split_train_dev_test(args.inp, args.out_dir, args.seed)))
    elif args.cmd == "recipe":
        write_training_recipe(args.data_dir, args.out)
        print(args.out)
    elif args.cmd == "monitor-lp7":
        print(json.dumps(monitor_lp7(args.pre, args.post, args.release_gates), ensure_ascii=False))
    elif args.cmd == "diagnose-semantic":
        print(json.dumps(lp_semantic_diagnostics(args.in_csv, args.out_json), ensure_ascii=False))


if __name__ == "__main__":
    main()
