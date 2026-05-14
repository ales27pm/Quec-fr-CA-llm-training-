import json
from pathlib import Path

from qfr_pipeline.io import load_yaml


def generate_lp9_minimal_pairs(rule_path: Path):
    rule = load_yaml(rule_path)
    if int(rule.get("lp_id", -1)) != 9:
        raise ValueError("Only LP9 generation is supported")
    positives = [p["pattern"] for p in rule.get("positive_patterns", [])]
    negatives = [n["pattern"] for n in rule.get("negative_patterns", [])]
    base = [
        "Merci de m'envoyer le {term} avant la fin de semaine.",
        "Un {term} officiel sera transmis en début de fin de semaine.",
    ]
    out = []
    i = 1
    for pos, neg in zip(positives, negatives):
        for templ in base:
            good = templ.format(term=pos)
            bad = templ.format(term=neg)
            out.append(
                {
                    "id": f"lp9-{i:03d}",
                    "lp_id": 9,
                    "phenomenon": rule.get("name"),
                    "good": good,
                    "bad": bad,
                    "expected": "good",
                    "source_rule": str(rule_path),
                    "metadata": {"exception_safe": True},
                }
            )
            i += 1
    return out


def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
