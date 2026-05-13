#!/usr/bin/env python3
"""Update AGENTS dynamic state, scaffold nested AGENTS, and run governance validations."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "project" / "status.json"
AGENTS_PATH = ROOT / "AGENTS.md"
EVAL_PATH = ROOT / "eval" / "evaluation_manifest.template.yaml"
START = "## Dynamic State (auto-generated)"
END = "## Non-Negotiable Rules (Drift Prevention)"

TASK_FILES = {
    "T1": ROOT / "manifests" / "dataset_manifest.template.yaml",
    "T2": ROOT / "rules" / "lp_rule_manifest.template.yaml",
    "T3": ROOT / "eval" / "evaluation_manifest.template.yaml",
    "T4": ROOT / "tools" / "minimal_pair_generator_spec.md",
    "T5": ROOT / "tools" / "update_agents.py",
    "T6": ROOT / "manifests" / "AGENTS.md",
}


def icon(status: str) -> str:
    return {
        "fully_implemented": "✅ fully_implemented",
        "partially_implemented": "🟡 partially_implemented",
        "not_implemented": "❌ not_implemented",
    }.get(status, status)


def sync_status() -> dict:
    data = json.loads(STATUS_PATH.read_text())
    for task in data["tasks"]:
        tid = task["id"]
        path = TASK_FILES.get(tid)
        task["status"] = "fully_implemented" if (path and path.exists()) else "not_implemented"
    data["last_updated"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    STATUS_PATH.write_text(json.dumps(data, indent=2) + "\n")
    return data


def render_dynamic(status_doc: dict) -> str:
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d UTC")
    lines = [
        START,
        f"> Last auto-update: **{date}**  ",
        "> Source: `project/status.json` (managed by `python3 tools/update_agents.py --write`)",
        "",
        "| Task ID | Goal | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for task in status_doc["tasks"]:
        lines.append(f"| {task['id']} | {task['goal']} | {icon(task['status'])} | `{task['evidence']}` |")
    lines.append("")
    return "\n".join(lines)


def write_root_agents() -> None:
    status_doc = sync_status()
    text = AGENTS_PATH.read_text()
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    AGENTS_PATH.write_text(pre + render_dynamic(status_doc) + END + post)


def init_nested(path_str: str) -> Path:
    target_dir = (ROOT / path_str).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    nested = target_dir / "AGENTS.md"
    if not nested.exists():
        nested.write_text(
            "# AGENTS.md (Nested)\n\n"
            "## Scope\n"
            f"Applies to `{target_dir.relative_to(ROOT)}/`.\n\n"
            "## Mission\nImplement subtree changes while respecting root AGENTS.md safeguards.\n\n"
            "## Local Rules\n"
            "1. Maintain schema consistency and versioning.\n"
            "2. Prohibit benchmark contamination and dialect neutralization.\n"
            "3. Record schema changes in root CHANGELOG.md.\n\n"
            "## Overrides\n"
            "No safety overrides; root AGENTS remains authoritative.\n"
            "Parent contract: nearest ancestor `AGENTS.md`.\n"
        )
    return nested


def validate() -> None:
    text = EVAL_PATH.read_text()
    required = ["qfrblimp", "qfrcore_eval", "qfrcort_eval", "overall_lp_accuracy_min: 0.88", '"9": 0.80', '"20": 0.70']
    missing = [r for r in required if r not in text]
    if missing:
        raise SystemExit(f"Validation failed; missing required release-gate/holdout markers: {missing}")
    print("Validation passed: contamination holdouts and release gates present.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--sync-status", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--init-nested", metavar="PATH")
    args = parser.parse_args()

    if args.sync_status:
        sync_status(); print("Status synced.")
    if args.write:
        write_root_agents(); print("Updated root AGENTS.md dynamic section.")
    if args.validate:
        validate()
    if args.init_nested:
        print(f"Nested AGENTS ready at: {init_nested(args.init_nested)}")
    if not any([args.sync_status, args.write, args.validate, args.init_nested]):
        parser.print_help()


if __name__ == "__main__":
    main()
