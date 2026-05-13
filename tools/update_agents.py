#!/usr/bin/env python3
"""Update AGENTS.md dynamic status section and scaffold nested AGENTS.md files."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "project" / "status.json"
AGENTS_PATH = ROOT / "AGENTS.md"
START = "## Dynamic State (auto-generated)"
END = "## Non-Negotiable Rules (Drift Prevention)"


def icon(status: str) -> str:
    return {
        "fully_implemented": "✅ fully_implemented",
        "partially_implemented": "🟡 partially_implemented",
        "not_implemented": "❌ not_implemented",
    }.get(status, status)


def render_dynamic(status_doc: dict) -> str:
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d UTC")
    lines = [
        START,
        f"> Last auto-update: **{date}**  ",
        "> Source: `project/status.json` (managed by `python3 tools/update_agents.py`)",
        "",
        "| Task ID | Goal | Status | Evidence |",
        "|---|---|---|---|",
    ]
    for task in status_doc["tasks"]:
        lines.append(
            f"| {task['id']} | {task['goal']} | {icon(task['status'])} | `{task['evidence']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_root_agents() -> None:
    data = json.loads(STATUS_PATH.read_text())
    text = AGENTS_PATH.read_text()
    if START not in text or END not in text:
        raise RuntimeError("AGENTS.md markers missing; cannot safely update dynamic section.")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    new_text = pre + render_dynamic(data) + END + post
    AGENTS_PATH.write_text(new_text)


def init_nested(path_str: str) -> Path:
    target_dir = (ROOT / path_str).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    nested = target_dir / "AGENTS.md"
    if not nested.exists():
        nested.write_text(
            "# AGENTS.md (Nested)\n\n"
            "## Scope\n"
            f"Applies to `{target_dir.relative_to(ROOT)}/`.\n\n"
            "## Mission\n"
            "Implement subtree-specific changes while respecting root AGENTS.md safety constraints.\n\n"
            "## Local Rules\n"
            "1. Keep files schema-consistent and versioned.\n"
            "2. Do not introduce benchmark contamination.\n"
            "3. Update relevant docs when contract changes.\n\n"
            "## Overrides\n"
            "No additional overrides; root contract remains authoritative for safety/compliance.\n"
            "Parent contract: `../AGENTS.md` (or nearest ancestor).\n"
        )
    return nested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Rewrite AGENTS.md dynamic state section")
    parser.add_argument("--init-nested", metavar="PATH", help="Create nested AGENTS.md in PATH if missing")
    args = parser.parse_args()

    if args.write:
        write_root_agents()
        print("Updated root AGENTS.md dynamic section.")

    if args.init_nested:
        out = init_nested(args.init_nested)
        print(f"Nested AGENTS ready at: {out}")

    if not args.write and not args.init_nested:
        parser.print_help()


if __name__ == "__main__":
    main()
