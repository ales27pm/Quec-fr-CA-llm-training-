#!/usr/bin/env python3
"""Update AGENTS dynamic state, scaffold nested AGENTS, and run governance validations."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "project" / "status.json"
AGENTS_PATH = ROOT / "AGENTS.md"
EVAL_PATH = ROOT / "eval" / "evaluation_manifest.template.yaml"
RELEASE_GATES_PATH = ROOT / "project" / "release_gates.yaml"
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
    for task in data.get("tasks", []):
        tid = task.get("id")
        path = TASK_FILES.get(tid)
        new_status = "fully_implemented" if (path and path.exists()) else "not_implemented"
        task["status"] = new_status
        task["evidence"] = "Provided" if new_status == "fully_implemented" else "Pending"
    data["last_updated"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    STATUS_PATH.write_text(json.dumps(data, indent=2) + "\n")
    return data
def render_dynamic(status_doc: dict) -> str:
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d UTC")
    lines = [
        START,
        f"> Last auto-update: **{date}**  ",
        "> Source: `project/status.json` (managed by `python3 tools/update_agents.py --write`)",
        "",
        "| Task ID | Goal | Status | Evidence |",
        "|---|---|---|---|",
    ]
    tasks = status_doc.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id", "unknown"))
        goal = str(task.get("goal", "unknown"))
        status = str(task.get("status", "unknown"))
        evidence = task.get("evidence", "unknown")
        if isinstance(evidence, list):
            evidence = ", ".join(str(x) for x in evidence)
        lines.append(f"| {tid} | {goal} | {icon(status)} | `{evidence}` |")
    lines.extend(["", ""])
    return "\n".join(lines)
def write_root_agents() -> None:
    try:
        status_doc = sync_status()
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to load/sync status JSON at {STATUS_PATH}: {exc}") from exc
    try:
        original_text = AGENTS_PATH.read_text()
    except OSError as exc:
        raise RuntimeError(f"Failed to read AGENTS file {AGENTS_PATH}: {exc}") from exc
    lines = original_text.splitlines(keepends=True)
    start_idx = next((i for i, line in enumerate(lines) if line.strip() == START), None)
    end_idx = next((i for i, line in enumerate(lines) if line.strip() == END), None)
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise RuntimeError("AGENTS.md markers missing or out of order; cannot safely update dynamic section.")

    replacement = render_dynamic(status_doc).splitlines(keepends=True)
    if replacement and not replacement[-1].endswith("\n"):
        replacement[-1] += "\n"
    new_lines = lines[:start_idx] + replacement + lines[end_idx:]
    new_text = "".join(new_lines)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = AGENTS_PATH.with_name(f"AGENTS.md.bak.{ts}")
    try:
        shutil.copy2(AGENTS_PATH, backup_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=AGENTS_PATH.parent, delete=False) as tmp:
            tmp.write(new_text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(AGENTS_PATH)
    except OSError as exc:
        raise RuntimeError(f"Failed atomic AGENTS update (backup: {backup_path}): {exc}") from exc
def init_nested(path_str: str) -> Path:
    root_resolved = ROOT.resolve()
    resolved_target = (ROOT / path_str).resolve()
    if os.path.commonpath([str(root_resolved), str(resolved_target)]) != str(root_resolved):
        raise RuntimeError(f"Refusing to create nested AGENTS outside repository root: {resolved_target}")
    target_dir = resolved_target
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create directory {target_dir}: {exc}") from exc
    nested = target_dir / "AGENTS.md"
    if not nested.exists():
        try:
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
        except OSError as exc:
            raise RuntimeError(f"Failed to write nested AGENTS file {nested}: {exc}") from exc
    return nested
def _extract_gate_values(yaml_text: str) -> dict[str, str]:
    patterns = {
        "asr_wer_max": r"^\s*wer_max:\s*([0-9.]+)\s*$",
        "overall_lp_accuracy_min": r"^\s*overall_accuracy_min:\s*([0-9.]+)\s*$",
        "lp9_min": r"^\s*lp9_lexical_semantics_min:\s*([0-9.]+)\s*$",
        "lp20_min": r"^\s*lp20_orphaned_preposition_min:\s*([0-9.]+)\s*$",
        "lp7_drop_ratio_max": r"^\s*lp7_standard_negation_max_post_alignment_drop_ratio:\s*([0-9.]+)\s*$",
    }
    out: dict[str, str] = {}
    import re
    for key, pattern in patterns.items():
        m = re.search(pattern, yaml_text, flags=re.MULTILINE)
        if not m:
            raise SystemExit(f"Validation failed; missing `{key}` in {RELEASE_GATES_PATH}.")
        out[key] = m.group(1)
    return out


def validate() -> None:
    eval_text = EVAL_PATH.read_text()
    gates_text = RELEASE_GATES_PATH.read_text()
    gate_values = _extract_gate_values(gates_text)

    required_benchmarks = ["qfrblimp", "multiblimp_fr", "qfrcore_eval", "qfrcort_eval"]
    missing_benchmarks = [name for name in required_benchmarks if name not in eval_text]
    if missing_benchmarks:
        raise SystemExit(f"Validation failed; missing required holdout benchmark markers: {missing_benchmarks}")

    required_eval_markers = [
        f"overall_lp_accuracy_min: {gate_values['overall_lp_accuracy_min']}",
        f'"9": {gate_values["lp9_min"]}',
        f'"20": {gate_values["lp20_min"]}',
    ]
    missing_eval_markers = [m for m in required_eval_markers if m not in eval_text]
    if missing_eval_markers:
        raise SystemExit(
            "Validation failed; evaluation manifest thresholds are out of sync with project/release_gates.yaml: "
            f"{missing_eval_markers}"
        )

    print("Validation passed: contamination holdouts present and release gates synchronized to project/release_gates.yaml.")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--sync-status", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--init-nested", metavar="PATH")
    args = parser.parse_args()
    if args.sync_status:
        sync_status()
        print("Status synced.")
    if args.write:
        write_root_agents()
        print("Updated root AGENTS.md dynamic section.")
    if args.validate:
        validate()
    if args.init_nested:
        print(f"Nested AGENTS ready at: {init_nested(args.init_nested)}")
    if not any([args.sync_status, args.write, args.validate, args.init_nested]):
        parser.print_help()
if __name__ == "__main__":
    main()
