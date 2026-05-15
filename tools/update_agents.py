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

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "project" / "status.json"
AGENTS_PATH = ROOT / "AGENTS.md"
EVAL_PATH = ROOT / "eval" / "evaluation_manifest.template.yaml"
RELEASE_GATES_PATH = ROOT / "project" / "release_gates.yaml"
START = "## Dynamic State (auto-generated)"
END = "## Non-Negotiable Rules (Drift Prevention)"
REQUIRED_NESTED_DIRS = ("manifests", "rules", "eval", "src", "project")
TASK_FILES = {
    "T1": ROOT / "manifests" / "dataset_manifest.template.yaml",
    "T2": ROOT / "rules" / "lp_rule_manifest.template.yaml",
    "T3": ROOT / "eval" / "evaluation_manifest.template.yaml",
    "T4": ROOT / "tools" / "minimal_pair_generator_spec.md",
    "T5": ROOT / "tools" / "update_agents.py",
    "T6": ROOT / "manifests" / "AGENTS.md",
    "T7": ROOT / "src" / "qfr_pipeline" / "validation.py",
    "T8": ROOT / "src" / "qfr_pipeline" / "contamination.py",
    "T9": ROOT / "src" / "qfr_pipeline" / "minimal_pairs.py",
    "T10": ROOT / "src" / "qfr_pipeline" / "release_report.py",
    "T11": ROOT / ".github" / "workflows" / "ci.yml",
    "T12": ROOT / "src" / "qfr_pipeline" / "minimal_pair_quality.py",
    "T13": ROOT / "src" / "qfr_pipeline" / "cli.py",
    "T14": ROOT / "rules" / "lp20_orphaned_preposition.contexts.yaml",
    "T15": ROOT / "src" / "qfr_pipeline" / "diagnostics.py",
    "T16": ROOT / "src" / "qfr_pipeline" / "release_candidate.py",
    "T17": ROOT / "src" / "qfr_pipeline" / "legacy_diagnostics.py",
    "T18": ROOT / "src" / "qfr_pipeline" / "data_pipeline.py",
    "T19": ROOT / "src" / "qfr_pipeline" / "corpus_sources.py",
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
    skip_backup = os.environ.get("QFR_NO_AGENTS_BACKUP") == "1"
    try:
        if not skip_backup:
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
    parsed = yaml.safe_load(yaml_text) or {}
    out = {
        "asr_wer_max": parsed.get("asr", {}).get("wer_max"),
        "overall_lp_accuracy_min": parsed.get("linguistic_phenomena", {}).get("overall_accuracy_min"),
        "lp9_min": parsed.get("linguistic_phenomena", {}).get("lp9_lexical_semantics_min"),
        "lp20_min": parsed.get("linguistic_phenomena", {}).get("lp20_orphaned_preposition_min"),
        "lp7_drop_ratio_max": parsed.get("alignment", {}).get("lp7_standard_negation_max_post_alignment_drop_ratio"),
    }
    missing = [name for name, value in out.items() if value is None]
    if missing:
        raise SystemExit(f"Validation failed; missing gate(s) in {RELEASE_GATES_PATH}: {missing}")
    return {k: str(v) for k, v in out.items()}


def ensure_required_nested_agents() -> list[Path]:
    created: list[Path] = []
    for rel in REQUIRED_NESTED_DIRS:
        nested = init_nested(rel)
        if nested.exists() and nested.stat().st_size > 0:
            # init_nested writes only when missing, so we detect newly created by mtime proximity is brittle;
            # instead rely on existence check before invocation in caller when needed.
            pass
    return created


def validate() -> None:
    if not RELEASE_GATES_PATH.exists():
        raise SystemExit(f"Validation failed; missing release gates file: {RELEASE_GATES_PATH}")

    eval_text = EVAL_PATH.read_text()
    try:
        gates_text = RELEASE_GATES_PATH.read_text()
    except OSError as exc:
        raise SystemExit(f"Validation failed; could not read release gates file {RELEASE_GATES_PATH}: {exc}") from exc
    gate_values = _extract_gate_values(gates_text)
    eval_doc = yaml.safe_load(eval_text) or {}

    required_benchmarks = ["qfrblimp", "multiblimp_fr", "qfrcore_eval", "qfrcort_eval"]
    benchmark_sets = eval_doc.get("benchmark_sets", [])
    holdouts = [item.get("name") for item in benchmark_sets if isinstance(item, dict)]
    missing_benchmarks = [name for name in required_benchmarks if name not in holdouts]
    if missing_benchmarks:
        raise SystemExit(f"Validation failed; missing required holdout benchmark markers: {missing_benchmarks}")

    thresholds = eval_doc.get("release_gates", {})
    lp_floors = thresholds.get("lp_floors", {})
    required_eval_markers = {
        "overall_lp_accuracy_min": str(thresholds.get("overall_lp_accuracy_min")),
        "lp9_min": str(lp_floors.get("9", lp_floors.get(9))),
        "lp20_min": str(lp_floors.get("20", lp_floors.get(20))),
    }
    mismatched_thresholds = [
        key for key in required_eval_markers if required_eval_markers[key] != gate_values[key]
    ]
    if mismatched_thresholds:
        raise SystemExit(
            "Validation failed; evaluation manifest thresholds are out of sync with project/release_gates.yaml: "
            f"{mismatched_thresholds}"
        )

    try:
        status_doc = json.loads(STATUS_PATH.read_text())
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Validation failed; could not load status file {STATUS_PATH}: {exc}") from exc
    for task in status_doc.get("tasks", []):
        if task.get("status") == "fully_implemented" and task.get("id") in TASK_FILES:
            if not TASK_FILES[task["id"]].exists():
                raise SystemExit(f"Validation failed; task {task['id']} marked fully_implemented but missing required file: {TASK_FILES[task['id']]}")

    print("Validation passed: contamination holdouts present and release gates synchronized to project/release_gates.yaml.")
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--sync-status", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--init-nested", metavar="PATH")
    parser.add_argument("--ensure-nested", action="store_true")
    args = parser.parse_args()
    if args.sync_status:
        sync_status()
        print("Status synced.")
    if args.write:
        write_root_agents()
        print("Updated root AGENTS.md dynamic section.")
    if args.validate:
        validate()
    if args.ensure_nested:
        ensured = [init_nested(path) for path in REQUIRED_NESTED_DIRS]
        print("Ensured nested AGENTS: " + ", ".join(str(p.relative_to(ROOT)) for p in ensured))
    if args.init_nested:
        print(f"Nested AGENTS ready at: {init_nested(args.init_nested)}")
    if not any([args.sync_status, args.write, args.validate, args.init_nested, args.ensure_nested]):
        parser.print_help()
if __name__ == "__main__":
    main()
