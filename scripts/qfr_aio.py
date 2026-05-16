#!/usr/bin/env python3
"""Interactive all-in-one QFR pipeline runner.

Production-oriented terminal console with saved defaults, ASCII dashboard,
individual task execution, full pipeline execution, live meters, verbose logs,
and final status summaries.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = ROOT / ".qfr_aio_defaults.json"
LOG_DIR = ROOT / "reports" / "aio"

ASCII = r"""
   ____  ______ ____        ___     ___   ____
  / __ \/ ____// __ \      /   |   /  /  / __ \
 / / / / /_   / /_/ /_____/ /| |  / /   / / / /
/ /_/ / __/  / _, _/_____/ ___ | / /   / /_/ /
\___\_\_/    /_/ |_|     /_/  |_|/_/    \____/
     Québec French AI Sovereignty Pipeline Console
"""

FULL_PIPELINE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

DEFAULTS = {
    "python": "python3",
    "download_manifest": (
        "manifests/corpus_source_manifest.real_downloads.yaml"
    ),
    "downloaded_manifest": (
        "manifests/corpus_source_manifest.real_downloaded.local.yaml"
    ),
    "download_report": "reports/corpus_ingestion/downloads.real.json",
    "external_catalog": "project/external_dataset_catalog.yaml",
    "external_report": "reports/external_datasets/materialization_report.json",
    "external_mode": "all-safe",
    "external_min_chars": "80",
    "external_max_records": "",
    "external_include_unreviewed_license": "false",
    "external_include_review_required": "false",
    "real_harvest": "reports/corpus_ingestion/harvest.real.jsonl",
    "real_ingestion_report": "reports/corpus_ingestion/report.real.json",
    "real_min_chars": "80",
    "curation_policy": "manifests/curation_policy_manifest.template.yaml",
    "real_curation_dir": "reports/corpus_curation_real",
    "split_policy": "manifests/split_policy_manifest.template.yaml",
    "split_input": "reports/corpus_curation_real/accepted.jsonl",
    "split_out_dir": "reports/curated_splits",
    "training_export_manifest": (
        "manifests/training_export_manifest.template.yaml"
    ),
    "training_export_dir": "reports/training_export",
    "metrics": "fixtures/valid_metrics.json",
    "diagnostics_input": "fixtures/diagnostics/lp9_lp20_eval_sample.jsonl",
    "release_candidate_json": "reports/release_candidate.json",
    "release_candidate_md": "reports/release_candidate.md",
    "dolphin_base_model": "dphn/Dolphin3.0-Qwen2.5-3b",
    "dolphin_train": "reports/curated_splits/train.jsonl",
    "dolphin_eval": "reports/curated_splits/test.jsonl",
    "dolphin_output_dir": "models/qfr-dolphin3-qwen25-3b-lora-smoke",
    "dolphin_max_steps": "10",
    "dolphin_quantization": "q4_k_m",
}


def load_defaults() -> dict[str, str]:
    """Load saved defaults, falling back to built-in defaults."""
    if DEFAULTS_PATH.exists():
        data = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    return dict(DEFAULTS)


def save_defaults(defaults: dict[str, str]) -> None:
    """Persist interactive defaults to disk."""
    payload = json.dumps(defaults, indent=2, ensure_ascii=False) + "\n"
    DEFAULTS_PATH.write_text(payload, encoding="utf-8")


def truthy(value: Any) -> bool:
    """Parse common truthy string values."""
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def line_count(path: Path) -> int | None:
    """Count lines in a file, or return None for non-files."""
    if not path.exists() or not path.is_file():
        return None
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))


def meter_bar(elapsed: float) -> str:
    """Render a lightweight animated elapsed-time meter."""
    width = 28
    pos = int(elapsed * 8) % width
    fill = "".join(
        "█" if i == pos else "▓" if i < pos else "░" for i in range(width)
    )
    return "[" + fill + "]"


def run(name: str, cmd: list[str], log_path: Path) -> dict[str, Any]:
    """Execute one command with streaming log + live terminal progress."""
    print(f"\n==> {name}")
    print("$ " + " ".join(cmd))
    start = time.monotonic()
    last_tick = start
    spinner = "|/-\\"
    spin = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {name} =====\n")
        start_time = datetime.now().isoformat(timespec="seconds")
        log.write("START " + start_time + "\n")
        log.write("CMD " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            now = time.monotonic()
            if now - last_tick > 0.15:
                spin += 1
                preview = raw.strip().replace("\t", " ")[:100]
                meter = meter_bar(now - start)
                elapsed_txt = f"{now - start:7.1f}s"
                print(
                    f"\r{spinner[spin % len(spinner)]} {meter} "
                    f"elapsed={elapsed_txt}  {preview:100}",
                    end="",
                )
                last_tick = now
        rc = proc.wait()
        elapsed = time.monotonic() - start
        log.write(f"END rc={rc} elapsed={elapsed:.2f}s\n")
    status = "✅" if rc == 0 else "❌"
    print(f"\r{status} {name} rc={rc} elapsed={elapsed:.1f}s" + " " * 40)
    return {"name": name, "rc": rc, "elapsed": round(elapsed, 2), "cmd": cmd}


def external_materializer_command(defaults: dict[str, str]) -> list[str]:
    """Build CLI args for external dataset materialization."""
    p = defaults["python"]
    cmd = [
        p,
        "scripts/materialize_external_datasets.py",
        "--catalog",
        defaults["external_catalog"],
        "--report",
        defaults["external_report"],
        "--mode",
        defaults["external_mode"],
        "--min-chars",
        defaults["external_min_chars"],
    ]
    if str(defaults.get("external_max_records", "")).strip():
        cmd += ["--max-records", str(defaults["external_max_records"])]
    if truthy(defaults.get("external_include_unreviewed_license")):
        cmd.append("--include-unreviewed-license")
    if truthy(defaults.get("external_include_review_required")):
        cmd.append("--include-review-required")
    return cmd


def commands(
    defaults: dict[str, str],
) -> dict[str, tuple[str, list[list[str]]]]:
    """Build the interactive menu command map."""
    p = defaults["python"]
    return {
        "1": ("Bootstrap", [["bash", "scripts/bootstrap_dev_env.sh"]]),
        "2": (
            "Validate",
            [
                [p, "tools/update_agents.py", "--ensure-nested"],
                [p, "tools/update_agents.py", "--validate"],
                ["qfr", "validate"],
            ],
        ),
        "3": (
            "Download Gutenberg real corpus",
            [
                [
                    p,
                    "tools/download_real_corpus_sources.py",
                    "--manifest",
                    defaults["download_manifest"],
                    "--out-manifest",
                    defaults["downloaded_manifest"],
                    "--report",
                    defaults["download_report"],
                    "--overwrite",
                ]
            ],
        ),
        "4": (
            "Materialize external dataset catalog",
            [external_materializer_command(defaults)],
        ),
        "5": (
            "Ingest real corpus",
            [
                [
                    "qfr",
                    "validate-corpus-sources",
                    "--manifest",
                    defaults["downloaded_manifest"],
                ],
                [
                    "qfr",
                    "ingest-corpus-sources",
                    "--manifest",
                    defaults["downloaded_manifest"],
                    "--out",
                    defaults["real_harvest"],
                    "--report",
                    defaults["real_ingestion_report"],
                    "--min-chars",
                    defaults["real_min_chars"],
                ],
            ],
        ),
        "6": (
            "Curate real corpus",
            [
                [
                    "qfr",
                    "curate-corpus",
                    "--input",
                    defaults["real_harvest"],
                    "--policy",
                    defaults["curation_policy"],
                    "--out-dir",
                    defaults["real_curation_dir"],
                ]
            ],
        ),
        "7": (
            "Split accepted corpus",
            [
                [
                    "qfr",
                    "validate-split-policy",
                    "--policy",
                    defaults["split_policy"],
                ],
                [
                    "qfr",
                    "split-curated-corpus",
                    "--input",
                    defaults["split_input"],
                    "--policy",
                    defaults["split_policy"],
                    "--out-dir",
                    defaults["split_out_dir"],
                ],
            ],
        ),
        "8": (
            "Export training dataset",
            [
                [
                    "qfr",
                    "validate-training-export",
                    "--manifest",
                    defaults["training_export_manifest"],
                ],
                [
                    "qfr",
                    "export-training-dataset",
                    "--manifest",
                    defaults["training_export_manifest"],
                    "--out-dir",
                    defaults["training_export_dir"],
                ],
            ],
        ),
        "9": (
            "Release candidate",
            [
                [
                    "qfr",
                    "release-candidate",
                    "--metrics",
                    defaults["metrics"],
                    "--diagnostics-input",
                    defaults["diagnostics_input"],
                    "--out-json",
                    defaults["release_candidate_json"],
                    "--out-md",
                    defaults["release_candidate_md"],
                ]
            ],
        ),
        "10": (
            "Dolphin Unsloth smoke train",
            [
                [
                    p,
                    "scripts/train_qfr_dolphin3_unsloth_lora.py",
                    "--base-model",
                    defaults["dolphin_base_model"],
                    "--train",
                    defaults["dolphin_train"],
                    "--eval",
                    defaults["dolphin_eval"],
                    "--output-dir",
                    defaults["dolphin_output_dir"],
                    "--max-steps",
                    defaults["dolphin_max_steps"],
                ]
            ],
        ),
    }


def _status_keys() -> list[str]:
    """Return tracked artifact keys for status output."""
    return [
        "real_harvest",
        "real_ingestion_report",
        "external_report",
        "real_curation_dir",
        "split_out_dir",
        "training_export_dir",
        "release_candidate_json",
        "dolphin_output_dir",
    ]


def _status_extra(path: Path) -> str:
    """Collect compact size details for known artifact file patterns."""
    extra = ""
    if path.is_file() and path.suffix == ".jsonl":
        extra = f" lines={line_count(path)}"
    if path.is_dir():
        for child in ["accepted.jsonl", "train.jsonl", "test.jsonl"]:
            child_path = path / child
            if child_path.exists():
                extra += f" {child}={line_count(child_path)}"
    return extra


def _relative_if_possible(path: Path) -> Path:
    """Return a root-relative path when possible for cleaner status output."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def show_status(defaults: dict[str, str]) -> None:
    """Print existence and lightweight counts for key pipeline artifacts."""
    print("\nArtifact status")
    for key in _status_keys():
        path = ROOT / str(defaults[key])
        extra = _status_extra(path)
        shown_path = _relative_if_possible(path)
        exists_label = "yes" if path.exists() else "no "
        print(f"- {key:28} {exists_label:3} {shown_path}{extra}")


def edit_defaults(defaults: dict[str, str]) -> None:
    """Interactively edit persisted defaults."""
    keys = sorted(defaults)
    while True:
        for i, key in enumerate(keys, 1):
            print(f"{i:2}. {key} = {defaults[key]}")
        choice = input("Edit number, s=save, Enter=back: ").strip()
        if choice == "":
            return
        if choice.lower() == "s":
            save_defaults(defaults)
            print(f"Saved {DEFAULTS_PATH.relative_to(ROOT)}")
            return
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            key = keys[int(choice) - 1]
            val = input(f"{key} [{defaults[key]}]: ").strip()
            if val:
                defaults[key] = val


def run_group(
    label: str, group: list[tuple[str, list[str]]], log_path: Path
) -> list[dict[str, Any]]:
    """Run a command group and stop at first non-zero return code."""
    print(f"\n### {label}")
    results = []
    for name, cmd in group:
        result = run(name, cmd, log_path)
        results.append(result)
        if result["rc"] != 0:
            break
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for qfr_aio interactive/runtime modes."""
    parser = argparse.ArgumentParser(
        description="Interactive QFR all-in-one console."
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--full-with-smoke-training", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser


def _full_order(include_smoke_training: bool) -> list[str]:
    """Return the non-interactive full-run command order."""
    order = list(FULL_PIPELINE_ORDER)
    if include_smoke_training:
        order.append("10")
    return order


def _run_sequence(
    *,
    defaults: dict[str, str],
    log_path: Path,
    all_results: list[dict[str, Any]],
    order: list[str],
) -> None:
    """Run ordered menu stages and append results in-place."""
    menu = commands(defaults)
    for item in order:
        label, cmds = menu[item]
        results = run_group(label, [(label, cmd) for cmd in cmds], log_path)
        all_results.extend(results)
        if results and results[-1]["rc"] != 0:
            break


def _print_menu(menu: dict[str, tuple[str, list[list[str]]]]) -> None:
    """Render the interactive top-level menu."""
    print(ASCII)
    for key, (label, _) in menu.items():
        print(f"{key}. {label}")
    print("11. FULL data pipeline")
    print("12. FULL + Dolphin smoke train")
    print("13. Status")
    print("14. Edit saved defaults")
    print("0. Exit")


def _print_completion(
    all_results: list[dict[str, Any]],
    log_path: Path,
) -> None:
    """Print condensed pass/fail summary for accumulated runs."""
    print("\nCompletion status")
    for result in all_results:
        state = "OK" if result["rc"] == 0 else "FAIL"
        print(f"- {state} {result['name']} {result['elapsed']}s")
    print(f"Log: {log_path.relative_to(ROOT)}")


def _handle_shortcut_choice(choice: str, defaults: dict[str, str]) -> str:
    """Handle choices that do not execute pipeline commands."""
    if choice == "0":
        return "exit"
    if choice == "13":
        show_status(defaults)
        input("Enter to continue...")
        return "continue"
    if choice == "14":
        edit_defaults(defaults)
        return "continue"
    return "unhandled"


def _run_single_menu_item(
    *,
    choice: str,
    menu: dict[str, tuple[str, list[list[str]]]],
    log_path: Path,
    all_results: list[dict[str, Any]],
) -> None:
    """Execute one specific command menu item."""
    label, cmds = menu[choice]
    runs = run_group(label, [(label, cmd) for cmd in cmds], log_path)
    all_results.extend(runs)


def _run_full_choice(
    *,
    choice: str,
    defaults: dict[str, str],
    log_path: Path,
    all_results: list[dict[str, Any]],
) -> None:
    """Execute FULL pipeline choices from interactive mode."""
    order = _full_order(include_smoke_training=choice == "12")
    _run_sequence(
        defaults=defaults,
        log_path=log_path,
        all_results=all_results,
        order=order,
    )


def _interactive_loop(
    *,
    defaults: dict[str, str],
    log_path: Path,
    all_results: list[dict[str, Any]],
) -> None:
    """Run interactive menu loop until user exits."""
    while True:
        menu = commands(defaults)
        _print_menu(menu)
        choice = input("Select: ").strip()

        shortcut = _handle_shortcut_choice(choice, defaults)
        if shortcut == "exit":
            break
        if shortcut == "continue":
            continue

        if choice in menu:
            _run_single_menu_item(
                choice=choice,
                menu=menu,
                log_path=log_path,
                all_results=all_results,
            )
        elif choice in {"11", "12"}:
            _run_full_choice(
                choice=choice,
                defaults=defaults,
                log_path=log_path,
                all_results=all_results,
            )

        _print_completion(all_results, log_path)
        input("Enter to continue...")


def _final_exit_code(all_results: list[dict[str, Any]]) -> int:
    """Return process exit code from accumulated command results."""
    return 0 if all(item["rc"] == 0 for item in all_results) else 1


def main() -> int:
    """Entry point for interactive and full-run workflows."""
    parser = _build_parser()
    args = parser.parse_args()

    defaults = load_defaults()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"qfr_aio_{ts}.log"
    all_results: list[dict[str, Any]] = []

    if args.status:
        show_status(defaults)
        return 0

    if args.full or args.full_with_smoke_training:
        order = _full_order(
            include_smoke_training=args.full_with_smoke_training
        )
        _run_sequence(
            defaults=defaults,
            log_path=log_path,
            all_results=all_results,
            order=order,
        )
        return _final_exit_code(all_results)

    _interactive_loop(
        defaults=defaults,
        log_path=log_path,
        all_results=all_results,
    )
    return _final_exit_code(all_results)


if __name__ == "__main__":
    raise SystemExit(main())
