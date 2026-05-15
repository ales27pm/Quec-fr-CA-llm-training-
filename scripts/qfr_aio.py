#!/usr/bin/env python3
"""Interactive all-in-one QFR pipeline runner."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

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

DEFAULTS = {
    "python": "python3",
    "download_manifest": "manifests/corpus_source_manifest.real_downloads.yaml",
    "downloaded_manifest": "manifests/corpus_source_manifest.real_downloaded.local.yaml",
    "download_report": "reports/corpus_ingestion/downloads.real.json",
    "real_harvest": "reports/corpus_ingestion/harvest.real.jsonl",
    "real_ingestion_report": "reports/corpus_ingestion/report.real.json",
    "real_min_chars": "80",
    "curation_policy": "manifests/curation_policy_manifest.template.yaml",
    "real_curation_dir": "reports/corpus_curation_real",
    "split_policy": "manifests/split_policy_manifest.template.yaml",
    "split_input": "reports/corpus_curation_real/accepted.jsonl",
    "split_out_dir": "reports/curated_splits",
    "training_export_manifest": "manifests/training_export_manifest.template.yaml",
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


def load_defaults():
    if DEFAULTS_PATH.exists():
        data = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    return dict(DEFAULTS)


def save_defaults(defaults):
    DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def line_count(path: Path):
    if not path.exists() or not path.is_file():
        return None
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))


def run(name, cmd, log_path):
    print(f"\n==> {name}")
    print("$ " + " ".join(cmd))
    start = time.monotonic()
    last_tick = start
    spinner = "|/-\\"
    spin = 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {name} =====\n")
        log.write("CMD " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            now = time.monotonic()
            if now - last_tick > 0.15:
                spin += 1
                print(f"\r{spinner[spin % len(spinner)]} elapsed={now-start:7.1f}s  {raw.strip()[:100]:100}", end="")
                last_tick = now
        rc = proc.wait()
        elapsed = time.monotonic() - start
        log.write(f"END rc={rc} elapsed={elapsed:.2f}s\n")
    print(f"\r{'✅' if rc == 0 else '❌'} {name} rc={rc} elapsed={elapsed:.1f}s" + " " * 20)
    return {"name": name, "rc": rc, "elapsed": round(elapsed, 2), "cmd": cmd}


def commands(defaults):
    p = defaults["python"]
    return {
        "1": ("Bootstrap", [["bash", "scripts/bootstrap_dev_env.sh"]]),
        "2": ("Validate", [[p, "tools/update_agents.py", "--ensure-nested"], [p, "tools/update_agents.py", "--validate"], ["qfr", "validate"]]),
        "3": ("Download real corpus", [[p, "tools/download_real_corpus_sources.py", "--manifest", defaults["download_manifest"], "--out-manifest", defaults["downloaded_manifest"], "--report", defaults["download_report"], "--overwrite"]]),
        "4": ("Ingest real corpus", [["qfr", "validate-corpus-sources", "--manifest", defaults["downloaded_manifest"]], ["qfr", "ingest-corpus-sources", "--manifest", defaults["downloaded_manifest"], "--out", defaults["real_harvest"], "--report", defaults["real_ingestion_report"], "--min-chars", defaults["real_min_chars"]]]),
        "5": ("Curate real corpus", [["qfr", "curate-corpus", "--input", defaults["real_harvest"], "--policy", defaults["curation_policy"], "--out-dir", defaults["real_curation_dir"]]]),
        "6": ("Split accepted corpus", [["qfr", "validate-split-policy", "--policy", defaults["split_policy"]], ["qfr", "split-curated-corpus", "--input", defaults["split_input"], "--policy", defaults["split_policy"], "--out-dir", defaults["split_out_dir"]]]),
        "7": ("Export training dataset", [["qfr", "validate-training-export", "--manifest", defaults["training_export_manifest"]], ["qfr", "export-training-dataset", "--manifest", defaults["training_export_manifest"], "--out-dir", defaults["training_export_dir"]]]),
        "8": ("Release candidate", [["qfr", "release-candidate", "--metrics", defaults["metrics"], "--diagnostics-input", defaults["diagnostics_input"], "--out-json", defaults["release_candidate_json"], "--out-md", defaults["release_candidate_md"]]]),
        "9": ("Dolphin Unsloth smoke train", [[p, "scripts/train_qfr_dolphin3_unsloth_lora.py", "--base-model", defaults["dolphin_base_model"], "--train", defaults["dolphin_train"], "--eval", defaults["dolphin_eval"], "--output-dir", defaults["dolphin_output_dir"], "--max-steps", defaults["dolphin_max_steps"]]]),
    }


def show_status(defaults):
    print("\nArtifact status")
    for key in ["real_harvest", "real_ingestion_report", "real_curation_dir", "split_out_dir", "training_export_dir", "release_candidate_json", "dolphin_output_dir"]:
        path = ROOT / defaults[key]
        extra = ""
        if path.is_file() and path.suffix == ".jsonl":
            extra = f" lines={line_count(path)}"
        if path.is_dir():
            for child in ["accepted.jsonl", "train.jsonl", "test.jsonl"]:
                p = path / child
                if p.exists():
                    extra += f" {child}={line_count(p)}"
        print(f"- {key:28} {'yes' if path.exists() else 'no ':3} {path.relative_to(ROOT)}{extra}")


def edit_defaults(defaults):
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


def run_group(label, group, log_path):
    print(f"\n### {label}")
    results = []
    for name, cmd in group:
        result = run(name, cmd, log_path)
        results.append(result)
        if result["rc"] != 0:
            break
    return results


def main():
    defaults = load_defaults()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"qfr_aio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    all_results = []
    while True:
        print(ASCII)
        menu = commands(defaults)
        for key, (label, _) in menu.items():
            print(f"{key}. {label}")
        print("10. FULL data pipeline")
        print("11. FULL + Dolphin smoke train")
        print("12. Status")
        print("13. Edit saved defaults")
        print("0. Exit")
        choice = input("Select: ").strip()
        if choice == "0":
            break
        if choice == "12":
            show_status(defaults)
            input("Enter to continue...")
            continue
        if choice == "13":
            edit_defaults(defaults)
            continue
        if choice in menu:
            label, cmds = menu[choice]
            all_results += run_group(label, [(label, c) for c in cmds], log_path)
        elif choice in {"10", "11"}:
            order = ["1", "2", "3", "4", "5", "6", "7", "8"]
            if choice == "11":
                order.append("9")
            for item in order:
                label, cmds = menu[item]
                results = run_group(label, [(label, c) for c in cmds], log_path)
                all_results += results
                if results and results[-1]["rc"] != 0:
                    break
        print("\nCompletion status")
        for r in all_results:
            print(f"- {'OK' if r['rc'] == 0 else 'FAIL'} {r['name']} {r['elapsed']}s")
        print(f"Log: {log_path.relative_to(ROOT)}")
        input("Enter to continue...")
    return 0 if all(r["rc"] == 0 for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
