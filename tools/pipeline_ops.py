#!/usr/bin/env python3
"""Legacy compatibility wrappers for data-pipeline and diagnostics commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qfr_pipeline.data_pipeline import curate, edit_normative, harvest, split_train_dev_test, write_training_recipe
from qfr_pipeline.lp7_monitor import monitor_lp7


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
    p_d.add_argument("--taxonomy", type=Path, action="append", default=None)
    p_d.add_argument("--allow-missing-phenomena", action="store_true")

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
        from qfr_pipeline.legacy_diagnostics import run_legacy_semantic_diagnostics

        payload = run_legacy_semantic_diagnostics(
            args.in_csv,
            args.out_json,
            taxonomy_paths=args.taxonomy,
            allow_missing_phenomena=args.allow_missing_phenomena,
        )
        print(json.dumps(payload, ensure_ascii=False))
        if not payload.get("ok", False):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
