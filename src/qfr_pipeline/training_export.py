from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import TrainingExportManifest

FORBIDDEN_LABELS = {"rejected", "review_required", "quarantine"}


@dataclass
class TrainingExportReport:
    ok: bool
    manifest: str
    dataset_name: str
    dataset_version: str
    records: dict[str, int]
    hashes: dict[str, str]
    lineage: dict[str, str]
    policy_refs: dict[str, str]
    issues: list[str]
    outputs: dict[str, str]


def _is_repo_relative(path: str) -> bool:
    return not Path(path).is_absolute()


def _assert_dataset_card_text(text: str) -> None:
    banned = ["neutralize", "fr_fr", "fr-fr", "standardize to france french"]
    lowered = text.casefold()
    if any(token in lowered for token in banned):
        raise ValueError("dataset card fields must not recommend dialect neutralization")


def load_training_export_manifest(path: Path) -> TrainingExportManifest:
    return TrainingExportManifest.model_validate(load_yaml(path))


def validate_training_export_manifest(path: Path) -> TrainingExportManifest:
    manifest = load_training_export_manifest(path)
    for p in [
        manifest.inputs.train,
        manifest.inputs.dev,
        manifest.inputs.test,
        manifest.inputs.split_report,
        manifest.inputs.corpus_source_manifest,
        manifest.inputs.curation_policy_manifest,
        manifest.inputs.split_policy_manifest,
        manifest.lineage.source_contract_report,
        manifest.lineage.curation_report,
        manifest.lineage.release_candidate_report,
    ]:
        if not _is_repo_relative(p):
            raise ValueError(f"path must be repo-relative: {p}")
        if not (ROOT / p).exists():
            raise ValueError(f"required input path does not exist: {p}")
    if manifest.quality_requirements.require_accepted_only is not True:
        raise ValueError("require_accepted_only must be true")
    if manifest.quality_requirements.require_no_holdout_material is not True:
        raise ValueError("require_no_holdout_material must be true")
    if "fr-ca" not in manifest.dataset_card.language.casefold() or "québécois" not in manifest.dataset_card.dialect.casefold():
        raise ValueError("dataset card must identify fr-CA / Québécois French")
    for txt in [manifest.description, manifest.intended_use, manifest.dataset_card.provenance_summary, *manifest.dataset_card.limitations, *manifest.dataset_card.ethical_notes]:
        _assert_dataset_card_text(txt)
    for split_path in (manifest.inputs.train, manifest.inputs.dev, manifest.inputs.test):
        if not split_path.startswith("reports/curated_splits/"):
            raise ValueError("train/dev/test must point to curated split files")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_training_dataset(manifest_path: Path, out_dir: Path) -> TrainingExportReport:
    manifest = validate_training_export_manifest(manifest_path)
    issues: list[str] = []
    counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    for split_name, split_rel in (("train", manifest.inputs.train), ("dev", manifest.inputs.dev), ("test", manifest.inputs.test)):
        rows = _read_jsonl(ROOT / split_rel)
        counts[split_name] = len(rows)
        for idx, row in enumerate(rows):
            label = row.get("curation_label")
            if label in FORBIDDEN_LABELS:
                issues.append(f"{split_name}[{idx}] forbidden label: {label}")
            if label != "accepted":
                issues.append(f"{split_name}[{idx}] curation_label must be accepted")
            for req in ("policy_id", "curation_score", "curation_reasons"):
                if req not in row:
                    issues.append(f"{split_name}[{idx}] missing {req}")
        split_hashes[f"{split_name}_sha256"] = _sha256_file(ROOT / split_rel)
    if manifest.quality_requirements.require_nonempty_train and counts["train"] == 0:
        issues.append("train split is empty")
    aggregate_input = json.dumps(
        {
            "dataset_name": manifest.dataset_name,
            "dataset_version": manifest.dataset_version,
            "export_id": manifest.export_id,
            "split_hashes": split_hashes,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    split_hashes["aggregate_sha256"] = hashlib.sha256(aggregate_input).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "training_manifest_json": repo_relative_path(out_dir / "training_manifest.json"),
        "training_manifest_yaml": repo_relative_path(out_dir / "training_manifest.yaml"),
        "dataset_card_md": repo_relative_path(out_dir / "dataset_card.md"),
        "export_report_json": repo_relative_path(out_dir / "export_report.json"),
    }
    report = TrainingExportReport(
        ok=not issues,
        manifest=repo_relative_path(manifest_path),
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        records={"train": counts["train"], "dev": counts["dev"], "test": counts["test"], "total": counts["train"] + counts["dev"] + counts["test"]},
        hashes=split_hashes,
        lineage=manifest.lineage.model_dump(),
        policy_refs={"corpus_source_manifest": manifest.inputs.corpus_source_manifest, "curation_policy_manifest": manifest.inputs.curation_policy_manifest, "split_policy_manifest": manifest.inputs.split_policy_manifest},
        issues=issues,
        outputs=outputs,
    )
    training_manifest_payload = {
        "kind": "training_dataset_manifest",
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "primary_language": manifest.primary_language,
        "records": report.records,
        "hashes": report.hashes,
        "inputs": manifest.inputs.model_dump(),
        "lineage": manifest.lineage.model_dump(),
        "quality_requirements": manifest.quality_requirements.model_dump(),
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(training_manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "training_manifest.yaml").write_text(yaml.safe_dump(training_manifest_payload, sort_keys=True, allow_unicode=True), encoding="utf-8")
    write_dataset_card(report, manifest, out_dir / "dataset_card.md")
    write_training_export_report(report, out_dir / "export_report.json")
    return report


def write_training_export_report(report: TrainingExportReport, path: Path) -> None:
    write_json(path, asdict(report))


def write_dataset_card(report: TrainingExportReport, manifest: TrainingExportManifest, path: Path) -> None:
    lines = [
        f"# Dataset Card — {manifest.dataset_name}",
        "",
        f"- Version: `{manifest.dataset_version}`",
        f"- Language: `{manifest.dataset_card.language}`",
        f"- Dialect: `{manifest.dataset_card.dialect}`",
        f"- Region: `{manifest.dataset_card.region}`",
        f"- Intended use: {manifest.intended_use}",
        "",
        "## Provenance",
        manifest.dataset_card.provenance_summary,
        f"- Source contract report: `{manifest.lineage.source_contract_report}`",
        f"- Curation report: `{manifest.lineage.curation_report}`",
        f"- Release candidate report: `{manifest.lineage.release_candidate_report}`",
        "",
        "## Limitations",
    ]
    lines.extend([f"- {x}" for x in manifest.dataset_card.limitations])
    lines.extend(["", "## Ethical notes"])
    lines.extend([f"- {x}" for x in manifest.dataset_card.ethical_notes])
    lines.extend(["", "## Records", f"- Train: `{report.records['train']}`", f"- Dev: `{report.records['dev']}`", f"- Test: `{report.records['test']}`", f"- Total: `{report.records['total']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
