from dataclasses import dataclass, field
from pathlib import Path

from qfr_pipeline.io import load_yaml
from qfr_pipeline.schemas import (
    DatasetManifest,
    EvaluationManifest,
    LPRuleManifest,
    ReleaseGates,
    ensure_eval_gates_sync,
)


@dataclass
class ValidationIssue:
    path: str
    kind: str
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)


def _validate(path: Path, model):
    doc = load_yaml(path)
    return model.model_validate(doc)


def validate_release_gates(path: Path) -> ReleaseGates:
    return _validate(path, ReleaseGates)


def validate_dataset_manifest(path: Path) -> DatasetManifest:
    return _validate(path, DatasetManifest)


def validate_lp_rule_manifest(path: Path) -> LPRuleManifest:
    return _validate(path, LPRuleManifest)


def validate_evaluation_manifest(path: Path, release_gates_path: Path) -> EvaluationManifest:
    eval_manifest = _validate(path, EvaluationManifest)
    gates = validate_release_gates(release_gates_path)
    try:
        ensure_eval_gates_sync(eval_manifest, gates)
    except ValueError as exc:
        raise ValueError(f"Release gate mismatch between {path} and {release_gates_path}: {exc}") from exc
    return eval_manifest


def validate_repository(root: Path) -> ValidationReport:
    report = ValidationReport(ok=True)
    gates_path = root / "project" / "release_gates.yaml"
    try:
        validate_release_gates(gates_path)
        report.checked_files.append(str(gates_path))
    except Exception as exc:
        report.ok = False
        report.issues.append(ValidationIssue(str(gates_path), "release_gates", str(exc)))

    for folder, fn, kind in [
        ("manifests", validate_dataset_manifest, "dataset_manifest"),
        ("rules", validate_lp_rule_manifest, "lp_rule_manifest"),
    ]:
        for path in sorted((root / folder).glob("*.y*ml")):
            try:
                fn(path)
                report.checked_files.append(str(path))
            except Exception as exc:
                report.ok = False
                report.issues.append(ValidationIssue(str(path), kind, str(exc)))

    for path in sorted((root / "eval").glob("*.y*ml")):
        try:
            validate_evaluation_manifest(path, gates_path)
            report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), "evaluation_manifest", str(exc)))

    return report
