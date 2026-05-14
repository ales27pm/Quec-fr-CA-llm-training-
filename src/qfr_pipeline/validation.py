from dataclasses import dataclass, field
from pathlib import Path

from qfr_pipeline.io import load_yaml
from qfr_pipeline.schemas import DatasetManifest, EvaluationManifest, LPContextManifest, LPRuleManifest, ReleaseGates, ensure_eval_gates_sync


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
    return model.model_validate(load_yaml(path))


def validate_release_gates(path: Path) -> ReleaseGates: return _validate(path, ReleaseGates)
def validate_dataset_manifest(path: Path) -> DatasetManifest: return _validate(path, DatasetManifest)
def validate_lp_rule_manifest(path: Path) -> LPRuleManifest: return _validate(path, LPRuleManifest)
def validate_lp_context_manifest(path: Path) -> LPContextManifest: return _validate(path, LPContextManifest)


def validate_evaluation_manifest(path: Path, release_gates_path: Path) -> EvaluationManifest:
    eval_manifest = _validate(path, EvaluationManifest)
    gates = validate_release_gates(release_gates_path)
    ensure_eval_gates_sync(eval_manifest, gates)
    return eval_manifest


def validate_repository(root: Path) -> ValidationReport:
    report = ValidationReport(ok=True)
    gates_path = root / "project" / "release_gates.yaml"
    for path, fn, kind in [
        (gates_path, validate_release_gates, "release_gates"),
    ]:
        try:
            fn(path)
            report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), kind, str(exc)))

    for path in sorted((root / "manifests").glob("*.y*ml")):
        try:
            validate_dataset_manifest(path)
            report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), "dataset_manifest", str(exc)))

    for path in sorted((root / "rules").glob("*.y*ml")):
        try:
            doc = load_yaml(path)
            kind = (doc or {}).get("kind") if isinstance(doc, dict) else None
            if kind == "lp_context_manifest" or "contexts" in path.name:
                validate_lp_context_manifest(path)
                report.checked_files.append(str(path))
            else:
                validate_lp_rule_manifest(path)
                report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), "rules_manifest", str(exc)))

    for path in sorted((root / "eval").glob("*.y*ml")):
        try:
            validate_evaluation_manifest(path, gates_path)
            report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), "evaluation_manifest", str(exc)))
    return report
