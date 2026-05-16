from dataclasses import dataclass, field
from pathlib import Path

from qfr_pipeline.io import load_yaml
from qfr_pipeline.schemas import (
    CorpusSourceManifest,
    CurationPolicyManifest,
    DatasetManifest,
    ErrorTaxonomyManifest,
    EvaluationManifest,
    LP9FailureMiningPolicyManifest,
    LP9LexicalPreferencePackManifest,
    LPContextManifest,
    LPRuleManifest,
    ModernCorpusAcquisitionManifest,
    ReleaseGates,
    SplitPolicyManifest,
    TrainingExportManifest,
    TrainingPackPolicyManifest,
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
    return model.model_validate(load_yaml(path))


def validate_release_gates(path: Path) -> ReleaseGates:
    return _validate(path, ReleaseGates)


def validate_dataset_manifest(path: Path) -> DatasetManifest:
    return _validate(path, DatasetManifest)


def validate_lp_rule_manifest(path: Path) -> LPRuleManifest:
    return _validate(path, LPRuleManifest)


def validate_lp_context_manifest(path: Path) -> LPContextManifest:
    return _validate(path, LPContextManifest)


def validate_error_taxonomy_manifest(path: Path) -> ErrorTaxonomyManifest:
    return _validate(path, ErrorTaxonomyManifest)


def validate_corpus_source_manifest(path: Path) -> CorpusSourceManifest:
    return _validate(path, CorpusSourceManifest)


def validate_curation_policy_manifest(path: Path) -> CurationPolicyManifest:
    return _validate(path, CurationPolicyManifest)


def validate_split_policy_manifest(path: Path) -> SplitPolicyManifest:
    return _validate(path, SplitPolicyManifest)


def validate_training_export_manifest(path: Path) -> TrainingExportManifest:
    return _validate(path, TrainingExportManifest)


def validate_training_pack_policy(path: Path) -> TrainingPackPolicyManifest:
    return _validate(path, TrainingPackPolicyManifest)


def validate_lp9_failure_mining_policy(path: Path) -> LP9FailureMiningPolicyManifest:
    return _validate(path, LP9FailureMiningPolicyManifest)


def validate_lp9_lexical_preference_pack_manifest(path: Path) -> LP9LexicalPreferencePackManifest:
    return _validate(path, LP9LexicalPreferencePackManifest)


def validate_evaluation_manifest(path: Path, release_gates_path: Path) -> EvaluationManifest:
    eval_manifest = _validate(path, EvaluationManifest)
    gates = validate_release_gates(release_gates_path)
    ensure_eval_gates_sync(eval_manifest, gates)
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

    for path in sorted((root / "manifests").glob("*.y*ml")):
        manifest_kind = "dataset_manifest"
        try:
            doc = load_yaml(path)
            if not isinstance(doc, dict) or "kind" not in doc:
                raise ValueError("Manifest must be a mapping and include a 'kind' field")
            kind = doc.get("kind")
            manifest_kind = str(kind)
            if kind == "dataset_manifest":
                validate_dataset_manifest(path)
            elif kind == "corpus_source_manifest":
                validate_corpus_source_manifest(path)
            elif kind == "curation_policy_manifest":
                validate_curation_policy_manifest(path)
            elif kind == "split_policy_manifest":
                validate_split_policy_manifest(path)
            elif kind == "training_export_manifest":
                validate_training_export_manifest(path)
            elif kind == "training_pack_policy_manifest":
                validate_training_pack_policy(path)
            elif kind == "lp9_failure_mining_policy":
                validate_lp9_failure_mining_policy(path)
            elif kind == "lp9_lexical_preference_pack":
                validate_lp9_lexical_preference_pack_manifest(path)
            elif kind == "modern_corpus_acquisition_manifest":
                validate_modern_corpus_manifest(path)
            else:
                raise ValueError(f"Unsupported manifests kind: {kind}")
            report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), manifest_kind, str(exc)))

    for path in sorted((root / "rules").glob("*.y*ml")):
        manifest_kind = "lp_manifest_unknown"
        try:
            doc = load_yaml(path)
            if not isinstance(doc, dict) or "kind" not in doc:
                raise ValueError("Rules manifest must be a mapping and include a 'kind' field")
            kind = doc.get("kind")
            manifest_kind = str(kind)
            if kind == "lp_context_manifest":
                validate_lp_context_manifest(path)
                report.checked_files.append(str(path))
            elif kind == "lp_rule_manifest":
                validate_lp_rule_manifest(path)
                report.checked_files.append(str(path))
            else:
                raise ValueError(f"Unsupported rules manifest kind: {kind}")
        except Exception as exc:
            report.ok = False
            report.issues.append(ValidationIssue(str(path), manifest_kind, str(exc)))

    for path in sorted((root / "eval").glob("*.y*ml")):
        try:
            if "taxonomy" in path.name:
                validate_error_taxonomy_manifest(path)
                report.checked_files.append(str(path))
            else:
                validate_evaluation_manifest(path, gates_path)
                report.checked_files.append(str(path))
        except Exception as exc:
            report.ok = False
            kind = "error_taxonomy_manifest" if "taxonomy" in path.name else "evaluation_manifest"
            report.issues.append(ValidationIssue(str(path), kind, str(exc)))
    return report


def validate_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return _validate(path, ModernCorpusAcquisitionManifest)
