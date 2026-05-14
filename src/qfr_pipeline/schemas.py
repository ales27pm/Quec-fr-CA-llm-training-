from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

REQUIRED_HOLDOUTS = {"qfrblimp", "multiblimp_fr", "qfrcore_eval", "qfrcort_eval"}
GATE_TOLERANCE = 1e-6


def _parse_unit_interval(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be a number in [0,1]")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number in [0,1]") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"{field_name} must be a number in [0,1]")
    return parsed


class AsrGates(BaseModel):
    wer_max: float = Field(ge=0, le=1)


class LPPhenomenaGates(BaseModel):
    overall_accuracy_min: float = Field(ge=0, le=1)
    lp9_lexical_semantics_min: float = Field(ge=0, le=1)
    lp20_orphaned_preposition_min: float = Field(ge=0, le=1)


class AlignmentGates(BaseModel):
    lp7_standard_negation_max_post_alignment_drop_ratio: float = Field(ge=0, le=1)


class BenchmarkingGates(BaseModel):
    scaling_plateau_expected_min: float = Field(ge=0, le=1)
    scaling_plateau_expected_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_order(self):
        if self.scaling_plateau_expected_min > self.scaling_plateau_expected_max:
            raise ValueError("scaling plateau min cannot exceed max")
        return self


class ReleaseGates(BaseModel):
    asr: AsrGates
    linguistic_phenomena: LPPhenomenaGates
    alignment: AlignmentGates
    benchmarking: BenchmarkingGates


class DatasetManifest(BaseModel):
    kind: str
    language: dict[str, Any]
    forbidden_transformations: list[str]
    contamination_checks: dict[str, Any]
    lp_coverage: dict[str, Any]
    quality_filters: dict[str, Any]

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, v: str) -> str:
        if v != "dataset_manifest":
            raise ValueError("kind must be dataset_manifest")
        return v

    @model_validator(mode="after")
    def validate_content(self):
        if self.language.get("primary") != "fr-CA":
            raise ValueError("primary language must be fr-CA")
        forbidden = set(self.forbidden_transformations)
        if "dialect_neutralization" not in forbidden or "translation_to_fr_FR" not in forbidden:
            raise ValueError("forbidden transformations must include dialect_neutralization and translation_to_fr_FR")
        blocklisted = set(self.contamination_checks.get("blocklisted_eval_sets", []))
        if not REQUIRED_HOLDOUTS.issubset(blocklisted):
            raise ValueError("missing required holdout benchmark names")
        fuzzy = self.contamination_checks.get("fuzzy_match_threshold")
        _parse_unit_interval(fuzzy, "fuzzy_match_threshold")
        confidence = self.quality_filters.get("language_id", {}).get("min_confidence")
        _parse_unit_interval(confidence, "min_confidence")
        lp_ids = self.lp_coverage.get("included_lp_ids", [])
        if any((not isinstance(i, int)) or i < 1 or i > 20 for i in lp_ids):
            raise ValueError("LP IDs must be integers in 1..20")
        return self


class LPRuleManifest(BaseModel):
    kind: str
    lp_id: int
    positive_patterns: list[dict[str, Any]]
    negative_patterns: list[dict[str, Any]]
    confidence_threshold: dict[str, float]

    @field_validator("kind")
    @classmethod
    def rule_kind(cls, v: str) -> str:
        if v != "lp_rule_manifest":
            raise ValueError("kind must be lp_rule_manifest")
        return v

    @model_validator(mode="after")
    def validate_rule(self):
        if not (1 <= self.lp_id <= 20):
            raise ValueError("lp_id must be in 1..20")
        for key in ("auto_accept", "auto_reject"):
            val = self.confidence_threshold.get(key)
            if val is None or not 0 <= float(val) <= 1:
                raise ValueError(f"confidence_threshold.{key} must be in [0,1]")
        return self


class EvaluationManifest(BaseModel):
    kind: str
    benchmark_sets: list[dict[str, Any]]
    release_gates: dict[str, Any]

    @field_validator("kind")
    @classmethod
    def eval_kind(cls, v: str) -> str:
        if v != "evaluation_manifest":
            raise ValueError("kind must be evaluation_manifest")
        return v

    @model_validator(mode="after")
    def holdouts_present(self):
        names = {b.get("name") for b in self.benchmark_sets if isinstance(b, dict)}
        if not REQUIRED_HOLDOUTS.issubset(names):
            raise ValueError("missing holdout benchmark names")
        return self


class StatusTask(BaseModel):
    id: str
    goal: str
    status: str
    evidence: Any


class ProjectStatus(BaseModel):
    last_updated: str
    tasks: list[StatusTask]


def ensure_eval_gates_sync(eval_manifest: EvaluationManifest, release_gates: ReleaseGates) -> None:
    rg = eval_manifest.release_gates
    lp_floors = rg.get("lp_floors", {})
    overall = rg.get("overall_lp_accuracy_min")
    lp9 = lp_floors.get("9", lp_floors.get(9))
    lp20 = lp_floors.get("20", lp_floors.get(20))
    try:
        overall_f = float(overall)
        lp9_f = float(lp9)
        lp20_f = float(lp20)
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation manifest release_gates must contain numeric overall/lp9/lp20 thresholds") from exc
    rg_overall = release_gates.linguistic_phenomena.overall_accuracy_min
    rg_lp9 = release_gates.linguistic_phenomena.lp9_lexical_semantics_min
    rg_lp20 = release_gates.linguistic_phenomena.lp20_orphaned_preposition_min
    if abs(overall_f - rg_overall) > GATE_TOLERANCE:
        raise ValueError(
            f"Mismatch in overall_lp_accuracy_min: eval={overall_f} release_gates={rg_overall}"
        )
    if abs(lp9_f - rg_lp9) > GATE_TOLERANCE:
        raise ValueError(f"Mismatch in lp_floors.9: eval={lp9_f} release_gates={rg_lp9}")
    if abs(lp20_f - rg_lp20) > GATE_TOLERANCE:
        raise ValueError(f"Mismatch in lp_floors.20: eval={lp20_f} release_gates={rg_lp20}")
