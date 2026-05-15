from typing import Any
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

REQUIRED_HOLDOUTS = {"qfrblimp", "multiblimp_fr", "qfrcore_eval", "qfrcort_eval"}
GATE_TOLERANCE = 1e-6


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


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








class CorpusSourceEntry(BaseModel):
    source_id: str
    name: str
    source_type: str
    path: str
    license: str
    license_url: str | None = None
    provenance: str
    collection_method: str
    allowed_for_training: bool
    allowed_for_evaluation: bool
    contains_holdout_material: bool
    holdout_only: bool = False
    contains_personal_data: bool
    requires_review: bool
    quality_tier: str
    register: str
    dialect_region: str
    notes: str

    @field_validator("source_type")
    @classmethod
    def valid_source_type(cls, v: str) -> str:
        allowed = {"local_text", "local_jsonl", "local_csv", "manual_fixture", "future_remote"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {sorted(allowed)}")
        return v

    @field_validator("quality_tier")
    @classmethod
    def valid_quality_tier(cls, v: str) -> str:
        if v not in {"gold", "silver", "bronze", "quarantine"}:
            raise ValueError("quality_tier must be one of gold|silver|bronze|quarantine")
        return v

    @field_validator("register")
    @classmethod
    def valid_register(cls, v: str) -> str:
        if v not in {"formal", "informal", "mixed", "unknown"}:
            raise ValueError("register must be one of formal|informal|mixed|unknown")
        return v

    @field_validator("dialect_region")
    @classmethod
    def valid_dialect_region(cls, v: str) -> str:
        if v not in {"Quebec", "Canada_fr", "unknown"}:
            raise ValueError("dialect_region must be one of Quebec|Canada_fr|unknown")
        return v

    @model_validator(mode="after")
    def validate_policy(self):
        for field_name in ("source_id", "name", "path", "license", "provenance", "collection_method", "notes"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if Path(self.path).is_absolute():
            raise ValueError("path must be repo-relative")
        if self.allowed_for_training and self.contains_holdout_material:
            raise ValueError("allowed_for_training and contains_holdout_material cannot both be true")
        if self.allowed_for_evaluation and self.contains_holdout_material and (not self.holdout_only):
            raise ValueError("allowed_for_evaluation with holdout material requires holdout_only=true")
        if self.quality_tier == "quarantine" and self.allowed_for_training:
            raise ValueError("quality_tier=quarantine cannot be allowed for training")
        if self.contains_personal_data and (not self.requires_review):
            raise ValueError("contains_personal_data=true requires requires_review=true")
        return self


class CorpusSourceManifest(BaseModel):
    kind: str
    schema_version: str
    primary_language: str
    sources: list[CorpusSourceEntry]

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, v: str) -> str:
        if v != "corpus_source_manifest":
            raise ValueError("kind must be corpus_source_manifest")
        return v

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.primary_language != "fr-CA":
            raise ValueError("primary_language must be fr-CA")
        ids = [s.source_id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source_id must be unique")
        return self

class ErrorTaxonomyItem(BaseModel):
    error_code: str
    label: str
    severity: str
    description: str
    detection_guidance: str
    remediation_guidance: str
    examples: list[str]
    applies_to: list[str]

    @field_validator("severity")
    @classmethod
    def valid_severity(cls, v: str) -> str:
        if v not in {"blocking", "non_blocking"}:
            raise ValueError("severity must be blocking or non_blocking")
        return v

    @model_validator(mode="after")
    def validate_nonempty(self):
        for field_name in ("error_code", "label", "description", "detection_guidance", "remediation_guidance"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if not self.examples or any((not isinstance(e, str)) or (not e.strip()) for e in self.examples):
            raise ValueError("examples must contain non-empty strings")
        if not self.applies_to or any((not isinstance(e, str)) or (not e.strip()) for e in self.applies_to):
            raise ValueError("applies_to must contain non-empty strings")
        return self


class ErrorTaxonomyManifest(BaseModel):
    kind: str
    schema_version: str
    lp_id: int
    phenomenon: str
    taxonomy: list[ErrorTaxonomyItem]

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, v: str) -> str:
        if v != "error_taxonomy_manifest":
            raise ValueError("kind must be error_taxonomy_manifest")
        return v

    @model_validator(mode="after")
    def validate_manifest(self):
        if not (1 <= self.lp_id <= 20):
            raise ValueError("lp_id must be in 1..20")
        codes=[item.error_code for item in self.taxonomy]
        if len(set(codes)) != len(codes):
            raise ValueError("error_code entries must be unique within taxonomy")
        return self
class LPContextContrast(BaseModel):
    model_config = {"populate_by_name": True}
    contrast_id: str
    positive_pattern: str
    negative_pattern: str
    register_value: str = Field(alias="register")
    term_type: str
    allowed_contexts: list[str]
    blocked_contexts: list[str]
    good_templates: list[str]
    bad_templates: list[str]
    notes: str
    source_authority: str
    phenomenon_tags: list[str] | None = None
    required_good_substrings: list[str] = Field(default_factory=list)
    required_bad_substrings: list[str] = Field(default_factory=list)
    forbidden_good_substrings: list[str] = Field(default_factory=list)
    forbidden_bad_substrings: list[str] = Field(default_factory=list)
    minimal_contrast_focus: str | None = None
    normalized_required_good_substrings: list[str] = Field(default_factory=list, exclude=True)
    normalized_required_bad_substrings: list[str] = Field(default_factory=list, exclude=True)
    normalized_forbidden_good_substrings: list[str] = Field(default_factory=list, exclude=True)
    normalized_forbidden_bad_substrings: list[str] = Field(default_factory=list, exclude=True)

    _allowed_contrast_focuses = {
        "preposition_attachment",
        "required_preposition_retention",
        "stranded_preposition",
    }

    @field_validator("register_value")
    @classmethod
    def valid_register(cls, v: str) -> str:
        if v not in {"formal", "informal", "neutral"}:
            raise ValueError("register must be one of formal|informal|neutral")
        return v

    @field_validator("term_type")
    @classmethod
    def valid_term_type(cls, v: str) -> str:
        if v not in {"noun", "noun_phrase", "idiom", "anglicism", "expression"}:
            raise ValueError("term_type must be one of noun|noun_phrase|idiom|anglicism|expression")
        return v

    @model_validator(mode="after")
    def validate_templates(self):
        if not self.positive_pattern.strip() or not self.negative_pattern.strip():
            raise ValueError("positive_pattern and negative_pattern must be non-empty")
        if not self.good_templates or not self.bad_templates:
            raise ValueError("good_templates and bad_templates must be non-empty")
        if any(not t.strip() for t in [*self.good_templates, *self.bad_templates]):
            raise ValueError("templates must be non-empty strings")
        if len(self.good_templates) != len(self.bad_templates):
            raise ValueError("good_templates and bad_templates must be the same length")
        for field_name in (
            "required_good_substrings",
            "required_bad_substrings",
            "forbidden_good_substrings",
            "forbidden_bad_substrings",
        ):
            values = getattr(self, field_name)
            stripped_values: list[str] = []
            for v in values:
                if not isinstance(v, str):
                    raise ValueError(f"{field_name} must contain non-empty strings")
                s = v.strip()
                if not s:
                    raise ValueError(f"{field_name} must contain non-empty strings")
                stripped_values.append(s)
            setattr(self, field_name, stripped_values)
        if self.phenomenon_tags is not None:
            stripped_tags: list[str] = []
            for v in self.phenomenon_tags:
                if not isinstance(v, str):
                    raise ValueError("phenomenon_tags must contain at least one non-empty string when provided")
                s = v.strip()
                if not s:
                    raise ValueError("phenomenon_tags must contain at least one non-empty string when provided")
                stripped_tags.append(s)
            if not stripped_tags:
                raise ValueError("phenomenon_tags must contain at least one non-empty string when provided")
            self.phenomenon_tags = stripped_tags
        if self.minimal_contrast_focus is not None:
            focus = self.minimal_contrast_focus.strip()
            if focus not in self._allowed_contrast_focuses:
                allowed = "|".join(sorted(self._allowed_contrast_focuses))
                raise ValueError(f"minimal_contrast_focus must be one of {allowed}")
            self.minimal_contrast_focus = focus
        self.normalized_required_good_substrings = [normalize_text(v) for v in self.required_good_substrings]
        self.normalized_required_bad_substrings = [normalize_text(v) for v in self.required_bad_substrings]
        self.normalized_forbidden_good_substrings = [normalize_text(v) for v in self.forbidden_good_substrings]
        self.normalized_forbidden_bad_substrings = [normalize_text(v) for v in self.forbidden_bad_substrings]
        return self


class LPContextManifest(BaseModel):
    kind: str
    lp_id: int
    name: str
    phenomenon: str
    contrasts: list[LPContextContrast]

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, v: str) -> str:
        if v != "lp_context_manifest":
            raise ValueError("kind must be lp_context_manifest")
        return v

    @model_validator(mode="after")
    def validate_lp_id(self):
        if not (1 <= self.lp_id <= 20):
            raise ValueError("lp_id must be in 1..20")
        return self
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
