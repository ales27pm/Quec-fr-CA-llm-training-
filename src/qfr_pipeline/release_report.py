from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qfr_pipeline.io import load_json
from qfr_pipeline.validation import validate_release_gates

REQUIRED_METRICS = {
    "asr_wer",
    "overall_lp_accuracy",
    "lp9_lexical_semantics",
    "lp20_orphaned_preposition",
    "lp7_standard_negation_post_alignment_drop_ratio",
}


def parse_numeric_metric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ReleaseReport:
    passed: bool
    per_gate: dict
    missing_metrics: list[str]
    diagnostics: dict[str, Any] | None = None

    def to_json(self):
        payload = {"passed": self.passed, "per_gate": self.per_gate, "missing_metrics": self.missing_metrics}
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        return payload

    def to_markdown(self):
        lines = [f"# Release Readiness: {'PASS' if self.passed else 'FAIL'}", "", "## Gate Results"]
        for k, v in self.per_gate.items():
            lines.append(f"- {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['actual']} vs {v['target']})")
        if self.missing_metrics:
            lines.extend(["", "## Missing Metrics", *[f"- {m}" for m in self.missing_metrics]])
        if self.diagnostics is not None:
            lines.extend(["", "## LP9/LP20 Diagnostics"])
            for key, p in sorted(self.diagnostics.get("phenomena", {}).items()):
                lines.extend([
                    f"### {key}",
                    f"- Binary accuracy: {p['binary_accuracy']:.4f}",
                    f"- Mean semantic similarity: {p['mean_semantic_similarity']}",
                    f"- Top recurring failure classes: {p['top_error_codes']}",
                    f"- Blocking/non-blocking error counts: {p['blocking_error_count']}/{p['non_blocking_error_count']}",
                ])
        return "\n".join(lines) + "\n"


def evaluate_release(metrics_path: Path, gates_path: Path) -> ReleaseReport:
    metrics = load_json(metrics_path)
    gates = validate_release_gates(gates_path)
    missing = sorted(list(REQUIRED_METRICS - set(metrics.keys())))
    asr_wer = parse_numeric_metric(metrics.get("asr_wer"))
    overall_lp = parse_numeric_metric(metrics.get("overall_lp_accuracy"))
    lp9 = parse_numeric_metric(metrics.get("lp9_lexical_semantics"))
    lp20 = parse_numeric_metric(metrics.get("lp20_orphaned_preposition"))
    lp7_drop = parse_numeric_metric(metrics.get("lp7_standard_negation_post_alignment_drop_ratio"))
    per = {
        "asr_wer": {"actual": metrics.get("asr_wer"), "target": gates.asr.wer_max, "pass": asr_wer is not None and asr_wer <= gates.asr.wer_max},
        "overall_lp_accuracy": {"actual": metrics.get("overall_lp_accuracy"), "target": gates.linguistic_phenomena.overall_accuracy_min, "pass": overall_lp is not None and overall_lp >= gates.linguistic_phenomena.overall_accuracy_min},
        "lp9_lexical_semantics": {"actual": metrics.get("lp9_lexical_semantics"), "target": gates.linguistic_phenomena.lp9_lexical_semantics_min, "pass": lp9 is not None and lp9 >= gates.linguistic_phenomena.lp9_lexical_semantics_min},
        "lp20_orphaned_preposition": {"actual": metrics.get("lp20_orphaned_preposition"), "target": gates.linguistic_phenomena.lp20_orphaned_preposition_min, "pass": lp20 is not None and lp20 >= gates.linguistic_phenomena.lp20_orphaned_preposition_min},
        "lp7_standard_negation_post_alignment_drop_ratio": {"actual": metrics.get("lp7_standard_negation_post_alignment_drop_ratio"), "target": gates.alignment.lp7_standard_negation_max_post_alignment_drop_ratio, "pass": lp7_drop is not None and lp7_drop <= gates.alignment.lp7_standard_negation_max_post_alignment_drop_ratio},
    }
    passed = not missing and all(v["pass"] for v in per.values())
    return ReleaseReport(passed=passed, per_gate=per, missing_metrics=missing)
