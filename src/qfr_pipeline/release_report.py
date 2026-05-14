from dataclasses import dataclass
from pathlib import Path

from qfr_pipeline.io import load_json
from qfr_pipeline.validation import validate_release_gates


REQUIRED_METRICS = {
    "asr_wer",
    "overall_lp_accuracy",
    "lp9_lexical_semantics",
    "lp20_orphaned_preposition",
    "lp7_standard_negation_post_alignment_drop_ratio",
}


@dataclass
class ReleaseReport:
    passed: bool
    per_gate: dict
    missing_metrics: list[str]

    def to_json(self):
        return {
            "passed": self.passed,
            "per_gate": self.per_gate,
            "missing_metrics": self.missing_metrics,
        }

    def to_markdown(self):
        lines = [f"# Release Readiness: {'PASS' if self.passed else 'FAIL'}", "", "## Gate Results"]
        for k, v in self.per_gate.items():
            lines.append(f"- {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['actual']} vs {v['target']})")
        if self.missing_metrics:
            lines.extend(["", "## Missing Metrics", *[f"- {m}" for m in self.missing_metrics]])
        return "\n".join(lines) + "\n"


def evaluate_release(metrics_path: Path, gates_path: Path) -> ReleaseReport:
    metrics = load_json(metrics_path)
    gates = validate_release_gates(gates_path)
    missing = sorted(list(REQUIRED_METRICS - set(metrics.keys())))
    per = {
        "asr_wer": {"actual": metrics.get("asr_wer"), "target": gates.asr.wer_max, "pass": metrics.get("asr_wer", 9) <= gates.asr.wer_max},
        "overall_lp_accuracy": {"actual": metrics.get("overall_lp_accuracy"), "target": gates.linguistic_phenomena.overall_accuracy_min, "pass": metrics.get("overall_lp_accuracy", -1) >= gates.linguistic_phenomena.overall_accuracy_min},
        "lp9_lexical_semantics": {"actual": metrics.get("lp9_lexical_semantics"), "target": gates.linguistic_phenomena.lp9_lexical_semantics_min, "pass": metrics.get("lp9_lexical_semantics", -1) >= gates.linguistic_phenomena.lp9_lexical_semantics_min},
        "lp20_orphaned_preposition": {"actual": metrics.get("lp20_orphaned_preposition"), "target": gates.linguistic_phenomena.lp20_orphaned_preposition_min, "pass": metrics.get("lp20_orphaned_preposition", -1) >= gates.linguistic_phenomena.lp20_orphaned_preposition_min},
        "lp7_standard_negation_post_alignment_drop_ratio": {"actual": metrics.get("lp7_standard_negation_post_alignment_drop_ratio"), "target": gates.alignment.lp7_standard_negation_max_post_alignment_drop_ratio, "pass": metrics.get("lp7_standard_negation_post_alignment_drop_ratio", 9) <= gates.alignment.lp7_standard_negation_max_post_alignment_drop_ratio},
    }
    passed = not missing and all(v["pass"] for v in per.values())
    return ReleaseReport(passed=passed, per_gate=per, missing_metrics=missing)
