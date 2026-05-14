from pathlib import Path


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "project" / "release_gates.yaml").exists() and (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not locate repository root from qfr_pipeline.paths")


ROOT = _find_root()
PROJECT_DIR = ROOT / "project"
MANIFESTS_DIR = ROOT / "manifests"
RULES_DIR = ROOT / "rules"
EVAL_DIR = ROOT / "eval"
RELEASE_GATES_PATH = PROJECT_DIR / "release_gates.yaml"
STATUS_PATH = PROJECT_DIR / "status.json"
AGENTS_PATH = ROOT / "AGENTS.md"


def repo_relative_path(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()
