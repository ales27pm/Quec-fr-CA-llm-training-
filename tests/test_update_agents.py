from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE_AGENTS_PATH = ROOT / "tools" / "update_agents.py"


spec = importlib.util.spec_from_file_location("qfr_update_agents_test", UPDATE_AGENTS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load update_agents module from {UPDATE_AGENTS_PATH}")
update_agents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_agents)


def test_ensure_required_nested_agents_returns_all_required_paths():
    ensured = update_agents.ensure_required_nested_agents()
    expected = [ROOT / rel / "AGENTS.md" for rel in update_agents.REQUIRED_NESTED_DIRS]
    assert ensured == expected
    assert all(path.exists() for path in ensured)


def test_ensure_nested_cli_output_lists_all_required_paths():
    proc = subprocess.run(
        [sys.executable, "tools/update_agents.py", "--ensure-nested"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    out = proc.stdout
    assert "manifests/AGENTS.md" in out
    assert "rules/AGENTS.md" in out
    assert "eval/AGENTS.md" in out
    assert "src/AGENTS.md" in out
    assert "project/AGENTS.md" in out
