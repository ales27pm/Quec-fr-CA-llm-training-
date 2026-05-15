#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
qfr --help >/dev/null
python3 - <<'PY'
import yaml
print("PyYAML available")
PY
