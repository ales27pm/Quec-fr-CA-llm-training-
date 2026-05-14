import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


class FileIOError(RuntimeError):
    pass


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    except OSError as exc:
        raise FileIOError(f"Failed atomic write for {path}: {exc}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileIOError(f"Failed to read YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FileIOError(f"YAML at {path} must be a mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileIOError(f"Failed to read JSON {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
