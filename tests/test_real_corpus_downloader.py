from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import urllib.error
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_PATH = ROOT / "tools" / "download_real_corpus_sources.py"

spec = importlib.util.spec_from_file_location("qfr_real_downloader_test", DOWNLOADER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load downloader module from {DOWNLOADER_PATH}")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self._payload = text.encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_manifest(path: Path, sources: list[dict[str, Any]]) -> None:
    payload = {
        "kind": "corpus_source_manifest",
        "schema_version": "1.0",
        "primary_language": "fr-CA",
        "sources": sources,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_out_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_downloader(
    manifest: Path,
    out_manifest: Path,
    report: Path,
    *,
    args: list[str],
    monkeypatch: Any,
) -> int:
    argv = [
        "download_real_corpus_sources.py",
        "--manifest",
        _repo_rel(manifest),
        "--out-manifest",
        _repo_rel(out_manifest),
        "--report",
        _repo_rel(report),
    ]
    argv.extend(args)
    monkeypatch.setattr(sys, "argv", argv)
    return downloader.main()


def test_existing_local_file_reused_as_cached_with_resume(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        local_source = tmp_dir / "cached_source.txt"
        local_source.write_text("Texte de cache local.\n", encoding="utf-8")

        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"
        _write_manifest(
            manifest,
            [
                {
                    "source_id": "cached_one",
                    "source_type": "future_remote",
                    "path": _repo_rel(local_source),
                    "download_url": "https://example.com/cached.txt",
                    "notes": "fixture",
                }
            ],
        )

        def _unexpected_urlopen(*_: Any, **__: Any) -> _FakeResponse:
            raise AssertionError("urlopen should not be called for cached resume flow")

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _unexpected_urlopen)
        rc = _run_downloader(manifest, out_manifest, report, args=["--resume"], monkeypatch=monkeypatch)
        payload = _read_report(report)

        assert rc == 0
        assert payload["cached_count"] == 1
        assert payload["downloaded_count"] == 0
        assert payload["failed_count"] == 0
        assert payload["sources"][0]["status"] == "cached"
        assert payload["sources"][0]["attempts"] == 0
        assert _read_out_manifest(out_manifest)["sources"][0]["source_id"] == "cached_one"


def test_transient_failure_retries_then_succeeds(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        local_source = tmp_dir / "retried_source.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        _write_manifest(
            manifest,
            [
                {
                    "source_id": "retry_success",
                    "source_type": "future_remote",
                    "path": _repo_rel(local_source),
                    "download_url": "https://example.com/retry.txt",
                }
            ],
        )

        attempts = {"count": 0}

        def _urlopen(*_: Any, **__: Any) -> _FakeResponse:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise urllib.error.URLError(OSError(104, "Connection reset by peer"))
            return _FakeResponse("Texte téléchargé après reprise.\n")

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
        monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=["--retries", "2", "--retry-delay-seconds", "0", "--timeout", "1"],
            monkeypatch=monkeypatch,
        )
        payload = _read_report(report)

        assert rc == 0
        assert payload["downloaded_count"] == 1
        assert payload["failed_count"] == 0
        assert payload["sources"][0]["status"] == "downloaded"
        assert payload["sources"][0]["attempts"] == 3


def test_transient_failure_exhausts_retries_and_fails(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        local_source = tmp_dir / "retry_fail_source.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        _write_manifest(
            manifest,
            [
                {
                    "source_id": "retry_fail",
                    "source_type": "future_remote",
                    "path": _repo_rel(local_source),
                    "download_url": "https://example.com/retry_fail.txt",
                }
            ],
        )

        def _urlopen(*_: Any, **__: Any) -> _FakeResponse:
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
        monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=["--retries", "1", "--retry-delay-seconds", "0"],
            monkeypatch=monkeypatch,
        )
        payload = _read_report(report)

        assert rc == 1
        assert payload["failed_count"] == 1
        assert payload["failed_sources"] == ["retry_fail"]
        assert payload["sources"][0]["status"] == "failed"
        assert payload["sources"][0]["attempts"] == 2
        assert "Network is unreachable" in payload["sources"][0]["last_error"]


def test_allow_partial_with_enough_successes_returns_ok(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"
        source_a = tmp_dir / "a.txt"
        source_b = tmp_dir / "b.txt"
        source_c = tmp_dir / "c.txt"

        _write_manifest(
            manifest,
            [
                {"source_id": "a", "source_type": "future_remote", "path": _repo_rel(source_a), "download_url": "https://example.com/a.txt"},
                {"source_id": "b", "source_type": "future_remote", "path": _repo_rel(source_b), "download_url": "https://example.com/b.txt"},
                {"source_id": "c", "source_type": "future_remote", "path": _repo_rel(source_c), "download_url": "https://example.com/c.txt"},
                {"source_id": "holdout_catalog", "source_type": "future_remote", "path": "data/eval/catalog/holdout.jsonl"},
            ],
        )

        calls = {"count": 0}

        def _urlopen(*_: Any, **__: Any) -> _FakeResponse:
            calls["count"] += 1
            if calls["count"] <= 2:
                return _FakeResponse("succès\n")
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
        monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=["--allow-partial", "--fail-under-downloaded", "2", "--retries", "0"],
            monkeypatch=monkeypatch,
        )
        payload = _read_report(report)

        assert rc == 0
        assert payload["ok"] is True
        assert payload["partial_ok"] is True
        assert payload["downloaded_count"] == 2
        assert payload["failed_count"] == 1
        assert payload["skipped_count"] == 1
        assert payload["failed_sources"] == ["c"]


def test_allow_partial_with_too_few_successes_returns_not_ok(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"
        source_a = tmp_dir / "a.txt"
        source_b = tmp_dir / "b.txt"
        source_c = tmp_dir / "c.txt"

        _write_manifest(
            manifest,
            [
                {"source_id": "a", "source_type": "future_remote", "path": _repo_rel(source_a), "download_url": "https://example.com/a.txt"},
                {"source_id": "b", "source_type": "future_remote", "path": _repo_rel(source_b), "download_url": "https://example.com/b.txt"},
                {"source_id": "c", "source_type": "future_remote", "path": _repo_rel(source_c), "download_url": "https://example.com/c.txt"},
            ],
        )

        calls = {"count": 0}

        def _urlopen(*_: Any, **__: Any) -> _FakeResponse:
            calls["count"] += 1
            if calls["count"] <= 2:
                return _FakeResponse("succès\n")
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
        monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=["--allow-partial", "--fail-under-downloaded", "3", "--retries", "0"],
            monkeypatch=monkeypatch,
        )
        payload = _read_report(report)

        assert rc == 1
        assert payload["ok"] is False
        assert payload["partial_ok"] is False
        assert payload["downloaded_count"] == 2
        assert payload["failed_count"] == 1


def test_out_manifest_contains_only_downloaded_or_cached_sources(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        cached_source = tmp_dir / "cached.txt"
        cached_source.write_text("cache local\n", encoding="utf-8")
        downloaded_source = tmp_dir / "downloaded.txt"
        failed_source = tmp_dir / "failed.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        _write_manifest(
            manifest,
            [
                {"source_id": "cached", "source_type": "future_remote", "path": _repo_rel(cached_source), "download_url": "https://example.com/cached.txt"},
                {"source_id": "downloaded", "source_type": "future_remote", "path": _repo_rel(downloaded_source), "download_url": "https://example.com/downloaded.txt"},
                {"source_id": "failed", "source_type": "future_remote", "path": _repo_rel(failed_source), "download_url": "https://example.com/failed.txt"},
                {"source_id": "holdout_catalog", "source_type": "future_remote", "path": "data/eval/catalog/holdout.jsonl"},
            ],
        )

        def _urlopen(request: Any, timeout: float) -> _FakeResponse:
            if request.full_url.endswith("downloaded.txt"):
                return _FakeResponse("contenu ok\n")
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)
        monkeypatch.setattr(downloader.time, "sleep", lambda _: None)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=[
                "--resume",
                "--allow-partial",
                "--fail-under-downloaded",
                "2",
                "--retries",
                "0",
            ],
            monkeypatch=monkeypatch,
        )

        out = _read_out_manifest(out_manifest)
        payload = _read_report(report)
        source_ids = [row["source_id"] for row in out["sources"]]

        assert rc == 0
        assert source_ids == ["cached", "downloaded"]
        assert "failed" not in source_ids
        assert "holdout_catalog" not in source_ids
        assert payload["failed_sources"] == ["failed"]


def test_skipped_holdout_and_catalog_sources_not_counted_as_failed(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        source_a = tmp_dir / "a.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        _write_manifest(
            manifest,
            [
                {"source_id": "ok", "source_type": "future_remote", "path": _repo_rel(source_a), "download_url": "https://example.com/ok.txt"},
                {"source_id": "holdout_catalog", "source_type": "future_remote", "path": "data/eval/catalog/holdout.jsonl"},
            ],
        )

        monkeypatch.setattr(
            downloader.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: _FakeResponse("ok\n"),
        )
        rc = _run_downloader(manifest, out_manifest, report, args=["--retries", "0"], monkeypatch=monkeypatch)
        payload = _read_report(report)

        assert rc == 0
        assert payload["downloaded_count"] == 1
        assert payload["skipped_count"] == 1
        assert payload["failed_count"] == 0
        assert payload["failed_sources"] == []


def test_report_is_deterministic_and_keeps_manifest_source_order(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        cached_source = tmp_dir / "cached.txt"
        cached_source.write_text("cache local\n", encoding="utf-8")
        downloaded_source = tmp_dir / "downloaded.txt"
        failed_source = tmp_dir / "failed.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        ordered_ids = ["cached", "downloaded", "failed", "holdout_catalog"]
        _write_manifest(
            manifest,
            [
                {"source_id": "cached", "source_type": "future_remote", "path": _repo_rel(cached_source), "download_url": "https://example.com/cached.txt"},
                {"source_id": "downloaded", "source_type": "future_remote", "path": _repo_rel(downloaded_source), "download_url": "https://example.com/downloaded.txt"},
                {"source_id": "failed", "source_type": "future_remote", "path": _repo_rel(failed_source), "download_url": "https://example.com/failed.txt"},
                {"source_id": "holdout_catalog", "source_type": "future_remote", "path": "data/eval/catalog/holdout.jsonl"},
            ],
        )

        def _urlopen(request: Any, timeout: float) -> _FakeResponse:
            if request.full_url.endswith("downloaded.txt"):
                return _FakeResponse("ok\n")
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))

        monkeypatch.setattr(downloader.urllib.request, "urlopen", _urlopen)

        rc = _run_downloader(
            manifest,
            out_manifest,
            report,
            args=["--resume", "--allow-partial", "--fail-under-downloaded", "2", "--retries", "0"],
            monkeypatch=monkeypatch,
        )
        payload = _read_report(report)
        actual_order = [row["source_id"] for row in payload["sources"]]

        assert rc == 0
        assert actual_order == ordered_ids


def test_report_has_no_workspace_or_home_runner_path_leaks(monkeypatch: Any) -> None:
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as tmp:
        tmp_dir = Path(tmp)
        source_a = tmp_dir / "a.txt"
        manifest = tmp_dir / "manifest.yaml"
        out_manifest = tmp_dir / "out.yaml"
        report = tmp_dir / "report.json"

        _write_manifest(
            manifest,
            [
                {"source_id": "ok", "source_type": "future_remote", "path": _repo_rel(source_a), "download_url": "https://example.com/ok.txt"},
            ],
        )

        monkeypatch.setattr(
            downloader.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: _FakeResponse("ok\n"),
        )

        rc = _run_downloader(manifest, out_manifest, report, args=["--retries", "0"], monkeypatch=monkeypatch)
        report_text = report.read_text(encoding="utf-8")

        assert rc == 0
        assert "/workspace" not in report_text
        assert "/home/runner" not in report_text
