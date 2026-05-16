from __future__ import annotations

import json
from pathlib import Path
import tempfile

import yaml

from qfr_pipeline.corpus_sources import ingest_corpus_sources
from qfr_pipeline.paths import ROOT


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_manifest(path: Path, source_path: Path) -> None:
    payload = {
        "kind": "corpus_source_manifest",
        "schema_version": "1.0",
        "primary_language": "fr-CA",
        "sources": [
            {
                "source_id": "gutenberg_local_fixture",
                "name": "Gutenberg local fixture",
                "source_type": "local_text",
                "path": _repo_rel(source_path),
                "license": "project_gutenberg_public_domain_us_verify_jurisdiction",
                "license_url": "https://www.gutenberg.org/policy/license.html",
                "provenance": "Project Gutenberg fixture",
                "collection_method": "fixture",
                "allowed_for_training": True,
                "allowed_for_evaluation": False,
                "contains_holdout_material": False,
                "holdout_only": False,
                "contains_personal_data": False,
                "requires_review": False,
                "quality_tier": "silver",
                "register": "mixed",
                "dialect_region": "Quebec",
                "domain": "literary",
                "notes": "fixture",
            }
        ],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_hard_wrapped_text_produces_paragraph_records() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        repo_tmp_dir = Path(tmp)
        source = repo_tmp_dir / "gutenberg.txt"
        source.write_text(
            (
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                "Produced by Example Team\n\n"
                "Ce passage est écrit sur plusieurs lignes pour simuler un texte\n"
                "Gutenberg à retour chariot dur et doit être reconstruit comme\n"
                "un paragraphe continu suffisamment long pour dépasser le seuil.\n\n"
                "Un deuxième paragraphe suit aussi avec des retours à la ligne\n"
                "forcés afin de confirmer la segmentation et la normalisation\n"
                "de l'ingestion en mode paragraphe pour le corpus réel local.\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            ),
            encoding="utf-8",
        )
        manifest = repo_tmp_dir / "manifest.yaml"
        out = repo_tmp_dir / "harvest.jsonl"
        report = repo_tmp_dir / "report.json"
        _write_manifest(manifest, source)

        payload = ingest_corpus_sources(manifest, out, min_chars=80)
        assert payload.records_written >= 2

        rows = _read_jsonl(out)
        assert rows
        assert all("project gutenberg" not in row["text"].casefold() for row in rows)
        assert all(len(row["text"]) >= 80 for row in rows)
        assert all(row["source_path"] == _repo_rel(source) for row in rows)

        report.write_text(
            json.dumps(payload.__dict__, ensure_ascii=False),
            encoding="utf-8",
        )
        report_text = report.read_text(encoding="utf-8")
        assert "/workspace" not in report_text
        assert "/home/runner" not in report_text


def test_no_blank_lines_falls_back_to_line_level() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        repo_tmp_dir = Path(tmp)
        source = repo_tmp_dir / "plain.txt"
        source.write_text(
            "\n".join(
                [
                    "Ligne courte.",
                    "Cette ligne est suffisamment longue pour passer le filtre de caractères"
                    " sans segmentation par paragraphe.",
                    "Une deuxième ligne longue est aussi présente pour valider le"
                    " comportement de secours déterministe.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = repo_tmp_dir / "manifest.yaml"
        out = repo_tmp_dir / "harvest.jsonl"
        _write_manifest(manifest, source)

        payload = ingest_corpus_sources(manifest, out, min_chars=80)
        assert payload.records_written == 2

        rows = _read_jsonl(out)
        assert len(rows) == 2
        assert all("segmentation:line_fallback" in row["quality_flags"] for row in rows)


def test_min_chars_and_normalized_duplicate_suppression() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=reports_dir) as tmp:
        repo_tmp_dir = Path(tmp)
        source = repo_tmp_dir / "dupes.txt"
        source.write_text(
            """
Texte très long avec plusieurs mots pour dépasser largement le seuil minimal.

  texte   très long avec plusieurs mots pour dépasser largement le seuil minimal.  

Court.
""",
            encoding="utf-8",
        )
        manifest = repo_tmp_dir / "manifest.yaml"
        out = repo_tmp_dir / "harvest.jsonl"
        _write_manifest(manifest, source)

        payload = ingest_corpus_sources(manifest, out, min_chars=30)
        assert payload.records_written == 1

        rows = _read_jsonl(out)
        assert len(rows) == 1
        assert len(rows[0]["text"]) >= 30
