from pathlib import Path
from qfr_pipeline.modern_corpus import validate_modern_corpus_manifest, acquire_modern_corpus

ROOT=Path(__file__).resolve().parents[1]

def test_valid_manifest_passes():
    validate_modern_corpus_manifest(ROOT/'manifests/modern_corpus_acquisition_manifest.template.yaml')

def test_catalog_only_sources_skipped(tmp_path: Path):
    out=tmp_path/'h.jsonl'; rep=tmp_path/'r.json'
    payload=acquire_modern_corpus(ROOT/'manifests/modern_corpus_acquisition_manifest.template.yaml', out, rep, max_documents=0)
    assert payload['ok']
    assert any(x['reason'] in {'catalog_only','holdout_only','permission_required','noncommercial_requires_explicit_flag'} for x in payload['skipped_sources'])
    assert '/workspace' not in rep.read_text(encoding='utf-8')
