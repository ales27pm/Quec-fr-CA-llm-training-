import json
from pathlib import Path
from qfr_pipeline.corpus_readiness import audit_corpus_readiness

def test_readiness_smoke_or_lower(tmp_path: Path):
    inp=tmp_path/'i.jsonl'; out=tmp_path/'o.json'
    txt='a'*2000000
    inp.write_text(json.dumps({'source_id':'s1','text':txt,'domain':'d1','register':'r1','commercial_use':'allowed','license_status':'open_compatible'})+'\n', encoding='utf-8')
    rep=audit_corpus_readiness(inp,out)
    assert rep['readiness_level'] in {'smoke_test','insufficient'}

def test_duplicates_detected(tmp_path: Path):
    inp=tmp_path/'i.jsonl'; out=tmp_path/'o.json'
    row={'source_id':'s1','text':'bonjour', 'domain':'d1','register':'r1','commercial_use':'allowed','license_status':'open_compatible'}
    inp.write_text('\n'.join([json.dumps(row),json.dumps(row)])+'\n', encoding='utf-8')
    rep=audit_corpus_readiness(inp,out)
    assert rep['duplicates_exact'] >= 1
