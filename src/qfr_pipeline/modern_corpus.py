
from __future__ import annotations
import hashlib, json, re, time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from qfr_pipeline.io import load_yaml, write_json
from qfr_pipeline.paths import ROOT, repo_relative_path
from qfr_pipeline.schemas import ModernCorpusAcquisitionManifest

def load_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return ModernCorpusAcquisitionManifest.model_validate(load_yaml(path))

def validate_modern_corpus_manifest(path: Path) -> ModernCorpusAcquisitionManifest:
    return load_modern_corpus_manifest(path)

class _TextExtractor(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self, data): self.parts.append(data)

def _sha(t:str)->str: return hashlib.sha256(t.encode('utf-8')).hexdigest()

def _record(source, text, idx, register='formal', domain='government', extra=None):
    extra=extra or {}
    rid=f"{source.source_id}:{idx}:{_sha(text)[:10]}"
    return {
        'record_id':rid,'source_id':source.source_id,'source_name':source.name,'text':text,'text_sha256':_sha(text),
        'language':'fr-CA','dialect_region':extra.get('dialect_region','Quebec'),'register':register,'domain':domain,
        'source_type':source.source_type,'acquisition_status':source.acquisition_status,'license_status':source.license_status,
        'license_name':source.license_name,'license_url':source.license_url,'commercial_use':source.commercial_use,
        'allowed_for_training':source.allowed_for_training,'allowed_for_evaluation':source.allowed_for_evaluation,
        'holdout_only':source.holdout_only,'pii_risk':source.pii_risk,'source_url':extra.get('source_url'),
        'source_path':extra.get('source_path'),'collected_via_adapter':source.adapter.name,'date_min':source.date_min,'date_max':source.date_max,
        'quality_flags':extra.get('quality_flags',[]),'requires_review':extra.get('requires_review',False)
    }

def acquire_modern_corpus(manifest_path: Path, out_jsonl: Path, report_path: Path, permission_manifest: Path|None=None, include_noncommercial: bool=False, max_documents:int|None=None, timeout:int=30):
    m=load_modern_corpus_manifest(manifest_path); perm={}
    if permission_manifest and permission_manifest.exists(): perm=load_yaml(permission_manifest) or {}
    out=[]; skipped=[]; per={}; hold=[]
    for s in m.sources:
      reason=None
      if s.acquisition_status in {'catalog_only','holdout_only','blocked_license'}: reason=s.acquisition_status
      if s.source_type=='permission_required' and not perm.get('sources',{}).get(s.source_id): reason='permission_required'
      if s.license_status=='noncommercial_only' and not include_noncommercial: reason='noncommercial_requires_explicit_flag'
      if reason: skipped.append({'source_id':s.source_id,'reason':reason});
      else:
        if s.adapter.name=='holdout_registry': hold.append(s.source_id)
        elif s.adapter.name=='local_text_bundle':
          if not perm.get('sources',{}).get(s.source_id): skipped.append({'source_id':s.source_id,'reason':'permission_required'}); continue
          idx=0
          for g in s.adapter.local_globs:
            for fp in sorted(ROOT.glob(g)):
              for para in [x.strip() for x in fp.read_text(encoding='utf-8').split('

') if x.strip()]:
                out.append(_record(s, para, idx, extra={'source_path':repo_relative_path(fp)})); idx+=1
        elif s.adapter.name=='assnat_journal_debats':
          idx=0
          for u in s.adapter.seed_urls:
            txt=''
            if u.startswith('file://'): txt=Path(u[7:]).read_text(encoding='utf-8')
            else: txt=urlopen(u, timeout=timeout).read().decode('utf-8','ignore')
            ex=_TextExtractor(); ex.feed(txt)
            for para in [re.sub(r'\s+',' ',p).strip() for p in '
'.join(ex.parts).split('
') if p.strip() and len(p.strip())>30]:
              out.append(_record(s, para, idx, register='parliamentary', domain='politics', extra={'source_url':u})); idx+=1
            if s.min_delay_seconds: time.sleep(float(s.min_delay_seconds))
        elif s.adapter.name=='donnees_quebec_ckan':
          base=s.adapter.base_url.rstrip('/')
          res=json.loads(urlopen(f"{base}/package_search?"+urlencode({'rows':s.adapter.rows,'q':s.adapter.query}), timeout=timeout).read())
          items=sorted(res.get('result',{}).get('results',[]), key=lambda x:(x.get('name',''),x.get('id','')))
          for idx,it in enumerate(items[: (max_documents if max_documents is not None else len(items))]):
            text=' '.join(filter(None,[it.get('title'),it.get('notes'),it.get('organization',{}).get('title'),' '.join(t.get('display_name','') for t in it.get('tags',[]))]))
            out.append(_record(s,text,idx,register='administrative',domain='public_data',extra={'source_url':it.get('url') or it.get('metadata_created')}))
        per[s.source_id]=sum(1 for r in out if r['source_id']==s.source_id)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(''.join(json.dumps(r, ensure_ascii=False)+'
' for r in out), encoding='utf-8')
    rep={'ok':True,'manifest':repo_relative_path(manifest_path),'sources_total':len(m.sources),'sources_active':sum(1 for s in m.sources if s.acquisition_status=='active'),'sources_acquired':len(per),'records_written':len(out),'skipped_sources':skipped,'issues':[],'per_source':per,'license_summary':{},'domain_summary':{},'register_summary':{},'holdout_registry':hold,'output':repo_relative_path(out_jsonl)}
    write_json(report_path, rep); return rep
