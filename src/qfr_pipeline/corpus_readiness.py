from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path
from qfr_pipeline.io import write_json
from qfr_pipeline.paths import repo_relative_path

def _norm(t:str)->str: return re.sub(r'\s+',' ',t.strip().lower())

def audit_corpus_readiness(input_jsonl: Path, out_report: Path, policy_manifest: Path|None=None):
    rows=[json.loads(l) for l in input_jsonl.read_text(encoding='utf-8').splitlines() if l.strip()]
    chars=sum(len(r.get('text','')) for r in rows); words=sum(len(r.get('text','').split()) for r in rows)
    est=max(chars/4, words*1.35)
    src=Counter(r.get('source_id','unknown') for r in rows); dom=Counter(r.get('domain','unknown') for r in rows); reg=Counter(r.get('register','unknown') for r in rows)
    dup_exact=len(rows)-len(set(r.get('text_sha256',r.get('text','')) for r in rows)); dup_norm=len(rows)-len(set(_norm(r.get('text','')) for r in rows))
    holdout=sum(1 for r in rows if r.get('holdout_only'))
    instr=sum(1 for r in rows if any(k in r for k in ['messages','user','assistant']) or r.get('task_type')=='instruction')
    dialog=sum(1 for r in rows if r.get('task_type')=='dialogue')
    legal_ok=sum(1 for r in rows if r.get('license_status') in {'open_compatible','noncommercial_only'})/(len(rows) or 1)
    comm_ready=sum(1 for r in rows if r.get('commercial_use')=='allowed')/(len(rows) or 1)
    lvl='insufficient'
    if est>=500000: lvl='smoke_test'
    if est>=20_000_000: lvl='pilot_lora_candidate'
    if est>=150_000_000: lvl='production_lora_candidate'
    blocking=[]
    if holdout>0: blocking.append('holdout_contamination_risk')
    if src and max(src.values())/len(rows)>0.35: blocking.append('single_source_dominance')
    if len(dom)<5: blocking.append('low_domain_diversity')
    if len(reg)<4: blocking.append('low_register_diversity')
    if instr<200000 and lvl=='production_lora_candidate': blocking.append('insufficient_instruction_turns')
    if blocking and lvl=='production_lora_candidate': lvl='production_blocked'
    rep={'ok':True,'input':repo_relative_path(input_jsonl),'records_total':len(rows),'estimated_tokens':int(est),'source_count':len(src),'domain_balance':dom,'register_balance':reg,'instruction_like_ratio':instr/(len(rows) or 1),'dialog_like_ratio':dialog/(len(rows) or 1),'legal_license_ok_ratio':legal_ok,'commercial_ready_ratio':comm_ready,'holdout_contamination_risk_count':holdout,'duplicates_exact':dup_exact,'duplicates_normalized':dup_norm,'readiness_level':lvl,'blocking_reasons':blocking,'recommendations':['Add modern institutional/admin/instructional sources for production readiness.']}
    write_json(out_report, rep)
    return rep
