#!/usr/bin/env python3
"""Qualify a ten-image SGG response using the pinned official PCIs tuple parser."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT/'.agentic_workflow/workspace/qwenvl-four-task'
PANEL=WORKSPACE/'qualification-sgg-panel.json'
PANEL_SHA='190ddbab331e892bfef7a4be1eec0d77b8b963117789be9af964735743be7219'
OFFICIAL_INFERENCE=ROOT/'.agentic_workflow/workspace/g-substrate-official/inference'
TUPLE=re.compile(r'\(\s*([^,()]+)\s*,\s*([^,()]+)\s*,\s*([^,()]+)\s*\)')
FALLBACK=re.compile(r'^(.*?)\s*\[obj\]\s*(.*?)\s*\[rel\]\s*(.*?)\s*\[obj\]',re.I)
CAP_ERROR=re.compile(r'^generation reached the [0-9,]+-token limit$')

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def official_triplets(text:str)->list[tuple[str,str,str]]:
    matches=TUPLE.findall(text)
    if matches:
        return [(s.strip().strip('"\''),p.strip().strip('"\''),o.strip().strip('"\'')) for s,p,o in matches]
    found=[]
    for segment in re.split(r',[ ]*|\[SEP\]',text):
        match=FALLBACK.search(segment.strip())
        if match: found.append(tuple(part.strip() for part in match.groups()))
    return found

def official_mapped_relation_counts(rows:list[dict])->dict[str,int]:
    if str(OFFICIAL_INFERENCE) not in sys.path:
        sys.path.insert(0,str(OFFICIAL_INFERENCE))
    from evaluate import process_one_ssg_record
    counts={}
    for row in rows:
        processed=process_one_ssg_record(dict(row))
        counts[str(row['image_id'])]=len(processed.get('parsed_triplets',[]))
    return counts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--attempt',type=int,choices=(0,1,2,3),required=True)
    p.add_argument('--run-record',type=Path,required=True)
    p.add_argument('--reassess-existing',action='store_true',
                   help='write a strict gate sidecar while preserving existing format-only artifacts')
    a=p.parse_args()
    if sha(PANEL)!=PANEL_SHA: raise ValueError('frozen SGG qualification panel hash mismatch')
    panel=json.loads(PANEL.read_text())['items']
    ids=[str(r['image_id']) for r in panel]
    run=json.loads(a.run_record.read_text())
    if run.get('model_id')!=a.model_id or run.get('attempt')!=a.attempt:
        raise ValueError('SGG run model or attempt mismatch')
    if (run.get('selected_image_ids')!=ids
            or run.get('phase_status') not in {'awaiting_official_pcis_score','running'}):
        raise ValueError('SGG run is incomplete or does not use the frozen ten images')
    input_path=Path(run.get('official_input_path','')).resolve(strict=True)
    if sha(input_path)!=run.get('official_input_sha256'):
        raise ValueError('official scorer input hash mismatch')
    scorer_rows=[json.loads(line) for line in input_path.read_text().splitlines() if line]
    if len(scorer_rows)!=10 or [str(r.get('image_id')) for r in scorer_rows]!=ids:
        raise ValueError('official scorer input does not match the frozen ten-row panel')
    rows=run.get('datasets',{}).get('vg150',{}).get('items',[])
    by_id={str(r.get('image_id')):r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!=set(ids): raise ValueError('SGG run rows do not exactly match panel IDs')
    for item in scorer_rows:
        saved=by_id[str(item['image_id'])]
        expected='' if saved.get('error') else saved.get('raw_response') or ''
        if item.get('predict')!=expected or item.get('inference_error')!=saved.get('error'):
            raise ValueError('official scorer input prediction differs from the saved inference row')
    mapped_counts=official_mapped_relation_counts(scorer_rows)
    item_results=[]
    for image_id in ids:
        row=by_id[image_id]
        raw=row.get('raw_response') or ''
        count=len(official_triplets(raw))
        empty_graph=raw.strip().casefold()=='none'
        cap_only=(row.get('finish_reason')=='length'
                  or bool(CAP_ERROR.fullmatch(row.get('error') or '')))
        transport_error=bool(row.get('error')) and not cap_only
        mapped_count=mapped_counts.get(image_id,0)
        valid=mapped_count>0 and not transport_error and not cap_only
        item_results.append({'image_id':image_id,'official_tuple_n':count,'official_mapped_triplet_n':mapped_count,'empty_graph':empty_graph,'token_cap_hit':cap_only,'transport_error':transport_error,'valid':valid,'finish_reason':row.get('finish_reason'),'error':row.get('error')})
    valid_n=sum(r['valid'] for r in item_results)
    record={
      'schema_version':1,'model_id':a.model_id,'model_revision':run.get('model_revision'),
      'attempt':a.attempt,'attempt_kind':'baseline' if a.attempt==0 else 'prompt_repair',
      'protocol':'pinned official PCIs mapping path; require at least one complete mapped relation tuple without transport error or token truncation; empty-graph responses do not qualify on this all-positive panel',
      'panel_path':str(PANEL.resolve()),'panel_sha256':PANEL_SHA,
      'run_record':str(a.run_record.resolve()),'run_record_sha256':sha(a.run_record),
      'prompt_version':run.get('prompt_version'),'prompt_sha256':run.get('prompt_sha256'),
      'official_router_sha256':run.get('official_router_sha256'),'official_scorer_sha256':run.get('official_scorer_sha256'),
      'n':10,'valid_n':valid_n,'token_cap_n':sum(x['token_cap_hit'] for x in item_results),'transport_error_n':sum(x['transport_error'] for x in item_results),'threshold':3,'passed':valid_n>=3,'items':item_results
    }
    qual=WORKSPACE/'qualification'/re.sub(r'[^A-Za-z0-9._-]+','_',a.model_id)
    qual.mkdir(parents=True,exist_ok=True)
    output=qual/f'attempt-{a.attempt}-sgg-official-parseability.json'
    gate=qual/f'attempt-{a.attempt}-SGG-gate.json'
    strict_gate=qual/f'attempt-{a.attempt}-SGG-gate-strict.json'
    if a.reassess_existing:
        if a.attempt != 0 or not output.is_file() or not gate.is_file() or strict_gate.exists():
            raise FileExistsError('strict reassessment requires preserved attempt-0 format artifacts and no prior strict sidecar')
        original=json.loads(gate.read_text())
        if (original.get('model_id')!=a.model_id or original.get('attempt')!=0
                or original.get('run_record_sha256')!=record['run_record_sha256']
                or original.get('panel_sha256')!=PANEL_SHA):
            raise ValueError('existing format-only gate does not match the saved baseline')
        record['gate_protocol_version']='strict-v2'
        record['reassesses_artifact']=str(gate.resolve())
        record['reassesses_sha256']=sha(gate)
        if original.get('pcis_score_provenance') is not None:
            record['pcis_score_provenance']=original['pcis_score_provenance']
        strict_gate.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'model_id':a.model_id,'attempt':a.attempt,'valid_n':valid_n,'threshold':3,'passed':valid_n>=3,'record':str(strict_gate)},indent=2))
        return
    if output.exists() or gate.exists() or strict_gate.exists(): raise FileExistsError('preserving an earlier SGG qualification record')
    if a.attempt>0:
        prior_strict=qual/f'attempt-{a.attempt-1}-SGG-gate-strict.json'
        prior_path=prior_strict if prior_strict.is_file() else qual/f'attempt-{a.attempt-1}-SGG-gate.json'
        if not prior_path.is_file(): raise ValueError('SGG repair lacks preceding gate')
        prior=json.loads(prior_path.read_text())
        if (prior.get('model_id')!=a.model_id or prior.get('attempt')!=a.attempt-1
                or prior.get('passed') is not False or prior.get('panel_sha256')!=PANEL_SHA
                or prior.get('model_revision')!=record['model_revision']):
            raise ValueError('SGG repair requires preceding failed gate for the same model revision and panel')
    record['gate_protocol_version']='strict-v2'
    payload=json.dumps(record,indent=2)+'\n'
    output.write_text(payload)
    gate.write_text(payload)
    strict_gate.write_text(payload)
    print(json.dumps({'model_id':a.model_id,'attempt':a.attempt,'valid_n':valid_n,'threshold':3,'passed':valid_n>=3,'record':str(output)},indent=2))
if __name__=='__main__': main()
