#!/usr/bin/env python3
"""Import exact fixed-panel rows from a preserved prior run as qualification attempt 0."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / '.agentic_workflow/workspace/qwenvl-four-task'
PANEL_PATH = WORKSPACE / 'qualification-text-panel.json'
PANEL_SHA256 = '4541d7ed8e641ca70485180a04c0645acf022cea7a2dd446bc1a502a9d172849'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--source-run-dir',type=Path,required=True)
    p.add_argument('--source-validation',type=Path,required=True)
    p.add_argument('--families',nargs='+',choices=('GAR','MGD','ERE'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if sha(PANEL_PATH)!=PANEL_SHA256: raise ValueError('frozen panel hash mismatch')
    panel=json.loads(PANEL_PATH.read_text())
    families=list(dict.fromkeys(a.families))
    selected=[r for r in panel if r['family'] in families]
    if any(sum(r['family']==f for r in selected)!=10 for f in families): raise ValueError('selected fixed panel family is not exactly ten rows')
    manifest_path=a.source_run_dir/'run_manifest.json'
    predictions_path=a.source_run_dir/'predictions.jsonl'
    source=json.loads(manifest_path.read_text())
    if source.get('model_id')!=a.model_id or source.get('status') not in {'complete','partial_stopped_by_qualification_policy'}:
        raise ValueError('source run model or terminal state is not eligible for exact-row import')
    old_model=source.get('model') or {}
    if old_model.get('id')!=a.model_id:
        raise ValueError('nested source model identity does not match the requested model')
    revision=old_model.get('revision')
    if not revision: raise ValueError('source run has no pinned model revision')
    validation=json.loads(a.source_validation.read_text())
    if validation.get('model_id')!=a.model_id or validation.get('source_model_revision')!=revision:
        raise ValueError('source validation report model identity or revision mismatch')
    if Path(validation.get('prediction_path','')).resolve()!=predictions_path.resolve() or sha(predictions_path)!=validation.get('prediction_sha256'):
        raise ValueError('source prediction file hash does not match its reviewed validation report')
    source_prediction_hash=(source.get('prediction_sha256') or source.get('predictions_sha256')
                            or source.get('partial_predictions_sha256'))
    if source_prediction_hash and sha(predictions_path)!=source_prediction_hash:
        raise ValueError('source prediction file hash does not match its run manifest')
    old=[json.loads(line) for line in predictions_path.read_text().splitlines() if line]
    by_id={r.get('sample_id'):r for r in old}
    if len(by_id)!=len(old): raise ValueError('source run contains duplicate sample IDs')
    out=[]
    for row in selected:
        pred=by_id.get(row['sample_id'])
        if pred is None: raise ValueError(f"source run lacks fixed qualification row {row['sample_id']}")
        for field in ('source_file','source_sha256','source_index','task_tag','family','reported_subtask','subtask','label'):
            if pred.get(field)!=row.get(field): raise ValueError(f"source row mismatch {row['sample_id']} field {field}")
        if pred.get('prompt_text')!=row['prompt_text'] or pred.get('prompt_sha256')!=hashlib.sha256(row['prompt_text'].encode()).hexdigest():
            raise ValueError(f"source prompt differs from fixed baseline prompt for {row['sample_id']}")
        imported=dict(pred)
        imported['repair_instruction']=''
        out.append(imported)
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    prediction_path=a.output_dir/'predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
    manifest={
      'schema_version':1,'model_id':a.model_id,'model_revision':revision,'attempt':0,
      'attempt_kind':'imported_exact_baseline_rows','families':families,
      'panel_path':str(PANEL_PATH.resolve()),'panel_sha256':PANEL_SHA256,
      'panel_n':len(out),'completed_rows':len(out),'status':'complete',
      'model':old_model,'prediction_sha256':sha(prediction_path),
      'official_evaluator_sha256':source.get('official_evaluator_sha256'),
      'implementation_sha256':source.get('implementation_sha256',{}),
      'import_provenance':{
        'source_manifest':str(manifest_path.resolve()),'source_manifest_sha256':sha(manifest_path),
        'source_predictions':str(predictions_path.resolve()),'source_predictions_sha256':sha(predictions_path),
        'source_validation':str(a.source_validation.resolve()),'source_validation_sha256':sha(a.source_validation),
        'source_status':source.get('status'),'source_completed_rows':source.get('completed_rows',source.get('partial_completed_rows')),
        'selection':'exact fixed qualification sample IDs and unchanged baseline prompts'
      }
    }
    (a.output_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output_dir':str(a.output_dir),'families':families,'rows':len(out),'revision':revision,'prediction_sha256':manifest['prediction_sha256']},indent=2))
if __name__=='__main__': main()
