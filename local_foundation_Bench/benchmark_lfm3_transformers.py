#!/usr/bin/env python3
"""Full shared-suite benchmark of the official LFM3-VL-2B Transformers checkpoint."""
from __future__ import annotations
import json, os, statistics, sys, time, threading, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qwenVL_Bench')); sys.path.insert(0,str(ROOT))
import benchmark as common
import benchmark_runners as shared
from gpu_queue import run_queued

MODEL='LiquidAI/LFM3-VL-2B'; REV='23387beda43c60310348e83e25f78f33c3140d3a'
SNAP=Path('/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/models--LiquidAI--LFM3-VL-2B/snapshots')/REV
SIZES={'ai2d':100,'textvqa':100,'vqa_truth':25,'vqa_count':25,'vqa_description':25,'vqa_affordance':25,'vqa_negation':25,'graph_grab_lite':500}
OUT=Path(__file__).resolve().parent/'results/runs/lfm3_refresh_20261112/transformers-original'
class Sampler:
 def __init__(self):self.stop=threading.Event();self.peak=0;self.samples=0
 def loop(self):
  while not self.stop.is_set():
   try:
    v=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0]);self.peak=max(self.peak,v);self.samples+=1
   except Exception:pass
   self.stop.wait(.5)
 def start(self):self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def close(self):self.stop.set();self.thread.join(5)
 def gib(self):return round(self.peak/1024,3)
def run():
 import torch
 from transformers import AutoModelForImageTextToText,AutoProcessor
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 OUT.mkdir(parents=True,exist_ok=True); path=OUT/'results.json'
 result={'generated_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'seed':3407,'model':{'model_id':MODEL,'revision':REV,'snapshot':str(SNAP),'backend':f'Transformers {__import__("transformers").__version__} AutoModelForImageTextToText, CUDA {torch.version.cuda}','quantization':'BF16','configured_context':8192,'generation':{'temperature':0.1,'top_p':1.0,'top_k':50,'min_p':0.0,'repetition_penalty':1.1,'max_new_tokens':4096,'seed':3407},'tasks':{},'status':'loading'},'preflight':{'passed':False,'required':['ai2d','textvqa'],'checks':{}}}
 def save():path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
 save(); sampler=Sampler();sampler.start()
 model=AutoModelForImageTextToText.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=True,device_map='cuda',torch_dtype='auto'); processor=AutoProcessor.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=True);model.eval()
 result['model']['status']='preflight'; cache={t:{} for t in SIZES};
 def generate(img,prompt,seed):
  msgs=[{'role':'user','content':[{'type':'image','image':img.convert('RGB')},{'type':'text','text':prompt}]}]
  inputs=processor.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(model.device)
  torch.manual_seed(seed); start=time.perf_counter()
  with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=4096,do_sample=True,temperature=.1,top_p=1.,top_k=50,repetition_penalty=1.1)
  torch.cuda.synchronize(); elapsed=time.perf_counter()-start; new=out[:,inputs['input_ids'].shape[-1]:]
  return processor.decode(new[0],skip_special_tokens=True).strip(),int(inputs['input_ids'].shape[-1]),int(new.shape[-1]),elapsed
 for task in ('ai2d','textvqa'):
  data=shared.dataset_for(task);idx=common.pick_indices(len(data),SIZES[task],3407)[0];row=data[idx];prompt,expected=shared.prompt_for(task,row)
  answer,it,ot,sec=generate(row['image'],prompt,3407);pred,score,err=shared.score_for(task,answer,expected)
  result['preflight']['checks'][task]={'index':idx,'passed':bool(answer) and (not err if task=='ai2d' else True),'answer':answer,'prompt':prompt,'score':score,'parse_error':err,'input_tokens':it,'output_tokens':ot,'generation_s':sec}
  cache[task][idx]=(answer,it,ot,sec,pred,score,err)
 result['preflight']['passed']=len(result['preflight']['checks'])==2 and all(x['passed'] for x in result['preflight']['checks'].values());save()
 if not result['preflight']['passed']:result['model'].update(status='preflight_failed');save();return
 result['model']['status']='in_progress';
 try:
  for task,n in SIZES.items():
   data=shared.dataset_for(task);indices=common.pick_indices(len(data),n,3407);items=[]
   for pos,idx in enumerate(indices,1):
    row=data[idx];prompt,expected=shared.prompt_for(task,row)
    try:
     a,it,ot,sec,pred,score,err=cache[task][idx] if idx in cache[task] else (*generate(row['image'],prompt,3407),None,None,None)
     if pred is None:pred,score,err=shared.score_for(task,a,expected)
     items.append({'index':idx,'output':a,'prompt':prompt,'input_tokens':it,'output_tokens':ot,'generation_s':round(sec,4),'prediction':pred,'score':score,'parse_error':err,'finish_reason':'length' if ot>=4096 else 'stop'})
    except Exception as e:items.append({'index':idx,'error':f'{type(e).__name__}: {e}'})
    if pos%10==0 or pos==n:result['model']['tasks'][task]={'requested_n':n,'completed_n':sum('score'in x for x in items),'failed_n':sum('error'in x for x in items),'score':statistics.mean(x['score'] for x in items if 'score'in x) if any('score'in x for x in items) else None,'parse_errors':sum(bool(x.get('parse_error')) for x in items),'items':items};save();print('PROGRESS',task,pos,n,flush=True)
  m=result['model'];m['full_benchmark_complete']=all(m['tasks'].get(t,{}).get('completed_n')==n for t,n in SIZES.items());m['full_benchmark_macro']=statistics.mean(m['tasks'][t]['score'] for t in SIZES) if m['full_benchmark_complete'] else None;m['status']='complete' if m['full_benchmark_complete'] else 'partial'
 except Exception as e:result['model'].update(status='failed',error=f'{type(e).__name__}: {e}');raise
 finally:
  sampler.close();result['model']['whole_device_peak_gib']=sampler.gib();result['model']['memory_sample_count']=sampler.samples;save()
if __name__=='__main__':run_queued(run,'lfm3-transformers')
