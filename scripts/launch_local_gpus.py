"""One worker per explicitly allocated GPU; respects an existing visibility mask."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--gpus',nargs='+',required=True,help='logical indices in the existing CUDA visibility mask')
p.add_argument('--db',required=True);p.add_argument('--runs',default='runs');p.add_argument('--data',default='prepared');p.add_argument('--analyze',action='store_true');a=p.parse_args()
visible=os.environ.get('CUDA_VISIBLE_DEVICES');mapping=visible.split(',') if visible else None;processes=[];logs=[]
if len(a.gpus)!=len(set(a.gpus)):raise ValueError('Duplicate GPU assignments')
Path(a.runs).mkdir(parents=True,exist_ok=True)
try:
    for gpu in a.gpus:
        actual=mapping[int(gpu)] if mapping is not None else gpu;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=actual
        log=open(Path(a.runs)/('worker_gpu'+gpu+'.log'),'a');logs.append(log)
        cmd=[sys.executable,'-m','snnbench','worker','--db',a.db,'--out',a.runs,'--data',a.data,'--device','cuda:0']
        if a.analyze:cmd.append('--analyze')
        processes.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    if any(codes):raise SystemExit('Worker failure: '+str(codes))
except KeyboardInterrupt:
    for proc in processes:proc.terminate()
    raise
finally:
    for log in logs:log.close()
