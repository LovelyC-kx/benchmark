"""Same frozen old body/head; actual native route and same-head alternatives."""
import argparse,copy
from pathlib import Path
from dataclasses import replace
import torch
from bridge35 import load_model,fixed_batch
from native_head_audit import audit_model,parameter_hash
from artifacts35 import write_json
ROOT=Path(__file__).resolve().parent

def run(run_directory=None):
    torch.set_num_threads(1);model,cfg,_=load_model(Path(run_directory) if run_directory else ROOT/'evidence/source_smoke_run')
    b=fixed_batch(cfg.data,4,'geometry_fit',8440000,0)
    with torch.no_grad():encoded=model.encode(torch.as_tensor(b['x']))
    before=parameter_hash(model);rows=[]
    for phase in ('output','pre','post'):
        for summary in ('last','mean','sum','exponential'):
            core=copy.deepcopy(model.core);core.cfg=replace(core.cfg,observation=phase,readout=summary)
            r=audit_model(core,encoded);r['actual_trained_configuration']=phase==model.core.cfg.observation and summary==model.core.cfg.readout
            rows.append(r)
    assert parameter_hash(model)==before
    return {'module':'L4.1 frozen native and same-head alternatives','cases':rows,'same_original_parameters':True,
        'scope':'Only post/last is the actual trained configuration. Eleven same-W alternatives audit G/P/R compatibility, not adaptation or trained performance. Trajectory is excluded because the original head has no compatible dimension.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--run-directory');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    write_json(out,run(a.run_directory));print('Frozen native access audit completed')
