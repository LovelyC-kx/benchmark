"""Validated factorial overlays and preflight checks of concrete run contracts."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from .specs import RunSpec
from .utils import atomic_json,file_hash,digest


def sweep(base_path,grid_path,out):
    base=json.loads(Path(base_path).read_text());grid=json.loads(Path(grid_path).read_text())
    if not grid or any(not isinstance(v,list) or not v for v in grid.values()):raise ValueError('Grid is a nonempty mapping of dotted fields to candidate lists')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);seen=set();blocked=[]
    with out.open('w') as f:
        for values in product(*grid.values()):
            row=deepcopy(base);choice=dict(zip(grid,values))
            for key,value in choice.items():
                section,field=key.split('.',1)
                if section not in ('task','model','train') or field not in row[section]:raise ValueError('Unknown training field; analysis-only choices use explicit analyze calls: '+key)
                row[section][field]=value
            try:spec=RunSpec.from_dict(row)
            except (ValueError,TypeError) as exc:blocked.append({'choices':choice,'reason':str(exc)});continue
            if spec.run_id not in seen:seen.add(spec.run_id);f.write(json.dumps(spec.to_dict(),ensure_ascii=False)+'\n')
    receipt={'runs':len(seen),'blocked':blocked,'base_sha256':file_hash(base_path),'grid_sha256':file_hash(grid_path),'manifest_sha256':file_hash(out)}
    atomic_json(out.with_suffix(out.suffix+'.receipt.json'),receipt);return receipt


def preflight(manifest,data_root='prepared',limit=None):
    import numpy as np
    import torch
    from .data import generate,DataStore
    from .models import create_model
    from .engine import native_losses
    cache={};rows=[];torch.set_num_threads(2)
    for line in Path(manifest).open(encoding='utf-8'):
        spec=RunSpec.from_dict(json.loads(line));task=spec.task
        try:
            batch=DataStore(task,data_root).ensure('train',2) if task.family=='archive' else generate(task,2)
            meta=dict(batch['meta'],input_shape=list(batch['x'].shape[2:]),steps=batch['x'].shape[1])
            key=digest({'shape':meta['input_shape'],'output':meta['output_shape'],'mode':meta['target_mode'],'model':spec.model,'steps':meta['steps'],'objective':task.objective})
            if key not in cache:
                model=create_model(spec,meta);p=model(torch.as_tensor(np.array(batch['x'],copy=True)));loss,_=native_losses(p,batch);loss.mean().backward()
                cache[key]=model.resources()
            rows.append({'run_id':spec.run_id,'status':'preflight_passed','parameters':cache[key]['parameters']})
        except Exception as exc:rows.append({'run_id':spec.run_id,'status':'blocked','reason':type(exc).__name__+': '+str(exc)})
        if limit is not None and len(rows)>=limit:break
    result={'schema':'preflight_v1','checked_rows':len(rows),'distinct_model_shape_checks':len(cache),'blocked':sum(r['status']=='blocked' for r in rows),'rows':rows,
            'scope':'small CPU forward/backward checks; no convergence, real corpus validation, or GPU readiness claim'}
    atomic_json(Path(manifest).with_suffix('.preflight.json'),result);return result


def export_encoder(run,out,checkpoint='best.pt'):
    import torch
    from .engine import load_run
    from .utils import atomic_torch
    spec,model,ck,path=load_run(run,checkpoint)
    if not ck['progress']['updates']:raise ValueError('Encoder export for training comparisons requires a trained checkpoint')
    if not model.encoder.state_dict():raise ValueError('This configuration has no learned frame encoder')
    atomic_torch(out,{'encoder':model.encoder.state_dict(),'source_run_id':spec.run_id,'source_checkpoint_sha256':file_hash(path),
        'source_training_updates':ck['progress']['updates'],'input_shape':ck['contract']['input_shape'],'encoder_config':spec.model.encoder})
    return {'path':str(out),'sha256':file_hash(out),'source_run_id':spec.run_id}
