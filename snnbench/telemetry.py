"""Per-update evidence with checkpoint-consistent, loss-preserving journal repair."""
import json
import os
from pathlib import Path
import uuid
import numpy as np
import torch
from .utils import jsonable


class UpdateJournal:
    def __init__(self,run,committed_updates,run_id):
        self.path=Path(run)/'updates.jsonl';self.run_id=run_id;self.last=committed_updates
        if self.path.exists():
            offset=0;last=0;orphan=None
            with self.path.open('rb') as f:
                for line in f:
                    try:
                        row=json.loads(line);u=row['update'];valid=row['run_id']==run_id and u==last+1 and u<=committed_updates
                    except (ValueError,KeyError,TypeError):valid=False
                    if valid and orphan is None:offset+=len(line);last=u
                    else:
                        if orphan is None:orphan=self.path.with_name('updates_uncommitted_'+uuid.uuid4().hex[:10]+'.jsonl').open('wb')
                        orphan.write(line)
            if orphan is not None:
                orphan.flush();os.fsync(orphan.fileno());orphan.close()
                with self.path.open('r+b') as f:f.truncate(offset)
            if last!=committed_updates:raise ValueError('Committed checkpoint lacks its per-update journal prefix')
        elif committed_updates:raise ValueError('Per-update journal is missing for a noninitial checkpoint')
    def append(self,row):
        if row['update']!=self.last+1:raise ValueError('Journal update order')
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(jsonable(dict(row,run_id=self.run_id)),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');f.flush()
        self.last=row['update']
    def sync(self):
        if self.path.exists():
            with self.path.open('rb') as f:os.fsync(f.fileno())


@torch.no_grad()
def gradient_statistics(model):
    result={}
    for name,p in model.named_parameters():
        if p.grad is None:result[name]={'has_gradient':False};continue
        g=p.grad.detach().float();n=g.numel()
        result[name]={'has_gradient':True,'numel':n,'l2':float(g.norm().cpu()),'rms':float(g.square().mean().sqrt().cpu()),
                      'max_abs':float(g.abs().max().cpu()),'zero_fraction':float((g==0).float().mean().cpu())}
    return result


def torch_safe(value):
    if isinstance(value,np.ndarray):return torch.from_numpy(np.array(value,copy=True)) if value.dtype.kind in 'biufc' else value.tolist()
    if isinstance(value,dict):return {k:torch_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [torch_safe(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value
