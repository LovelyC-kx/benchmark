"""Save/read nested NumPy arrays without object arrays or pickle."""
import json
from pathlib import Path
import numpy as np

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def save_tree(path,value):
    path=Path(path);arrays={}
    def convert(v):
        if isinstance(v,np.ndarray):
            key='array_'+str(len(arrays));arrays[key]=v;return {'__array__':key}
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {str(k):convert(w) for k,w in v.items()}
        if isinstance(v,(list,tuple)):return [convert(w) for w in v]
        return v
    metadata=convert(value);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    write_json(path.with_suffix('.json'),metadata)

def load_tree(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as arrays:
        def restore(v):
            if isinstance(v,dict) and set(v)=={'__array__'}:return arrays[v['__array__']].copy()
            if isinstance(v,dict):return {k:restore(w) for k,w in v.items()}
            if isinstance(v,list):return [restore(w) for w in v]
            return v
        return restore(meta)
