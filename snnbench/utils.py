"""Strict identities and atomic outputs used by all execution stages."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import contextlib
import numpy as np


def jsonable(value):
    if is_dataclass(value):return jsonable(asdict(value))
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def canonical(value):
    return json.dumps(jsonable(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path, block=2**20):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(block),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            f.write(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_torch(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent);os.close(fd)
    try:
        torch.save(value,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def parameter_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def seeded(seed,*keys):
    # Stable independent streams; never use Python's randomized hash().
    key=int.from_bytes(hashlib.sha256(canonical(keys).encode()).digest()[:8],'little')
    return np.random.default_rng(np.random.SeedSequence([int(seed),key&0xffffffff,key>>32]))


@contextlib.contextmanager
def exclusive_file(path):
    import socket
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        owner=json.loads(path.read_text())
        if owner.get('host')==socket.gethostname():
            try:os.kill(owner['pid'],0)
            except ProcessLookupError:path.unlink()
    try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:raise RuntimeError('Operation already leased: '+str(path))
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'host':socket.gethostname()},f)
    try:yield
    finally:path.unlink(missing_ok=True)
