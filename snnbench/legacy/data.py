"""Data-only handoff: canonical archives and explicit adapters, no auto downloads."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

def load_archive(path,objective='classification'):
    p=Path(path)
    with np.load(p,allow_pickle=False) as a:d={k:a[k] for k in a.files}
    for key in ('x','y'):
        if key not in d:raise ValueError(f'{p}: missing {key}')
    if d['x'].ndim not in (3,5):raise ValueError('x requires N,T,F or N,T,C,H,W')
    if len(d['x'])!=len(d['y']) or not np.isfinite(d['x']).all():raise ValueError('length or finite check')
    if objective=='classification':
        if d['y'].ndim!=1 or not np.issubdtype(d['y'].dtype,np.integer) or np.min(d['y'])<0:
            raise ValueError('classification y must nonnegative integer N labels')
    elif not np.isfinite(d['y']).all():raise ValueError('nonfinite regression target')
    d['x']=d['x'].astype('float32');d.setdefault('target',d['y'])
    d['meta']={'source':str(p),'preprocessed_archive':True};return d

def load_splits(root,objective):
    def one(s):
        archive=Path(root)/f'{s}.npz';directory=Path(root)/s
        if archive.exists() and (directory/'x.npy').exists():raise ValueError('ambiguous NPZ/NPY copies; use a fresh prepared directory')
        return load_archive(archive,objective) if archive.exists() else load_array_directory(directory,objective)
    d={s:one(s) for s in ('train','val','test')}
    if any(a['x'].shape[1:]!=d['train']['x'].shape[1:] for a in d.values()):
        raise ValueError('train/val/test incompatible input shapes')
    grouped=all('group_id' in a for a in d.values())
    if grouped:
        sets={s:set(a['group_id'].astype(str).tolist()) for s,a in d.items()}
        if any(sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('group leakage across splits')
    for a in d.values():a['meta']['group_isolation_checked']=grouped
    return d

def load_array_directory(path,objective):
    """Read-only memmaps for large datasets; training indexes only each batch."""
    p=Path(path)
    d={f.stem:np.load(f,mmap_mode='r',allow_pickle=False) for f in p.glob('*.npy')}
    if 'x' not in d or 'y' not in d:raise ValueError('directory requires x.npy and y.npy')
    if d['x'].ndim not in (3,5) or d['x'].dtype!=np.float32:raise ValueError('memmap x must float32 N,T,F or N,T,C,H,W')
    if len(d['x'])!=len(d['y']):raise ValueError('length mismatch')
    if objective=='classification' and (d['y'].ndim!=1 or not np.issubdtype(d['y'].dtype,np.integer)):
        raise ValueError('classification labels must N integers')
    # Bounded-memory validation. No full-dataset boolean tensor allocation.
    for i in range(0,len(d['x']),32):
        if not np.isfinite(d['x'][i:i+32]).all() or not np.isfinite(d['y'][i:i+32]).all():raise ValueError('nonfinite archive')
        if objective=='classification' and np.any(d['y'][i:i+32]<0):raise ValueError('classification labels must be nonnegative')
    d.setdefault('target',d['y']);d['meta']={'source':str(p),'storage':'read-only NPY memmap','validation':'finite values checked in chunks'}
    return d

def convert_uci_har(root,out):
    """Official inertial signal files, subject-disjoint validation, no feature-axis guessing."""
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    signals=['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z',
             'total_acc_x','total_acc_y','total_acc_z']
    for s in ('train','test'):
        x=np.stack([np.loadtxt(root/s/'Inertial Signals'/f'{sig}_{s}.txt',dtype='float32') for sig in signals],-1)
        y=np.loadtxt(root/s/f'y_{s}.txt',dtype='int64')-1
        group=np.loadtxt(root/s/f'subject_{s}.txt',dtype='int64')
        if s=='train':
            subjects=sorted(np.unique(group));val_subjects=subjects[-max(1,len(subjects)//5):]
            v=np.isin(group,val_subjects)
            for name,mask in [('train',~v),('val',v)]:
                np.savez_compressed(out/f'{name}.npz',x=x[mask],y=y[mask],group_id=group[mask],
                                    timestamps=np.arange(x.shape[1],dtype='float32')/50)
        else:np.savez_compressed(out/'test.npz',x=x,y=y,group_id=group,
                                timestamps=np.arange(x.shape[1],dtype='float32')/50)
    (out/'preprocessing.json').write_text(json.dumps({'source':'UCI HAR inertial signals',
         'axes':'N,T,9','sampling_hz':50,'validation':'last sorted training subjects',
         'normalization':'not fitted here; fit training only'},indent=2))

def bin_events(times,units,T,channels,window_seconds=None,mode='fixed_dt'):
    times=np.asarray(times);units=np.asarray(units,int)
    if np.any(np.diff(times)<0):raise ValueError('sort events before conversion')
    valid=(units>=0)&(units<channels)
    if window_seconds is not None:valid &= (times>=0)&(times<window_seconds)
    times=times[valid];units=units[valid]
    if mode=='fixed_dt':
        if window_seconds is None or window_seconds<=0:raise ValueError('fixed_dt needs physical window')
        idx=np.floor(times/window_seconds*T).astype(int).clip(0,T-1)
        stamps=np.arange(T,dtype='float32')*window_seconds/T
    elif mode=='event_count':
        idx=np.minimum(np.arange(len(times))*T//max(1,len(times)),T-1)
        stamps=np.array([times[idx==t].mean() if (idx==t).any() else np.nan for t in range(T)],dtype='float32')
    else:raise ValueError(mode)
    x=np.zeros((T,channels),'float32');np.add.at(x,(idx,units),1)
    return x,stamps

def convert_shd(files,out,T=100,window_seconds=1.,channels=700,mode='fixed_dt'):
    """HDF5 spike times/units. Existing official split retained; val from train samples.
    Speaker-independent validation needs source speaker metadata and is NOT claimed here.
    """
    import h5py
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for split,path in files.items():
        with h5py.File(path,'r') as f:
            xs=[];ts=[]
            for t,u in zip(f['spikes/times'],f['spikes/units']):
                xx,tt=bin_events(t,u,T,channels,window_seconds,mode);xs.append(xx);ts.append(tt)
            x=np.stack(xs);y=np.asarray(f['labels'],dtype='int64');stamps=np.stack(ts)
        if split=='train':
            ids=np.random.default_rng(2026).permutation(len(y));v=ids[:max(1,len(y)//5)];tr=ids[len(v):]
            for name,ind in [('train',tr),('val',v)]:np.savez_compressed(out/f'{name}.npz',x=x[ind],y=y[ind],timestamps=stamps[ind])
        else:np.savez_compressed(out/f'{split}.npz',x=x,y=y,timestamps=stamps)
    (out/'preprocessing.json').write_text(json.dumps({'mode':mode,'T':T,'window_seconds':window_seconds,
           'speaker_disjoint_validation':False,'event_count_steps_not_uniform_physical_time':mode=='event_count'},indent=2))

def convert_frame_manifest(manifest,out,T=8,size=64):
    """JSON rows: split, frames(list paths), label, group_id, timestamps(optional).
    Deterministic evenly spaced frames, one consistent resize, no label-dependent sampling.
    """
    path=Path(manifest);rows=json.loads(path.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    packs={s:[] for s in ('train','val','test')}
    for row in rows:
        frames=row['frames'];idx=np.rint(np.linspace(0,len(frames)-1,T)).astype(int)
        imgs=[]
        for i in idx:
            p=Path(frames[i]);p=p if p.is_absolute() else path.parent/p
            imgs.append(np.asarray(Image.open(p).convert('RGB').resize((size,size)),dtype='float32').transpose(2,0,1)/255)
        stamps=np.array(row.get('timestamps',list(range(len(frames)))))[idx]
        packs[row['split']].append((np.stack(imgs),row['label'],str(row['group_id']),stamps))
    for s,pack in packs.items():
        if not pack:raise ValueError(f'empty {s}; independent validation required')
        x,y,g,t=zip(*pack);np.savez_compressed(out/f'{s}.npz',x=np.stack(x),y=np.asarray(y,dtype='int64'),group_id=np.asarray(g),timestamps=np.stack(t))
    load_splits(out,'classification')

def convert_tensor_manifest(manifest,out,objective='reconstruction',storage='npz'):
    """Explicit key/axes adapter for NPY/NPZ/MAT/HDF5/TIFF. No inferred CSI/SAR axes.
    JSON rows provide split, input or inputs, target or label, group_id and layout.
    """
    path=Path(manifest);rows=json.loads(path.read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def load(spec):
        if isinstance(spec,str):spec={'path':spec}
        p=Path(spec['path']);p=p if p.is_absolute() else path.parent/p;key=spec.get('key')
        ext='.'+spec['format'].lstrip('.') if spec.get('format') else p.suffix.lower()
        if ext=='.npy':a=np.load(p,allow_pickle=False)
        elif ext=='.npz':
            if key is None:raise ValueError('NPZ key must be explicit')
            with np.load(p,allow_pickle=False) as z:a=z[key]
        elif ext=='.mat':
            from scipy.io import loadmat
            if key is None:raise ValueError('MAT key must be explicit')
            a=loadmat(p)[key]
        elif ext in ('.h5','.hdf5'):
            import h5py
            if key is None:raise ValueError('HDF5 dataset key must be explicit')
            with h5py.File(p,'r') as z:a=z[key][:]
        elif ext in ('.tif','.tiff'):
            import tifffile
            a=tifffile.imread(p)
        else:raise ValueError(f'unsupported tensor format: {ext}')
        a=np.asarray(a)
        if 'transpose' in spec:a=a.transpose(spec['transpose'])
        if np.iscomplexobj(a):
            encoding=spec.get('complex_encoding')
            if encoding=='amplitude':a=np.abs(a)
            elif encoding in ('real_imag','amplitude_phase'):
                if 'complex_axis' not in spec:raise ValueError('complex_axis must be explicit')
                u,v=(a.real,a.imag) if encoding=='real_imag' else (np.abs(a),np.angle(a))
                a=np.concatenate([u,v],axis=spec['complex_axis'])
            else:raise ValueError('complex input requires declared amplitude/real_imag/amplitude_phase encoding')
        return a.astype('float32')/spec.get('divisor',1.)
    if storage not in ('npz','npy'):raise ValueError('storage')
    packs={s:[] for s in ['train','val','test']}
    split_rows={s:[r for r in rows if r['split']==s] for s in packs}
    maps={};indices={s:0 for s in packs}
    for row in rows:
        x=np.stack([load(v) for v in row['inputs']]) if 'inputs' in row else load(row['input'])
        if x.ndim not in (2,4):raise ValueError('per-sequence input requires T,F or T,C,H,W; provide transpose')
        y=row['label'] if objective=='classification' else load(row['target'])
        stamps=np.array(row.get('timestamps',list(range(len(x)))),dtype='float32')
        if len(stamps)!=len(x) or np.any(np.diff(stamps)<0):raise ValueError('timestamps invalid')
        if 'target_timestamp' in row and not row.get('allow_future',False) and np.max(stamps)>row['target_timestamp']:
            raise ValueError('future observation in a declared causal task')
        split=row['split']
        if storage=='npz':packs[split].append((x,y,str(row['group_id']),stamps))
        else:
            if split not in maps:
                folder=out/split;folder.mkdir(parents=True,exist_ok=True);N=len(split_rows[split])
                maps[split]={
                    'x':np.lib.format.open_memmap(folder/'x.npy',mode='w+',dtype='float32',shape=(N,*x.shape)),
                    'y':np.lib.format.open_memmap(folder/'y.npy',mode='w+',dtype='int64' if objective=='classification' else 'float32',shape=(N,*np.asarray(y).shape)),
                    'timestamps':np.lib.format.open_memmap(folder/'timestamps.npy',mode='w+',dtype='float32',shape=(N,len(stamps)))}
                np.save(folder/'group_id.npy',np.asarray([str(r['group_id']) for r in split_rows[split]]))
            j=indices[split]
            for k,v in [('x',x),('y',y),('timestamps',stamps)]:maps[split][k][j]=v
            indices[split]+=1
    if storage=='npz':
        for split,items in packs.items():
            if not items:raise ValueError('all three independent splits required')
            x,y,g,t=zip(*items);np.savez_compressed(out/f'{split}.npz',x=np.stack(x),
                y=np.asarray(y,dtype='int64' if objective=='classification' else 'float32'),group_id=np.asarray(g),timestamps=np.stack(t))
    else:
        if set(maps)!=set(packs):raise ValueError('all three independent splits required')
        for split,d in maps.items():
            for a in d.values():a.flush()
    (out/'preprocessing_manifest.json').write_text(json.dumps({'source':str(path),'storage':storage,'objective':objective,'rows':rows},indent=2))
    load_splits(out,objective)
