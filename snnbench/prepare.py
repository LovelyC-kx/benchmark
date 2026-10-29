"""Raw adapters and a common group-isolated, checksummed NPY interchange.

No adapter downloads data. Official test sources are never mixed into fitting.
Rows sharing a subject/video/ROI group are assigned together before windowing.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import wave
import numpy as np
from .utils import atomic_json,digest,file_hash,seeded

TRAIN_ROLES=('train','val','calibration','probe_train','probe_val','verification_train','verification_val','development')
TEST_ROLES=('confirmation','audit')


def partition_groups(records,seed=20261029):
    """Official val, if available, supplies val+calibration; test supplies holdouts.

    Uses group ranks, not per-window random splits. Insufficient independent
    groups is an error rather than recycling one subject into multiple roles.
    """
    official={};mapping={};has_val=any(r['official_split']=='val' for r in records)
    for r in records:
        group=str(r['group_id']);split=r['official_split']
        if split not in ('train','val','test'):raise ValueError('Official split train/val/test required')
        if group in official and official[group]!=split:raise ValueError('Source group crosses official splits: '+group)
        official[group]=split
    for split in ('train','val','test'):
        groups=[g for g,s in official.items() if s==split]
        if not groups:continue
        roles=TEST_ROLES if split=='test' else ('val','calibration') if split=='val' else tuple(r for r in TRAIN_ROLES if not has_val or r not in ('val','calibration'))
        if len(groups)<len(roles):raise ValueError(f'{split}: {len(groups)} independent groups cannot supply {len(roles)} disjoint analysis roles')
        groups=sorted(groups,key=lambda g:hashlib.sha256((str(seed)+'|'+g).encode()).hexdigest())
        if split=='train':
            # Reserve one group per small role, then allocate 60% of the remainder to model fitting.
            remaining=len(groups)-len(roles);weights=np.array([.6 if r=='train' else .4/(len(roles)-1) for r in roles]);counts=1+np.floor(remaining*weights).astype(int)
            counts[0]+=len(groups)-int(counts.sum())
        else:
            counts=np.full(len(roles),len(groups)//len(roles));counts[:len(groups)%len(roles)]+=1
        start=0
        for role,count in zip(roles,counts):
            for group in groups[start:start+int(count)]:mapping[group]=role
            start+=int(count)
    if set(mapping.values())!=set(TRAIN_ROLES+TEST_ROLES):raise ValueError('All ten disjoint data roles are required')
    return mapping


def write_records(records,loader,out,contract,seed=20261029):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a fresh canonical output directory: '+str(out))
    temp=out.with_name(out.name+'.preparing-'+str(os.getpid()));temp.mkdir(parents=True)
    mapping=partition_groups(records,seed);byrole=defaultdict(list)
    if len({str(r['scene_id']) for r in records})!=len(records):raise ValueError('Unique original-window scene IDs required')
    if any(len(str(r[k]))>256 for r in records for k in ('scene_id','group_id')):raise ValueError('Scene/group IDs exceed canonical Unicode storage limit')
    for r in records:byrole[mapping[str(r['group_id'])]].append(r)
    manifest={'schema':'snnbench_canonical_v2','contract':dict(contract,model_fields=['x'],independent_factors=False),
              'split_seed':seed,'group_assignment':mapping,'splits':{},'raw_sources':contract.get('raw_sources',[])}
    expected=None
    try:
        for role,rows in byrole.items():
            rows=sorted(rows,key=lambda r:hashlib.sha256((str(seed)+'|'+str(r['scene_id'])).encode()).hexdigest())
            folder=temp/role;folder.mkdir();maps={};N=len(rows)
            for j,r in enumerate(rows):
                values=loader(r);x=np.asarray(values['x'],'float32');y=np.asarray(values['y'],'int64' if contract['target_mode']=='class_index' else 'float32')
                if contract['target_mode']=='class_index' and (y.ndim!=0 or not 0<=int(y)<contract['classes']):raise ValueError('Scalar label outside declared classes')
                if x.ndim not in (2,4) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite T,F or T,C,H,W and native target required')
                shape=(x.shape,y.shape)
                if expected is None:expected=shape
                if shape!=expected:raise ValueError('Canonical shapes differ; explicitly resize/pad in the raw adapter')
                times=np.asarray(values.get('timestamps',np.arange(len(x))),'float32');available=np.asarray(values.get('availability_times',times),'float32')
                if times.shape!=(len(x),) or available.shape!=times.shape or not np.isfinite(times).all() or not np.isfinite(available).all() or (np.diff(times)<0).any():raise ValueError('Finite nondecreasing observation times required')
                if (available<times-1e-6).any():raise ValueError('Observation cannot be available before measurement time')
                if 'target_deadline' in values and (available>values['target_deadline']+1e-6).any():raise ValueError('Future/unavailable input at the target deadline')
                target=np.eye(contract['classes'],dtype='float32')[int(y)] if contract['target_mode']=='class_index' else y.reshape(-1)
                fields={'x':x,'y':y,'targets':target,'loss_mask':np.asarray(values.get('loss_mask',np.ones_like(y)),'float32'),
                    'scene_ids':np.array(j,'int64'),'group_ids':np.array(str(r['group_id']),dtype='<U256'),
                    'source_ids':np.array(str(r['scene_id']),dtype='<U256'),'timestamps':times,'availability_times':available,
                    'source_times':np.array([len(x)-1],'int64'),'source_present':np.ones(1,bool)}
                fields.update({k:np.asarray(v) for k,v in values.items() if k.startswith('region_')})
                for key,value in fields.items():
                    if key not in maps:maps[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype=value.dtype,shape=(N,*value.shape))
                    if maps[key].shape[1:]!=value.shape:raise ValueError('Inconsistent field shape: '+key)
                    maps[key][j]=value
            for a in maps.values():a.flush()
            maps.clear()
            manifest['splits'][role]={'directory':role,'samples':N,'groups':len(set(str(r['group_id']) for r in rows)),
                'files':{p.name:{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in folder.glob('*.npy')}}
        c=manifest['contract'];c.update(source_names=['native_target'],source_dims=[contract['classes'] if c['target_mode']=='class_index' else int(np.prod(expected[1]))],
            requested_sources=['native_target'],input_shape=list(expected[0][1:]),steps=expected[0][0],
            target_semantics='one-hot native class label' if c['target_mode']=='class_index' else 'native continuous target flattened for probes')
        atomic_json(temp/'dataset.json',manifest);os.replace(temp,out)
    except BaseException:shutil.rmtree(temp,ignore_errors=True);raise
    return {'root':str(out.resolve()),'manifest_sha256':file_hash(out/'dataset.json'),'samples':{k:v['samples'] for k,v in manifest['splits'].items()}}


def verify_dataset(root):
    root=Path(root);manifest=json.loads((root/'dataset.json').read_text());seen={};files=0
    for split,entry in manifest['splits'].items():
        for name,info in entry['files'].items():
            p=root/entry['directory']/name
            if p.stat().st_size!=info['bytes'] or file_hash(p)!=info['sha256']:raise ValueError('Prepared file mismatch: '+str(p))
            files+=1
        groups=set(np.load(root/entry['directory']/'group_ids.npy',allow_pickle=False).tolist())
        for g in groups:
            if g in seen:raise ValueError('Group overlaps '+seen[g]+' and '+split)
            seen[g]=split
    receipt={'status':'verified','manifest_sha256':file_hash(root/'dataset.json'),'files':files,'groups':len(seen)}
    atomic_json(root/'verification.json',receipt);return receipt


def prepare_har(root,out,seed=20261029):
    root=Path(root)
    if (root/'UCI HAR Dataset').is_dir():root=root/'UCI HAR Dataset'
    signals=['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y','body_gyro_z','total_acc_x','total_acc_y','total_acc_z']
    data={};records=[];raw=[]
    for split in ('train','test'):
        xs=[]
        for signal in signals:
            p=root/split/'Inertial Signals'/f'{signal}_{split}.txt';xs.append(np.loadtxt(p,dtype='float32'));raw.append({'path':str(p),'sha256':file_hash(p)})
        data[split]=np.stack(xs,-1);y=np.loadtxt(root/split/f'y_{split}.txt',dtype='int64')-1;subjects=np.loadtxt(root/split/f'subject_{split}.txt',dtype='int64')
        for j in range(len(y)):records.append({'official_split':split,'index':j,'label':int(y[j]),'group_id':'subject:'+str(subjects[j]),'scene_id':split+':'+str(j)})
    def loader(r):return {'x':data[r['official_split']][r['index']],'y':r['label'],'timestamps':np.arange(128)/50}
    contract={'dataset_id':'D46','name':'UCI HAR raw inertial','target_mode':'class_index','classes':6,'output_shape':[6],
        'group_unit':'subject','sampling_hz':50,'channels':signals,'raw_sources':raw,'normalization':'as released; no 561-feature aggregate input'}
    return write_records(records,loader,out,contract,seed)


def prepare_spikes(files,out,T=100,window=1.4,channels=700,seed=20261029,dataset='SHD'):
    import h5py
    handles={k:h5py.File(path,'r') for k,path in files.items()};records=[]
    try:
        classes=20 if dataset.upper()=='SHD' else 35
        for split,h in handles.items():
            for j,label in enumerate(h['labels']):records.append({'official_split':split,'index':j,'label':int(label),'group_id':split+':recording:'+str(j),'scene_id':split+':'+str(j)})
        def loader(r):
            h=handles[r['official_split']];times=np.asarray(h['spikes/times'][r['index']],float);units=np.asarray(h['spikes/units'][r['index']],int)
            if (units<0).any() or (units>=channels).any() or not np.isfinite(times).all():raise ValueError('Invalid spike unit/time')
            valid=(times>=0)&(times<window);idx=np.floor(times[valid]/window*T).astype(int);x=np.zeros((T,channels),'float32');np.add.at(x,(idx,units[valid]),1)
            # Count bins are available at their RIGHT boundary, not their opening time.
            return {'x':x,'y':r['label'],'timestamps':(np.arange(T)+1)*window/T}
        contract={'dataset_id':'D31' if dataset.upper()=='SHD' else 'D32','name':dataset,'target_mode':'class_index','classes':classes,'output_shape':[classes],
            'group_unit':'original recording','speaker_disjoint_claim':False,'physical_window_seconds':window,'time_bins':T,'representation':'event count per bin/unit; right-boundary availability',
            'raw_sources':[{'path':str(p),'sha256':file_hash(p)} for p in files.values()]}
        return write_records(records,loader,out,contract,seed)
    finally:
        for h in handles.values():h.close()


def prepare_gsc(root,out,seed=20261029):
    root=Path(root);validation=set((root/'validation_list.txt').read_text().splitlines());testing=set((root/'testing_list.txt').read_text().splitlines())
    labels=sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('_'));records=[]
    if len(labels)!=35:raise ValueError('This adapter freezes the 35-word Speech Commands v2 protocol; different taxonomy needs a new version')
    for label in labels:
        for p in sorted((root/label).glob('*.wav')):
            rel=p.relative_to(root).as_posix();split='val' if rel in validation else 'test' if rel in testing else 'train';speaker=p.stem.split('_nohash_')[0]
            records.append({'official_split':split,'path':str(p),'label':labels.index(label),'group_id':'speaker:'+speaker,'scene_id':rel})
    def loader(r):
        with wave.open(r['path'],'rb') as w:
            if w.getframerate()!=16000 or w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('16kHz mono PCM16 required')
            audio=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype('float32')/32768
        audio=np.pad(audio[:16000],(0,max(0,16000-len(audio))));frames=np.lib.stride_tricks.sliding_window_view(audio,400)[::160]
        power=np.abs(np.fft.rfft(frames*np.hanning(400),n=512))**2
        bands=np.array_split(np.arange(power.shape[-1]),40);x=np.stack([np.log1p(power[:,b].mean(-1)) for b in bands],-1).astype('float32')
        return {'x':x,'y':r['label'],'timestamps':(400+160*np.arange(len(x)))/16000}
    contract={'dataset_id':'D33','name':'Speech Commands v2 35 words','target_mode':'class_index','classes':35,'output_shape':[35],'label_names':labels,
        'group_unit':'speaker','feature':'causal 25ms FFT power, 10ms hop, 40 linear-frequency bands, log1p; NOT Mel',
        'padding':'right zero pad/crop to one second','raw_list_sha256':digest([(r['scene_id'],Path(r['path']).stat().st_size) for r in records])}
    return write_records(records,loader,out,contract,seed)


def prepare_gesture(root,out,T=100,window=3.,size=32,seed=20261029):
    """Tonic presegmented NPY archives, with original subject groups retained."""
    root=Path(root);records=[]
    for official,folder in (('train','ibmGestureTrain'),('test','ibmGestureTest')):
        candidates=list(root.rglob(folder))
        if len(candidates)!=1:raise ValueError('Locate one extracted '+folder+' directory')
        for p in sorted(candidates[0].rglob('*.npy')):
            rel=p.relative_to(candidates[0]);user=rel.parts[0].split('_',1)[0]
            if not user.startswith('user'):raise ValueError('Missing original user directory')
            label=int(p.stem)
            if not 0<=label<11:raise ValueError('DVS Gesture class must be 0..10')
            records.append({'official_split':official,'path':str(p),'label':label,'group_id':user,'scene_id':official+':'+str(rel)})
    def loader(r):
        events=np.load(r['path'],allow_pickle=False)
        if events.ndim!=2 or events.shape[1]!=4 or not np.isfinite(events).all():raise ValueError('Tonic Nx4 x,y,p,t in milliseconds expected')
        xx,yy,pp,ms=events.T
        if (xx<0).any() or (xx>=128).any() or (yy<0).any() or (yy>=128).any() or not np.isin(pp,[0,1]).all():raise ValueError('Event coordinate/polarity bounds')
        seconds=(ms-ms.min())/1000;valid=(seconds>=0)&(seconds<window);time=np.floor(seconds[valid]/window*T).astype(int)
        x=np.zeros((T,2,size,size),'float32')
        np.add.at(x,(time,pp[valid].astype(int),(yy[valid]*size//128).astype(int),(xx[valid]*size//128).astype(int)),1)
        return {'x':x,'y':r['label'],'timestamps':(np.arange(T)+1)*window/T}
    contract={'dataset_id':'D35','name':'DVS Gesture Tonic presegmented','target_mode':'class_index','classes':11,'output_shape':[11],
        'group_unit':'original subject','temporal_precision':'milliseconds in source','physical_window_seconds':window,'window_rule':'fixed prefix relative to first event; longer gestures are cropped',
        'spatial_count_binning':[size,size],'official_raw_ibm_reproduction':False,
        'raw_list_sha256':digest([(r['scene_id'],Path(r['path']).stat().st_size) for r in records])}
    return write_records(records,loader,out,contract,seed)


def prepare_manifest(path,out,objective='classification',classes=None,seed=20261029):
    """Explicit TF/TCHW NPY/NPZ rows for video, HAR, wireless, or restoration.

    Each row: official_split, group_id, scene_id, input {path,key?}, layout,
    label OR target {path,key?}, timestamps, availability_times, target_deadline.
    Reconstruction additionally requires target_is_verified_clean=true.
    """
    path=Path(path);payload=json.loads(path.read_text());records=payload['records'];base=path.parent
    if objective=='classification' and (classes is None or classes<2):raise ValueError('Declare class count')
    def load(spec):
        p=Path(spec['path']);p=p if p.is_absolute() else base/p
        if p.suffix=='.npy':a=np.load(p,allow_pickle=False)
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:a=z[spec['key']]
        else:raise ValueError('Convert native MAT/HDF5/GeoTIFF/event containers with the official source loader, then export NPY/NPZ explicitly')
        if 'transpose' in spec:a=a.transpose(spec['transpose'])
        if np.iscomplexobj(a):raise ValueError('Complex CSI must be explicitly encoded before canonical export')
        return np.asarray(a,'float32')/spec.get('divisor',1.)
    def loader(r):
        x=load(r['input']);layout=r['layout']
        if layout not in ('TF','TCHW') or x.ndim!=len(layout):raise ValueError('Explicit TF or TCHW required')
        if 'target_deadline' not in r or 'timestamps' not in r or 'availability_times' not in r:raise ValueError('Causal manifest requires timestamps, actual availability, and target deadline')
        if objective=='reconstruction' and r.get('target_is_verified_clean') is not True:raise ValueError('Unverified clean target cannot define restoration risk')
        y=np.array(r['label'],'int64') if objective=='classification' else load(r['target'])
        result={'x':x,'y':y,'timestamps':r['timestamps'],'availability_times':r['availability_times'],'target_deadline':r['target_deadline']}
        for name,spec in r.get('regions',{}).items():result['region_'+name]=load(spec).astype(bool)
        return result
    first=loader(records[0]);contract={'dataset_id':payload['dataset_id'],'name':payload.get('name',payload['dataset_id']),
        'target_mode':'class_index' if objective=='classification' else 'continuous','classes':classes,
        'output_shape':[classes] if objective=='classification' else list(first['y'].shape),
        'group_unit':payload['group_unit'],'raw_sources':[{'path':str(path),'sha256':file_hash(path)}],
        'native_raw_adapter':False,'causal_current_target':True}
    return write_records(records,loader,out,contract,seed)
