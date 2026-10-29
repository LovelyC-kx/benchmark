"""Real-source adapters. Derived protocols are named; authors' splits are not invented.

All normalization is fitted on the actual model-training role. Forecast windows
never cross a role/block boundary; physical measurement time is not simulation T.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import struct
import wave
from pathlib import Path
import numpy as np
from .prepare import write_records,partition_groups,TRAIN_ROLES,TEST_ROLES
from .utils import file_hash


def _idx(path,dimensions):
    path=Path(path)
    with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as f:raw=f.read()
    if len(raw)<4 or raw[:2]!=b'\x00\x00' or raw[2]!=8 or raw[3]!=dimensions:raise ValueError('Expected uint8 IDX of declared rank')
    shape=struct.unpack('>'+'I'*dimensions,raw[4:4+4*dimensions]);data=np.frombuffer(raw,offset=4+4*dimensions,dtype=np.uint8)
    if data.size!=int(np.prod(shape)):raise ValueError('IDX payload size mismatch')
    return data.reshape(shape)


def prepare_mnist(root,out,mode='sequential',permutation_seed=20261029,steps=16,seed=20261029):
    """Real images; scan/permutation/repeat are explicitly derived, nonphysical time."""
    root=Path(root);data={};records=[];raw=[]
    for split,prefix in (('train','train'),('test','t10k')):
        def find(suffix):
            name=prefix+'-'+suffix;paths=[root/name,root/(name+'.gz')]
            matches=[p for p in paths if p.is_file()]
            if not matches:raise FileNotFoundError(str(paths))
            return matches[0]
        image=find('images-idx3-ubyte');label=find('labels-idx1-ubyte');x=_idx(image,3);y=_idx(label,1)
        if x.shape[1:]!=(28,28) or len(x)!=len(y):raise ValueError('MNIST image/label shape')
        data[split]=(x,y);raw.extend({'path':str(p),'sha256':file_hash(p)} for p in (image,label))
        for j,label in enumerate(y):records.append({'official_split':split,'index':j,'label':int(label),'group_id':split+':image:'+str(j),'scene_id':split+':'+str(j)})
    if mode not in ('sequential','permuted','repeat') or steps<1:raise ValueError('MNIST mode/steps')
    permutation=np.random.default_rng(permutation_seed).permutation(784) if mode=='permuted' else np.arange(784)
    def loader(r):
        image=data[r['official_split']][0][r['index']].astype('float32')/255
        x=np.broadcast_to(image[None,None],(steps,1,28,28)).copy() if mode=='repeat' else image.reshape(-1)[permutation,None]
        return {'x':x,'y':r['label']}
    contract={'dataset_id':'D89_'+mode,'name':'MNIST '+mode,'target_mode':'class_index','classes':10,'output_shape':[10],
        'group_unit':'original image','dataset_nature':'real static images with derived sequence','physical_time':False,
        'protocol':'sNNbench derived pixel scan/permutation/static repeat; not natural video',
        'permutation_seed':permutation_seed if mode=='permuted' else None,
        'permutation':permutation.tolist() if mode=='permuted' else None,'raw_sources':raw}
    return write_records(records,loader,out,contract,seed)


def prepare_esc50(root,out,test_fold=5,seed=20261029):
    root=Path(root);records=[]
    if not (root/'meta/esc50.csv').is_file():
        found=list(root.rglob('esc50.csv'))
        if len(found)==1:root=found[0].parent.parent
    if test_fold not in range(1,6):raise ValueError('ESC-50 held-out fold 1..5')
    meta=root/'meta'/'esc50.csv'
    with meta.open(newline='') as f:
        for r in csv.DictReader(f):
            records.append({'official_split':'test' if int(r['fold'])==test_fold else 'train','path':str(root/'audio'/r['filename']),
                'label':int(r['target']),'group_id':'freesound:'+r['src_file'],'scene_id':r['filename']})
    def loader(r):
        with wave.open(r['path'],'rb') as w:
            hz=w.getframerate()
            if hz!=44100 or w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('ESC-50 44.1kHz mono PCM16 required; no implicit resampling')
            audio=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype('float32')/32768
        if len(audio)!=5*hz:raise ValueError('Expected released five-second clip')
        win=1102;hop=441;frames=np.lib.stride_tricks.sliding_window_view(audio,win)[::hop]
        power=np.abs(np.fft.rfft(frames*np.hanning(win),n=2048))**2
        x=np.stack([np.log1p(power[:,idx].mean(-1)) for idx in np.array_split(np.arange(power.shape[-1]),64)],-1).astype('float32')
        return {'x':x,'y':r['label'],'timestamps':(win+hop*np.arange(len(x)))/hz}
    c={'dataset_id':'D41','name':'ESC-50 source-group isolated fold '+str(test_fold),'target_mode':'class_index','classes':50,'output_shape':[50],
       'group_unit':'original Freesound recording','held_out_official_fold':test_fold,'dataset_nature':'real environmental sound',
       'protocol':'official fold held out; training fold groups subdivided into eight research roles; derived research score',
       'feature':'causal 1102-sample FFT / 441-sample hop, 64 linear frequency bands log1p; NOT Mel',
       'raw_sources':[{'path':str(meta),'sha256':file_hash(meta)}],
       'audio_inventory_sha256':hashlib.sha256('\n'.join(sorted(r['scene_id']+':'+file_hash(r['path']) for r in records)).encode()).hexdigest()}
    return write_records(records,loader,out,c,seed)


def prepare_forecast_csv(path,out,context=96,horizon=24,stride=24,columns=None,date_column='date',
                         dataset_id='real_forecast',embargo=0,block_size=1024,seed=20261029,role_allocation='proportional'):
    """Purged chronological research protocol, separate from official LTSF scores.

    Ten consecutive roles have fractions 60/5/3/8/3/6/3/4/4/4 percent. Both
    context and targets must lie in a single nonoverlapping block. No target
    from another role can be reintroduced as input. No interpolation/imputation.
    """
    if min(context,horizon,stride,block_size)<1 or embargo<0:raise ValueError('Positive forecast lengths; nonnegative embargo')
    path=Path(path)
    with path.open(newline='') as f:
        reader=csv.DictReader(f);names=reader.fieldnames or []
        if date_column not in names:raise ValueError('Explicit timestamp column required')
        columns=list(columns) if columns else [n for n in names if n!=date_column]
        if not columns or len(set(columns))!=len(columns) or set(columns)-set(names):raise ValueError('Distinct numeric value columns required')
        rows=list(reader)
    values=np.array([[float(r[n]) for n in columns] for r in rows],dtype='float64')
    dates=np.array([r[date_column] for r in rows],dtype='datetime64[ns]');ns=dates.astype('int64')
    if np.isnat(dates).any() or not np.isfinite(values).all() or (np.diff(ns)<=0).any():raise ValueError('Finite values and strictly increasing timestamps; repair missing data explicitly upstream')
    fractions=np.array([.60,.05,.03,.08,.03,.06,.03,.04,.04,.04]);roles=TRAIN_ROLES+TEST_ROLES
    if role_allocation=='proportional':
        boundaries=np.r_[0,np.floor(np.cumsum(fractions)*len(rows)).astype(int)];boundaries[-1]=len(rows)
    elif role_allocation=='minimum_first':
        # Explicit alternative for long horizons: at least two distinct windows
        # per role, then allocate remaining rows by the stated fractions.
        minimum=context+horizon+stride;reserved=10*minimum+9*embargo;remaining=len(rows)-reserved
        if remaining<0:raise ValueError('Series cannot supply ten isolated roles with two windows each and embargo')
        counts=np.full(10,minimum)+np.floor(remaining*fractions).astype(int);counts[1:]+=embargo;counts[0]+=len(rows)-int(counts.sum())
        boundaries=np.r_[0,np.cumsum(counts)]
    else:raise ValueError('Forecast role allocation proportional or minimum_first')
    train_end=int(boundaries[1]);mean=values[:train_end].mean(0);std=values[:train_end].std(0);std=np.where(std<1e-6,1.,std)
    standardized=((values-mean)/std).astype('float32');records=[];mapping={};ranges={}
    for i,role in enumerate(roles):
        lo=int(boundaries[i])+(embargo if i else 0);hi=int(boundaries[i+1]);length=hi-lo;window=context+horizon
        if length<window:raise ValueError(f'{role}: {length} rows cannot supply context+horizon={window}; reduce lengths or use a larger series')
        # Balanced blocks avoid discarding a short final tail. Groups are statistical
        # time blocks, not an assertion that the underlying process is independent.
        blocks=max(1,length//max(block_size,window));edges=np.linspace(lo,hi,blocks+1,dtype=int);ranges[role]=[lo,hi]
        for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            group=f'{dataset_id}:{role}:block:{j}';mapping[group]=role
            for start in range(int(a),int(b)-window+1,stride):
                records.append({'official_split':'test' if role in TEST_ROLES else 'train','start':start,
                    'group_id':group,'scene_id':f'{dataset_id}:{start}:{context}:{horizon}'})
    def loader(r):
        s=r['start'];deadline=s+context-1;ix=np.arange(s,s+context);iy=np.arange(s+context,s+context+horizon)
        # Relative seconds avoid float32 loss of precision at Unix timestamps.
        times=(ns[ix]-ns[s])/1e9;targets=(ns[iy]-ns[s])/1e9
        return {'x':standardized[ix],'y':standardized[iy],'timestamps':times,'availability_times':times,
                'target_deadline':float((ns[deadline]-ns[s])/1e9),'target_timestamps':targets,'input_indices':ix,'target_indices':iy}
    c={'dataset_id':dataset_id,'name':dataset_id+' purged blocked forecasting','target_mode':'continuous','classes':None,
       'output_shape':[horizon,len(columns)],'objective':'regression','dataset_nature':'real measured time series',
       'group_unit':'nonoverlapping contiguous time block','serial_dependence_claim':'blocks are not guaranteed independent; report counts and sensitivity to block length',
       'native_benchmark_score':False,'protocol':'chronological_ten_role_no_crossing_v1','role_row_ranges':ranges,
       'role_allocation':role_allocation,'allocation_weight_fractions':fractions.tolist(),
       'actual_role_fractions':{r:(hi-lo)/len(rows) for r,(lo,hi) in ranges.items()},
       'context':context,'horizon':horizon,'stride':stride,'embargo_rows':embargo,'block_size_requested':block_size,
       'columns':columns,'date_column':date_column,'origin_timestamp':str(dates[0]),'sampling_interval_ns':int(np.diff(ns)[0]) if np.all(np.diff(ns)==np.diff(ns)[0]) else None,
       'normalization':{'fit_role':'train','fit_raw_rows':[0,train_end],'mean':mean.tolist(),'std':std.tolist(),'inverse_metric_required':True},
       'raw_sources':[{'path':str(path),'sha256':file_hash(path)}]}
    return write_records(records,loader,out,c,seed,group_assignment=mapping)


def _subject_records(data,labels,seed):
    subjects=sorted(data,key=lambda s:hashlib.sha256((str(seed)+'|'+s).encode()).hexdigest())
    ntest=max(2,int(round(.2*len(subjects))))
    if len(subjects)-ntest<8:raise ValueError('At least eight fitting and two held-out subjects required')
    held=set(subjects[-ntest:]);records=[]
    for subject in subjects:
        for i,label in enumerate(labels[subject]):records.append({'official_split':'test' if subject in held else 'train','subject':subject,
            'index':i,'label':int(label),'group_id':'subject:'+subject,'scene_id':subject+':'+str(i)})
    return records


def _normalize_subject_train(data,records,seed):
    mapping=partition_groups(records,seed);subjects={r['subject'] for r in records if mapping[r['group_id']]=='train'}
    n=sum(data[s].reshape(-1,data[s].shape[-1]).shape[0] for s in subjects)
    sums=sum(data[s].reshape(-1,data[s].shape[-1]).astype('float64').sum(0) for s in subjects)
    squares=sum(np.square(data[s].reshape(-1,data[s].shape[-1]).astype('float64')).sum(0) for s in subjects)
    mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean**2,0));std=np.where(std<1e-6,1.,std)
    return mapping,mean,std,sorted(subjects)


def prepare_openbmi(root,out,sampling_hz,downsample=10,seed=20261029):
    """Author-exported per-subject data_N.mat/label_N.mat, before pooled split.

    Explicit input sampling rate is required: different author exports can have
    different rates. The provided pooled x_train.npy has lost subject identity
    and cannot support this subject-disjoint protocol.
    """
    from scipy.io import loadmat
    if sampling_hz<=0 or downsample<1:raise ValueError('Known sampling rate/downsample factor required')
    root=Path(root);data={};labels={};raw=[]
    for path in sorted(root.rglob('data_*.mat')):
        sid=path.stem[5:];yp=path.with_name('label_'+sid+'.mat')
        if not yp.is_file():raise FileNotFoundError(yp)
        x=np.asarray(loadmat(path)[path.stem],'float32');y=np.asarray(loadmat(yp)[yp.stem]).reshape(-1).astype('int64')-1
        if x.ndim!=3 or x.shape[0]!=len(y) or x.shape[1]%downsample or set(np.unique(y))-set((0,1)):raise ValueError('OpenBMI N,L,C / 1-based binary labels / divisible time length required')
        if not np.isfinite(x).all():raise ValueError('Nonfinite EEG')
        data[sid]=x.reshape(len(x),x.shape[1]//downsample,downsample,x.shape[2]).mean(2);labels[sid]=y
        raw.extend({'path':str(p),'sha256':file_hash(p)} for p in (path,yp))
    records=_subject_records(data,labels,seed);mapping,mean,std,subjects=_normalize_subject_train(data,records,seed)
    def loader(r):
        x=data[r['subject']][r['index']];times=(np.arange(len(x))+1)*downsample/sampling_hz
        return {'x':(x-mean)/std,'y':r['label'],'timestamps':times}
    c={'dataset_id':'D64','name':'OpenBMI MI subject-disjoint','target_mode':'class_index','classes':2,'output_shape':[2],
       'dataset_nature':'real EEG motor imagery','group_unit':'subject','native_benchmark_score':False,
       'protocol':'subject split before channel normalization; differs from NSA pooled trial split',
       'sampling_hz_input':sampling_hz,'downsample':'nonoverlapping causal block average of '+str(downsample)+' observations, timestamp at right boundary',
       'normalization':{'fit_role':'train','fit_subjects':subjects,'mean':mean.tolist(),'std':std.tolist()},'raw_sources':raw}
    return write_records(records,loader,out,c,seed,group_assignment=mapping)


def prepare_wisdm(root,out,window=200,stride=100,device='watch',sensor='gyro',seed=20261029):
    """WISDM2019 raw text: windows remain inside one subject/activity/time run."""
    if min(window,stride)<1 or device not in ('watch','phone') or sensor not in ('gyro','accel'):raise ValueError('WISDM window/device/sensor')
    root=Path(root);candidates=[p for p in root.rglob('*.txt') if p.parent.name==sensor and p.parent.parent.name==device]
    activities='ABCDEFGHIJKL MOPQRS'.replace(' ','');data={};labels={};times={};raw=[]
    for p in sorted(candidates):
        with p.open() as f:rows=[line.strip().rstrip(';').split(',') for line in f if line.strip()]
        if any(len(r)!=6 or r[1] not in activities for r in rows):raise ValueError('WISDM ID,activity,timestamp,x,y,z format required')
        if not rows:continue
        sid=rows[0][0]
        if sid in data or any(r[0]!=sid for r in rows):raise ValueError('Exactly one source file per subject/device/sensor required')
        x=np.array([[float(v) for v in r[3:6]] for r in rows],dtype='float32');stamp=np.array([int(r[2]) for r in rows],dtype='int64');y=np.array([activities.index(r[1]) for r in rows])
        if not np.isfinite(x).all():raise ValueError('Nonfinite WISDM measurement')
        boundaries=np.r_[0,np.flatnonzero((y[1:]!=y[:-1])|(np.diff(stamp)<=0)|(np.diff(stamp)>1_000_000_000))+1,len(x)]
        windows=[];ys=[];ts=[]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            for start in range(int(a),int(b)-window+1,stride):
                windows.append(x[start:start+window]);ys.append(y[start]);ts.append((stamp[start:start+window]-stamp[start])/1e9)
        if windows:data[sid]=np.stack(windows);labels[sid]=np.array(ys);times[sid]=np.stack(ts)
        raw.append({'path':str(p),'sha256':file_hash(p)})
    records=_subject_records(data,labels,seed);mapping,mean,std,subjects=_normalize_subject_train(data,records,seed)
    def loader(r):return {'x':(data[r['subject']][r['index']]-mean)/std,'y':r['label'],'timestamps':times[r['subject']][r['index']]}
    c={'dataset_id':'D47','name':'WISDM2019 '+device+' '+sensor+' subject-disjoint','target_mode':'class_index','classes':18,'output_shape':[18],
       'dataset_nature':'real wearable inertial','group_unit':'subject','native_benchmark_score':False,
       'protocol':'subject split before normalization; windows cannot cross activity, subject, nonmonotone timestamp or >1s acquisition gap',
       'window':window,'stride':stride,'timestamp_unit_input':'nanoseconds','normalization':{'fit_role':'train','fit_subjects':subjects,'mean':mean.tolist(),'std':std.tolist()},
       'normalization_weighting':'training windows including their within-role overlap; no other-role data','raw_sources':raw}
    return write_records(records,loader,out,c,seed,group_assignment=mapping)
