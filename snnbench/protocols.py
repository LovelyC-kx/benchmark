"""Versioned G01--G09 controlled protocols, derived from the archived task plan.

Generators are ours. Reference inspiration is not claimed as reproduction or
dataset novelty. All returned truth, schedules, and pairing IDs are analysis
fields. Only the explicit x array enters the model.
"""
from dataclasses import replace
import numpy as np
from .utils import seeded

PARAMS={
 'G01':{'items','age','writes','key_capacity'},
 'G02':{'items','age','versions','invalidate_probability','query','key_capacity','version_slots'},
 'G03':{'items'},'G04':{'items','physical_time','show_clock','interval_boundary'},
 'G05':{'measurements','acquisition'},
 'G06':{'target_frequency','distractor_frequency','waveform_target'},
 'G07':set(),'G08':set(),
 'G09':{'duration','tau','measurements','acquisition'}}


def validate_protocol(t):
    if t.name not in PARAMS or set(t.params)-PARAMS[t.name]:raise ValueError('Unknown protocol or parameters')
    if t.name in ('G07','G08'):
        if t.T!=6 or t.image_size<8 or t.image_size%2 or t.objective not in ('classification','reconstruction'):raise ValueError('Spatial G07/G08: T6, even H>=8, class/reconstruction')
    elif t.objective not in ('classification','regression'):raise ValueError('Vector protocols use classification or regression')
    if t.name in ('G01','G02'):
        k=int(t.params.get('items',4));age=int(t.params.get('age',min(8,t.T//2)));write=t.T-1-age
        if not k<=int(t.params.get('key_capacity',16)):raise ValueError('Active items must fit a fixed key interface')
        if k<2 or age<2 or write<k+1:raise ValueError('Keyed memory needs items>=2 and room before/after the target write')
        if t.name=='G01' and not 0<=int(t.params.get('writes',2))<=age-1:raise ValueError('Post-target writes must fit before query')
        if t.name=='G02':
            if not 0<=float(t.params.get('invalidate_probability',.25))<=1 or int(t.params.get('versions',2))<0:raise ValueError('Version/invalid range')
            if int(t.params.get('versions',2))>=write-k:raise ValueError('Earlier versions must fit before latest write')
            if not int(t.params.get('versions',2))<=int(t.params.get('version_slots',4))<write-k:raise ValueError('Version slots must include all versions and fit the common schedule')
            if t.params.get('query','latest') not in ('latest','protected'):raise ValueError('G02 query latest/protected')
    if t.name in ('G03','G04') and not 4<=int(t.params.get('items',8))<t.T-1:raise ValueError('At least four events and room for a query')
    if t.name=='G04' and t.params.get('physical_time',False) and not t.params.get('show_clock',True):raise ValueError('Irregular physical intervals need an observable clock; clock-hidden is a separate unidentifiable control')
    if t.name in ('G05','G09'):
        if not 1<=int(t.params.get('measurements',max(1,t.T//4)))<=t.T:raise ValueError('Measurement count 1..T')
        choices=('independent','repeat','causal_hold') if t.name=='G05' else ('new','repeat','causal_hold','delayed_linear')
        if t.params.get('acquisition',choices[0]) not in choices:raise ValueError('Acquisition mode')
    if t.name=='G09' and (t.params.get('duration',1.)<=0 or t.params.get('tau',.3)<=0):raise ValueError('Physical duration/tau')


def _base_meta(mode,shape,names,dims,requested,independent=False):
    return {'target_mode':mode,'output_shape':shape,'classes':shape[0] if mode=='class_index' else None,
        'source_names':names,'source_dims':dims,'requested_sources':requested,'independent_factors':independent,'model_fields':['x']}


def _value_target(t,value):
    if t.objective=='classification':return np.array(int(value[0]>=0),'int64')
    return value.astype('float32')


def _keyed(t,g):
    K=int(t.params.get('items',4));D=t.k;age=int(t.params.get('age',min(8,t.T//2)));write=t.T-1-age
    capacity=int(t.params.get('key_capacity',16));keys=g.permutation(capacity)[:K]
    values=g.normal(size=(K,D));old=g.normal(size=(K,D));x=np.zeros((t.T,capacity+D+5),'float32');event_times=np.full(K,-1,int)
    def event(step,key,value,op):
        x[step,keys[key]]=1;x[step,capacity:capacity+D]=value;x[step,capacity+D+op]=1;x[step,-1]=1
    slots=np.sort(g.choice(np.arange(write),K-1,replace=False))
    for key,step in zip(range(1,K),slots):event(step,key,values[key],0);event_times[key]=step
    event(write,0,values[0],0);event_times[0]=write
    if t.name=='G01':
        L=int(t.params.get('writes',2));later=np.sort(g.choice(np.arange(write+1,t.T-1),L,replace=False))
        # Repeated writes preserve other legal items; they change write load without changing the queried value.
        for step in later:
            key=int(g.integers(1,K));event(step,key,values[key],0);event_times[key]=step
        event(t.T-1,0,np.zeros(D),3);y=_value_target(t,values[0]);targets=values.reshape(-1);names=['queried']+[f'other{j}' for j in range(1,K)]
        meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[2] if t.objective=='classification' else [D],names,[D]*K,['queried'],True)
        meta['valid_feature_index']=x.shape[-1]-1
        meta['requested_coordinate_indices']={name:([0] if t.objective=='classification' else list(range(D))) if name=='queried' else [] for name in names}
        return x,y,targets,event_times,np.ones(K,bool),meta
    versions=int(t.params.get('versions',2));occupied=set(slots.tolist());free=np.array([s for s in range(write) if s not in occupied])
    slots_count=int(t.params.get('version_slots',4));version_times=sorted(g.choice(free,slots_count,replace=False))
    for version,step in enumerate(version_times):
        if version>=slots_count-versions:event(step,0,old[0] if version==slots_count-1 else seeded(t.data_seed,'older_version',int(step),float(old[0,0])).normal(size=D),1)
        else:event(step,K-1,values[K-1],0)
    invalid=g.random()<float(t.params.get('invalidate_probability',.25))
    if invalid:event(write+1,0,np.zeros(D),2)
    else:event(write+1,K-1,values[K-1],0)
    query=0 if t.params.get('query','latest')=='latest' else 1;event(t.T-1,query,np.zeros(D),3)
    value=values[query];is_valid=not(invalid and query==0)
    y=np.array(int(value[0]>=0) if is_valid else 2,'int64') if t.objective=='classification' else np.r_[value if is_valid else np.zeros(D),float(is_valid)].astype('float32')
    targets=np.r_[values[0],old[0],values[1:].reshape(-1),float(not invalid)]
    names=['latest','obsolete']+[f'protected{j}' for j in range(1,K)]+['validity'];requested=['latest','validity'] if query==0 else ['protected1']
    meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[3] if t.objective=='classification' else [D+1],names,[D]*(len(names)-1)+[1],requested,True)
    meta['requested_coordinate_indices']={name:([0] if name=='validity' or t.objective=='classification' else list(range(D))) if name in requested else [] for name in names}
    meta['valid_feature_index']=x.shape[-1]-1
    meta['validity_target']='explicit validity coordinate or class2; latest requested includes valid and invalid trials'
    meta['matched_event_schedule']='version_slots constant; unused version slots write a different key; invalidate/no-invalidate both contain one event'
    return x,y,targets,np.r_[event_times[0],max(0,write-1),event_times[1:],write+1],np.r_[True,versions>0,np.ones(K-1,bool),True],meta


def _relation(t,g):
    K=int(t.params.get('items',8));slots=np.sort(g.choice(t.T-1,K,replace=False));order=g.permutation(K);position=np.argsort(order)
    pair=g.choice(K,2,replace=False);physical=t.name=='G04' and t.params.get('physical_time',False)
    times=np.cumsum(g.uniform(.1,2.,t.T)) if physical else np.arange(t.T,dtype=float)
    clock=t.name=='G04' and t.params.get('show_clock',True);x=np.zeros((t.T,3*K+1+int(clock)),'float32')
    x[slots,order]=1;x[-1,K+pair[0]]=1;x[-1,2*K+pair[1]]=1;x[-1,3*K]=1
    time_scale=2*t.T if physical else max(1,t.T-1)
    if clock:x[:,-1]=times/time_scale  # fixed known units; no future-endpoint normalization
    a,b=(position[pair[j]] for j in (0,1));margin=(b-a)/max(1,K-1) if t.name=='G03' else (times[slots[b]]-times[slots[a]])/time_scale
    value=margin if t.name=='G03' else abs(margin)
    label=int(margin>0) if t.name=='G03' else int(value>float(t.params.get('interval_boundary',.25)))
    y=np.array(label,'int64') if t.objective=='classification' else np.array([value],'float32')
    meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[2] if t.objective=='classification' else [1],['queried_relation'],[1],['queried_relation'])
    meta.update(relation='signed relative rank' if t.name=='G03' else 'event-pair interval magnitude',physical_time=bool(physical),clock_visible=bool(clock),
                classification_rule='relative event order' if t.name=='G03' else 'normalized interval > declared boundary')
    return x,y,np.array([value]),np.array([max(slots[a],slots[b])]),np.ones(1,bool),meta,times


def _evidence(t,g):
    count=int(t.params.get('measurements',max(1,t.T//4)));mode=t.params.get('acquisition','independent');z=g.normal(size=t.k)
    noises=g.normal(0,t.sigma,(t.T,t.k));obs=z+noises[:count];x=np.zeros((t.T,t.k+1),'float32')
    if mode=='independent':
        idx=np.rint(np.linspace(0,t.T-1,count)).astype(int);x[idx,:-1]=obs;x[idx,-1]=1;effective=count
    else:
        if mode=='repeat':obs=np.repeat(obs[:1],count,axis=0);effective=1
        else:effective=count
        assignments=np.minimum(np.arange(t.T)*count//t.T,count-1);x[:,:-1]=obs[assignments];x[:,-1]=1
    y=_value_target(t,z);meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[2] if t.objective=='classification' else [t.k],['signal'],[t.k],['signal'],True)
    meta.update(independent_measurements=effective,simulation_steps=t.T,acquisition=mode,valid_feature_index=x.shape[-1]-1,
                gaussian_posterior_MSE_reference=float(1/(1+effective/t.sigma**2)) if t.sigma else 0.)
    meta['requested_coordinate_indices']={'signal':[0] if t.objective=='classification' else list(range(t.k))}
    return x,y,z,np.array([0]),np.ones(1,bool),meta


def _frequency(t,g):
    f=float(t.params.get('target_frequency',2.));d=float(t.params.get('distractor_frequency',5.));times=np.arange(t.T)/t.T
    if not 0<f<t.T/2 or not 0<d<t.T/2:raise ValueError('Frequencies must be strictly below Nyquist')
    z=g.normal(size=4);basis=lambda freq:np.stack([np.cos(2*np.pi*freq*times),np.sin(2*np.pi*freq*times)],1)
    target=basis(f)@z[:2];nuisance=basis(d)@z[2:];x=np.stack([target+nuisance+g.normal(0,t.sigma,t.T),np.full(t.T,f/(t.T/2))],1).astype('float32')
    value=target if t.params.get('waveform_target',True) else z[:2]
    y=_value_target(t,z[:2]) if t.objective=='classification' else value.astype('float32')
    meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[2] if t.objective=='classification' else [len(value)],['target_quadratures','distractor_quadratures'],[2,2],['target_quadratures'],True)
    meta.update(target_frequency=f,distractor_frequency=d,frequency_units='cycles per observation window',overlap_unidentifiable=f==d)
    meta['requested_coordinate_indices']={'target_quadratures':[0] if t.objective=='classification' else [0,1],'distractor_quadratures':[]}
    return x,y,z,np.array([t.T-1,t.T-1]),np.ones(2,bool),meta


def _physical(t,g):
    steps=1024;duration=float(t.params.get('duration',1.));tau=float(t.params.get('tau',.3));rho=np.exp(-duration/steps/tau)
    z=np.empty((steps+1,t.k));z[0]=g.normal(size=t.k)
    driving=g.normal(size=(steps,t.k));measurement_noise=g.normal(0,t.sigma,(steps+1,t.k))
    for j in range(1,steps+1):z[j]=rho*z[j-1]+np.sqrt(1-rho*rho)*driving[j-1]
    fine=z+measurement_noise;count=int(t.params.get('measurements',max(1,t.T//4)));mode=t.params.get('acquisition','new')
    rawtimes=np.linspace(duration/count,duration,count);grid=np.linspace(duration/t.T,duration,t.T);rawidx=np.rint(rawtimes/duration*steps).astype(int);raw=fine[rawidx]
    if mode=='new':idx=np.rint(grid/duration*steps).astype(int);x=fine[idx];availability=grid
    elif mode=='repeat':
        assignments=np.minimum(np.arange(t.T)*count//t.T,count-1);x=raw[assignments];grid=rawtimes[assignments];availability=grid
    elif mode=='causal_hold':
        assignments=np.searchsorted(rawtimes,grid,side='right')-1;valid=assignments>=0;x=raw[np.maximum(assignments,0)].copy();x[~valid]=0;availability=grid
    else:
        x=np.stack([np.interp(grid,rawtimes,raw[:,j]) for j in range(t.k)],1)
        availability=rawtimes[np.minimum(np.searchsorted(rawtimes,grid),count-1)]
    y=_value_target(t,z[-1]);meta=_base_meta('class_index' if t.objective=='classification' else 'continuous',[2] if t.objective=='classification' else [t.k],['terminal_state'],[t.k],['terminal_state'])
    meta.update(duration=duration,tau=tau,acquisition=mode,independent_measurements=t.T if mode=='new' else count,
        online_causal_at_nominal_grid=mode!='delayed_linear',availability_rule='linear interpolation available only after its right supporting measurement; final-deadline protocol',
        fine_grid_reference_steps=steps,physical_simulation_claim='OU process discretized on a common fine grid; not exact continuous path')
    meta['requested_coordinate_indices']={'terminal_state':[0] if t.objective=='classification' else list(range(t.k))}
    return x.astype('float32'),y,z[-1],np.array([t.T-1]),np.ones(1,bool),meta,grid,availability


def protocol_batch(task,ids,split):
    if task.name in ('G07','G08'):
        from .data import spatial_batch,_spatial_render
        b=spatial_batch(replace(task,family='spatial',name='spatial_v2'),ids,split)
        if task.name=='G07':
            target=_spatial_render(b['z'],task.image_size)
            for step in (0,2):b['x'][:,step,:4]=np.where(b['visible'][:,step],np.clip(target+b['noise'][:,step],0,1),0)
            b['meta']['independent_factors']=False  # obsolete factors are not separate causes in static branch
            b['source_present'][:,4:]=False;b['source_times'][:,4:]=-1
            if task.objective=='classification':
                b['y']=(b['z'][:,1,1]>=0).astype('int64');b['meta']['requested_sources']=['H'];b['meta']['requested_coordinate_indices']={name:([1] if name=='H' else []) for name in b['meta']['source_names']}
        b['meta']['protocol']=task.name+'_v1';return b
    rows=[];meta=None
    for sid in ids:
        g=seeded(task.data_seed,split,int(sid),'protocol_world',task.name)
        fn=_keyed if task.name in ('G01','G02') else _relation if task.name in ('G03','G04') else _evidence if task.name=='G05' else _frequency if task.name=='G06' else _physical
        result=fn(task,g);x,y,z,times,present,meta=result[:6]
        row={'x':x,'y':y,'targets':np.asarray(z,'float32'),'loss_mask':np.ones_like(y,'float32'),'source_times':times,'source_present':present}
        if len(result)>6:row['timestamps']=result[6].astype('float32')
        if len(result)>7:row['availability_times']=result[7].astype('float32')
        rows.append(row)
    out={k:np.stack([r[k] for r in rows]) for k in rows[0]};meta.update(protocol=task.name+'_v1',generator='counter_scene_v2',official_generator_reproduction=False)
    out['meta']=meta;return out
