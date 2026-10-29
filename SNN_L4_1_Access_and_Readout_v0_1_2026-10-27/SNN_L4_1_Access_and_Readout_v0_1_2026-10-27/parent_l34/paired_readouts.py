"""Shared labeled-example plans and scene-level evaluation uncertainty.

The final loss is averaged within each independent evaluation scene across
declared tasks, frozen training draws and initializations. Its Hoeffding
bound conditions on those fixed objects; it is not an architecture CI.
"""
import hashlib
import numpy as np
from task_moments import classification_error


def draw_plan(pool_size,sizes,repeats=4,seed=35041):
    if min(pool_size,repeats)<1 or any(not isinstance(p,(int,np.integer)) or isinstance(p,bool) or p<1 for p in sizes) or max(sizes)>pool_size or len(set(sizes))!=len(sizes):raise ValueError('Valid unique label counts and common train pool')
    rng=np.random.default_rng(seed)
    indices=np.stack([rng.permutation(pool_size)[:max(sizes)] for _ in range(repeats)])
    return {'indices':indices,'pool_size':pool_size,'sample_sizes':tuple(sizes),'repeats':repeats,'seed':seed,
        'sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()}


def fit_linear_family(x,y,plan,family='hebbian',alpha=1e-3):
    x,y=map(lambda a:np.asarray(a,float),(x,y));result={}
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(x)!=plan['pool_size'] or not np.isin(y,[-1,1]).all():raise ValueError('Common training data/labels and plan')
    for p in plan['sample_sizes']:
        weights=[]
        for idx in plan['indices']:
            a,b=x[idx[:p]],y[idx[:p]];m=a.T@b/p
            if family=='hebbian':w=m
            elif family=='ridge':
                if alpha<=0:raise ValueError('Positive fixed regularizer')
                w=np.linalg.solve(a.T@a/p+alpha*np.eye(a.shape[1]),m)
            else:raise ValueError('Declared Hebbian or ridge family')
            weights.append(w)
        result[p]={'family':family,'weights':np.stack(weights),'new_task_labels':p,'draw_plan_sha256':plan['sha256'],
            'alpha':alpha if family=='ridge' else None,'intercept':False}
    return result


def fit_independent_mlps(x,y,plan,width=16,epochs=80,seeds=(1,2)):
    import torch
    x,y=map(lambda a:np.asarray(a,float),(x,y));n,d=x.shape;tasks=y.shape[1];models={}
    if n!=plan['pool_size']:raise ValueError('Identical p-example plan required')
    for p in plan['sample_sizes']:
        state=[]
        for repetition,idx in enumerate(plan['indices']):
            for seed in seeds:
                # Data and initialization seed are independent axes.
                torch.manual_seed(seed)
                a=torch.tensor(x[idx[:p]],dtype=torch.float32);b=torch.tensor(y[idx[:p]],dtype=torch.float32)
                w1=torch.nn.Parameter(torch.randn(tasks,d,width)/np.sqrt(d));b1=torch.nn.Parameter(torch.zeros(tasks,width))
                w2=torch.nn.Parameter(torch.randn(tasks,width)/np.sqrt(width));b2=torch.nn.Parameter(torch.zeros(tasks))
                opt=torch.optim.Adam((w1,b1,w2,b2),lr=.02)
                for _ in range(epochs):
                    opt.zero_grad();s=(torch.tanh(torch.einsum('nd,tdh->nth',a,w1)+b1)*w2).sum(-1)+b2
                    loss=torch.nn.functional.softplus(-b*s).mean(0).sum()+1e-4*(w1.square().sum()+w2.square().sum());loss.backward();opt.step()
                state.append({'w1':w1.detach().numpy(),'b1':b1.detach().numpy(),'w2':w2.detach().numpy(),'b2':b2.detach().numpy(),
                    'training_draw':repetition,'initialization_seed':seed})
        models[p]={'family':'mlp','models':state,'new_task_labels':p,'draw_plan_sha256':plan['sha256'],
            'epochs':epochs,'width':width,'initialization_seeds':list(seeds),'vectorized_optimizer_steps':epochs*len(state),
            'individual_task_head_fits':len(state)*tasks,'validation_labels':0,'scale':'Shared geometry-fit scale; no extra pool fitted here.'}
    return models


def evaluate(fit,x,y,batch_size=512):
    x,y=map(lambda a:np.asarray(a,float),(x,y));task=np.zeros(y.shape[1]);scene=np.zeros(len(x));split=[];tie=0.
    states=fit['models'] if fit['family']=='mlp' else list(fit['weights'])
    for state in states:
        total=0.;count_tie=0
        for first in range(0,len(x),batch_size):
            last=min(first+batch_size,len(x));a=x[first:last]
            if fit['family']=='mlp':scores=(np.tanh(np.einsum('nd,tdh->nth',a,state['w1'])+state['b1'])*state['w2']).sum(-1)+state['b2']
            else:scores=a@state
            labels=y[first:last];loss=(scores*labels<0)+.5*(scores==0)
            scene[first:last]+=loss.mean(1)/len(states);task+=loss.sum(0)/(len(x)*len(states))
            total+=float(loss.sum());count_tie+=int(np.sum(scores==0))
        split.append(total/y.size);tie+=count_tie/(y.size*len(states))
    return {'mean_error':float(scene.mean()),'per_independent_scene_loss':scene,
        'per_task_error':task,'per_frozen_reader_error':np.array(split),'tie_fraction':tie,
        'new_task_labels':fit['new_task_labels'],'draw_plan_sha256':fit['draw_plan_sha256'],'tie_policy':'half'}


def paired_gain_certificate(baseline_scene_loss,chosen_scene_loss,delta=.05,family_size=1):
    a,b=map(lambda x:np.asarray(x,float),(baseline_scene_loss,chosen_scene_loss))
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a<0)|(a>1)|(b<0)|(b>1)) or not 0<delta<1 or family_size<1:raise ValueError('Independent scene losses in [0,1], declared alpha/family')
    difference=a-b;radius=float(np.sqrt(2*np.log(2*family_size/delta)/len(a)));mean=float(difference.mean())
    return {'mean_error_reduction':mean,'simultaneous_Hoeffding_interval':[max(-1.,mean-radius),min(1.,mean+radius)],
        'family_failure_probability':delta,'planned_comparison_count':int(family_size),'independent_evaluation_scenes':len(a),
        'radius':radius,'positive_lower_bound':mean-radius>0,
        'scope':'Distribution-free conditional bound for preselected frozen readers and fixed task/train collections, over independent evaluation scenes only. No training-seed or architecture guarantee.'}
