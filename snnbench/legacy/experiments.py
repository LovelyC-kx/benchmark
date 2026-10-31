"""Executable E00/E01 diagnostics, gradients, profiling and results aggregation."""
from pathlib import Path
from dataclasses import asdict
import json,time
import numpy as np
import torch
from torch import nn
from .analysis import *
from .tasks import TaskConfig,generate
from .runner import write_json,train_run,source_hash
from .neurons import Cell

def calibration(out,n=150000):
    rng=np.random.default_rng(42);T=16;beta=.9;sigma=.5;m=3
    z=rng.normal(size=n);q=rng.normal(size=n);eps=rng.normal(scale=sigma,size=(n,T))
    seq=np.repeat(z[:,None],T,1);seq[:,:-m]=q[:,None]
    w=beta**np.arange(T-1,-1,-1);u=(seq+eps)@w
    theory=stable_capacity(T,beta,sigma,m)
    actual=float(np.corrcoef(u,z)[0,1]**2)
    if abs(actual-theory['target'])>.01:raise AssertionError('analytic calibration failed')
    # Finite label enumeration, not an empirical MI estimator.
    from itertools import product
    rows=[]
    for a,b in product((0,1),repeat=2):rows.append({'a':a,'b':b,'xor':a^b})
    result={'theory':theory,'empirical_corr_squared':actual,'samples':n,'xor_enumeration':rows,
            'formal_discovery':False,'purpose':'known formula / task oracle calibration'}
    write_json(Path(out)/'calibration.json',result);return result

def fixed_scan(out,quick=True):
    """Fixed encoder mechanistic panel; no architecture training or SOTA claim."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    seeds=[0,1] if quick else list(range(5))
    betas=[.5,.9] if quick else [0,.25,.5,.75,.9,.97]
    gains=[1.,3.] if quick else [.25,1.,4.]
    sizes={'train':384,'val':128,'test':512} if quick else {'train':5000,'val':2000,'test':10000}
    for seed in seeds:
        for task in ('repeat','stale','complement','order'):
            tc=TaskConfig(name=task,T=12,new_steps=3,input_dim=8,latent_dim=4,
                          sigma=.5,data_seed=seed+2100,projection_seed=seed+9100)
            a={s:generate(tc,sizes[s],s,paired=s=='test') for s in sizes}
            # input-task audit: current, bag, and ordered observation baselines.
            for obs,fs in [('input_current',{s:d['x'][:,-1] for s,d in a.items()}),
                           ('input_bag',{s:d['x'].mean(1) for s,d in a.items()}),
                           ('input_ordered',{s:d['x'].reshape(len(d['x']),-1) for s,d in a.items()})]:
                cp,_,_=class_probe(fs['train'],fs['val'],fs['test'],a['train']['y'],a['val']['y'],a['test']['y'],Cs=(1.,))
                rows.append({'stage':'input_audit','task':task,'seed':seed,'observation':obs,**cp})
            for beta in betas:
                for gain in gains:
                    for neuron in ('continuous','threshold','lif_soft','lif_hard'):
                        tr={s:simulate(d['x'],neuron,beta,gain,dc=True) for s,d in a.items()}
                        paired=simulate(a['test']['x_pair'],neuron,beta,gain,dc=True)
                        for obs in ('post','count','trajectory'):
                            fs={s:observe(t,obs) for s,t in tr.items()}
                            r,reg,sc=linear_probe(fs['train'],fs['val'],fs['test'],a['train']['target'],a['val']['target'],a['test']['target'],alphas=(1.,))
                            cp,cl,csc=class_probe(fs['train'],fs['val'],fs['test'],a['train']['y'],a['val']['y'],a['test']['y'],Cs=(1.,))
                            row={'stage':'fixed_encoder_screen','task':task,'seed':seed,'beta':beta,'gain':gain,'neuron':neuron,
                                 'observation':obs,'target_probe':r,'label_probe':cp,
                                 'output_mean':float(tr['test']['output'].mean()),
                                 'output_mean_is_firing_rate':neuron!='continuous',
                                 'post_std':float(tr['test']['post'].std()),
                                 'margin_mean_abs':float(np.abs(tr['test']['pre']-1).mean()),
                                 'observed_features':fs['test'].shape[1],
                                 'primary_dc_gain_matched':True}
                            if task=='stale':
                                dr,_,_=linear_probe(fs['train'],fs['val'],fs['test'],a['train']['distractor'],a['val']['distractor'],a['test']['distractor'],alphas=(1.,))
                                row['distractor_probe']=dr
                                fp=observe(paired,obs)
                                p=reg.predict(fp);target=flat(a['test']['target'])
                                row['paired_history']={'feature_rms_change':float(np.sqrt(np.mean((fp-fs['test'])**2))),
                                    'target_mse_under_alternative_history':float(np.mean((p-target)**2)),
                                    'label_agreement':float(np.mean(cl.predict(csc.transform(fp))==cl.predict(csc.transform(fs['test'])))),
                                    'target_and_new_segment_held_fixed':True}
                                if beta==.9 and gain==3. and obs=='post':row['geometry']=geometry(fs['test'],a['test']['target'],a['test']['distractor'],permutations=16,seed=seed)
                            rows.append(row)
        write_json(out/'fixed_scan.json',{'rows':rows,'sizes':sizes,'formal_benchmark':False,
                                        'seeds_completed':seed+1,'source_hash':source_hash()})
        print(f'fixed scan seed {seed} complete: {len(rows)} rows',flush=True)
    return rows

def gradient_diagnostics(model,x,y,objective):
    """VJPs, not complete Jacobian spectral norms; graph scope explicit."""
    model.eval();p=model(x,capture=True)
    cells=[(n,m) for n,m in model.named_modules() if isinstance(m,Cell)]
    from .runner import loss_fn
    loss=loss_fn(p,y,objective);records=[]
    for name,c in cells:
        T=len(c.trace['pre']);last=c.trace['pre'][-1]
        for lag in (1,2,4,8):
            if lag>=T:continue
            past=c.trace['pre'][-1-lag]
            g=torch.autograd.grad(last.sum(),past,retain_graph=True,allow_unused=True)[0]
            records.append({'layer':name,'lag':lag,'ones_vjp_rms':0. if g is None else float(g.detach().square().mean().sqrt()),
                            'disconnected':g is None})
    params=[p for p in model.parameters() if p.requires_grad]
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    return {'state_vjps':records,'loss_parameter_gradient_squared':sum(float(g.detach().square().sum()) for g in grads if g is not None),
            'surrogate_training_diagnostic_not_forward_information':True,'batch_size':len(x)}

def fisher_diagnostic(model,x,y,objective,max_samples=8):
    from .runner import loss_fn
    params=[p for p in model.parameters() if p.requires_grad];gs=[]
    for i in range(min(len(x),max_samples)):
        loss=loss_fn(model(x[i:i+1]),y[i:i+1],objective)
        grad=torch.autograd.grad(loss,params,allow_unused=True)
        gs.append(torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grad)]).cpu())
    g=torch.stack(gs)
    return {'empirical_ground_truth_loss_gradient_mean_square':float(g.square().sum(1).mean()),
            'mean_gradient_square':float(g.mean(0).square().sum()),
            'between_sample_gradient_dispersion':float((g-g.mean(0)).square().sum(1).mean()),
            'samples':len(g),'is_true_model_fisher':False,'is_label_MI':False}

def profile(model,x,repeats=10):
    model.eval();ledger=[];hooks=[]
    def record(name,m,inputs,output):
        a=inputs[0]
        if isinstance(m,nn.Linear):mac=output.numel()*m.in_features
        elif isinstance(m,(nn.Conv1d,nn.Conv2d)):
            mac=output.numel()*m.in_channels//m.groups*int(np.prod(m.kernel_size))
        elif isinstance(m,nn.ConvTranspose2d):
            mac=a.numel()*m.out_channels//m.groups*int(np.prod(m.kernel_size))
        else:return
        # Numerical binary eligibility is only a proxy; hardware execution is dense.
        binary=bool(torch.all((a==0)|(a==1)))
        density=float((a!=0).float().mean())
        ledger.append({'module':name,'dense_MAC':int(mac),'binary_input_numerical_proxy':binary,
                       'nonzero_fraction':density,'active_synaptic_proxy':float(mac*density) if binary else None})
    for n,m in model.named_modules():
        if isinstance(m,(nn.Linear,nn.Conv1d,nn.Conv2d,nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(lambda m,a,o,n=n:record(n,m,a,o)))
    with torch.no_grad():model(x)
    for h in hooks:h.remove()
    with torch.no_grad():
        for _ in range(3):model(x)
        if x.is_cuda:torch.cuda.synchronize(x.device)
        start=time.perf_counter()
        for _ in range(repeats):model(x)
        if x.is_cuda:torch.cuda.synchronize(x.device)
        elapsed=time.perf_counter()-start
    eligible=sum(r['active_synaptic_proxy'] or 0 for r in ledger)
    dense=sum(r['dense_MAC'] for r in ledger if not r['binary_input_numerical_proxy'])
    return {'ledger':ledger,'full_forward_batch_latency_ms':1000*elapsed/repeats,
            'batch_size':len(x),'input_window_T':x.shape[1],'repeats':repeats,
            'dense_affine_MAC':sum(r['dense_MAC'] for r in ledger),
            'affine_only_45nm_proxy_pJ':dense*4.6+eligible*.9,
            'energy_definition':'4.6 pJ/MAC, 0.9 pJ/active binary-input AC proxy; arithmetic estimate only',
            'excluded_costs':['attention matmul','normalization','leak and reset','spike generation','readout aggregation','memory access','control'],
            'measured_hardware_energy':False,'asynchronous_event_execution':False,**model.resources()}

def suite(configs,out,resume=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,c in enumerate(configs):
        name=c.get('id',f'run_{i:04d}');p=out/name
        if resume and (p/'result.json').exists():
            r=json.loads((p/'result.json').read_text())
            if r.get('config')!=c or r.get('source_hash')!=source_hash():
                raise ValueError(f'{name}: config/source changed; choose a new output or --no-resume')
        else:
            print(f'[{i+1}/{len(configs)}] {name}',flush=True)
            try:r=train_run(c,p)
            except Exception as e:
                write_json(p/'failure.json',{'type':type(e).__name__,'message':str(e),'config':c});raise
        rows.append({'id':name,'seed':r['seed'],'task':r['config']['task']['name'],
                     'objective':r['config']['task']['objective'],'model':r['config']['model'],
                     'test':r['test'],'seconds':r['wall_seconds'],'stage':r['stage']})
        write_json(out/'summary.json',{'runs':rows,'formal_benchmark':False,'completed':len(rows),'requested':len(configs)})
    return rows
