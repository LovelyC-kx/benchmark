"""Freeze all choices before fresh confirmation. No SNN optimizer is created."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents
from temporal_capture import base_capture,temporal_views
from temporal_alignment import fit_transport,push_reader
from temporal_contracts import independent_ids,source_clock
from temporal_probes import fit_time_heads,score,loss_by_group,paired_bound,tree_hash
from artifacts35 import save_tree,load_tree,write_json

FAMILIES=('raw','center','orthogonal','affine')
CANDIDATES=('anchor_raw','anchor_center','anchor_orthogonal','anchor_affine','pooled','time_local')
SOURCES=('H1','H2','H3','H4','C')
ROOT=Path(__file__).resolve().parent

def target_bank(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]%5 or z.shape[1]//5 not in (1,2,4):raise ValueError('Five source blocks, each 1/2/4 coordinates')
    q=z.shape[1]//5;offset=5*q
    rng=np.random.default_rng(360904);theta=rng.normal(size=(5,q,16));theta/=np.linalg.norm(theta,axis=1,keepdims=True)
    binary=np.concatenate([np.where(z[:,q*j:q*j+q]@theta[j]>=0,1.,-1.) for j in range(5)],1)
    groups=[]
    for j,name in enumerate(SOURCES):
        groups.append({'source':name,'role':'supervised' if name in ('H1','C') else 'originally_unrequested',
                       'continuous_columns':list(range(q*j,q*j+q)),
                       'binary_columns':list(range(offset+16*j,offset+16*j+16))})
    return np.c_[z,binary],theta,groups

def map_digest(views,scales):
    h=hashlib.sha256();counts=0;ranks={}
    for name,x in views.items():
        x=np.asarray(x,float)/scales[name];rows=[]
        for s in range(x.shape[1]):
            for t in range(x.shape[1]):
                for family in FAMILIES:
                    f=fit_transport(x[:,t],x[:,s],family)
                    h.update(tree_hash(f).encode());counts+=1
                    if family=='orthogonal':rows.append(f['rank_cross_covariance'])
        ranks[name]={'minimum':min(rows),'maximum':max(rows),'dimension':x.shape[2]}
    return h.hexdigest(),counts,ranks

def candidate_losses(fits,cal,x,y,groups,anchor):
    """Per-scene mean risk over each source's available target times."""
    result={}
    for j,g in enumerate(groups):
        times=g['available_times'];cc=g['continuous_columns'];bc=g['binary_columns']
        result[g['source']]={k:[np.zeros(len(x)),np.zeros(len(x))] for k in CANDIDATES}
        for t in times:
            for family in FAMILIES:
                m=fit_transport(cal[:,t],cal[:,anchor],family)
                w,b=push_reader(m,fits['weights'][:,anchor],fits['origins'][:,anchor],fits['bias'][:,anchor])
                loss=loss_by_group(score(w,b,x[:,t]),y,bc,cc)
                for k in (0,1):result[g['source']]['anchor_'+family][k]+=loss[k]/len(times)
            w=fits['weights'][:,t];b=fits['bias'][:,t]-np.einsum('rd,rdo->ro',fits['origins'][:,t],w)
            loss=loss_by_group(score(w,b,x[:,t]),y,bc,cc)
            for k in (0,1):result[g['source']]['time_local'][k]+=loss[k]/len(times)
            columns=cc+bc
            w=np.stack([v[j] for v in fits['pooled_weights']]);o=np.stack([v[j] for v in fits['pooled_origins']]);b=np.stack([v[j] for v in fits['pooled_bias']])
            b=b-np.einsum('rd,rdo->ro',o,w)
            pred=score(w,b,x[:,t]);local_y=y[:,columns]
            q=len(cc);loss=loss_by_group(pred,local_y,list(range(q,len(columns))),list(range(q)))
            for k in (0,1):result[g['source']]['pooled'][k]+=loss[k]/len(times)
    return result

def run(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.perf_counter()
    run_directory=Path(getattr(args,'run_directory',None) or ROOT/'evidence/source_smoke_run')
    model,cfg,meta=load_model(run_directory)
    checkpoint=run_directory/'best.pt';checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cb=fixed_batch(cfg.data,512,'geometry_fit',7600000,0)
    tb=fixed_batch(cfg.data,512,'probe_train',7610000,0)
    db=fixed_batch(cfg.data,2048,'probe_val',7620000,0)
    independent_ids(cb['sample_ids'],tb['sample_ids'],db['sample_ids'],np.arange(7630000,7630000+args.scenes))
    cbase=base_capture(model,cb['x']);cv=temporal_views(cbase);tv=temporal_views(base_capture(model,tb['x']))
    dv=temporal_views(base_capture(model,db['x']));y,theta,groups=target_bank(flatten_contents(tb,cfg.data));dy,_,_=target_bank(flatten_contents(db,cfg.data))
    clock=source_clock(tb['event_times'],tb['source_present'],tb['x'].shape[1]);T=tb['x'].shape[1]
    for j,g in enumerate(groups):
        assert np.all(tb['event_times'][:,j]==tb['event_times'][0,j])
        g['onset']=int(tb['event_times'][0,j]);g['available_times']=list(range(g['onset'],T))
    rng=np.random.default_rng(360103);indices=np.stack([rng.choice(512,128,False) for _ in range(2)])
    bank={};scales={};selection={}
    for name,c in cv.items():
        scale=max(float(np.sqrt(np.mean(np.asarray(c,float)**2))),1e-8);scales[name]=scale
        masks=[(g['continuous_columns']+g['binary_columns'],g['available_times']) for g in groups]
        bank[name]=fit_time_heads(np.asarray(tv[name],float)/scale,y,indices,masks)
        losses=candidate_losses(bank[name],np.asarray(c,float)/scale,np.asarray(dv[name],float)/scale,dy,groups,T-1)
        selection[name]={}
        for source,rows in losses.items():
            means={k:float(v[0].mean()) for k,v in rows.items()}
            chosen=min(CANDIDATES,key=lambda k:(means[k],CANDIDATES.index(k)))
            selection[name][source]={'chosen':chosen,'selection_classification_error':means}
        print('prepared '+name,flush=True)
    del tv,dv
    save_tree(out/'frozen_readers',bank);np.savez_compressed(out/'paired_calibration.npz',**cbase)
    digest,map_count,map_ranks=map_digest(cv,scales)
    freeze={'module':'L3.6','checkpoint_sha256':checkpoint_sha,'source_original_optimizer_updates':meta['optimizer_updates'],
            'reader_tree_sha256':tree_hash(bank),'alignment_full_bank_sha256':digest,'alignment_maps_in_digest':map_count,
            'alignment_map_families':FAMILIES,'scales':scales,'groups':groups,'theta':theta,'training_indices':indices,
            'selection':selection,'evaluation_pool':[args.scenes,'development_test',7630000],
            'comparison_family_size':len(bank)*len(groups),
            'selection_pool':[2048,'probe_val',7620000],'calibration_pool':[512,'geometry_fit',7600000],
            'train_pool':[512,'probe_train',7610000],'unique_train_scenes_across_two_draws':len(np.unique(indices)),
            'source_times':list(range(T)),'map_rank_summary':map_ranks,'choice_frozen_before_confirmation_generation':True,
            'statistical_scope':'Conditional on this old representation, chosen readers and fixed task directions; independent evaluation scenes only.'}
    write_json(out/'freeze_contract.json',freeze)
    # Fresh confirmation is first generated below this line.
    eb=fixed_batch(cfg.data,args.scenes,'development_test',7630000,0)
    ev=temporal_views(base_capture(model,eb['x']));ey,_,_=target_bank(flatten_contents(eb,cfg.data))
    risk={};selected={};saved_scene={};second_hash=hashlib.sha256()
    for name,c in cv.items():
        scale=scales[name];cal=np.asarray(c,float)/scale;x=np.asarray(ev[name],float)/scale;f=bank[name]
        arrays={family:{g['source']:{'classification':np.empty((T,T)),'MSE':np.empty((T,T))} for g in groups} for family in FAMILIES}
        for s in range(T):
            for t in range(T):
                for family in FAMILIES:
                    m=fit_transport(cal[:,t],cal[:,s],family);second_hash.update(tree_hash(m).encode())
                    w,b=push_reader(m,f['weights'][:,s],f['origins'][:,s],f['bias'][:,s]);pred=score(w,b,x[:,t])
                    for g in groups:
                        loss=loss_by_group(pred,ey,g['binary_columns'],g['continuous_columns'])
                        arrays[family][g['source']]['classification'][s,t]=loss[0].mean();arrays[family][g['source']]['MSE'][s,t]=loss[1].mean()
        risk[name]=arrays
        dl=candidate_losses(f,cal,x,ey,groups,T-1);selected[name]={}
        for g in groups:
            source=g['source'];chosen=selection[name][source]['chosen'];base=dl[source]['anchor_raw'];new=dl[source][chosen]
            selected[name][source]={'chosen':chosen,'baseline_error':float(base[0].mean()),'chosen_error':float(new[0].mean()),
                                    'baseline_MSE':float(base[1].mean()),'chosen_MSE':float(new[1].mean()),
                                    'paired_classification_bound':paired_bound(base[0],new[0],len(bank)*len(groups))}
            saved_scene[name+'__'+source+'__baseline']=base[0];saved_scene[name+'__'+source+'__selected']=new[0]
        print('evaluated '+name,flush=True)
    assert second_hash.hexdigest()==digest and tree_hash(bank)==freeze['reader_tree_sha256']
    np.savez_compressed(out/'selected_scene_losses.npz',**saved_scene)
    report={'module':'L3.6','version':'0.2','views':len(bank),'contents':5,'times':T,
            'source_optimizer_updates':meta['optimizer_updates'],
            'matrix_cells_per_objective':len(bank)*5*T*T*len(FAMILIES),'reader_matrix_solves':len(bank)*2*(T+5),
            'logical_content_objective_reader_fits':len(bank)*2*5*2*(T+1),
            'risk_matrices':risk,'selected_reader_results':selected,'groups':groups,
            'all_map_hashes_replayed_exactly':True,'checkpoint_unchanged':hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_sha,
            'new_SNN_optimizer_updates':0,'formal_benchmark_completed':0,'formal_benchmark_planned':435,'audit_test_accessed':False,
            'evaluation_scenes':args.scenes,'elapsed_seconds':time.perf_counter()-start,
            'scope':f'One frozen two-cell project MLP, source optimizer updates={meta["optimizer_updates"]}. All {T}x{T} cells include pre-write negative controls; retention/selection uses available times only. No architecture ranking.'}
    write_json(out/'report.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--scenes',type=int,default=8192);parser.add_argument('--run-directory')
    result=run(parser.parse_args());print({k:result[k] for k in ('views','times','matrix_cells_per_objective','elapsed_seconds')})
