"""Content recoverability inside the actual current head's row/null spaces."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge42 import (ROOT,PARENT,load_model,fixed_batch,collect,targets,fit_head,
    parameter_hash,save_tree,write_json,tree_hash,independent_ids)
from access_operators import output_subspaces
from readers42 import linear_bundle,predict
from run_reader_ladder import ALPHAS,DRAW_SEEDS,group_losses

def run(output,evaluation_scenes=4096):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    directory=PARENT/'evidence/source_smoke_run';model,cfg,meta=load_model(directory)
    before=parameter_hash(model);k=cfg.data.k
    if k!=4 or cfg.objective!='regression':raise ValueError('Declared native task adapter required')
    W=model.core.head.weight.detach().numpy().astype(float);bias=model.core.head.bias.detach().numpy().astype(float)
    wanted=list(range(k))+list(range(16,16+k));definitions={};basis={};rng=np.random.default_rng(42117)
    for label,rows in [('requested8',wanted),('all20',list(range(len(W))))]:
        info=output_subspaces(W[rows])
        for kind,key in [('row','potent_basis'),('null','null_basis')]:
            b=info[key];name=label+'__'+kind;basis[name]=b
            definitions[name]={'contract':label,'kind':kind,'rank':b.shape[1],'output_rows':rows}
            for seed in range(5):
                q=np.linalg.qr(rng.normal(size=(W.shape[1],b.shape[1])))[0];name=label+'__random_'+kind+'_'+str(seed)
                basis[name]=q;definitions[name]={'contract':label,'kind':'matched_rank_random_'+kind,'rank':b.shape[1],'output_rows':rows,'random_draw':seed}
    cb=fixed_batch(cfg.data,512,'geometry_fit',8500000,0);tb=fixed_batch(cfg.data,1024,'probe_train',8510000,0);sb=fixed_batch(cfg.data,256,'probe_val',8520000,0)
    independent_ids(cb['sample_ids'],tb['sample_ids'],sb['sample_ids'],np.arange(8650000,8650000+evaluation_scenes))
    cv=collect(model,cb)[0]['post__last'];tv=collect(model,tb)[0]['post__last'];sv=collect(model,sb)[0]['post__last']
    y,theta,_=targets(tb,k);y=y[:,:8];sy=targets(sb,k)[0][:,:8]
    bank={};scales={};selection={};cases=[];constant_means={}
    for draw,seed in enumerate(DRAW_SEEDS):
        ids=np.random.default_rng(seed).permutation(len(y))[:512]
        constant_means[str(draw)]=y[ids].mean(0)
    for name,b in basis.items():
        scales[name]=max(float(np.sqrt(np.mean((cv@b)**2))),1e-8)
        for draw,seed in enumerate(DRAW_SEEDS):
            rng=np.random.default_rng(seed);idx=rng.permutation(len(y))[:512];sel=rng.permutation(len(sy))[:128]
            a=tv[idx]@b/scales[name];c=sv[sel]@b/scales[name]
            heads={str(alpha):fit_head(a,y[idx],alpha) for alpha in ALPHAS}
            alpha=min(heads,key=lambda z:float(np.mean((predict(linear_bundle(heads[z]),c)-sy[sel])**2)))
            key=name+'__draw'+str(draw);bank[key]=linear_bundle(heads[alpha]);selection[key]=float(alpha)
            cases.append({'key':key,'view':name,'draw':draw,'fit_labels':512,'selection_labels':128,
                'parameters':bank[key]['parameters'],'diagnostic_projection_scalars':int(b.size)})
    save_tree(out/'frozen_readers',bank);save_tree(out/'projection_bases',basis)
    freeze={'checkpoint_sha256':hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest(),
        'original_optimizer_updates':meta['optimizer_updates'],'reader_sha256':tree_hash(bank),'projection_sha256':tree_hash(basis),
        'definitions':definitions,'scales':scales,'selection':selection,'cases':cases,'theta':theta,'constant_means':constant_means,
        'fit_pool':[1024,'probe_train',8510000],'selection_pool':[256,'probe_val',8520000],
        'evaluation':[evaluation_scenes,'development_test',8650000],'choices_frozen_before_evaluation_generation':True,
        'scope':'Offline current affine-head input; projected observations are diagnostic access. No internal state or future continuation is edited.'}
    write_json(out/'freeze_contract.json',freeze)
    eb=fixed_batch(cfg.data,evaluation_scenes,'development_test',8650000,0);ev,native=collect(model,eb);h=np.asarray(ev['post__last'],float);ey=targets(eb,k)[0][:,:8]
    risks={};scene={};head_identity={};constants={};constant_scene={}
    for draw,mean in constant_means.items():
        lo=group_losses(np.broadcast_to(mean,ey.shape),ey,theta,k)
        constants[draw]={s:{o:float(a.mean()) for o,a in row.items()} for s,row in lo.items()}
        for source,row in lo.items():
            for obj,a in row.items():constant_scene[draw+'::'+source+'::'+obj]=a
    for label,rows in [('requested8',wanted),('all20',list(range(len(W))))]:
        br,bn=basis[label+'__row'],basis[label+'__null'];pr=h@br@br.T;pn=h@bn@bn.T
        head_identity[label]={'row_only_head_prediction_max_abs':float(np.max(np.abs(pr@W[rows].T-h@W[rows].T))),
            'null_component_head_effect_max_abs':float(np.max(np.abs(pn@W[rows].T))),
            'feature_reconstruction_max_abs':float(np.max(np.abs(pr+pn-h)))}
    for c in cases:
        key,name=c['key'],c['view'];pred=predict(bank[key],h@basis[name]/scales[name]);lo=group_losses(pred,ey,theta,k)
        risks[key]={s:{o:float(a.mean()) for o,a in row.items()} for s,row in lo.items()}
        for source,row in lo.items():
            for obj,a in row.items():scene[key+'::'+source+'::'+obj]=a
    np.savez_compressed(out/'scene_losses.npz',**scene)
    np.savez_compressed(out/'constant_baseline_losses.npz',**constant_scene)
    assert parameter_hash(model)==before and tree_hash(bank)==freeze['reader_sha256']
    assert max(v for row in head_identity.values() for v in row.values())<1e-10
    report={'module':'L4.2 task/readout subspaces','subspace_configurations':len(basis),'selected_readers':len(cases),
        'matrix_solves':len(cases)*len(ALPHAS),'risk_cells':len(scene),'evaluation_scenes':evaluation_scenes,
        'head_identities':head_identity,'risks':risks,'constant_risks':constants,'constant_baseline_risk_cells':len(constant_scene),'core_parameters_unchanged':True,
        'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'Independent new evaluation pool. Recoverable content in a current head-null component does not imply unique storage, whole-system disuse or future output-nullness. Matched-rank random controls are all retained, never selected by test performance.'}
    write_json(out/'report.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--evaluation-scenes',type=int,default=4096);a=p.parse_args()
    r=run(a.output,a.evaluation_scenes);print('Task subspace readers',r['selected_readers'],'risk cells',r['risk_cells'])
