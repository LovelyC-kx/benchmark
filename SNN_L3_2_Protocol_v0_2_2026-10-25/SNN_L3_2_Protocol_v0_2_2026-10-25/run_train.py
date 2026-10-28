"""Train one predefined L3.2 run. Requires PyTorch; not run in this session."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import os
import numpy as np

from experiment_plan import build_plan, config_from_dict, SPLIT_SIZES
from load_data import generate, model_payload, classification_labels

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--objective',choices=['regression','classification'])
    ap.add_argument('--frozen-encoder',type=Path)
    ap.add_argument('--export-audit',action='store_true',help='Only after protocol choices are frozen; log held-out access')
    ap.add_argument('--smoke',action='store_true',help='One epoch on tiny development data; never a benchmark result')
    args=ap.parse_args()
    rows=[r for r in build_plan()['runs'] if r['run_id']==args.run_id]
    if not rows:raise ValueError('Unknown plan ID; inspect experiment_plan.json')
    from dataclasses import replace
    run=config_from_dict(rows[0]['config'])
    if args.objective:run=replace(run,objective=args.objective)
    if args.frozen_encoder:run=replace(run,encoder_mode='frozen')
    if args.smoke:
        if args.export_audit:raise ValueError('Smoke must not access audit split')
        run=replace(run,epochs=1)
    sizes=dict(SPLIT_SIZES)
    if args.smoke:sizes.update(encoder_train=64,encoder_val=32,geometry_fit=64,probe_train=64,probe_val=32,development_test=32)
    frozen_sha=hashlib.sha256(args.frozen_encoder.read_bytes()).hexdigest() if args.frozen_encoder else None
    effective_id=hashlib.sha256((run.run_id()+str(frozen_sha)+str(args.smoke)).encode()).hexdigest()[:16]
    out=args.output/effective_id
    if out.exists():raise FileExistsError('Existing run not overwritten: '+str(out))
    out.mkdir(parents=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from torch_adapter import LoadModel,task_loss,collect_features,frozen_operations
    random.seed(run.seed);np.random.seed(run.seed);torch.manual_seed(run.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(run.seed)
    torch.use_deterministic_algorithms(True)
    model=LoadModel(run).to(args.device)
    if args.frozen_encoder:
        ckpt=torch.load(args.frozen_encoder,map_location='cpu',weights_only=True)
        state={k.removeprefix('encoder.'):v for k,v in ckpt['model_state'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(state)
        model.encoder.requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=run.parameters:raise AssertionError((parameters,run.parameters))
    meta={'effective_id':effective_id,'plan_run_id':args.run_id,'config':asdict(run),
          'frozen_encoder_sha256':frozen_sha,'parameters':parameters,
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'persistent_cell_scalars_per_sample':2*run.width,
          'observation_dim_last_post':run.width,'observation_dim_joint_post':2*run.width,
          'torch':torch.__version__,'device':args.device,'status':'running',
          'run_kind':'smoke' if args.smoke else 'planned_exploratory_fit','split_sizes':sizes,
          'interpretation':'separate learned models across L; same raw inputs, task weights differ'}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    train=generate(run.data,sizes['encoder_train'],'encoder_train')
    val=generate(run.data,sizes['encoder_val'],'encoder_val')
    tx=model_payload(train)['x'];vx=model_payload(val)['x']
    ty=train['y_all'] if run.objective=='regression' else classification_labels(run.data,train)
    vy=val['y_all'] if run.objective=='regression' else classification_labels(run.data,val)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    generator=np.random.default_rng(run.seed+1000);best=float('inf');best_state=None;best_epoch=None
    for epoch in range(run.epochs):
        model.train();order=generator.permutation(len(tx));train_loss=0.
        for start in range(0,len(tx),128):
            ix=order[start:start+128]
            xb=torch.as_tensor(tx[ix],device=args.device);yb=torch.as_tensor(ty[ix],device=args.device)
            optimizer.zero_grad(set_to_none=True);pred=model(xb)
            loss,_=task_loss(pred,yb,run)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            train_loss+=float(loss.detach())*len(ix)
        model.eval();value=0.
        with torch.no_grad():
            for start in range(0,len(vx),128):
                xb=torch.as_tensor(vx[start:start+128],device=args.device)
                yb=torch.as_tensor(vy[start:start+128],device=args.device)
                loss,_=task_loss(model(xb),yb,run);value+=float(loss)*len(xb)
        value/=len(vx)
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
        if value<best:
            best=value;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row={'epoch':epoch+1,'train_weighted_loss':train_loss/len(tx),'val_weighted_loss':value}
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    model.load_state_dict(best_state);model.eval()
    torch.save({'model_state':best_state,'config':asdict(run),'frozen_encoder_sha256':frozen_sha},out/'best.pt')
    meta.update(status='fit_completed',best_epoch=best_epoch,best_val_weighted_loss=best,
                optimizer_updates=run.epochs*((len(tx)+127)//128))
    # Geometry/probe data are independent of model fitting and selection.
    for split in ('geometry_fit','probe_train','probe_val','development_test')+ (('audit_test',) if args.export_audit else ()):
        batch=generate(run.data,sizes[split],split)
        features,stats=collect_features(model,model_payload(batch)['x'])
        np.savez_compressed(out/(split+'.npz'),**features,**stats,
                            targets=batch['latents'][:,:,:run.data.visible_content_dim].reshape(len(batch['x']),-1),
                            classification_targets=classification_labels(run.data,batch),
                            source_present=batch['source_present'])
        if split=='audit_test':
            (out/'audit_access.json').write_text(json.dumps({'phase':'after_checkpoint_selection','split':split,
                'protocol_frozen_is_callers_responsibility':True,'best_epoch':best_epoch},indent=2)+'\n')
    # Development scope check, not a held-out performance estimate.
    ops=frozen_operations(model,vx[:128]);np.savez_compressed(out/'development_operations.npz',**ops)
    meta['noop_max_abs_error']=float(np.max(np.abs(ops['baseline']-ops['noop'])))
    if not np.allclose(ops['baseline'],ops['noop'],atol=1e-6,rtol=1e-5):raise AssertionError('No-op replay failed')
    meta['status']='fit_and_feature_export_completed'
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(str(out))

if __name__=='__main__':main()
