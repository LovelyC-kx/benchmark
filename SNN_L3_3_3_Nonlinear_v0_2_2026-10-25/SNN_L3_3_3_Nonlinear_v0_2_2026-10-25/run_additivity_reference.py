"""Test the cross-write additive null in the ACTUAL delivered architecture.

Same checkpoint weights, no training. A deep-copied forward control changes
cell kind to continuous; this is an operator override, NOT a fairly trained
continuous architecture baseline. Double precision checks algebra, not rank.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from bridge import load_model,collect
from factor_sampling import full_lattice
from run_calibrations import serial


def analyze(run_directory):
    import torch
    torch.set_num_threads(1)
    path=Path(run_directory)/'best.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
    model,run,metadata=load_model(run_directory,'cpu');model=model.double()
    if run.cell!='lif_hard' or run.data.k!=4 or run.data.visible_content_dim!=4 or run.data.extra_writes!=3:
        raise ValueError('This pinned architecture check expects the delivered full-presence K4 lif_hard source')
    continuous=copy.deepcopy(model)
    continuous.core.cfg.neuron='continuous'
    for cell in continuous.core.backbone.cells:cell.kind='continuous'
    if list(model.state_dict())!=list(continuous.state_dict()):raise AssertionError('Control changed parameter schema')
    if any(not torch.equal(v,continuous.state_dict()[k]) for k,v in model.state_dict().items()):raise AssertionError('Control weights differ')
    results=[]
    for name,cfg in [('original_k4',run.data),('within_write_unrequested_k2',replace(run.data,k=2,content_mode='demand_only'))]:
        batches,schema,_=full_lattice(cfg,64,'development_test',3800000,3900000,0,'C')
        responses={}
        for op,m in [('source_lif_hard',model),('continuous_forward_override',continuous)]:
            # Four corners of F/G and F/U rectangles with every other factor
            # fixed to B. Sample each legal input, then use a finite contrast.
            masks=[0,1,2,3,4,5]
            values={mask:collect(m,batches[mask]['x'].astype(np.float64))['last_post'] for mask in masks}
            d_fg=values[3]-values[1]-values[2]+values[0]
            d_fu=values[5]-values[1]-values[4]+values[0]
            responses[op]={'cross_write_FG_max_abs':float(np.max(np.abs(d_fg))),
                'cross_write_FG_rms':float(np.sqrt(np.mean(d_fg*d_fg))),
                'F_with_U_max_abs':float(np.max(np.abs(d_fu))),
                'F_with_U_rms':float(np.sqrt(np.mean(d_fu*d_fu)))}
        if responses['continuous_forward_override']['cross_write_FG_max_abs']>1e-10:
            raise AssertionError('Linear-core cross-write additive null failed')
        if name=='within_write_unrequested_k2' and responses['continuous_forward_override']['F_with_U_max_abs']<=1e-10:
            raise AssertionError('The known nonlinear encoder did not exercise the within-write comparison')
        results.append({'data_case':name,'visible_coordinates':cfg.visible_content_dim,'focal_coordinates':cfg.k,
            'group_coordinates':schema['coordinates'],'responses':responses,
            'note':'K2 changes analysis roles on full visible inputs for this null check; it is not a trained K2 benchmark condition.'})
    after=hashlib.sha256(path.read_bytes()).hexdigest()
    if before!=after:raise AssertionError('Source checkpoint changed')
    return {'kind':'actual-architecture algebraic reference with a forward operator override; not SNN scientific comparison',
        'source_checkpoint_id':metadata['effective_id'],'checkpoint_sha256':before,'checkpoint_unchanged':True,
        'source_checkpoint_original_optimizer_updates':metadata.get('optimizer_updates'),
        'new_snn_training_runs':0,'new_snn_optimizer_updates':0,'audit_test_accessed':False,
        'dtype':'float64','independent_pairs_per_data_case':64,'source_schedule_index':0,
        'control_weights_identical':True,'cases':results,
        'interpretation':'Cross-write FG finite contrast is zero for a linear core with independent per-write encoding. F/U may interact inside one encoder call. A forward override is not a trained baseline or a causal specificity proof.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);result=analyze(args.run_directory)
    out.write_text(json.dumps(serial(result),indent=2))
    print(json.dumps({'output':str(out),'cases':[{k:v for k,v in c.items() if k in ('data_case','responses')} for c in result['cases']]}))
