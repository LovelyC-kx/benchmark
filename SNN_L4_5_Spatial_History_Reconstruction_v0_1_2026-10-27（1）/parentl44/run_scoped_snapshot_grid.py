"""Real scoped restorations on untrained interfaces, not trained semantic evidence."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bridge44 import BenchmarkModel,ModelConfig,NEURONS,BACKBONES,StreamSession,parameter_hash,write_json
from snapshot_recovery44 import SCOPES,zero_snapshot,restore_scope

def run():
    torch.set_num_threads(1);rows=[]
    for objective,bones in [('classification',BACKBONES),('reconstruction',('cnn2d','unet'))]:
        for backbone in bones:
            for cell in NEURONS:
                for readout in ('last','mean'):
                    torch.manual_seed(440240);image=backbone in ('cnn2d','sew_resnet18','spike_attention','unet')
                    shape=(1,8,8) if image else (4,);target=(1,8,8) if objective=='reconstruction' else (3,)
                    cfg=ModelConfig(backbone=backbone,neuron=cell,width=2,depth=2,observation='post',readout=readout)
                    model=BenchmarkModel(shape,target,4,cfg,objective=objective).eval();before=parameter_hash(model)
                    x=torch.tensor(np.random.default_rng(440240).normal(size=(2,4,*shape)).astype(np.float32));session=StreamSession(model)
                    session.step(x[:,0]);session.step(x[:,1]);clean=session.snapshot();bad=zero_snapshot(clean)
                    session.step(x[:,2]);session.step(x[:,3]);original=session.finish().numpy();values={}
                    for scope in SCOPES:
                        session.restore(restore_scope(bad,clean,scope));session.step(x[:,2]);session.step(x[:,3]);p=session.finish().numpy()
                        values[scope]={'clean_prediction_MSE':float(np.mean((p-original)**2)),'max_abs':float(np.max(np.abs(p-original)))}
                    assert values['system']['max_abs']==0 and parameter_hash(model)==before
                    if readout=='last':assert values['core']['max_abs']==0
                    rows.append({'objective':objective,'backbone':backbone,'cell':cell,'readout':readout,'scopes':values,'trained':False,
                        'scope':'Tensor-history values cleared at step1; clock/count/denominator preserved. Restore after consumed observation, then two common future steps.'})
        print('scoped restoration interfaces',objective,flush=True)
    return {'module':'L4.4 operational snapshot scopes','configurations':len(rows),'scoped_conditions':len(rows)*len(SCOPES),
        'classification_configurations':192,'image_reconstruction_configurations':48,'results':rows,
        'new_SNN_training':0,'scope':'All models untrained. Clean-output fidelity controls only; no classification accuracy, image quality, content-selectivity or architecture ranking.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=run();write_json(a.output,r);print('Scoped configurations',r['configurations'],'conditions',r['scoped_conditions'])
