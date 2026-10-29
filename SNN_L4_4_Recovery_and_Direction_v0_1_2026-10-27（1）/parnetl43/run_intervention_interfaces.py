"""Untrained no-op/phase interfaces, not trained mechanism comparisons."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bridge43 import BenchmarkModel,ModelConfig,Cell,NEURONS,BACKBONES,parameter_hash,write_json
from runtime_interventions43 import run_post_edit

def run():
    torch.set_num_threads(1);rows=[]
    for objective,bones in [('classification',BACKBONES),('reconstruction',('cnn2d','unet'))]:
        for backbone in bones:
            for cell in NEURONS:
                for readout in ('last','mean'):
                    torch.manual_seed(430240)
                    image=backbone in ('cnn2d','sew_resnet18','spike_attention','unet')
                    shape=(1,8,8) if image else (4,);target=(1,8,8) if objective=='reconstruction' else (3,)
                    cfg=ModelConfig(backbone=backbone,neuron=cell,width=2,depth=2,observation='post',readout=readout)
                    model=BenchmarkModel(shape,target,3,cfg,objective=objective).eval()
                    x=np.random.default_rng(430240).normal(size=(2,3,*shape)).astype(np.float32)
                    # Attention selects different kernels with/without autograd.
                    # Both members of a no-op control must use inference mode.
                    with torch.no_grad():original=model(torch.as_tensor(x)).numpy()
                    before=parameter_hash(model)
                    path=next(n for n,m in model.named_modules() if m is model.backbone.last_cell)
                    # Match the reference execution's batching, dtype and capture flag.
                    # Batch-size-dependent GEMM kernels need not be bitwise identical.
                    changed=run_post_edit(model,x,path,1,lambda h,start:h,batch_size=len(x),keep_traces=False)
                    gap=float(np.max(np.abs(changed['prediction']-original)))
                    if gap!=0 or before!=parameter_hash(model):raise AssertionError((objective,backbone,cell,readout,gap))
                    rows.append({'objective':objective,'backbone':backbone,'cell':cell,'readout':readout,'path':path,'noop_max_abs':gap,'parameters_unchanged':True,'trained':False})
        print('verified hook interfaces',objective,flush=True)
    return {'module':'L4.3 hook interface controls','cases':rows,'case_count':len(rows),'classification_cases':192,'image_reconstruction_cases':48,
        'scope':'Width2,T3,normalization none, post observation, one Cell call per step. No-op hook compatibility only; not content-selective intervention validity or trained performance.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=run();write_json(a.output,r);print('Intervention interface cases',r['case_count'])
