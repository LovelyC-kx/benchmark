"""Untrained backbone x cell x phase x head factorization/stream checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bridge35 import BenchmarkModel,ModelConfig,BACKBONES,NEURONS
from access_operators import MODES
from native_head_audit import audit_model
from artifacts35 import write_json

def run():
    torch.set_num_threads(1);rows=[]
    for objective,backbones in [('classification',BACKBONES),('reconstruction',('cnn2d','unet'))]:
        for backbone in backbones:
            for cell in NEURONS:
                for phase in ('output','pre','post'):
                    modes=MODES if objective=='classification' else MODES[:-1]
                    for mode in modes:
                        torch.manual_seed(4101728)
                        image=backbone in ('cnn2d','sew_resnet18','spike_attention','unet')
                        shape=(1,8,8) if image else (4,);output_shape=(1,8,8) if objective=='reconstruction' else (3,)
                        cfg=ModelConfig(backbone=backbone,neuron=cell,width=2,depth=2,observation=phase,readout=mode)
                        model=BenchmarkModel(shape,output_shape,3,cfg,objective=objective).eval()
                        x=np.random.default_rng(4101728).normal(size=(2,3,*shape)).astype(np.float32)
                        audit=audit_model(model,x)
                        if audit['carrier_to_features_max_abs']>2e-5 or audit['manual_GPR_prediction_max_abs']>2e-5 or audit['stream_vs_batch_prediction_max_abs']>2e-5:
                            raise AssertionError((objective,backbone,cell,phase,mode,audit))
                        if mode!='trajectory':assert not any('features.' in r['path'] for r in audit['stream_registry']['rows'])
                        rows.append({'objective':objective,'backbone':backbone,'cell':cell,'trained':False,**audit})
            print('completed interface grid',objective,backbone,flush=True)
    return {'module':'L4.1 deployed G/P/R interface grid','classification_cases':1440,'image_reconstruction_cases':288,
            'cases':rows,'new_SNN_optimizer_updates':0,
            'scope':'1728 untrained numerical factorization/streaming cases; source implementations are reference skeletons, not author full-model reproductions or trained performance comparisons. Fixed-T trajectory image head is explicitly unsupported, not padded into the grid.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    write_json(out,run());print('Interface grid completed')
