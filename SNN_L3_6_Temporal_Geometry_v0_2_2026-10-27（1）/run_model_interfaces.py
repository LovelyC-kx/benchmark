"""Untrained passive temporal capture across cells, backbones and native heads."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bridge35 import NEURONS,BACKBONES,BenchmarkModel,ModelConfig
from temporal_capture import named_temporal_bank
from artifacts35 import write_json

def run():
    torch.set_num_threads(1);rows=[]
    for backbone in BACKBONES:
        for kind in NEURONS:
            torch.manual_seed(360800)
            image=backbone in ('cnn2d','sew_resnet18','spike_attention','unet')
            shape=(1,8,8) if image else (4,)
            cfg=ModelConfig(backbone=backbone,neuron=kind,width=2,depth=2,normalization='none')
            model=BenchmarkModel(shape,(3,),3,cfg).eval()
            x=np.random.default_rng(3608).normal(size=(2,3,*shape)).astype(np.float32)
            with torch.no_grad():plain=model(torch.from_numpy(x)).numpy()
            observed=named_temporal_bank(model,x,batch_size=2)
            np.testing.assert_array_equal(observed['prediction'],plain)
            for key,a in observed['carriers'].items():
                spec=observed['view_specification'][key]
                assert a.shape[:2]==(2,3) and a.shape[2]==np.prod(spec['unflattened_feature_shape'])
            outputs=[k for k in observed['carriers'] if k.endswith('/output')]
            rows.append({'backbone':backbone,'cell':kind,'objective':'classification','trained':False,
                'prediction_max_abs':float(np.max(np.abs(plain-observed['prediction']))),
                'cell_modules':len(outputs),'views':len(observed['carriers']),
                'complete_spatial_coordinates':True,'actual_output_included':True})
    for backbone in ('cnn2d','unet'):
        for kind in NEURONS:
            torch.manual_seed(360801);cfg=ModelConfig(backbone=backbone,neuron=kind,width=2,depth=2)
            model=BenchmarkModel((1,8,8),(1,8,8),3,cfg,objective='reconstruction').eval()
            x=np.random.default_rng(3609).normal(size=(2,3,1,8,8)).astype(np.float32)
            with torch.no_grad():plain=model(torch.from_numpy(x)).numpy()
            observed=named_temporal_bank(model,x,batch_size=2)
            np.testing.assert_array_equal(observed['prediction'],plain)
            assert plain.shape==(2,1,8,8)
            rows.append({'backbone':backbone,'cell':kind,'objective':'reconstruction','trained':False,
                'prediction_max_abs':float(np.max(np.abs(plain-observed['prediction']))),
                'native_output_shape':list(plain.shape),'actual_output_included':True})
    return {'module':'L3.6 named temporal observation compatibility','classification_cases':96,'native_image_reconstruction_cases':24,
            'cases':rows,'new_SNN_optimizer_updates':0,'scope':'120 untrained forward interface cases, not 120 trained benchmark runs. Cell-state observations retain spatial axes; caches, functional merges and readout accumulators are outside this bank. Primary LoadModel loader remains a two-layer MLP adapter.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    r=run();write_json(dst,r);print('Model interface cases',len(r['cases']))
