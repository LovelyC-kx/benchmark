"""Standard SSIM, shape interfaces and geometry checks, using the frozen evidence contract."""
import argparse
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity
import skimage
from bridge45 import ROOT,CONFIGS,VIEWS,NEURONS,model_for,site_paths,parameter_hash,load_tree,read_json,dump
from spatial_data45 import batch,latest_visible
from spatial_readers45 import aggregate,decode
from spatial_runtime45 import run

def ssim_scenes(pred,target):
    pred=np.clip(np.asarray(pred,float),0,1);target=np.asarray(target,float)
    if pred.shape!=target.shape or pred.ndim!=4 or not np.isfinite(pred).all() or not np.isfinite(target).all():raise ValueError('Matching finite NCHW')
    return np.array([structural_similarity(y,p,data_range=1.,channel_axis=0,win_size=7) for p,y in zip(pred,target)])

def execute(output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);f=read_json(ROOT/'evidence/fitted/freeze_contract.json');bank=load_tree(ROOT/'evidence/fitted/bank')
    a=batch(128,f['pools']['confirmation']['first_id']);rows={};arrays={}
    for name,p in [('current_only',latest_visible(a['x'][:,-1:])),('latest_valid_history',latest_visible(a['x']))]:
        z=ssim_scenes(p,a['y']);rows[name]={'mean_sequence_SSIM':float(z.mean()),'n_sequences':len(z)};arrays[name]=z
    from bridge45 import config_key
    for b,n in CONFIGS:
        key=config_key(b,n);m=model_for(b,n);r=run(m,a['x'],sites=())
        for view in VIEWS:
            for kind,reader in bank[key]['decoders'][view].items():
                p=decode(reader,aggregate(r['features'],view));z=ssim_scenes(p,a['y']);tag=key+'::'+view+'::'+kind
                rows[tag]={'mean_sequence_SSIM':float(z.mean()),'n_sequences':len(z)};arrays[tag]=z
    interfaces=[]
    for b in ('cnn2d','unet'):
        for n in NEURONS:
            m=model_for(b,n);before=parameter_hash(m);v=run(m,a['x'][:2],sites=site_paths(m))
            assert v['features'].shape==(2,6,8,16,16) and v['prediction'].shape==(2,4,16,16) and parameter_hash(m)==before
            interfaces.append({'backbone':b,'neuron':n,'features_shape':list(v['features'].shape),'image_shape':list(v['prediction'].shape),
                'state_owners':v['captured_cell_state_sizes_per_scene'],'trained':False})
    # Test the scientific primitive against known identical and contrast-reversed images.
    assert np.array_equal(ssim_scenes(a['y'],a['y']),np.ones(len(a['y'])))
    x=np.arange(16*16,dtype=float).reshape(1,1,16,16)/255.;assert float(ssim_scenes(1-x,x)[0])<0
    np.savez_compressed(out/'scene_SSIM.npz',**arrays)
    result={'module':'L4.5 additional quality and image interfaces','skimage_version':skimage.__version__,
        'SSIM_definition':'skimage structural_similarity; NCHW per scene, channel_axis=0, win_size=7, data_range=1, default uniform weights/sample covariance; clipped predictions; arithmetic mean over independent scenes.',
        'confirmation_subset_scenes':128,'SSIM_conditions':len(rows),'SSIM':rows,'untrained_spatial_interfaces':len(interfaces),'interfaces':interfaces,
        'LPIPS':'Not computed: no pretrained perceptual weights downloaded.','scope':'SSIM diagnostic and new 5-channel/4-band shape validation only; no new SNN training, no dataset or energy result.'}
    dump(out/'report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();r=execute(a.output);print('SSIM',r['SSIM_conditions'],'interfaces',r['untrained_spatial_interfaces'])
