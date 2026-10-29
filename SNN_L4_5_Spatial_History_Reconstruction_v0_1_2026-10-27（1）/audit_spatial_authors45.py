"""Selected original definitions on artificial inputs; no external models or datasets."""
import argparse,ast,hashlib,importlib.util,math,warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional,Tuple,Literal
import numpy as np
import torch
from torch import nn,Tensor
from bridge45 import ROOT,read_json,dump
from spatial_data45 import batch,latest_visible
from spatial_metrics45 import spectral_angle

def definitions(path,names,class_name=None,namespace=None):
    tree=ast.parse(Path(path).read_text());nodes=tree.body
    if class_name:nodes=next(v for v in nodes if isinstance(v,ast.ClassDef) and v.name==class_name).body
    chosen=[v for v in nodes if isinstance(v,(ast.FunctionDef,ast.ClassDef)) and v.name in names]
    assert set(v.name for v in chosen)==set(names)
    ns={'torch':torch,'nn':nn,'Tensor':Tensor,'np':np,'math':math,'Optional':Optional,'Tuple':Tuple,'Literal':Literal,**(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),str(path),'exec'),ns)
    return ns

def audit(source_root):
    torch.set_num_threads(1);r=Path(source_root);manifest=read_json(ROOT/'references/selected_author_sources.json');verified=0
    for row in manifest['files']:
        p=r/row['repository']/row['path'];b=p.read_bytes()
        assert hashlib.sha256(b).hexdigest()==row['sha256'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['blob_sha1'];verified+=1
    z=definitions(r/'End2End-Diffusion/iREPA/jit/spnorm.py',['spatial_zscore'])['spatial_zscore']
    g=torch.Generator().manual_seed(451234);x=torch.randn(2,16,4,generator=g,dtype=torch.float64);a=z(x);offset=z(x+3.)
    shift_gap=float((a-offset).abs().max());assert shift_gap<1e-12
    constant=z(torch.ones_like(x));assert torch.all(constant==0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');single=z(x[:,:1])
    mean=x.mean(1,keepdim=True);std=x.std(1,keepdim=True);recovered=a*(std+1e-6)+mean
    recover_gap=float((recovered-x).abs().max());assert recover_gap<1e-12
    source=r/'End2End-Diffusion/iREPA/metrics/spatial_metrics.py';spec=importlib.util.spec_from_file_location('author_spatial_metrics45',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokens=torch.randn(2,64,4,generator=g);lds=module.metric_lds(tokens,8,8,far_dist=4)
    t=torch.nn.functional.normalize(tokens,dim=-1).reshape(2,8,8,4);local=torch.cat([(t[:,1:]*t[:,:-1]).sum(-1).flatten(1),(t[:,:,1:]*t[:,:,:-1]).sum(-1).flatten(1)],1).mean(1)
    coords=torch.tensor(list(np.ndindex(8,8)));dist=(coords[:,None]-coords[None]).abs().sum(-1);mask=dist>=4
    u=t.reshape(2,64,4);manual=local-(u@u.transpose(1,2))[:,mask].mean(1);lds_gap=float((lds-manual).abs().max());assert lds_gap<1e-6
    sam=definitions(r/'prs-eth/U-TILISE/lib/metrics.py',['compute_sam'])['compute_sam']
    p=torch.rand(2,4,4,4,generator=g,dtype=torch.float64);target=torch.rand(2,4,4,4,generator=g,dtype=torch.float64)
    original=float(sam(p,target));ours=np.mean(spectral_angle(p.numpy(),target.numpy())['scene_mean_radians']);sam_gap=abs(original-ours);assert sam_gap<1e-12
    assert sam(torch.zeros_like(p),target) is None
    interp=definitions(r/'prs-eth/U-TILISE/lib/models/ImageSeriesInterpolator.py',['_find_last_visible'],class_name='ImageSeriesInterpolator')['_find_last_visible']
    data=batch(4,45430000,noise=0);visible=interp(data['x'][:,:,:4],1-data['x'][:,:,4:5]);last=visible[:,-1]
    assert np.all(last[:,data['masks']['H']]==2) and np.all(last[:,data['masks']['C']]==5) and np.isnan(last[:,data['masks']['U_vis']]).all()
    decoder_source=r/'bytetriper/RAE/src/stage1/decoders/decoder.py';unpatch=definitions(decoder_source,['unpatchify'],class_name='GeneralDecoder')['unpatchify']
    image=torch.arange(2*3*4*4,dtype=torch.float32).reshape(2,3,4,4)
    patches=image.reshape(2,3,2,2,2,2).permute(0,2,4,3,5,1).reshape(2,4,12)
    fixture=SimpleNamespace(config=SimpleNamespace(patch_size=2,num_channels=3,image_size=4))
    restored=unpatch(fixture,patches);assert torch.equal(restored,image)
    invalid=False
    try:unpatch(fixture,patches[:,:3])
    except ValueError:invalid=True
    assert invalid
    evsource=r/'LinZhu111/EVSNN/neurons/spiking_neuron.py'
    charge=definitions(evsource,['neuronal_charge'],class_name='MpLIFNode')['neuronal_charge']
    fixture=SimpleNamespace(v=torch.tensor([.4,.8]),v_reset=0.,tau=2.);inp=torch.tensor([.2,1.2]);expected=fixture.v+(inp-fixture.v)/2
    charge(fixture,inp);assert torch.equal(fixture.v,expected)
    mpforward=definitions(evsource,['forward'],class_name='MpNode')['forward']
    fixture.neuronal_charge=lambda x:charge(fixture,x);fixture.register_memory=lambda name,v:setattr(fixture,name,v)
    out=mpforward(fixture,torch.tensor([.2,1.2]),torch.tensor([.6,.4]));assert torch.allclose(out,torch.tensor([.4,.8]),atol=1e-7,rtol=0)
    cdist=definitions(r/'lqnankai/MuSpike/evaluation/objective_evaluation/mgeval/utils.py',['c_dist'])['c_dist']
    distances=cdist(np.array([1.,2.]),np.array([[1.,2.],[2.,2.]]));assert np.array_equal(distances,np.array([0.,1.]))
    return {'module':'L4.5 selected author code audit','fixed_source_files_verified':verified,
        'iREPA_spatial_zscore_translation_invariance_max_abs':shift_gap,'iREPA_mean_scale_restore_max_abs':recover_gap,
        'iREPA_single_token_sample_std_is_nonfinite':bool(not torch.isfinite(single).all()),
        'iREPA_original_LDS_manual_4neighbor_max_abs':lds_gap,
        'LDS_scope':'Author code uses 4-neighbor local average; our broader local-distance contrast is a named adapted variant, not numerically equated.',
        'UTILISE_original_SAM_max_abs':sam_gap,'UTILISE_zero_norm_returns_None':True,'UTILISE_original_last_visible_indices_verified':True,
        'RAE_original_unpatchify_exact':True,'RAE_invalid_patch_count_rejected':True,
        'EVSNN_original_MP_charge_and_state_handoff_verified':True,
        'EVSNN_scope':'Selected original methods; artificial state fixture and supplied registration, not the full SpikingJelly class or paper model.',
        'MuSpike_original_Euclidean_metric_verified':True,
        'scope':'Original selected AST definitions and standalone iREPA metric module on artificial inputs. No pretrained vision model, satellite/event/music dataset, diffusion training, SpikingJelly model training or full paper reproduction.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=audit(a.source_root);dump(a.output,v);print('Author text files verified',v['fixed_source_files_verified'])
