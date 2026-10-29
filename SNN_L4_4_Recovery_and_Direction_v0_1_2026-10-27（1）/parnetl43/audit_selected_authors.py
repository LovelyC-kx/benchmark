"""Execute selected author definitions, not whole model or dataset pipelines."""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from bridge42 import ROOT, write_json
from latent_theory42 import spectral_ranks,safe_whitener

def definitions(path,names,scope):
    tree=ast.parse(Path(path).read_text())
    chosen=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in chosen}!=set(names):raise ValueError('Selected author definitions missing')
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),scope)
    return scope

def audit(source_directory):
    src=Path(source_directory);manifest=json.loads((ROOT/'references/selected_source_manifest.json').read_text());verified={}
    for key,m in manifest.items():
        data=(src/m['file']).read_bytes();sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if sha!=m['git_blob_sha']:raise ValueError('Author bytes differ: '+key)
        verified[key]={'git_blob_sha':sha,'bytes':len(data),'bytes_match':True}
    env=definitions(src/'wakhloo_utils.py',['theta','theory_part','emp_error','get_geometry'],{'np':np})
    rng=np.random.default_rng(42);b=rng.normal(size=(6,3));om=np.diag([1.,.5,2.]);phi=b@om;psi=b@om@b.T+.4*np.eye(6)
    raw=float(env['theory_part'](psi,phi,om,128));geo=env['get_geometry'](psi,phi,om,128)
    np.random.seed(42);z=np.random.randn(1024,3)
    tie=float(env['emp_error'](np.zeros((1024,6)),z,128,n_task=256,min_task=100)[0])
    norm=definitions(src/'layer_metrics.py',['normalize','entropy_normalization'],{'torch':torch,'math':math})
    with torch.no_grad():normalized=norm['normalize'](torch.zeros((8,4)))
    category=definitions(src/'concept_from_models.py',['get_g'],{'torch':torch})
    # Stub only the pretrained embedding loader; selected get_g runs unchanged.
    gamma=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],dtype=torch.float64)
    category['get_gamma']=lambda name,device:gamma
    g,inv,root=category['get_g']('artificial full-rank embeddings','cpu')
    fullgap=float(torch.max(torch.abs(g.T@g/len(g)-torch.eye(2))).item())
    category['get_gamma']=lambda name,device:torch.tensor([[1.,0.],[2.,0.],[3.,0.]],dtype=torch.float64)
    deficient=category['get_g']('artificial rank-deficient embeddings','cpu')[0]
    own=safe_whitener(np.array([[1.,0.],[2.,0.],[3.,0.]]))
    repa=definitions(src/'repa_loss.py',['mean_flat','sum_flat','SILoss'],{'torch':torch,'np':np})
    class FakeModel:
        def __init__(self,features):self.features=features
        def __call__(self,x,t,**kw):return torch.zeros_like(x),self.features
    zs=[torch.ones((2,3,4))];images=torch.zeros((2,1,2,2));repa_results={}
    for name,features in [('aligned',[zs[0]*2]),('reversed',[-zs[0]]),('zero',[zs[0]*0])]:
        torch.manual_seed(42);den,pro=repa['SILoss']()(FakeModel(features),images,zs=zs)
        repa_results[name]={'projection_loss':float(pro),'denoising_loss_finite':bool(torch.isfinite(den).all())}
    reg=ast.parse((src/'reg_loss.py').read_text())
    reg_symbols=[n.name for n in reg.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    rae=ast.parse((src/'rae_stage1.py').read_text())
    rc=next(n for n in rae.body if isinstance(n,ast.ClassDef) and n.name=='RAE')
    methods=[n for n in rc.body if isinstance(n,ast.FunctionDef) and n.name in ('encode','decode','noising')]
    re={'torch':torch,'nn':torch.nn,'sqrt':math.sqrt}
    exec(compile(ast.Module(body=methods,type_ignores=[]),'selected RAE methods','exec'),re)
    FakeRAE=type('FakeRAE',(),{n.name:re[n.name] for n in methods})
    class FakeDecoder:
        def __call__(self,z,**kwargs):
            class Result:pass
            r=Result();r.logits=z;return r
        def unpatchify(self,z):return z.transpose(1,2).reshape(len(z),3,2,2)
    r=FakeRAE();r.encoder_input_size=2;r.encoder_mean=torch.zeros(1,3,1,1);r.encoder_std=torch.ones(1,3,1,1)
    r.encoder=lambda x:x.flatten(2).transpose(1,2);r.decoder=FakeDecoder();r.training=False;r.noise_tau=0.
    r.reshape_to_2d=True;r.do_normalization=True;r.eps=1e-5;r.latent_mean=torch.ones(1,3,1,1)*.2;r.latent_var=torch.ones(1,3,1,1)*2
    image=torch.arange(24,dtype=torch.float32).reshape(2,3,2,2).requires_grad_(True)
    latent=r.encode(image);reconstructed=r.decode(latent)
    result={'module':'L4.2 selected author checks','source_count':len(manifest),'verified_sources':verified,
        'wakhloo_theory_formulations_absolute_difference':abs(raw-float(geo[-1])),
        'wakhloo_Gaussian_Hebbian_predicted_error':raw,'wakhloo_zero_code_original_emp_error':tie,
        'our_zero_prediction_balanced_classification_tie_loss':.5,
        'layer_normalize_zero_input_nonfinite_elements':int((~torch.isfinite(normalized)).sum()),
        'concept_get_g_full_rank_whitening_max_abs':fullgap,
        'concept_get_g_rank_deficient_is_finite':bool(torch.isfinite(deficient).all()),
        'adapted_range_whitener_rank':own['rank'],'REPA_artificial_forward':repa_results,
        'REG_top_level_symbols_read_not_executed':reg_symbols,
        'RAE_selected_methods_artificial_identity_roundtrip_max_abs':float(torch.max(torch.abs(reconstructed-image)).detach()),
        'RAE_selected_encode_detaches_gradient':not latent.requires_grad,
        'spectral_inequality_definition_check':spectral_ranks(np.array([[1.,0.],[.6,.8],[-1.,0.],[-.6,-.8]])),
        'execution_scope':'AST-selected original definitions. get_gamma uses artificial tensors; RAE encode/decode use identity stubs and explicit normalization. No original LLM loading, sklearn LDA, repitl entropy, REG model, full RAE, diffusion training or author datasets executed.',
        'interpretation':'Degenerate-input boundary checks motivate explicit SNN tie/silence/rank handling. They do not establish failures in the papers reported experiments.'}
    assert result['wakhloo_theory_formulations_absolute_difference']<1e-12
    assert fullgap<1e-12 and repa_results['aligned']['projection_loss']==-1.
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=audit(a.source_directory);write_json(a.output,r);print('Verified selected author sources:',r['source_count'])
