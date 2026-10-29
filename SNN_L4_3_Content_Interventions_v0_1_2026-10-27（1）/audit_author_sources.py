"""Execute pinned author functions only, on artificial or declared local inputs."""
import argparse,ast,contextlib,hashlib,importlib,io,json,sys,types,typing
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from bridge43 import ROOT,write_json
from intervention_math43 import fit_eraser,apply_eraser

def check_sources(root):
    records=json.loads((ROOT/'references/selected_author_sources.json').read_text())['files']
    for row in records:
        b=(Path(root)/row['repository']/row['path']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==row['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['blob_sha1']
    return records

def original_leace(root):
    folder=Path(root)/'EleutherAI/concept-erasure/concept_erasure'
    package=types.ModuleType('_l43_pinned_concept_erasure');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    return importlib.import_module(package.__name__+'.leace')

def selected_definitions(path,names):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    env=dict(vars(typing));env.update(np=np,scipy=scipy,torch=torch,nn=nn,sys=sys,gc=importlib.import_module('gc'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env

def audit(root):
    torch.set_num_threads(1);records=check_sources(root);root=Path(root);rng=np.random.default_rng(43043)
    author=original_leace(root);h=rng.normal(size=(2000,12))@rng.normal(size=(12,12));f=h[:,:2]+rng.normal(size=(2000,2))
    original=author.LeaceEraser.fit(torch.tensor(h),torch.tensor(f),shrinkage=False,constrain_cov_trace=False,svd_tol=1e-10)
    own=fit_eraser(h,f);gap=float(np.max(np.abs(original(torch.tensor(h)).numpy()-apply_eraser(h,own))))
    assert gap<1e-10
    tiny=f*1e-8;small=author.LeaceEraser.fit(torch.tensor(h),torch.tensor(tiny))
    default_rank=int(torch.linalg.matrix_rank(torch.eye(12,dtype=torch.float64)-small.P))
    own_rank=fit_eraser(h,tiny)['removed_rank']
    mp=selected_definitions(root/'efemeryds/amnesic-probing-with-single-projection/projection_methods/MPMethod.py',('MPMethod','assign_dep','get_labels','get_directions'))
    with np.errstate(invalid='ignore',divide='ignore'):
        zero=mp['get_directions'](np.tile([[1.,2.,3.]],(8,1)),np.array([0]*4+[1]*4))
    d=np.array([[1.,0.,0.]])
    with contextlib.redirect_stdout(io.StringIO()):pm,_=mp['MPMethod']('audit',[d]).debias_by_specific_directions([d],3)
    assert np.allclose(pm,np.diag([0,1,1]))
    inlp=selected_definitions(root/'yanaiela/amnesic_probing/amnesic_probing/debias/debias.py',('get_rowspace_projection','get_projection_to_intersection_of_nullspaces','debias_by_specific_directions'))
    pi=inlp['debias_by_specific_directions']([np.array([[1.,0.,0.]]),np.array([[0.,1.,0.]])],3)
    assert np.allclose(pi,np.diag([0,0,1]))
    pef=selected_definitions(root/'brcsomnath/PEF/src/pef.py',('_min_entropy','min_entropy'))
    p=np.array([.5,.3,.2]);q=np.array([.6,.4]);coupling=pef['min_entropy'](p,q)
    assert np.allclose(coupling.sum(1),np.sort(p)[::-1]) and np.allclose(coupling.sum(0),np.sort(q)[::-1])
    ds=selected_definitions(root/'Zishan-Shao/decodeshare/decodeshare/subspace.py',('JointSubspaceRemovalHook',))['JointSubspaceRemovalHook']
    x=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]]);u=np.array([[1.],[0.],[0.]])
    result=ds(0,u)(None,(),(x,'aux'))
    assert result[1]=='aux' and torch.equal(result[0],x*torch.tensor([0.,1.,1.]))
    degenerate=ds(0,np.zeros((3,1)))(None,(),x)
    assert ds(0,u,enabled=False)(None,(),x) is x
    return {'module':'L4.3 pinned author function audit','verified_source_files':len(records),
        'LEACE_full_rank_float64_equivalence_max_abs':gap,'LEACE_tiny_continuous_units':{'author_default_removed_rank':default_rank,'our_relative_removed_rank':own_rank,'scope':'Continuous diagnostic scaling; no claim that the one-hot paper theorem fails.'},
        'MP_equal_centroid_raw_direction_finite':bool(np.isfinite(zero).all()),'MP_explicit_dimension_projection_max_abs':float(np.max(np.abs(pm-np.diag([0,1,1])))),
        'MP_published_convenience_width':768,'INLP_two_direction_projection_max_abs':float(np.max(np.abs(pi-np.diag([0,0,1])))),
        'PEF_greedy_coupling_sorted_marginals_verified':True,
        'DecodeShare_valid_basis_max_abs':float(torch.max(torch.abs(result[0]-x*torch.tensor([0.,1.,1.])))),
        'DecodeShare_zero_basis_hook_change_max_abs':float(torch.max(torch.abs(degenerate-x))),
        'scope':'Original LEACE classes imported without edits; MP/INLP/coupling/hook definitions executed unchanged via AST extraction. No author model, dataset or full paper training was run. Degenerate hook input is an adapter boundary, not evidence that paper experiments used invalid bases. Bundled ACL LEACE wrapper and README inspected but not executed.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=audit(a.source_root);write_json(a.output,r);print(json.dumps(r,indent=2))
