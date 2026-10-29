"""Selected author functions only; no author top-level training is executed."""
import argparse,ast,json,types,hashlib
from pathlib import Path
from typing import Tuple
from unittest.mock import patch
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from artifacts35 import write_json

def functions(source,names,namespace):
    tree=ast.parse(source);selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:selected.append(node)
    if len(selected)!=len(names):raise ValueError('Pinned selected function absent')
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<selected author functions>','exec'),namespace)
    return namespace

def run(root):
    torch.set_num_threads(1);root=Path(root);meta=json.loads((Path(__file__).resolve().parent/'references/selected_source_manifest.json').read_text());checks={}
    for name,row in meta.items():
        b=(root/row['file']).read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if blob!=row['git_blob_sha']:raise ValueError((name,'source bytes do not match pinned blob'))
        checks[name]={'blob_bytes_match':True,'git_blob_sha':blob,'read_scope':row['read_scope']}
    nb=json.loads((root/'superposition_fig02.ipynb').read_text());code=''.join(nb['cells'][7]['source'])
    env=functions(code,['compute_ws_vectors','compute_theoretical_loss_components'],{'np':np,'torch':torch,'Tuple':Tuple,'BatchedRNN':object})
    obj=types.SimpleNamespace(Wx=torch.tensor([[[1.],[.2]]],dtype=torch.float64),Wh=torch.tensor([[[.6,-.2],[.2,.6]]],dtype=torch.float64),Wy=torch.tensor([[[.7],[-.3]]],dtype=torch.float64))
    got=env['compute_theoretical_loss_components'](obj,8,3,.3,.5,1/3)
    from temporal_theory import markov_parameters,sparse_delay_terms
    a=obj.Wh[0].numpy();b=obj.Wx[0].numpy();c=obj.Wy[0].numpy().T;kernel=markov_parameters(a,b,c,8).reshape(-1)
    own=np.zeros(5)
    for t in range(1,9):
        row=sparse_delay_terms(kernel[:t],3,.3,.5,1/3);own+=np.array([row[k] for k in ('total','task','projection','mean_correction','composition')])
    out={'selected_sources':checks,'source_count':len(checks),'superposition_four_terms_max_abs':float(np.max(np.abs(np.array(got)-own)))}
    env=functions((root/'rhythm_align.py').read_text(),['_slice_with_pad','_resample_to_bins'],{'np':np,'PadMode':str})
    after=env['_slice_with_pad'](np.ones((1,3)),5,7,'edge')
    out['rhythm_wholly_outside_slice']={'requested_length':2,'returned_length':after.shape[1],
        'scope':'Selected helper edge case. Does not establish that paper analyses requested such a window.'}
    trace=np.array([[0.,0.,1.]])
    out['offline_linear_resampling_example']=env['_resample_to_bins'](trace,5).tolist()
    env=functions((root/'mne_search_light.py').read_text(),['_gl_transform','_gl_init_pred'],{'np':np})
    class FakeEstimator:
        def __init__(self,bias):self.bias=bias
        def predict(self,x):return np.sum(x,axis=1)+self.bias
    x=np.arange(24.).reshape(3,2,4);pb=types.SimpleNamespace(update=lambda n:None);ests=[FakeEstimator(0),FakeEstimator(100)]
    got=env['_gl_transform'](ests,x,'predict',pb)
    expected=np.stack([np.stack([e.predict(x[:,:,t]) for t in range(4)],1) for e in ests],1)
    out['MNE_source_time_target_time_axes']={'shape':got.shape,'max_abs':float(np.max(np.abs(got-expected)))}
    env=functions((root/'rhythmsnn_model.py').read_text(),['mem_update_hidden_skip_woDecay','mem_update_adp_skip'],
        {'torch':torch,'decay':.8,'act_fun':lambda x:(x>=1).to(x.dtype),'act_fun_adp':lambda x:(x>=0).to(x.dtype),'b_j0':.01,'R_m':1.})
    v=torch.tensor([[.2]]);s=torch.zeros_like(v);mask=torch.zeros(1);a=torch.tensor([[.5]])
    mem,event=env['mem_update_hidden_skip_woDecay'](torch.ones_like(v),v,s,mask)
    # Device-only adaptation of extracted CUDA calls, explicitly scoped.
    with patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self):
        mv,se,threshold,aux=env['mem_update_adp_skip'](torch.zeros_like(v),v,s,torch.tensor([10.]),a,torch.tensor([20.]),mask)
    out['RhythmSNN_OFF_state']={'membrane':float(mem.item()),'event':float(event.item()),'ALIF_membrane':float(mv.item()),'ALIF_aux_before':.5,'ALIF_aux_after':float(aux.item()),
        'scope':'Two extracted update functions; Tensor.cuda replaced by identity only for CPU device placement. No author training/SG backward/model runs.'}
    rng=np.random.default_rng(3608);x=rng.normal(size=(32,3));y=2*x
    q,scale=orthogonal_procrustes(y,x)
    out['SciPy_orthogonal_scale_semantics']={'returned_scale':float(scale),'sum_singular_values':float(np.linalg.svd(y.T@x,compute_uv=False).sum()),
        'similarity_multiplier_would_be':.5,'matrix_determinant':float(np.linalg.det(q))}
    out['scope']='Pinned source checks and selected functions. No complete author paper reproduction, training, EEG or hardware.'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    write_json(dst,run(a.source_directory));print('Selected author function audit completed')
