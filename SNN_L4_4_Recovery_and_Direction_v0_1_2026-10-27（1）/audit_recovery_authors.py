"""Execute selected pinned author definitions; no author datasets or checkpoints."""
import argparse,ast,collections,hashlib,importlib.util,json,sys,types,typing
from pathlib import Path
import numpy as np
import torch
from bridge44 import ROOT,write_json

def definitions(path,names,extra=None):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    env=dict(vars(typing));env.update(np=np,numpy=np,torch=torch,defaultdict=collections.defaultdict,__name__='_l44_original_definitions')
    if extra:env.update(extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env

def audit(source_root):
    torch.set_num_threads(1);root=Path(source_root);manifest=json.loads((ROOT/'references/selected_author_sources.json').read_text())
    for row in manifest['files']:
        b=(root/row['repository']/row['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['blob_sha1']
    ds=root/'Zishan-Shao/decodeshare/experiments/03_patchback/subspace_patching_transfer.py'
    env=definitions(ds,('SubspacePatchHook','orthonormalize_np','project_cpu','sample_random_orthonormal_basis','energy_matched_random_vector_in_subspace','shuffle_coeffs_in_subspace','summarize_rescue'))
    q=np.eye(3,dtype=np.float32)[:,:1];x=torch.tensor([[[1.,2.,3.]],[[4.,5.,6.]]]);donor=torch.tensor([[9.,0,0],[8.,0,0]])
    hook=env['SubspacePatchHook'](q,{0:donor},{0});expected=x.clone();expected[:,0,0]=torch.tensor([9.,8.])
    patched=hook(None,(),(x,'cache'));assert torch.equal(patched[0],expected) and patched[1]=='cache'
    hook.reset();prefill=x.repeat(1,2,1);assert hook(None,(),prefill) is prefill and hook.step==0
    target=torch.tensor([1.,2.,3.,4.]);basis=env['sample_random_orthonormal_basis'](5,2,44);sample=env['energy_matched_random_vector_in_subspace'](basis,target,44)
    norm_gap=float(torch.max(torch.abs(torch.linalg.norm(sample,dim=1)-target)));assert norm_gap<2e-6
    zero_span=env['energy_matched_random_vector_in_subspace'](np.zeros((5,0),np.float32),torch.ones(3),44)
    empty=env['summarize_rescue']([],'patched');assert np.isnan(empty['rescued_pct'])
    summary=env['summarize_rescue']([{'patched':{'correct':True,'margin':2.},'ablated':{'margin':0.}},
        {'patched':{'correct':False,'margin':-1.},'ablated':{'margin':-2.}}],'patched');assert summary['rescued_pct']==50.
    # Load original instrumentation dependency, then execute the original tracing functions
    # on a small artificial named Torch network. This is not a GPT reproduction.
    netfile=root/'kmeng01/rome/util/nethook.py';spec=importlib.util.spec_from_file_location('_l44_original_nethook',netfile);net=importlib.util.module_from_spec(spec);sys.modules[spec.name]=net;spec.loader.exec_module(net)
    re=definitions(root/'kmeng01/rome/experiments/causal_trace.py',('trace_with_patch','trace_with_repatch','layername'),{'nethook':net})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.transformer=torch.nn.Module();self.transformer.wte=torch.nn.Embedding(8,3);self.transformer.h=torch.nn.ModuleList([torch.nn.Identity()]);self.head=torch.nn.Linear(3,8,bias=False)
        def forward(self,input_ids):
            h=self.transformer.wte(input_ids);h=self.transformer.h[0](h);return types.SimpleNamespace(logits=self.head(h))
    torch.manual_seed(44444);model=Tiny().eval();inp={'input_ids':torch.tensor([[1,2]]*4)};answer=torch.tensor([2])
    with torch.no_grad():clean=torch.softmax(model(**inp).logits[0,-1],dim=0)[answer]
    damaged=re['trace_with_patch'](model,inp,[],answer,(1,2),noise=1.)
    restored=re['trace_with_patch'](model,inp,[(1,'transformer.h.0')],answer,(1,2),noise=1.)
    blocked=re['trace_with_repatch'](model,inp,[(1,'transformer.h.0')],[(1,'transformer.h.0')],answer,(1,2),noise=1.)
    restore_gap=float(torch.max(torch.abs(clean-restored)));block_gap=float(torch.max(torch.abs(blocked-damaged)))
    assert restore_gap<1e-7 and block_gap<1e-7
    return {'module':'L4.4 author source audit','verified_source_files':len(manifest['files']),
        'DecodeShare_canonical_patch_hook_max_abs':float(torch.max(torch.abs(patched[0]-expected))),
        'DecodeShare_tuple_and_prefill_policies_verified':True,'DecodeShare_per_row_random_norm_gap':norm_gap,
        'DecodeShare_zero_dimensional_span_nonzero_requested_norm_returned_norm':float(torch.linalg.norm(zero_span)),
        'DecodeShare_empty_flip_set_percentage_is_NaN':bool(np.isnan(empty['rescued_pct'])),
        'DecodeShare_nonempty_conditional_rescue_percentage':summary['rescued_pct'],
        'ROME_original_trace_restore_clean_probability_max_abs':restore_gap,
        'ROME_original_repatch_damage_probability_max_abs':block_gap,
        'scope':'Canonical DecodeShare definitions and ROME trace_with_patch/trace_with_repatch/layername executed unchanged via AST; original nethook module imported unchanged. Artificial Torch inputs only; no pretrained LLM or author dataset. Legacy DecodeShare files reviewed for provenance, not counted as another experiment.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    r=audit(a.source_root);write_json(a.output,r);print(json.dumps(r,indent=2))
