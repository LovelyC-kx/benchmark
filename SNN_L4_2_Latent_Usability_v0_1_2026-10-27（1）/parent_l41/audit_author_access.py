"""Selected pinned author functions/expressions; no training/model reproduction."""
import argparse,ast,hashlib,json
from pathlib import Path
import torch
from torch import nn
from artifacts35 import write_json
ROOT=Path(__file__).resolve().parent

def exact_discrete_mi(x,y):
    # Declared dependency substitution: finite contingency MI in natural units.
    x,y=x.flatten().long(),y.flatten().long();_,a=torch.unique(x,return_inverse=True);_,b=torch.unique(y,return_inverse=True)
    n_b=int(b.max())+1;p=torch.bincount(a*n_b+b,minlength=(int(a.max())+1)*n_b).double().reshape(-1,n_b);p/=p.sum()
    independent=p.sum(1,keepdim=True)@p.sum(0,keepdim=True);mask=p>0
    return (p[mask]*torch.log(p[mask]/independent[mask])).sum().to(x.device)

def extract(source,names,environment):
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('Pinned selected definitions absent')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned author definitions>','exec'),environment)
    return environment

def run(source_directory):
    torch.set_num_threads(1);source=Path(source_directory);manifest=json.loads((ROOT/'references/selected_source_manifest.json').read_text());verified={}
    for name,row in manifest.items():
        b=(source/row['file']).read_bytes();sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if sha!=row['git_blob_sha']:raise ValueError((name,'Git blob mismatch'))
        verified[name]={'git_blob_sha':sha,'bytes_match':True,'scope':row['read_scope']}
    text=(source/'trqr_models.py').read_text()
    env=extract(text,['cal_batch_both_mi_mask_devidet0','TMIModule'],{'torch':torch,'nn':nn,'mutual_info_score':exact_discrete_mi})
    torch.manual_seed(41010);x=torch.randint(0,2,(4,3,2,2,3)).float()
    probs=env['cal_batch_both_mi_mask_devidet0'](x)
    constant=env['cal_batch_both_mi_mask_devidet0'](torch.zeros_like(x))
    module=env['TMIModule']().eval();original=x.clone();evaluated=module(x,100)
    assert torch.equal(x,original) and evaluated is x
    tet=extract((source/'trqr_functions.py').read_text(),['TET_loss'],{'torch':torch,'nn':nn})['TET_loss']
    logits=torch.tensor([[[20.,0.],[0.,2.],[0.,2.]]]);labels=torch.tensor([0]);criterion=nn.CrossEntropyLoss()
    actual=tet(logits,labels,criterion,1.,0.)
    expected=sum(criterion(logits[:,t],labels) for t in range(3))/3
    # Execute the exact three weighted-output assignments, without their train loop.
    train_tree=ast.parse((source/'trqr_imagenet_train.py').read_text());assignments={}
    for node in ast.walk(train_tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('firing_num','firing_num_tw') or (name=='output' and 'firing_num_tw' in ast.unparse(node.value)):
                assignments[name]=node
    chosen=[assignments[n] for n in ('firing_num','firing_num_tw','output')]
    code=compile(ast.Module(body=chosen,type_ignores=[]),'<selected author recalibration assignments>','exec')
    outputs=torch.tensor([[[1.,0.]],[[0.,1.]]]);env2={'output':outputs.clone(),'firing_num_t':torch.tensor([1.,3.])};exec(code,env2)
    return {'module':'L4.1 selected author access audit','verified_sources':verified,'source_count':len(verified),
       'MI_sample_axis':'CIFAR function reshapes each time to (batch*channels, spatial positions); global MI flattens all entries, local MI samples batch*channels at one position',
       'dependency_substitution':'mutual_info_score replaced with an exact finite-contingency MI implementation in natural units; original TorchMetrics package/runtime not executed',
       'mask_probability_shape':list(probs.shape),'constant_input_nonfinite_probability_entries':int((~torch.isfinite(constant)).sum()),
       'eval_mask_module_returns_input_unchanged':True,'TET_loss_lambda0_max_abs':float(torch.abs(actual-expected)),
       'TET_loss_value':float(actual),'weighted_training_assignment_output':env2['output'].numpy(),
       'selected_code_contracts':{'cifar_train_and_eval':'outputs.mean(1), except TET changes the training loss',
          'imagenet_selected_amp_training':'batch-aggregated firing weights then sum over time',
          'imagenet_selected_evaluation':'conditional output.dim()==3 branch uses output[0] after loss computation'},
       'scope':'Selected definitions and expressions executed, with a disclosed MI dependency substitution. No author full model, CUDA/AMP training, validation accuracy or dataset reproduction. Source-specific branch observations do not establish that a paper result is invalid.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    write_json(out,run(a.source_directory));print('Author access audit completed')
