"""Scoped operational snapshot restoration after a fully consumed step."""
import torch
from bridge44 import clone_tree

SCOPES=('none','v','aux','cell','cache','readout','core','system')
def zero_snapshot(snapshot):
    result=clone_tree(snapshot)
    for row in result['cells'].values():
        row['state']={k:torch.zeros_like(v) for k,v in row['state'].items()}
        row['last']={k:torch.zeros_like(v) for k,v in row['last'].items()}
    result['cache']=[torch.zeros_like(v) for v in result['cache']]
    for key in ('accumulator','last_feature'):
        if result['readout'][key] is not None:result['readout'][key]=torch.zeros_like(result['readout'][key])
    result['readout']['features']=[torch.zeros_like(v) for v in result['readout']['features']]
    # Count, absolute clock and normalization denominator retain their meanings.
    return result

def restore_scope(recipient,clean,scope):
    if scope not in SCOPES:raise ValueError('Declared operational scope')
    if recipient['model_fingerprint']!=clean['model_fingerprint'] or set(recipient['cells'])!=set(clean['cells']):raise ValueError('Same model and owner topology')
    if recipient['readout']['cursor']!=clean['readout']['cursor']:raise ValueError('Same absolute time required')
    result=clone_tree(recipient)
    if scope=='system':return clone_tree(clean)
    if scope in ('cell','core'):result['cells']=clone_tree(clean['cells'])
    elif scope in ('v','aux'):
        for path,row in result['cells'].items():
            for key in row['state']:
                if (scope=='v' and key=='v') or (scope=='aux' and key in ('a','syn')):
                    row['state'][key]=clone_tree(clean['cells'][path]['state'][key]);alias='post' if key=='v' else key
                    if alias in row['last']:row['last'][alias]=clone_tree(clean['cells'][path]['last'][alias])
    if scope in ('cache','core'):result['cache']=clone_tree(clean['cache'])
    if scope=='readout':result['readout']=clone_tree(clean['readout'])
    return result
