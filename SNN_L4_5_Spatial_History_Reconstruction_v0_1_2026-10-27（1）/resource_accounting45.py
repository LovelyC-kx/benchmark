"""Intrinsic state and decoder ledger; deliberately not a hardware energy estimator."""
import argparse
import torch
from bridge45 import ROOT,CONFIGS,VIEWS,Cell,model_for,config_key,load_tree,dump
from spatial_data45 import batch

def execute():
    torch.set_num_threads(1);bank=load_tree(ROOT/'evidence/fitted/bank');a=batch(1,45450000);rows={}
    for b,n in CONFIGS:
        key=config_key(b,n);m=model_for(b,n)
        with torch.no_grad():m(torch.as_tensor(a['x']))
        owners={p:{k:int(v.numel()) for k,v in cell.state.items()} for p,cell in m.named_modules() if isinstance(cell,Cell)}
        core_state=sum(sum(v.values()) for v in owners.values());head_params=sum(v.numel() for v in m.head.parameters());all_params=sum(v.numel() for v in m.parameters())
        rows[key]={'intrinsic_recurrent_state_scalars_per_scene':core_state,'intrinsic_recurrent_state_float32_bytes_per_scene':4*core_state,
            'actual_all_cell_owners':owners,'learned_core_parameters':all_params-head_params,'original_random_head_parameters':head_params,
            'diagnostic_full_feature_trajectory_scalars':int(m.last_features[0].numel()),
            'online_known_affine_mean_output_accumulator_scalars':4*16*16,'mean_count_scalar_extra':1,
            'reader_parameters':{v:{k:r['parameter_count'] for k,r in methods.items()} for v,methods in bank[key]['decoders'].items()},
            'last_observation':'Reads final post state already counted in core; output storage is separate.',
            'online_mean':'A fixed affine decoder may accumulate its 4-band output instead of storing every feature. This is a derived deployment option, not a measured optimized implementation.'}
    return {'module':'L4.5 state and decoder resource ledger','results':rows,
        'scope':'Per-scene scalar/parameter counts only. Diagnostic runtime stores full feature trajectories; no measured deployment peak memory, latency, asynchronous SOP or hardware energy claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();dump(a.output,execute())
