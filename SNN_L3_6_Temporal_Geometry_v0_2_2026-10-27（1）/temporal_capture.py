"""Project model adapter. Full spatial axes survive capture; current bank is MLP."""
import numpy as np
from capture35 import capture

def base_capture(model,x):
    observed=capture(model,x);c=observed['carriers'];out={}
    if set(observed['cell_inventory'])!={'backbone.cells.0','backbone.cells.1'}:
        raise ValueError('first/last aliases require the two-layer project MLP; use named_temporal_bank for other backbones')
    for i,name in enumerate(('first','last')):
        path=f'backbone.cells.{i}/'
        for phase in ('pre','post','natural_event'):
            if path+phase in c:
                out[name+'_'+('spike' if phase=='natural_event' else phase)]=c[path+phase]
        if observed['cell_inventory'][path[:-1]]['kind'] in ('continuous','tanh','analog_hard_ref'):
            out[name+'_analog_output']=c[path+'output']
        for auxiliary in ('a','syn'):
            if path+auxiliary in c:out[name+'_'+auxiliary]=c[path+auxiliary]
    return out

def named_temporal_bank(model,x,batch_size=128):
    """Every named cell, actual output and full spatial coordinates, all backbones.

    This is an observation bank, not a full recurrent-system state or causal DAG.
    Flattening preserves each spatial coordinate; it never averages image axes.
    """
    result=capture(model,x,batch_size=batch_size)
    inventory=result['cell_inventory'];spec={}
    for key,value in result['carriers'].items():
        module,phase=key.rsplit('/',1);entry=inventory[module]
        shape=entry['state_shape_without_batch']['v']
        spec[key]={'module':module,'phase':phase,'scene_time_features':list(value.shape),
                   'unflattened_feature_shape':shape,'cell_kind':entry['kind'],
                   'actual_transmitted_output':phase=='output','spatial_pooling':False,
                   'binary':phase in ('natural_event','emit_event','reset_event','aux_event')}
    result['view_specification']=spec
    result['complete_runtime_state']=False
    return result

def temporal_views(base):
    out=dict(base)
    for tag in ('first','last'):
        if tag+'_spike' in base:
            out[tag+'_post_spike']=np.concatenate([base[tag+'_post'],base[tag+'_spike']],-1)
        auxiliary=[base[tag+'_'+key] for key in ('a','syn') if tag+'_'+key in base]
        if auxiliary:out[tag+'_post_aux']=np.concatenate([base[tag+'_post']]+auxiliary,-1)
    out['core_post']=np.concatenate([base['first_post'],base['last_post']],-1)
    return out
