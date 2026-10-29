"""Phase-aligned carrier extraction; spatial dimensions are flattened, not pooled."""
import numpy as np
import torch
from bridge35 import Cell

@torch.no_grad()
def capture(model,x,batch_size=128):
    model.eval();device=next(model.parameters()).device
    result={};predictions=[];inventory=None
    core=getattr(model,'core',model)
    wanted=('pre','post','output','natural_event','emit_event','reset_event','aux_event','theta','a','previous_a','syn','previous_syn')
    for start in range(0,len(x),batch_size):
        xb=torch.as_tensor(x[start:start+batch_size],device=device)
        pred=model(xb,capture=True);predictions.append(pred.cpu().numpy())
        if inventory is None:
            inventory={}
            for name,cell in core.named_modules():
                if isinstance(cell,Cell):
                    inventory[name]={'kind':cell.kind,'persistent_cell_keys':list(cell.state),
                        'state_shape_without_batch':{k:list(v.shape[1:]) for k,v in cell.state.items()},
                        'state_scalars_per_scene':sum(v[0].numel() for v in cell.state.values()),
                        'state_bytes_per_scene':sum(v[0].numel()*v.element_size() for v in cell.state.values()),
                        'has_binary_event':cell.kind not in ('continuous','tanh'),
                        'phase':'post-state after each call; theta uses previous_a when ALIF',
                        'reset_strength':cell.reset_strength}
        for name,trace in core.traces().items():
            for key in wanted:
                if key not in trace:continue
                value=trace[key].detach().cpu().numpy()
                value=value.reshape(value.shape[0],value.shape[1],-1)
                if key in ('natural_event','emit_event','reset_event','aux_event'):value=value.astype(np.uint8)
                result.setdefault(name+'/'+key,[]).append(value)
    return {'carriers':{k:np.concatenate(v) for k,v in result.items()},
        'prediction':np.concatenate(predictions),'cell_inventory':inventory,
        'native_head_observation':core.cfg.observation,
        'native_head_time_readout':core.cfg.readout,
        'flattened_spatial_axes_preserved':True,
        'scope':'Exported diagnostics include nonrecurrent phases. Complete system state additionally includes backbone cache and readout bookkeeping.'}

def views(captured):
    c=captured['carriers'];result={};spec={}
    for layer in ('backbone.cells.0','backbone.cells.1'):
        tag='first' if layer.endswith('.0') else 'last'
        pre=c[layer+'/pre'][:,-1];post=c[layer+'/post'][:,-1];event=c[layer+'/natural_event'][:,-1].astype(float)
        values={'pre':pre,'post':post,'spike':event,'post_spike':np.c_[post,event]}
        window=c[layer+'/natural_event'][:,-8:]
        values.update(count8=window.sum(1).astype(float),
            bins8_4=window.reshape(len(window),4,2,-1).sum(2).reshape(len(window),-1).astype(float),
            raster8=window.reshape(len(window),-1).astype(float))
        for name,value in values.items():
            key=tag+'_'+name;result[key]=value
            spec[key]={'layer':layer,'phase':name,'dimension':value.shape[1],
                'window_steps':8 if name in ('count8','bins8_4','raster8') else 1,
                'transform':'natural','rank':None,'quantization_bits':None}
        # Same-dimensional train-only compression has its own interpretation.
        for source in ('post_spike','raster8'):
            key=tag+'_'+source+'_rank32';result[key]=values[source]
            spec[key]=dict(spec[tag+'_'+source],transform='PCA32',rank=32,dimension=32)
    result['core_two_layer_post']=np.c_[c['backbone.cells.0/post'][:,-1],c['backbone.cells.1/post'][:,-1]]
    spec['core_two_layer_post']={'layer':'both','phase':'post','dimension':result['core_two_layer_post'].shape[1],
        'window_steps':1,'transform':'natural','rank':None,'quantization_bits':None}
    for source in ('pre','post'):
        key='last_'+source+'_q4';result[key]=result['last_'+source]
        spec[key]=dict(spec['last_'+source],transform='uniform_q4',quantization_bits=4)
    # Quantize membrane only; emitted spike remains an exact one-bit channel.
    result['last_post_q3_spike']=result['last_post_spike']
    spec['last_post_q3_spike']=dict(spec['last_post_spike'],transform='post_q3_plus_event1',quantization_bits=3)
    return result,spec

