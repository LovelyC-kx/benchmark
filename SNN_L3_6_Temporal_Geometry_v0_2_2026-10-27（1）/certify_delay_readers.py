"""Replay existing supplementary scenes for all 900 bounded comparisons."""
import json,argparse,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from bridge35 import load_model,fixed_batch,flatten_contents
from temporal_capture import base_capture,temporal_views
from temporal_alignment import push_reader
from temporal_probes import score,loss_by_group,paired_bound,tree_hash
from artifacts35 import load_tree,write_json
from run_temporal_review import target_bank
ROOT=Path(__file__).resolve().parent

def certify(primary=None,extensions=None,run_directory=None):
    torch.set_num_threads(1)
    primary=Path(primary) if primary is not None else ROOT/'evidence/frozen_temporal_review'
    ext=Path(extensions) if extensions is not None else ROOT/'evidence/temporal_extensions'
    lock=json.loads((primary/'freeze_contract.json').read_text());contract=json.loads((ext/'freeze_contract.json').read_text())
    original=json.loads((ext/'report.json').read_text());bank=load_tree(primary/'frozen_readers');maps=load_tree(ext/'delay_maps')
    assert tree_hash(bank)==contract['primary_reader_sha256'] and tree_hash(maps)==contract['delay_map_sha256']
    run_directory=Path(run_directory) if run_directory is not None else ROOT/'evidence/source_smoke_run'
    checkpoint_hash=hashlib.sha256((run_directory/'best.pt').read_bytes()).hexdigest()
    if checkpoint_hash!=lock['checkpoint_sha256']:
        raise ValueError('Certification checkpoint differs from the frozen primary checkpoint')
    if contract.get('checkpoint_sha256',checkpoint_hash)!=checkpoint_hash:
        raise ValueError('Supplementary checkpoint differs from primary')
    model,cfg,_=load_model(run_directory)
    n,split,first_id=contract['delay_evaluation']
    b=fixed_batch(replace(cfg.data,blank_delay=contract.get('long_blank_delay',11)),n,split,first_id,0)
    views=temporal_views(base_capture(model,b['x']));y,_,groups=target_bank(flatten_contents(b,cfg.data));results={};maximum=0.
    anchor=contract.get('source_reader_time',max(lock['source_times']))
    family_size=sum((len(entries)-1)*len(groups) for times in maps.values() for entries in times.values())
    for name,x in views.items():
        x=np.asarray(x,float)/lock['scales'][name];f=bank[name]
        for time,entries in maps[name].items():
            per={};t=int(time)
            for family,m in entries.items():
                w,bias=push_reader(m,f['weights'][:,anchor],f['origins'][:,anchor],f['bias'][:,anchor]);p=score(w,bias,x[:,t]);per[family]={}
                for g in groups:
                    scene,_=loss_by_group(p,y,g['binary_columns'],g['continuous_columns']);per[family][g['source']]=scene
                    expected=original['blank_delay_scan'][name][time][family][g['source']]['classification_error']
                    maximum=max(maximum,abs(float(scene.mean())-expected))
            for family in entries:
                if family=='raw':continue
                for g in groups:
                    source=g['source'];key=f'{name}__t{t}__{source}__raw_vs_{family}'
                    results[key]=paired_bound(per['raw'][source],per[family][source],family_size)
    assert len(results)==family_size and maximum==0
    return {'comparisons':results,'family_size':family_size,'delta':.05,'mean_risk_replay_max_abs':maximum,
            'checkpoint_sha256':checkpoint_hash,
            'positive_lower_bounds':sum(v['interval'][0]>0 for v in results.values()),
            'new_fits':0,'independent_confirmation_by_replay':False,
            'scope':'Simultaneous conditional bounds across all frozen supplementary views, times, contents and non-raw transports. Separate from the primary family; no joint whole-study 95% claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--primary');p.add_argument('--extensions');p.add_argument('--run-directory');a=p.parse_args();dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    r=certify(a.primary,a.extensions,a.run_directory);write_json(dst,r);print('Certified comparisons:',r['family_size'],'positive lower bounds:',r['positive_lower_bounds'])
