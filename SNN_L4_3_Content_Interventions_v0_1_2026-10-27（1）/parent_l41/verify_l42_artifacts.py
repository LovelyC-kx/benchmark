"""Read-only protocol checks and frozen-reader replay on saved scene prefixes."""
import argparse,ast,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from bridge42 import ROOT,PARENT,load_model,fixed_batch,collect,targets,load_tree,tree_hash,write_json,paired_bound
from readers42 import predict
from run_reader_ladder import group_losses

def read(p):return json.loads(Path(p).read_text())
def verify():
    torch.set_num_threads(1)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in ROOT.rglob('*.json'):json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    provenance=read(ROOT/'references/project_provenance.json')
    for rel,sha in provenance['inherited_files'].items():
        assert hashlib.sha256((PARENT/rel).read_bytes()).hexdigest()==sha,rel
    model,cfg,meta=load_model(PARENT/'evidence/source_smoke_run')
    checkpoint=hashlib.sha256((PARENT/'evidence/source_smoke_run/best.pt').read_bytes()).hexdigest()
    gaps={};counts={}
    for name,first,prefix in [('reader_ladder',8540000,128),('task_subspaces',8650000,128)]:
        path=ROOT/'evidence'/name;r=read(path/'report.json');f=read(path/'freeze_contract.json');bank=load_tree(path/'frozen_readers')
        assert f['reader_sha256']==tree_hash(bank) and f['checkpoint_sha256']==checkpoint
        assert f['choices_frozen_before_evaluation_generation'] and meta['optimizer_updates']==1
        batch=fixed_batch(cfg.data,prefix,'development_test',first,0);views,_=collect(model,batch);y=targets(batch,4)[0][:,:8]
        theta=np.asarray(f['theta']);gap=0.;arrays=0
        basis=None if name=='reader_ladder' else load_tree(path/'projection_bases')
        if basis is not None:assert tree_hash(basis)==f['projection_sha256']
        with np.load(path/'scene_losses.npz',allow_pickle=False) as saved:
            for c in f['cases']:
                key,view=c['key'],c['view']
                x=views[view] if basis is None else views['post__last']@basis[view]
                lo=group_losses(predict(bank[key],np.asarray(x,float)/f['scales'][view]),y,theta,4)
                for source,row in lo.items():
                    for obj,a in row.items():
                        stored=saved[key+'::'+source+'::'+obj]
                        assert stored.shape==(r['evaluation_scenes'],) and np.isfinite(stored).all()
                        assert abs(float(stored.mean())-r['risks'][key][source][obj])<1e-12
                        gap=max(gap,float(np.max(np.abs(a-stored[:prefix]))));arrays+=1
            if name=='reader_ladder':
                for key,c in r['classification_comparisons'].items():
                    pair,source=key.rsplit('::',1);a,b=pair.split('_vs_')
                    assert paired_bound(saved[a+'::'+source+'::derived_classification'],saved[b+'::'+source+'::derived_classification'],r['classification_comparison_family'])==c
                verification=load_tree(path/'independent_verification_readers')
                assert tree_hash(verification)==f['verification_sha256'] and len(verification)==2
                for c in f['cases']:assert c['total_head_supervised_labels']==c['fit_labels']+c['selection_labels']
            else:
                assert max(v for row in r['head_identities'].values() for v in row.values())<1e-10
                with np.load(path/'constant_baseline_losses.npz',allow_pickle=False) as baseline:
                    assert len(baseline)==8
                    for key in baseline.files:
                        draw,source,obj=key.split('::');assert abs(float(baseline[key].mean())-r['constant_risks'][draw][source][obj])<1e-12
        assert gap<1e-9;gaps[name]=gap;counts[name]=arrays
    controls=read(ROOT/'evidence/known_latent_controls.json')
    assert min(controls['risk_chain']['pieces'].values())>0
    author=read(ROOT/'evidence/selected_author_checks.json')
    assert author['source_count']==8 and all(x['bytes_match'] for x in author['verified_sources'].values())
    assert author['RAE_selected_encode_detaches_gradient'] and author['RAE_selected_methods_artificial_identity_roundtrip_max_abs']<1e-5
    text=(ROOT/'evidence/test_log.txt').read_text();m=re.search(r'Ran (\d+) tests',text)
    assert m and int(m.group(1))>=25 and text.rstrip().endswith('OK')
    return {'module':'L4.2','unit_tests_passed':int(m.group(1)),'unit_test_skips':0,
        'analytic_control_families':12,'replayed_scene_prefix_per_analysis':128,'replayed_saved_loss_arrays':counts,
        'replay_max_abs':gaps,'frozen_hashes_match':True,'inherited_files_verified':len(provenance['inherited_files']),
        'pinned_author_sources':8,'checkpoint_sha256':checkpoint,'original_optimizer_updates':1,
        'new_SNN_training':0,'formal_benchmark_completed':0,'audit_test_accessed':False,
        'scope':'Consistency replay on the same saved development scenes, not independent scientific replication.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r=verify();write_json(out,r);print(json.dumps(r,indent=2))
