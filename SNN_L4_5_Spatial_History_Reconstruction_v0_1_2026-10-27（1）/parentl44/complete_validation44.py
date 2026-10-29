"""Complete delayed R10 semantic diagnostics without refitting or changing outcomes."""
import json
import torch
from bridge44 import ROOT,RUN_DIRECTORY,load_model,fixed_batch,hybrid,load_tree,write_json
from recovery_runtime44 import scheduled_run,trace_site,damage_edit
from run_recovery_main import validation

def complete():
    torch.set_num_threads(1);f=json.loads((ROOT/'evidence/fitted/freeze_contract.json').read_text());bank=load_tree(ROOT/'evidence/fitted/frozen_bank');counts={}
    for pool,settings in f['evaluation_pools'].items():
        path=ROOT/'evidence'/pool/'report.json';report=json.loads(path.read_text());model,cfg,_=load_model(RUN_DIRECTORY)
        a=fixed_batch(cfg.data,settings['n'],'development_test',settings['recipient'],0);b=fixed_batch(cfg.data,settings['n'],'development_test',settings['donor'],0)
        clean=scheduled_run(model,a['x']);damaged={}
        for c in f['damage_contexts']:damaged[c['key']]=scheduled_run(model,a['x'],[damage_edit(c,bank['damage_erasers'][c['key']])])
        for c in f['recovery_contexts']:
            h=trace_site(clean,c['layer'],c['patch_time']);dh=trace_site(damaged[c['key']],c['layer'],c['patch_time']);site=c['patch_site'];d=hybrid(cfg.data,a,b,c['source'])
            report['results'][c['key44']]['damaged_state_validation']=validation(dh,h,d,a,c['source'],bank['validation_probes'][site],f['scales'][site],c['patch_time'])
        write_json(path,report);counts[pool]=len(f['recovery_contexts'])
    write_json(ROOT/'evidence/semantic_validation_completion.json',{'context_baselines_added':counts,'reason':'Explicit delayed damaged-state probe baseline is needed to distinguish recovered content from a mere return toward the clean state.','new_probe_fitting':0,'predictions_or_risks_changed':False,'method_choices_changed':False,'evaluation_cohorts_changed':False})

if __name__=='__main__':complete();print('Delayed damaged-state semantic baselines completed')
