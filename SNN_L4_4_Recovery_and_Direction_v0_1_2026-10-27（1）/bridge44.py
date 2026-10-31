"""Explicitly inherited L4.3 models, data, frozen operations and probe routines."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'parent_l43'
if not PARENT.is_dir():
    if (ROOT/'parnetl43'/'bridge43.py').is_file(): PARENT=ROOT/'parnetl43'
    else:
        PARENT=next((base/'SNN_L4_3_Content_Interventions_v0_1_2026-10-27（1）' for base in ROOT.parents
                     if (base/'SNN_L4_3_Content_Interventions_v0_1_2026-10-27（1）'/'bridge43.py').is_file()),PARENT)
if not PARENT.is_dir(): raise FileNotFoundError('Missing archived L4.3 parent; use the canonical snnbench pipeline for new runs')
sys.path.insert(0,str(PARENT))
from bridge43 import (RUN_DIRECTORY,SOURCES,load_model,fixed_batch,hybrid,labels23,
    parameter_hash,fit_head,fit_mlp,predict,linear_bundle,save_tree,load_tree,
    tree_hash,write_json,independent_ids,Cell,StreamSession,BenchmarkModel,ModelConfig,
    NEURONS,BACKBONES,clone_tree,capture_edge)
from intervention_math43 import apply_eraser,matched_binary_message
from regularized_intervention43 import fit_covariance_regularized_eraser
from fit_intervention_bank import choose_ridge

TIMES=(3,4,12,13,14,15)
def original_contract():
    import json
    return json.loads((PARENT/'evidence/fitted/freeze_contract.json').read_text())

def site_name(layer,time):return f'l{layer}_t{time}'
def context_name(source,layer,time):return f'{source}_l{layer}_t{time}'

def recovery_contexts():
    result=[]
    for c in original_contract()['contexts']:
        t=c['time'];patches=(3,4,14,15) if t==3 else (12,13,14,15) if t==12 else (15,)
        for p in patches:
            result.append(dict(c,damage_time=t,patch_time=p,patch_site=site_name(c['layer'],p),patch_key=context_name(c['source'],c['layer'],p),key44=c['key']+f'_p{p}'))
    return result
