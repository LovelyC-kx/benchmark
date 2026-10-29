"""Keep original six-layer plans addressable without conflating names or versions."""
import re
import json
from pathlib import Path
from .utils import file_hash,atomic_json

ROUTES={
 'L1.1':['task_controls'],'L1.2':['task_controls','temporal'],'L1.C':['task_controls'],'L1.D':['recovery'],'L1.E':['task_controls'],'L1.F':['task_controls','reconstruction'],'L1.G':[],
 'L2.1':['mechanism'],'L2.2':['working_point'],'L2.3':['mechanism','theory'],'L2.4':['mechanism'],'L2.5':['working_point','probes'],'L2.6':['communication','interventions'],'L2.7':[],
 'L3.1':['probes','geometry'],'L3.2':['probes','geometry'],'L3.3':['geometry','selective'],'L3.4':['theory'],'L3.5':['probes','access'],'L3.6':['temporal'],'L3.7':['communication'],
 'L4.1':['access'],'L4.2':['access','theory'],'L4.3':['selective','interventions'],'L4.4':['recovery'],'L4.5':['reconstruction'],
 'L5.1':['probes'],'L5.2':['interventions','recovery'],'L5.3':['task_controls'],'L5.4':[],'L5.5':['working_point'],
 'L6.1':['mechanism','probes'],'L6.2':['mechanism','probes'],'L6.3':['working_point'],'L6.4':[],'L6.5':['geometry','temporal'],
 'X.PAPER':[],'X.STATS':[],'X.COMP':[],'X.CODE':[]}


def plan_inventory(root,out=None):
    root=Path(root);master=root/'SNN_Research_Master_Index_2026-10-24.md';text=master.read_text();cards=[]
    matches=list(re.finditer(r'^## ((?:L[1-6]\.[A-Za-z0-9]+|X\.[A-Z]+)) ([^\n]+)',text,re.M))
    for i,m in enumerate(matches):
        stop=matches[i+1].start() if i+1<len(matches) else text.find('# 按论文逻辑',m.end())
        body=text[m.end():stop if stop>0 else len(text)];module=m[1]
        action=re.search(r'\*\*执行逻辑。\*\*\s*([^\n]+)',body)
        cards.append({'id':'master:'+module,'source_module_id':module,'title':m[2].strip(),'source':str(master.relative_to(root)),
            'source_sha256':file_hash(master),'execution_logic':action[1] if action else '',
            'shared_runner_components':ROUTES.get(module,[]),'status':'awaiting_campaign_execution_and_specialized_adapters',
            'complete_section_claim':False,'completion_requires':'Run every applicable original subsection/control; attach outputs or a specific applicability reason. Shared-module execution alone does not close the section.'})
    sources=[];subsections={};headings=[]
    for p in sorted(root.glob('SNN*.md')):
        content=p.read_text(errors='replace');sources.append({'path':str(p.relative_to(root)),'sha256':file_hash(p),'kind':'research_document'})
        for line in content.splitlines():
            if re.match(r'^#{1,6} ',line):headings.append({'source':str(p.relative_to(root)),'heading':line})
    for p in sorted(root.rglob('*coverage*.json')):
        relative=str(p.relative_to(root))
        if any(s in relative for s in ('/parent','/parnet','docs/verification')):continue
        content=json.loads(p.read_text());sources.append({'path':relative,'sha256':file_hash(p),'kind':'original_coverage_map'})
        def walk(value):
            if isinstance(value,dict):
                ident=str(value.get('id',''))
                if re.fullmatch(r'[34](?:\.\d+){2,3}',ident):
                    row=subsections.setdefault(ident,{'id':'section:'+ident,'source_section_id':ident,'sources':[],'code_candidates':[],
                        'title':value.get('topic',value.get('purpose',ident)),'status':'specialized_checkpoint_adapter_required','complete_section_claim':False})
                    row['sources'].append(relative)
                    for name in value.get('code',[]):
                        candidate=p.parent.parent/name
                        row['code_candidates'].append({'path':str(candidate.relative_to(root)),'exists':candidate.exists()})
                for key,v in value.items():
                    if key=='sections':
                        for ident in re.findall(r'(?<!\d)[34](?:\.\d+){2,3}',str(v)):
                            row=subsections.setdefault(ident,{'id':'section:'+ident,'source_section_id':ident,'sources':[],'code_candidates':[],
                                'title':'Subsection referenced by original literature/plan map','status':'specialized_checkpoint_adapter_required','complete_section_claim':False})
                            if relative not in row['sources']:row['sources'].append(relative)
                    walk(v)
            elif isinstance(value,list):
                for v in value:walk(v)
        walk(content)
    result={'schema':'all_six_layer_plan_inventory_v1','source_root':str(root.resolve()),'master_cards':cards,
        'specialized_subsections':list(subsections.values()),'source_documents':sources,'reference_heading_index':headings,
        'counts':{'master_cards':len(cards),'specialized_subsections':len(subsections),'source_documents':len(sources),'reference_headings':len(headings)},
        'all_planned_analysis_complete':False,
        'meaning':'Executable shared components, source-specific requirements and future work remain separate; every original plan source stays addressable.'}
    if out:atomic_json(out,result)
    return result
