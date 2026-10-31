#!/usr/bin/env python3
"""Read-only failure diagnosis for an original campaign and a selected new stage."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sqlite3


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=60)


def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def tail(path,budget=8192):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-budget))
            return stream.read(budget).decode('utf-8',errors='replace')
    except OSError:return ''


def evidence(run_id,raw_spec,queue_error,runs):
    spec=json.loads(raw_spec);model=spec.get('model',{});task=spec.get('task',{})
    row={'run_id':run_id,'task':task.get('name'),'family':task.get('family'),
         'objective':task.get('objective'),'T':task.get('T'),'params':task.get('params',{}),
         'backbone':model.get('backbone'),'neuron':model.get('neuron'),
         'gain':model.get('gain'),'beta':model.get('beta'),'seed':spec.get('train',{}).get('seed'),
         'queue_error':queue_error,'cause_source':'queue_only','cause':queue_error or 'Unresolved failure'}
    if not re.fullmatch(r'[0-9a-f]{24}',run_id):return row
    run=runs/run_id
    for name in ('failure.json','status.json'):
        value=read_json(run/name)
        if value.get('error'):
            row.update(cause=str(value['error']),cause_source=str(run/name),failure_stage=value.get('stage',value.get('failure_stage')))
            break
    logs=[runs/'_queue_specs'/(run_id+'.log')]
    for path in logs:
        text=tail(path)
        if text:
            row['log']=str(path);row['log_tail']=text
            if row['cause_source']=='queue_only':
                matches=re.findall(r'(?m)^((?:[\w.]+(?:Error|Exception)|Error|Exception):[^\n]*)',text)
                if matches:row.update(cause=matches[-1],cause_source=str(path))
            break
    row['cause_class']=row['cause'].split(':',1)[0][:160]
    return row


def audit(root,out,examples=4):
    root=Path(root).resolve();plan=json.loads((root/'campaign.json').read_text())
    with readonly(root/'train.sqlite') as con:selected={r[0] for r in con.execute('SELECT id FROM jobs')}
    with readonly(Path(plan['source_root'])/'train.sqlite') as con:
        con.execute('BEGIN')
        counts=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        failed=con.execute("SELECT id,spec,error FROM jobs WHERE state='failed'").fetchall()
    runs=Path(plan['training_output'])
    rows=[evidence(key,spec,error,runs) for key,spec,error in failed]
    classes=Counter(r.get('cause_class','unresolved') for r in rows)
    signatures=Counter(re.sub(r'\b[0-9a-f]{24}\b','<run_id>',r['cause'])[:500] for r in rows)
    matched=[r for r in rows if r['run_id'] in selected]
    representatives=[];seen=set()
    for row in rows:
        cls=row.get('cause_class','unresolved')
        if cls in seen:continue
        seen.add(cls);representatives.append(row)
        if len(representatives)>=examples:break
    report={'schema':'campaign_failure_audit_v1','stage':str(root),'source_root':plan['source_root'],
        'source_counts':counts,'failed_rows_examined':len(rows),'cause_classes':classes.most_common(20),
        'common_causes':signatures.most_common(15),'selected_stage_failures':matched,
        'representative_failures':representatives,'all_failed_rows':rows,
        'policy':'No database, task, checkpoint, model, or failure evidence modified. No retries launched.',
        'scope':'Recorded exceptions, not inferred root-cause diagnoses. A missing log stays unresolved.'}
    out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_name(out.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,out)
    preview={k:report[k] for k in ('source_counts','failed_rows_examined','cause_classes','common_causes')}
    preview['selected_stage_failures']=matched
    preview['representative_failures']=[{**r,'log_tail':r.get('log_tail','')[-3000:]} for r in representatives]
    preview['report']=str(out)
    print(json.dumps(preview,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();audit(args.root,args.out)
