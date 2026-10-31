"""Read actual receipts into flat CSV/JSON; specifications never imply success."""
import csv
import json
from pathlib import Path
from .utils import atomic_json,file_hash


def summarize(root,out):
    rows=[];analyses=[]
    for p in sorted(Path(root).glob('*/spec.json')):
        run=p.parent;spec=json.loads(p.read_text());status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'incomplete'}
        history=json.loads((run/'history.json').read_text()) if (run/'history.json').exists() else []
        row={'run_id':run.name,'status':status['status'],'family':spec['task']['family'],'task':spec['task']['name'],'objective':spec['task']['objective'],
             'backbone':spec['model']['backbone'],'cell':spec['model']['neuron'],'seed':spec['train']['seed'],'epochs_completed':status.get('epochs',status.get('progress',{}).get('epoch',0)),
             'updates':status.get('updates',status.get('progress',{}).get('updates',0)),'parameters_changed':status.get('parameters_changed'),
             'last_val_loss':history[-1]['val']['native_loss'] if history else None,'checkpoint_sha256':file_hash(run/'best.pt') if (run/'best.pt').exists() else None}
        rows.append(row)
        for r in sorted(run.glob('analysis/*/report.json')):
            a=json.loads(r.read_text());analyses.append({'run_id':run.name,'analysis_id':r.parent.name,'split':a['contract']['analysis']['eval_split'],
                'checkpoint_sha256':a['contract']['checkpoint_sha256'],'native_loss':a['native']['loss']['mean'],'accuracy':a['native']['accuracy'],
                'source':str(r),'modules':list(a['modules'])})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'runs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(out/'ledger.json',{'runs':rows,'analyses':analyses,'counts':{'observed_runs':len(rows),'trained':sum(r['status']=='trained' for r in rows),
        'trained_and_weights_changed':sum(r['status']=='trained' and r['parameters_changed'] is True for r in rows),'analysis_receipts':len(analyses)},
        'claim_rule':'No performance ranking across different data/label/permission/compute contracts; seed-level paired comparisons precede discovery claims.'})
    return {'runs':len(rows),'analyses':len(analyses),'out':str(out)}
