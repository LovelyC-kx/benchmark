import argparse,json
from pathlib import Path
import yaml

def read_config(path):
    p=Path(path);return json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text())

def main():
    p=argparse.ArgumentParser(description='SNN internal mechanism benchmark')
    sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--config',required=True);tr.add_argument('--out',required=True);tr.add_argument('--device')
    su=sub.add_parser('suite');su.add_argument('--config',required=True);su.add_argument('--out',required=True);su.add_argument('--no-resume',action='store_true');su.add_argument('--device');su.add_argument('--shard-index',type=int,default=0);su.add_argument('--num-shards',type=int,default=1)
    ca=sub.add_parser('calibrate');ca.add_argument('--out',required=True);ca.add_argument('--samples',type=int,default=150000)
    fs=sub.add_parser('fixed-scan');fs.add_argument('--out',required=True);fs.add_argument('--full',action='store_true')
    di=sub.add_parser('diagnose');di.add_argument('--checkpoint',required=True);di.add_argument('--out',required=True);di.add_argument('--n',type=int,default=128);di.add_argument('--data-root');di.add_argument('--device',default='cpu')
    cv=sub.add_parser('convert');cv.add_argument('kind',choices=['uci-har','shd','frames','tensors']);cv.add_argument('--root');cv.add_argument('--train');cv.add_argument('--test');cv.add_argument('--manifest');cv.add_argument('--objective',default='reconstruction');cv.add_argument('--out',required=True);cv.add_argument('--T',type=int,default=16);cv.add_argument('--size',type=int,default=64);cv.add_argument('--window',type=float,default=1.);cv.add_argument('--mode',choices=['fixed_dt','event_count'],default='fixed_dt')
    va=sub.add_parser('validate-data');va.add_argument('--root',required=True);va.add_argument('--objective',default='classification')
    cv.add_argument('--storage',choices=['npz','npy'],default='npy')
    a=p.parse_args()
    if a.command=='train':
        from .runner import train_run
        c=read_config(a.config)
        if a.device:c['device']=a.device
        r=train_run(c,a.out);print(json.dumps(r['test'],indent=2))
    elif a.command=='suite':
        from .experiments import suite
        c=read_config(a.config);runs=c.get('runs',c) if isinstance(c,dict) else c
        if a.num_shards<1 or not 0<=a.shard_index<a.num_shards:raise ValueError('invalid shard')
        runs=runs[a.shard_index::a.num_shards]
        if a.device:
            for r in runs:r['device']=a.device
        out=str(Path(a.out)/f'shard_{a.shard_index}') if a.num_shards>1 else a.out
        suite(runs,out,not a.no_resume)
    elif a.command=='calibrate':
        from .experiments import calibration
        Path(a.out).mkdir(parents=True,exist_ok=True);print(calibration(a.out,a.samples))
    elif a.command=='fixed-scan':
        from .experiments import fixed_scan
        fixed_scan(a.out,quick=not a.full)
    elif a.command=='diagnose':
        import torch
        torch.set_num_threads(2)
        from .runner import load_checkpoint,export_features,write_json,diagnostic_datasets
        from .experiments import gradient_diagnostics,fisher_diagnostic,profile
        m,ck=load_checkpoint(a.checkpoint,a.device)
        sets=diagnostic_datasets(ck,a.n,a.data_root)
        export_features(m,sets,Path(a.out)/'features',device=a.device)
        x=torch.tensor(sets['test']['x'][:8],dtype=torch.float32,device=a.device);y=torch.tensor(sets['test']['y'][:8],device=a.device)
        if ck['objective']!='classification':y=y.float()
        write_json(Path(a.out)/'gradient.json',gradient_diagnostics(m,x,y,ck['objective']))
        write_json(Path(a.out)/'fisher.json',fisher_diagnostic(m,x,y,ck['objective']))
        write_json(Path(a.out)/'profile.json',profile(m,x))
    elif a.command=='validate-data':
        from .data import load_splits
        d=load_splits(a.root,a.objective);print({s:a['x'].shape for s,a in d.items()})
    elif a.command=='convert':
        from .data import convert_uci_har,convert_shd,convert_frame_manifest,convert_tensor_manifest
        if a.kind=='uci-har':convert_uci_har(a.root,a.out)
        elif a.kind=='shd':convert_shd({'train':a.train,'test':a.test},a.out,a.T,a.window,mode=a.mode)
        elif a.kind=='frames':convert_frame_manifest(a.manifest,a.out,a.T,a.size)
        else:convert_tensor_manifest(a.manifest,a.out,a.objective,a.storage)

if __name__=='__main__':main()
