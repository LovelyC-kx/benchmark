"""Executable entry point. Public datasets are fetched only by `download`."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(prog='snnbench');sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('download');p.add_argument('args',nargs=argparse.REMAINDER)
    p=sub.add_parser('datasets');p.add_argument('--tier',default='all');p.add_argument('--ids',nargs='*')
    p=sub.add_parser('doctor')
    p=sub.add_parser('sweep');p.add_argument('--base',required=True);p.add_argument('--grid',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('preflight');p.add_argument('--manifest',required=True);p.add_argument('--data',default='prepared');p.add_argument('--limit',type=int)
    p=sub.add_parser('export-encoder');p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',default='best.pt')
    p=sub.add_parser('init-spec');p.add_argument('--out',required=True);p.add_argument('--image',action='store_true');p.add_argument('--classification',action='store_true')
    p=sub.add_parser('compile');p.add_argument('--out',required=True);p.add_argument('--profile',choices=('smoke','screen','full','paired','learning'),default='screen');p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);p.add_argument('--archives',nargs='*',default=[]);p.add_argument('--limit',type=int)
    p=sub.add_parser('train');p.add_argument('--spec',required=True);p.add_argument('--out',default='runs');p.add_argument('--data',default='prepared');p.add_argument('--device',default='cpu');p.add_argument('--max-updates',type=int)
    p=sub.add_parser('analyze');p.add_argument('--run',required=True);p.add_argument('--data',default='prepared');p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',default='best.pt');p.add_argument('--split',choices=('development','confirmation','audit'));p.add_argument('--modules',nargs='+');p.add_argument('--fit-scenes',type=int);p.add_argument('--select-scenes',type=int);p.add_argument('--eval-scenes',type=int)
    p=sub.add_parser('prepare');prep=p.add_subparsers(dest='adapter',required=True)
    for name in ('har','gsc','gesture'):
        q=prep.add_parser(name);q.add_argument('--root',required=True);q.add_argument('--out',required=True)
        if name=='gesture':q.add_argument('--steps',type=int,default=100);q.add_argument('--window',type=float,default=3.);q.add_argument('--size',type=int,default=32)
    q=prep.add_parser('spikes');q.add_argument('--train',required=True);q.add_argument('--val');q.add_argument('--test',required=True);q.add_argument('--out',required=True);q.add_argument('--dataset',choices=('SHD','SSC'),default='SHD');q.add_argument('--steps',type=int,default=100);q.add_argument('--window',type=float,default=1.4)
    q=prep.add_parser('manifest');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--objective',choices=('classification','regression','reconstruction'),default='classification');q.add_argument('--classes',type=int)
    q=prep.add_parser('earthnet');q.add_argument('--train',required=True);q.add_argument('--test',required=True);q.add_argument('--out',required=True);q.add_argument('--steps',type=int,default=6);q.add_argument('--size',type=int,default=32);q.add_argument('--group-unit',choices=('tile','cube'),default='tile')
    p=sub.add_parser('verify-data');p.add_argument('--root',required=True)
    p=sub.add_parser('queue-init');p.add_argument('--db',required=True);p.add_argument('--manifest',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p=sub.add_parser('worker');p.add_argument('--db',required=True);p.add_argument('--out',default='runs');p.add_argument('--data',default='prepared');p.add_argument('--device',default='cuda:0');p.add_argument('--analyze',action='store_true');p.add_argument('--one',action='store_true')
    p=sub.add_parser('queue-status');p.add_argument('--db',required=True)
    p=sub.add_parser('queue-retry');p.add_argument('--db',required=True)
    p=sub.add_parser('summarize');p.add_argument('--runs',default='runs');p.add_argument('--out',default='results_summary')
    args=parser.parse_args(argv);result=None
    if args.command=='download':
        from .download import main as download_main
        return download_main(args.args)
    if args.command=='datasets':
        from .download import main as download_main
        return download_main(['--list','--tier',args.tier]+(['--ids',*args.ids] if args.ids else []))
    from .specs import RunSpec,TaskSpec,ModelSpec
    from .utils import atomic_json,jsonable
    if args.command=='doctor':
        import importlib,torch,numpy,sys
        modules=[p.stem for p in (Path(__file__).parent/'kernels').glob('*.py') if p.stem!='__init__']
        for m in modules:importlib.import_module('snnbench.kernels.'+m)
        result={'python':sys.version,'torch':str(torch.__version__),'numpy':numpy.__version__,'cuda_available':torch.cuda.is_available(),'gpus':torch.cuda.device_count(),'kernel_imports':len(modules)}
    elif args.command=='sweep':
        from .study import sweep
        result=sweep(args.base,args.grid,args.out)
    elif args.command=='preflight':
        from .study import preflight
        result=preflight(args.manifest,args.data,args.limit)
    elif args.command=='export-encoder':
        from .study import export_encoder
        result=export_encoder(args.run,args.out,args.checkpoint)
    elif args.command=='init-spec':
        task=TaskSpec(family='spatial',name='spatial_v2',T=6,objective='classification' if args.classification else 'reconstruction') if args.image else TaskSpec(objective='classification' if args.classification else 'regression')
        s=RunSpec(task=task,model=ModelSpec(backbone='cnn2d' if args.image else 'mlp'));atomic_json(args.out,s.to_dict());result={'spec':args.out,'run_id':s.run_id}
    elif args.command=='compile':
        from .catalogue import compile_manifest
        result=compile_manifest(args.out,args.profile,args.seeds,args.archives,args.limit)
    elif args.command=='train':
        from .engine import train
        result={'run':str(train(RunSpec.load(args.spec),args.out,args.data,args.device,max_updates=args.max_updates))}
    elif args.command=='analyze':
        from .analysis import analyze
        s=RunSpec.load(Path(args.run)/'spec.json');changes={k:getattr(args,k) for k in ('fit_scenes','select_scenes','eval_scenes') if getattr(args,k) is not None}
        if args.split:changes['eval_split']=args.split
        if args.modules:changes['modules']=tuple(args.modules)
        a=replace(s.analysis,**changes);RunSpec(task=s.task,model=s.model,train=s.train,analysis=a).validate()
        result={'analysis':str(analyze(args.run,args.data,args.checkpoint,args.device,a))}
    elif args.command=='prepare':
        from .prepare import prepare_har,prepare_gsc,prepare_gesture,prepare_spikes,prepare_manifest
        if args.adapter=='har':result=prepare_har(args.root,args.out)
        elif args.adapter=='gsc':result=prepare_gsc(args.root,args.out)
        elif args.adapter=='gesture':result=prepare_gesture(args.root,args.out,args.steps,args.window,args.size)
        elif args.adapter=='spikes':
            files={'train':args.train,'test':args.test}
            if args.val:files['val']=args.val
            result=prepare_spikes(files,args.out,args.steps,args.window,dataset=args.dataset)
        elif args.adapter=='earthnet':
            from .earthnet import prepare_earthnet
            result=prepare_earthnet(args.train,args.test,args.out,args.steps,args.size,args.group_unit)
        else:result=prepare_manifest(args.manifest,args.out,args.objective,args.classes)
    elif args.command=='verify-data':
        from .prepare import verify_dataset
        result=verify_dataset(args.root)
    elif args.command=='queue-init':
        from .queue import enqueue
        result=enqueue(args.db,args.manifest,args.shard,args.shards)
    elif args.command=='worker':
        from .queue import worker
        result=worker(args.db,args.out,args.data,args.device,args.analyze,one=args.one)
        if result.get('failed',0):
            print(json.dumps(result));raise SystemExit(1)
    elif args.command in ('queue-status','queue-retry'):
        from .queue import connect
        with connect(args.db) as con:
            if args.command=='queue-retry':con.execute("UPDATE jobs SET state='pending',error=NULL,attempts=0 WHERE state='failed'")
            result=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())
    elif args.command=='summarize':
        from .ledger import summarize
        result=summarize(args.runs,args.out)
    print(json.dumps(jsonable(result),ensure_ascii=False,indent=2))
