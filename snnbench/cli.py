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
    p=sub.add_parser('campaign-plan');p.add_argument('--root',required=True);p.add_argument('--data-root');p.add_argument('--archives',nargs='*',default=[]);p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55]);p.add_argument('--profiles',nargs='+',default=['full','paired','learning']);p.add_argument('--materialize',action='store_true')
    p=sub.add_parser('campaign-stage');p.add_argument('--root',required=True);p.add_argument('--phase',choices=('preliminary','all_shared'),default='preliminary');p.add_argument('--checkpoints',choices=('all','terminal'),default='all');p.add_argument('--archive-scenes',type=int);p.add_argument('--archive-batch',type=int,default=8)
    p=sub.add_parser('post-worker');p.add_argument('--db',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--one',action='store_true')
    p=sub.add_parser('post-job');p.add_argument('--job',required=True);p.add_argument('--device',default='cpu')
    p=sub.add_parser('archive');p.add_argument('--run',required=True);p.add_argument('--data',default='prepared');p.add_argument('--checkpoint',default='best.pt');p.add_argument('--device',default='cpu');p.add_argument('--roles',nargs='+');p.add_argument('--scenes',type=int);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--owners-only',action='store_true')
    p=sub.add_parser('models');p.add_argument('--sources',action='store_true')
    p=sub.add_parser('upstream-checkout');p.add_argument('--suite',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('upstream-run');p.add_argument('--suite',required=True);p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('sweep');p.add_argument('--base',required=True);p.add_argument('--grid',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('preflight');p.add_argument('--manifest',required=True);p.add_argument('--data',default='prepared');p.add_argument('--limit',type=int)
    p=sub.add_parser('export-encoder');p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',default='best.pt')
    p=sub.add_parser('init-spec');p.add_argument('--out',required=True);p.add_argument('--image',action='store_true');p.add_argument('--classification',action='store_true')
    p=sub.add_parser('compile');p.add_argument('--out',required=True);p.add_argument('--profile',choices=('smoke','screen','full','paired','learning','expanded'),default='screen');p.add_argument('--model-roster',choices=('legacy','expanded'));p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);p.add_argument('--archives',nargs='*',default=[]);p.add_argument('--limit',type=int)
    p=sub.add_parser('train');p.add_argument('--spec',required=True);p.add_argument('--out',default='runs');p.add_argument('--data',default='prepared');p.add_argument('--device',default='cpu');p.add_argument('--max-updates',type=int)
    p=sub.add_parser('analyze');p.add_argument('--run',required=True);p.add_argument('--data',default='prepared');p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',default='best.pt');p.add_argument('--split',choices=('development','confirmation','audit'));p.add_argument('--modules',nargs='+');p.add_argument('--fit-scenes',type=int);p.add_argument('--select-scenes',type=int);p.add_argument('--eval-scenes',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--max-features',type=int);p.add_argument('--max-targets',type=int)
    p=sub.add_parser('prepare');prep=p.add_subparsers(dest='adapter',required=True)
    for name in ('har','gsc','gesture'):
        q=prep.add_parser(name);q.add_argument('--root',required=True);q.add_argument('--out',required=True)
        if name=='gesture':q.add_argument('--steps',type=int,default=100);q.add_argument('--window',type=float,default=3.);q.add_argument('--size',type=int,default=32)
    q=prep.add_parser('spikes');q.add_argument('--train',required=True);q.add_argument('--val');q.add_argument('--test',required=True);q.add_argument('--out',required=True);q.add_argument('--dataset',choices=('SHD','SSC'),default='SHD');q.add_argument('--steps',type=int,default=100);q.add_argument('--window',type=float,default=1.4)
    q=prep.add_parser('manifest');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--objective',choices=('classification','regression','reconstruction'),default='classification');q.add_argument('--classes',type=int)
    q=prep.add_parser('earthnet');q.add_argument('--train',required=True);q.add_argument('--test',required=True);q.add_argument('--out',required=True);q.add_argument('--steps',type=int,default=6);q.add_argument('--size',type=int,default=32);q.add_argument('--group-unit',choices=('tile','cube'),default='tile')
    q=prep.add_parser('mnist');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.add_argument('--mode',choices=('sequential','permuted','repeat'),default='sequential');q.add_argument('--permutation-seed',type=int,default=20261029);q.add_argument('--steps',type=int,default=16)
    q=prep.add_parser('esc50');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.add_argument('--test-fold',type=int,default=5)
    q=prep.add_parser('forecast-csv');q.add_argument('--csv',required=True);q.add_argument('--out',required=True);q.add_argument('--context',type=int,default=96);q.add_argument('--horizon',type=int,default=24);q.add_argument('--stride',type=int,default=24);q.add_argument('--columns',nargs='+');q.add_argument('--date-column',default='date');q.add_argument('--dataset-id',default='real_forecast');q.add_argument('--embargo',type=int,default=0);q.add_argument('--block-size',type=int,default=1024);q.add_argument('--role-allocation',choices=('proportional','minimum_first'),default='proportional')
    q=prep.add_parser('wisdm');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.add_argument('--window',type=int,default=200);q.add_argument('--stride',type=int,default=100);q.add_argument('--device',choices=('watch','phone'),default='watch');q.add_argument('--sensor',choices=('gyro','accel'),default='gyro')
    q=prep.add_parser('openbmi');q.add_argument('--root',required=True);q.add_argument('--out',required=True);q.add_argument('--sampling-hz',type=float,required=True);q.add_argument('--downsample',type=int,default=10)
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
    if args.command=='models':
        from .specs import REFERENCE_BACKBONES,BASELINES,CELLS
        result=json.loads((Path(__file__).parent/'resources'/'model_sources.json').read_text()) if args.sources else {'snn_reference_backbones':REFERENCE_BACKBONES,'continuous_reference_backbones':BASELINES,'injectable_reference_cells':CELLS,'official_reproduction':False}
    elif args.command=='upstream-checkout':
        from .upstream import checkout_suite
        result=checkout_suite(args.suite,args.out)
    elif args.command=='upstream-run':
        from .upstream import run_suite
        result=run_suite(args.suite,args.source,args.config,args.out)
    elif args.command=='campaign-plan':
        from .campaign import campaign_plan
        result=campaign_plan(args.root,args.archives,args.seeds,args.profiles,args.materialize,args.data_root)
    elif args.command=='campaign-stage':
        from .campaign import stage_postprocessing
        result=stage_postprocessing(args.root,args.phase,args.checkpoints,args.archive_scenes,args.archive_batch)
    elif args.command=='post-worker':
        from .campaign import post_worker
        result=post_worker(args.db,args.device,args.one)
        if result.get('failed',0):print(json.dumps(result));raise SystemExit(1)
    elif args.command=='post-job':
        from .campaign import run_post_job
        result=run_post_job(json.loads(Path(args.job).read_text()),args.device)
    elif args.command=='archive':
        from .archive import archive_run
        result={'archive':str(archive_run(args.run,args.data,args.checkpoint,args.device,args.roles,args.scenes,args.batch_size,not args.owners_only))}
    elif args.command=='doctor':
        import importlib,torch,numpy,sys
        modules=[p.stem for p in (Path(__file__).parent/'kernels').glob('*.py') if p.stem!='__init__']
        for m in modules:importlib.import_module('snnbench.kernels.'+m)
        result={'python':sys.version,'torch':str(torch.__version__),'numpy':numpy.__version__,'cuda_available':torch.cuda.is_available(),'gpus':torch.cuda.device_count(),'kernel_imports':len(modules)}
        from .campaign import hardware_inventory
        result['hardware']=hardware_inventory()
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
        result=compile_manifest(args.out,args.profile,args.seeds,args.archives,args.limit,args.model_roster)
    elif args.command=='train':
        from .engine import train
        result={'run':str(train(RunSpec.load(args.spec),args.out,args.data,args.device,max_updates=args.max_updates))}
    elif args.command=='analyze':
        from .analysis import analyze
        s=RunSpec.load(Path(args.run)/'spec.json');changes={k:getattr(args,k) for k in ('fit_scenes','select_scenes','eval_scenes','batch_size','max_features','max_targets') if getattr(args,k) is not None}
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
        elif args.adapter in ('mnist','esc50','forecast-csv','wisdm','openbmi'):
            from .real_data import prepare_mnist,prepare_esc50,prepare_forecast_csv,prepare_wisdm,prepare_openbmi
            if args.adapter=='mnist':result=prepare_mnist(args.root,args.out,args.mode,args.permutation_seed,args.steps)
            elif args.adapter=='esc50':result=prepare_esc50(args.root,args.out,args.test_fold)
            elif args.adapter=='wisdm':result=prepare_wisdm(args.root,args.out,args.window,args.stride,args.device,args.sensor)
            elif args.adapter=='openbmi':result=prepare_openbmi(args.root,args.out,args.sampling_hz,args.downsample)
            else:result=prepare_forecast_csv(args.csv,args.out,args.context,args.horizon,args.stride,args.columns,args.date_column,args.dataset_id,args.embargo,args.block_size,role_allocation=args.role_allocation)
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
