"""Pinned author-code execution, separate from our intervention-ready references.

This runner records code/input/command/log provenance. It does not relabel native
results as common-protocol results or pretend to expose unimplemented state hooks.
Install each source's dependencies in its own environment before running.
"""
import json
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time
from .utils import atomic_json,file_hash


def suites():return json.loads((Path(__file__).parent/'resources/model_sources.json').read_text())['upstream_suites']


def _git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def checkout_suite(name,out):
    suite=suites()[name];root=Path(out)
    if root.exists():raise FileExistsError('Use a new source checkout: '+str(root))
    root.mkdir(parents=True)
    for args in (('init',),('remote','add','origin',suite['repository']),('fetch','--depth','1','origin',suite['commit']),('checkout','--detach','FETCH_HEAD')):
        subprocess.run(['git','-C',str(root),*args],check=True)
    if _git(root,'rev-parse','HEAD')!=suite['commit']:raise ValueError('Author commit mismatch')
    return {'suite':name,'checkout':str(root.resolve()),'commit':suite['commit'],'status':'source checked out; dependencies and data still required'}


def run_suite(name,source,config,out):
    """config JSON: argv, input_files. Paths resolve against author checkout.

    argv[0] is an allowed Python script, or the allowed '-m', module pair; the
    current Python interpreter is prepended. No shell command interpolation.
    input_files includes at least one actual data manifest/config to hash.
    """
    suite=suites()[name];root=Path(source).resolve();plan=json.loads(Path(config).read_text());argv=plan.get('argv',[])
    working=(root/suite.get('working_directory','.')).resolve()
    if working!=root and root not in working.parents:raise ValueError('Working directory must stay inside author checkout')
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv):raise ValueError('argv must be a nonempty JSON string list')
    prefix=argv[:2] if argv[0]=='-m' else argv[:1]
    if prefix not in suite['entrypoints']:raise ValueError('Entry point not registered for this pinned source')
    if argv[0]!='-m':
        script=(working/argv[0]).resolve()
        if root not in script.parents or not script.is_file():raise ValueError('Author entry script must exist inside checkout')
    commit=_git(root,'rev-parse','HEAD')
    if commit!=suite['commit']:raise ValueError('Checkout differs from pinned author commit')
    if _git(root,'status','--porcelain','--untracked-files=no'):raise ValueError('Tracked author files changed; use a separate registered patch revision')
    untracked=_git(root,'ls-files','--others','--exclude-standard').splitlines()
    if any(p.endswith('.py') for p in untracked):raise ValueError('Untracked Python could change imports; move it outside the source checkout')
    inputs=plan.get('input_files',[])
    if not inputs or not isinstance(inputs,list):raise ValueError('At least one explicit data/config manifest must be pinned')
    provenance=[]
    for p in inputs:
        path=Path(p);path=path if path.is_absolute() else root/path
        if not path.is_file():raise FileNotFoundError(path)
        provenance.append({'path':str(path.resolve()),'sha256':file_hash(path)})
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new run receipt directory')
    out.mkdir(parents=True);command=[sys.executable,*argv];started=time.time();log=out/'native.log'
    receipt={'schema':'author_native_execution_v1','suite':name,'commit':commit,'repository':suite['repository'],
        'command':command,'cwd':str(working),'source_root':str(root),'inputs':provenance,'plan_sha256':file_hash(config),
        'protocol':plan.get('protocol','author native; exact split/settings must be recorded in input files'),
        'status':'running','common_state_analysis':False,'official_reproduction_verified':False,
        'environment':{'python':sys.version,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},
        'input_integrity_scope':'only listed files; raw data must have a separate checksummed inventory'}
    for package in ('torch','numpy','spikingjelly','timm','triton','mamba-ssm'):
        try:receipt['environment'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:receipt['environment'][package]=None
    atomic_json(out/'receipt.json',receipt)
    try:
        with log.open('w') as f:result=subprocess.run(command,cwd=working,stdout=f,stderr=subprocess.STDOUT,check=False)
        receipt.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        if result.returncode:raise RuntimeError('Native author command failed; inspect '+str(log))
    except BaseException:
        if receipt['status']=='running':receipt['status']='interrupted_or_launcher_error'
        raise
    finally:
        receipt.update(elapsed_seconds=time.time()-started,log_sha256=file_hash(log) if log.exists() else None)
        receipt['tracked_source_changed_during_run']=bool(_git(root,'status','--porcelain','--untracked-files=no'))
        if receipt['tracked_source_changed_during_run'] and receipt['status']=='completed':receipt['status']='completed_but_source_changed'
        atomic_json(out/'receipt.json',receipt)
    return receipt
