"""Reproducible CPU integration acceptance; never presented as full experiments."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
import numpy as np
from snnbench.catalogue import compile_manifest
from snnbench.queue import enqueue,worker
from snnbench.ledger import summarize
from snnbench.engine import implementation_hash
from snnbench.utils import atomic_json,file_hash

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--skip-unit',action='store_true');a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
if not a.skip_unit:
    with (root/'unit_tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit('Unit tests failed; see '+str(root/'unit_tests.log'))
manifest=root/'smoke.jsonl';compiled=compile_manifest(manifest,'smoke',(11,));enqueue(root/'queue.sqlite',manifest)
states=worker(root/'queue.sqlite',root/'runs',root/'data','cpu',analyze_after=True);summary=summarize(root/'runs',root/'summary')
ledger=json.loads((root/'summary/ledger.json').read_text())
if states!={'done':8} or ledger['counts']['trained_and_weights_changed']!=8 or ledger['counts']['analysis_receipts']!=8:raise SystemExit('Smoke integration failed: '+str(states))
receipt={'schema':'cpu_engineering_acceptance_v1','status':'passed','date':'2026-10-29','implementation_sha256':implementation_hash(),
    'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu'},
    'manifest':compiled,'queue_states':states,'ledger_counts':ledger['counts'],
    'runs':[dict(r,checkpoint_sha256=file_hash(root/'runs'/r['run_id']/'best.pt')) for r in ledger['runs']],
    'scope':'8 short CPU runs, 2 epochs, 32 training scenes each; verifies execution and artifacts. No GPU convergence/performance/novelty claim.'}
atomic_json(root/'acceptance.json',receipt);print(json.dumps(receipt,indent=2))
