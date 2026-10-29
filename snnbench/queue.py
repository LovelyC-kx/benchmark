"""Single-coordinator SQLite leases, bounded attempts, resumable subprocess jobs.

Place this database on one host's local filesystem. Multi-node use partitions
immutable manifests with --shard/--shards; it does not share SQLite over NFS.
"""
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
import signal
from .specs import RunSpec


def connect(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(path,timeout=60);con.execute('PRAGMA journal_mode=WAL');con.execute('PRAGMA busy_timeout=60000')
    con.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,spec TEXT NOT NULL,state TEXT NOT NULL DEFAULT \'pending\',owner TEXT,lease REAL,attempts INTEGER NOT NULL DEFAULT 0,error TEXT)');return con


def enqueue(database,manifest,shard=0,shards=1):
    if not 0<=shard<shards:raise ValueError('0 <= shard < shards')
    with connect(database) as con:
        for line in Path(manifest).open(encoding='utf-8'):
            spec=RunSpec.from_dict(json.loads(line))
            if int(spec.run_id,16)%shards==shard:con.execute('INSERT OR IGNORE INTO jobs(id,spec) VALUES(?,?)',(spec.run_id,json.dumps(spec.to_dict())))
        return dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall())


def claim(con,owner,lease_seconds,max_attempts):
    con.execute('BEGIN IMMEDIATE');now=time.time()
    con.execute("UPDATE jobs SET state='pending',owner=NULL WHERE state='running' AND lease<? AND attempts<?",(now,max_attempts))
    con.execute("UPDATE jobs SET state='failed',error='lease expired after maximum attempts' WHERE state='running' AND lease<? AND attempts>=?",(now,max_attempts))
    row=con.execute("SELECT id,spec FROM jobs WHERE state='pending' AND attempts<? ORDER BY id LIMIT 1",(max_attempts,)).fetchone()
    if row:con.execute("UPDATE jobs SET state='running',owner=?,lease=?,attempts=attempts+1 WHERE id=?",(owner,now+lease_seconds,row[0]))
    con.commit();return row


def worker(database,out='runs',data='prepared',device='cuda:0',analyze_after=False,lease_seconds=300,max_attempts=2,one=False):
    owner=socket.gethostname()+':'+str(os.getpid())+':'+uuid.uuid4().hex[:8];con=connect(database);work=Path(out)/'_queue_specs';work.mkdir(parents=True,exist_ok=True)
    active={'process':None,'job':None}
    def stop(signum,frame):
        process=active['process']
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if active['job'] is not None:
            con.execute("UPDATE jobs SET state='pending',owner=NULL,lease=NULL WHERE id=? AND owner=?",(active['job'],owner));con.commit()
        raise SystemExit(128+signum)
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    while True:
        row=claim(con,owner,lease_seconds,max_attempts)
        if row is None:break
        run_id,spec=row;path=work/(run_id+'.json');path.write_text(spec);log=work/(run_id+'.log');error=None
        active['job']=run_id
        cmd=[sys.executable,'-m','snnbench','train','--spec',str(path),'--out',str(out),'--data',str(data),'--device',device]
        with log.open('a') as f:
            process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            active['process']=process
            while process.poll() is None:
                time.sleep(2);con.execute("UPDATE jobs SET lease=? WHERE id=? AND owner=? AND state='running'",(time.time()+lease_seconds,run_id,owner));con.commit()
            if process.returncode:error='train exit '+str(process.returncode)+'; '+str(log)
            elif analyze_after:
                process=subprocess.Popen([sys.executable,'-m','snnbench','analyze','--run',str(Path(out)/run_id),'--data',str(data),'--device',device],stdout=f,stderr=subprocess.STDOUT)
                active['process']=process
                while process.poll() is None:
                    time.sleep(2);con.execute('UPDATE jobs SET lease=? WHERE id=? AND owner=?',(time.time()+lease_seconds,run_id,owner));con.commit()
                if process.returncode:error='analysis exit '+str(process.returncode)+'; '+str(log)
        con.execute("UPDATE jobs SET state=?,error=?,lease=NULL WHERE id=? AND owner=?",('failed' if error else 'done',error,run_id,owner));con.commit()
        active.update(process=None,job=None)
        print(json.dumps({'run_id':run_id,'status':'failed' if error else 'done','error':error}),flush=True)
        if one:break
    result=dict(con.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state').fetchall());con.close()
    for s,h in handlers.items():signal.signal(s,h)
    return result
