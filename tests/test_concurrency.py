from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from snnbench.data import DataStore
from snnbench.specs import *
from snnbench.catalogue import compile_manifest
from snnbench.queue import connect,enqueue,claim


class Concurrency(unittest.TestCase):
    def test_shared_data_preparation_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            def get(_):return np.array(DataStore(TaskSpec(),tmp).ensure('train',70)['x'])
            with ThreadPoolExecutor(max_workers=4) as ex:arrays=list(ex.map(get,range(4)))
            self.assertTrue(all(np.array_equal(arrays[0],a) for a in arrays[1:]))

    def test_atomic_queue_claim_and_idempotent_enqueue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'smoke.jsonl';compile_manifest(manifest,'smoke',(11,));db=root/'local/queue.sqlite'
            self.assertEqual(enqueue(db,manifest),{'pending':8});self.assertEqual(enqueue(db,manifest),{'pending':8})
            def get(i):
                c=connect(db)
                try:
                    r=claim(c,'worker'+str(i),300,2);return r[0] if r else None
                finally:c.close()
            with ThreadPoolExecutor(max_workers=8) as ex:ids=list(ex.map(get,range(8)))
            self.assertEqual(len(set(ids)),8);self.assertNotIn(None,ids)

    def test_stable_multinode_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'smoke.jsonl';compile_manifest(manifest,'smoke',(11,));sets=[]
            for i in range(3):
                db=root/(str(i)+'.sqlite');enqueue(db,manifest,i,3)
                with connect(db) as c:sets.append(set(x[0] for x in c.execute('SELECT id FROM jobs')))
            self.assertEqual(len(set.union(*sets)),8)
            for i in range(3):
                for j in range(i):self.assertFalse(sets[i]&sets[j])


if __name__=='__main__':unittest.main()
