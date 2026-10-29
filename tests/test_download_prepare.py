import functools
import gzip
import hashlib
import http.server
import io
import json
from pathlib import Path
import socketserver
import tarfile
import tempfile
import threading
import unittest
import zipfile
import numpy as np
from snnbench.download import download_file,extract_archive,checksum,write_json
from snnbench.prepare import write_records,verify_dataset,partition_groups
from snnbench.data import DataStore
from snnbench.specs import TaskSpec


class Downloader(unittest.TestCase):
    def test_mixed_csv_and_gzip_batch_extracts_only_container(self):
        from snnbench.download import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serve=root/'serve';serve.mkdir();out=root/'raw';csv=b'date,value\n2020-01-01,1\n'
            (serve/'ETTh1.csv').write_bytes(csv)
            with gzip.open(serve/'spikes.h5.gz','wb') as f:f.write(b'fixture h5')
            handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(serve))
            with socketserver.TCPServer(('127.0.0.1',0),handler) as server:
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_address[1]}'
                manifest=root/'sources.json';manifest.write_text(json.dumps({'datasets':[{'id':'fixture','priority':'P0','files':[
                    {'filename':'ETTh1.csv','url':base+'/ETTh1.csv'},{'filename':'spikes.h5.gz','url':base+'/spikes.h5.gz'}]}]}))
                try:
                    main(['--manifest',str(manifest),'--out',str(out),'--extract'])
                    self.assertEqual((out/'fixture/ETTh1.csv').read_bytes(),csv)
                    self.assertEqual((out/'fixture/extracted/spikes.h5').read_bytes(),b'fixture h5')
                    self.assertTrue((out/'fixture/ETTh1.csv.receipt.json').is_file())
                finally:server.shutdown();thread.join()

    def test_ignored_range_restarts_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serve=root/'serve';serve.mkdir();data=bytes(range(256))*50;(serve/'data.bin').write_bytes(data);out=root/'out';out.mkdir()
            handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(serve))
            with socketserver.TCPServer(('127.0.0.1',0),handler) as server:
                t=threading.Thread(target=server.serve_forever,daemon=True);t.start();url=f'http://127.0.0.1:{server.server_address[1]}/data.bin'
                (out/'data.bin.part').write_bytes(data[:301]);write_json(out/'data.bin.part.json',{'url':url})
                item={'filename':'data.bin','url':url,'sha256':hashlib.sha256(data).hexdigest()};r=download_file(item,out,attempts=1)
                self.assertEqual((out/'data.bin').read_bytes(),data);self.assertEqual(r['bytes'],len(data));download_file(item,out,attempts=1)
                with self.assertRaises(ValueError):download_file(dict(item,sha256='0'*64),out,attempts=1)
                server.shutdown();t.join()

    def test_valid_range_resume(self):
        payload=b'0123456789'*100
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0]);self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{len(payload)-1}/{len(payload)}');self.send_header('Content-Length',str(len(payload)-start));self.end_headers();self.wfile.write(payload[start:])
            def log_message(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp,socketserver.TCPServer(('127.0.0.1',0),Handler) as server:
            t=threading.Thread(target=server.serve_forever,daemon=True);t.start();out=Path(tmp);url=f'http://127.0.0.1:{server.server_address[1]}/x'
            (out/'x.part').write_bytes(payload[:77]);write_json(out/'x.part.json',{'url':url});download_file({'filename':'x','url':url,'md5':hashlib.md5(payload).hexdigest()},out,attempts=1)
            self.assertEqual((out/'x').read_bytes(),payload);server.shutdown();t.join()

    def test_archive_traversal_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);z=root/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escaped',b'x')
            with self.assertRaises(ValueError):extract_archive(z,root/'extract')
            t=root/'bad.tar'
            with tarfile.open(t,'w') as f:
                m=tarfile.TarInfo('link');m.type=tarfile.SYMTYPE;m.linkname='../escaped';f.addfile(m)
            with self.assertRaises(ValueError):extract_archive(t,root/'extract')
            g=root/'x.h5.gz'
            with gzip.open(g,'wb') as f:f.write(b'example h5 bytes')
            extract_archive(g,root/'extract');self.assertEqual((root/'extract/x.h5').read_bytes(),b'example h5 bytes')


class Canonical(unittest.TestCase):
    def test_group_isolation_hashes_and_memmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=[]
            for split,count in (('train',12),('test',4)):
                for g in range(count):
                    for j in range(3):records.append({'official_split':split,'group_id':split+':'+str(g),'scene_id':f'{split}:{g}:{j}','label':j%2})
            loader=lambda r:{'x':np.ones((4,3),'float32'),'y':r['label'],'timestamps':np.arange(4),'availability_times':np.arange(4)}
            out=Path(tmp)/'canonical';receipt=write_records(records,loader,out,{'name':'fixture','dataset_id':'test','target_mode':'class_index','classes':2,'output_shape':[2]})
            verify_dataset(out);t=TaskSpec(family='archive',name='fixture',archive_root=str(out),archive_manifest_sha256=receipt['manifest_sha256']);b=DataStore(t,Path(tmp)/'data').ensure('probe_train',2)
            self.assertEqual(b['x'].shape,(2,4,3));self.assertFalse(b['x'].flags.writeable)
            with self.assertRaises(ValueError):DataStore(t,Path(tmp)/'data').ensure('audit',1000)
            (out/'audit/x.npy').write_bytes(b'broken')
            with self.assertRaises(ValueError):verify_dataset(out)

    def test_future_and_group_leak_rejected(self):
        records=[{'official_split':'train','group_id':'x'},{'official_split':'test','group_id':'x'}]
        with self.assertRaises(ValueError):partition_groups(records)


if __name__=='__main__':unittest.main()
