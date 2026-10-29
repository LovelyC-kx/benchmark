"""Fetch only exact inspected text files; verify SHA256 and Git blob digest."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def fetch(directory):
    v=json.loads((ROOT/'references/selected_author_sources.json').read_text())
    for row in v['files']:
        path=Path(directory)/row['repository']/row['path'];url=f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{row['path']}"
        b=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
        assert hashlib.sha256(b).hexdigest()==row['sha256'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['blob_sha1']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);print('verified',row['repository'],row['path'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();fetch(a.directory)
