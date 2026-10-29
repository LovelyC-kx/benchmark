"""Download only the exact listed public author files, verifying both digests."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

def main(directory):
    rows=json.loads((Path(__file__).parent/'references/selected_author_sources.json').read_text())['files']
    for r in rows:
        p=Path(directory)/r['repository']/r['path'];url=f"https://raw.githubusercontent.com/{r['repository']}/{r['commit']}/{r['path']}"
        b=p.read_bytes() if p.exists() else urllib.request.urlopen(url,timeout=60).read()
        assert hashlib.sha256(b).hexdigest()==r['sha256'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['blob_sha1']
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);print('verified',r['repository'],r['path'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();main(a.directory)
