"""Retrieve only the eight declared author files, verifying their Git blob hashes."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(destination):
    dst=Path(destination)
    if dst.exists():raise FileExistsError(dst)
    dst.mkdir(parents=True);manifest=json.loads((ROOT/'references/selected_source_manifest.json').read_text());receipts={}
    for name,row in manifest.items():
        url=f'https://raw.githubusercontent.com/{row["repository"]}/{row["ref"]}/{row["path"]}'
        req=urllib.request.Request(url,headers={'User-Agent':'SNN-L3.6-pinned-source-audit'})
        with urllib.request.urlopen(req,timeout=60) as response:blob=response.read()
        sha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        if sha!=row['git_blob_sha']:raise ValueError(f'{name}: downloaded bytes differ from pinned Git blob')
        (dst/row['file']).write_bytes(blob);receipts[name]={'url':url,'git_blob_sha':sha,'bytes':len(blob)}
    (dst/'download_receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return receipts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print('Verified author files',len(run(a.output)))
