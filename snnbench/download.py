"""Standard-library server downloader. No model/data dependencies required.

Resume accepts only a correctly matching Content-Range; a server ignoring Range
restarts the file. MD5 is retained only where the dataset provider publishes it;
every completed file also gets our local SHA256 receipt.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile


def checksum(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def verify(path,item):
    if item.get('bytes') is not None and Path(path).stat().st_size!=item['bytes']:raise ValueError('File size mismatch: '+str(path))
    for alg in ('md5','sha256'):
        if item.get(alg) and checksum(path,alg)!=item[alg]:raise ValueError(alg+' mismatch: '+str(path))
    return {'bytes':Path(path).stat().st_size,'sha256':checksum(path),'provider_checksum':bool(item.get('md5') or item.get('sha256'))}


def download_file(item,out,attempts=3,timeout=60):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);name=item['filename']
    if Path(name).name!=name:raise ValueError('A download filename must be a basename')
    dest=out/name;part=out/(name+'.part');metadata=out/(name+'.part.json');receipt=out/(name+'.receipt.json')
    url=item['url']
    if dest.exists():
        info=verify(dest,item)
        if info['provider_checksum']:return info
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['url']==url and old['sha256']==info['sha256']:return info
        raise ValueError('Existing file has no matching checksum/receipt; move it aside before downloading: '+str(dest))
    for attempt in range(attempts):
        try:
            previous=json.loads(metadata.read_text()) if metadata.exists() else {}
            start=part.stat().st_size if part.exists() and previous.get('url')==url else 0
            headers={'User-Agent':'snnbench-research/0.2','Accept-Encoding':'identity'}
            if start:
                headers['Range']=f'bytes={start}-'
                validator=previous.get('etag') or previous.get('last_modified')
                if validator:headers['If-Range']=validator
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=timeout) as response:
                status=response.status;expected=None;mode='wb'
                if status==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=start:raise ValueError('Invalid resumed Content-Range')
                    if match[3]!='*':expected=int(match[3])
                    if int(match[2])<int(match[1]):raise ValueError('Invalid byte range')
                    mode='ab' if start else 'wb'
                elif status==200:
                    start=0
                    if response.headers.get('Content-Length'):expected=int(response.headers['Content-Length'])
                else:raise ValueError('Unexpected HTTP status '+str(status))
                write_json(metadata,{'url':url,'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified'),'expected_bytes':expected})
                with part.open(mode) as f:
                    for block in iter(lambda:response.read(2**20),b''):f.write(block)
                    f.flush();os.fsync(f.fileno())
                if expected is not None and part.stat().st_size!=expected:raise IOError('Incomplete response; partial file retained for retry')
            info=verify(part,item);write_json(receipt,dict(info,url=url,filename=name,checked_on='2026-10-29',verification='provider checksum' if info['provider_checksum'] else 'transport length + locally recorded SHA256; no published checksum claimed'))
            os.replace(part,dest);metadata.unlink(missing_ok=True);return info
        except urllib.error.HTTPError as exc:
            if exc.code==416 and part.exists():
                match=re.fullmatch(r'bytes \*/(\d+)',exc.headers.get('Content-Range',''))
                if match and part.stat().st_size==int(match[1]):
                    info=verify(part,item);write_json(receipt,dict(info,url=url,filename=name));os.replace(part,dest);metadata.unlink(missing_ok=True);return info
            if attempt+1==attempts:raise
        except (OSError,ValueError):
            if attempt+1==attempts:raise
        time.sleep(min(2**attempt,4))
    raise RuntimeError('Download attempts exhausted')


def _safe_path(root,name):
    rel=PurePosixPath(name.replace('\\','/'))
    if rel.is_absolute() or '..' in rel.parts or (rel.parts and ':' in rel.parts[0]):raise ValueError('Unsafe archive path: '+name)
    result=(root/str(rel)).resolve()
    if result!=root.resolve() and root.resolve() not in result.parents:raise ValueError('Archive escape')
    return result


def extract_archive(path,out):
    path=Path(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                _safe_path(out,info.filename)
                if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink is not accepted')
            z.extractall(out)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            for member in t.getmembers():
                _safe_path(out,member.name)
                if not (member.isfile() or member.isdir()):raise ValueError('Archive links/devices are not accepted')
            t.extractall(out,filter='data')
    elif path.suffix=='.gz':
        dest=_safe_path(out,path.stem)
        with gzip.open(path,'rb') as src,dest.with_suffix(dest.suffix+'.part').open('wb') as dst:shutil.copyfileobj(src,dst)
        os.replace(dest.with_suffix(dest.suffix+'.part'),dest)
    else:raise ValueError('Unsupported archive: '+str(path))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default=str(Path(__file__).parent/'resources'/'datasets.json'))
    p.add_argument('--out',default='downloads');p.add_argument('--ids',nargs='*');p.add_argument('--tier',choices=('P0','P1','P2','all'),default='P0')
    p.add_argument('--list',action='store_true');p.add_argument('--extract',action='store_true');p.add_argument('--verify-only',action='store_true');p.add_argument('--include-native',action='store_true',help='Also fetch optional author-native preprocessed archives; these may omit required group identities');args=p.parse_args(argv)
    manifest=json.loads(Path(args.manifest).read_text());items=[x for x in manifest['datasets'] if (not args.ids or x['id'] in args.ids) and (args.tier=='all' or x['priority']==args.tier or bool(args.ids))]
    if args.ids and set(args.ids)-{x['id'] for x in items}:raise ValueError('Unknown dataset ID')
    for item in items:
        print(json.dumps({k:item.get(k) for k in ('id','name','priority','access','adapter','source_url')},ensure_ascii=False),flush=True)
        if args.list:continue
        if not item.get('files'):
            print('No direct binary file registered; follow the official source/generator instructions above.',flush=True);continue
        folder=Path(args.out)/item['id']
        for f in item['files']:
            if f.get('optional') and not args.include_native:continue
            info=verify(folder/f['filename'],f) if args.verify_only else download_file(f,folder)
            print(json.dumps(dict(filename=f['filename'],**info)),flush=True)
            if args.extract:extract_archive(folder/f['filename'],folder/'extracted')


if __name__=='__main__':main()
