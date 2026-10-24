"""Bounded real mounted-byte verification; not a synthetic fixture."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256();count=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            count+=len(block)
            if count>6*1024**3:raise ValueError('Unexpected cache-file size')
            h.update(block)
    return h.hexdigest(),count

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text());result={}
    allowed={'data_stats.json','validation.json','validation_legacy.json','tokenizer.json','train_source.npy','train_target.npy','train_row_ids.npy'}
    for name,root in [('b033',Path(a.old)),('b034',Path(a.new))]:
        digest,_=sha(root/'cache_manifest.json')
        if digest!=m['cache_manifests_sha256'][name]:raise ValueError('Cache manifest mismatch')
        cm=json.loads((root/'cache_manifest.json').read_text())
        if not cm['complete'] or set(cm['checksums'])!=allowed:raise ValueError('Incomplete/unexpected cache manifest')
        verified={};total=0
        for filename,expected in cm['checksums'].items():
            actual,count=sha(root/filename)
            if actual!=expected:raise ValueError('Mounted byte mismatch: '+name+'/'+filename)
            total+=count;verified[filename]=actual
        result[name]={'files':verified,'verified_bytes':total,'cache_manifest_sha256':digest}
    print(json.dumps({'event':'mounted_cache_bytes_verified','lanes':result}),flush=True)
