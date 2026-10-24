"""Fresh private CPU cache stage, preserving every frozen training-source member."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stage(project,tag='20261004c'):
    if tag!='20261004c':raise ValueError('Only reviewed fresh stage tag supported')
    project=Path(project);tools=project/('artifacts/kaggle/tools-cache-6m-'+tag);kernel=project/('kaggle/cache-6m-'+tag)
    if tools.exists() or kernel.exists():raise ValueError('Fresh immutable stage required')
    original=project/'artifacts/kaggle/b031-b032-v1/training_source.bin'
    if sha(original)!='6c840a501233f425099b6d9a313f532f66800591aa15c7d6030b166cf0465dea':raise ValueError('Frozen B031 source mismatch')
    tools.mkdir(parents=True);kernel.mkdir(parents=True)
    with zipfile.ZipFile(original) as z,zipfile.ZipFile(tools/'source.bin','w',zipfile.ZIP_DEFLATED) as out:
        if z.testzip() is not None:raise ValueError('Original CRC failure')
        for name in z.namelist():out.writestr(name,z.read(name))
        for name in ('prepare_cache_6m.py','check_cache_coverage.py'):
            target='scripts/'+name
            if target in z.namelist():raise ValueError('Addition would overwrite original source')
            out.writestr(target,(ROOT/'scripts'/name).read_bytes())
    with zipfile.ZipFile(tools/'source.bin') as z:
        names=z.namelist()
        if len(names)!=41 or len(set(names))!=len(names) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe/unexpected source')
        member_hashes={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
    with zipfile.ZipFile(original) as z:
        for n in z.namelist():
            if member_hashes[n]!=hashlib.sha256(z.read(n)).hexdigest():raise ValueError('Frozen source changed')
    parent=project/'artifacts/kaggle/prepare-v2/cache'
    pinned=['cache_manifest.json','data_stats.json','validation.json','validation_legacy.json','tokenizer.json']
    parent_hashes={n:sha(parent/n) for n in pinned};parent_hashes['train_row_ids.npy']=json.loads((parent/'cache_manifest.json').read_text())['checksums']['train_row_ids.npy']
    assignment=json.loads((project/'assignment/dataset_manifest.json').read_text())
    shutil.copyfile(ROOT/'scripts/kaggle_cache_6m.py',kernel/'run.py')
    manifest={'train_rows':6000000,'validation_rows':2048,'seed':42,'outer_seconds':5400,'worker_seconds':4800,
              'dataset_revision':assignment['revision'],'raw_bytes':sum(f['size'] for f in assignment['files'] if f['path'].startswith('data/train/')),
              'source_sha256':sha(tools/'source.bin'),'original_training_source_sha256':sha(original),
              'source_members_sha256':member_hashes,'driver_sha256':sha(kernel/'run.py'),
              'parent_kernel':'arsen4ikvar/nlp-hw02-prepare-v2/1','parent_cache_sha256':parent_hashes,
              'GPU_requested':False,'no_local_arrays':True,'training':False,'no_submission':True}
    (tools/'cache_6m_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    slug='nlp-hw02-tools-cache-6m-'+tag;(tools/'dataset-metadata.json').write_text(json.dumps({'id':'arsen4ikvar/'+slug,'title':slug.replace('-',' '),'licenses':[{'name':'other'}]},indent=2)+'\n')
    slug='nlp-hw02-cache-6m-'+tag;metadata={'id':'arsen4ikvar/'+slug,'title':slug.replace('-',' '),'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':True,'dataset_sources':['arsen4ikvar/nlp-hw02-tools-cache-6m-'+tag],'kernel_sources':['arsen4ikvar/nlp-hw02-prepare-v2'],'competition_sources':[],'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','docker_image_pinning_type':'original'}
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return {'utc':datetime.now(timezone.utc).isoformat(),'tools':str(tools),'kernel':str(kernel),'manifest':manifest,'metadata':metadata,'tools_input_bytes':sum(p.stat().st_size for p in tools.iterdir() if p.name!='dataset-metadata.json'),'original39_source_members_unchanged':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);a=p.parse_args();print(json.dumps(stage(a.project_root),indent=2))
