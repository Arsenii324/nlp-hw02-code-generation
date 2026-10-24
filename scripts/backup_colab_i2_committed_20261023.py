"""Native actual committed backup; selected model + active fullstate, no arrays."""
import argparse,hashlib,importlib.util,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--owner-sha',required=True);p.add_argument('--helper',required=True);a=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timeout(*_):raise TimeoutError('Backup deadline')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
try:
 helper=Path(a.helper);assert sha(helper)=='442a382a14359a4373d5df9b277f914ed65003a60e6a63bb913242023d000167'
 spec=importlib.util.spec_from_file_location('approved_snapshot',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 capture=m.snapshot(a.name,a.owner_sha,sha(helper));dest=Path(capture['snapshot_directory']);pointer=json.loads((dest/'pointer_before.json').read_text());slot=pointer['slot'];assert slot in ('durable-slot0','durable-slot1')
 source=m.BASE/'result/worker'/slot/'last_training.pt';assert source.is_file() and not source.is_symlink() and 0<source.stat().st_size<=256*1024**2
 target=dest/'last_training.pt';os.link(source,target);assert sha(target)==pointer['files_sha256']['last_training.pt']
 assert not(m.BASE/'result/session_result.json').exists()
 original=json.loads((dest/'snapshot_manifest.json').read_text());files=dict(original['files']);files['snapshot_manifest.json']={'sha256':sha(dest/'snapshot_manifest.json'),'size_bytes':(dest/'snapshot_manifest.json').stat().st_size};files['last_training.pt']={'sha256':sha(target),'size_bytes':target.stat().st_size}
 total=sum(x['size_bytes'] for x in files.values());assert total<=256*1024**2
 result={'schema':'colab_i2_actual_resumable_backup_v1','complete':True,'training_complete':False,'owner_manifest_sha256':a.owner_sha,'pointer_sha256':original['pointer_sha256'],'durable_absolute_step':pointer['absolute_step'],'selected_step':original['selected_step'],'source_snapshot_manifest_sha256':sha(dest/'snapshot_manifest.json'),'fullstate_included':True,'arrays_included':False,'files':files,'bytes':total}
 m.atomic_json(dest/'resumable_backup_manifest.json',result);print(json.dumps({'complete':True,'directory':str(dest),'manifest_sha256':sha(dest/'resumable_backup_manifest.json'),'bytes':total,'absolute_step':pointer['absolute_step']}))
except BaseException as e:
 print(json.dumps({'complete':False,'error_type':type(e).__name__}));raise SystemExit(1)
