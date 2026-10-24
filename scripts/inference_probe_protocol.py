"""Strict mounted parent identity and bounded local child finalization."""
import hashlib,json,os,re,signal,subprocess
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def verified_inference_parents(input_root,manifest):
 pair=set(manifest['parents'])
 if pair=={'b019','b024'}:completed=120000
 elif pair=={'b027','b028'}:completed=180000
 else:raise ValueError('Exact declared120k or180k inference pair required')
 terminals=list(Path(input_root).rglob('session_result.json'));paths={}
 for name,pins in manifest['parents'].items():
  run=pins['run']
  if run!=name or not re.fullmatch('b[0-9]+',run):raise ValueError('Invalid parent name')
  matches=[p for p in terminals if sha(p)==pins['terminal_sha256']]
  if len(matches)!=1:raise ValueError('Exactly one pinned parent terminal required: '+run)
  terminal=json.loads(matches[0].read_text());root=matches[0].parent
  if (terminal.get('complete') is not True or terminal.get('training_complete') is not True or
      terminal['training_source_sha256']!=pins['source_sha256'] or
      terminal['cache_manifest_sha256']!=manifest['cache_manifest_sha256']):raise ValueError('Parent completion/source/cache mismatch')
  required={f'run-{run}/{p}' for p in ['model.pt','run_config.json','metrics.json','tokenizer.json','validation.json']}
  if set(pins['files'])!=required:raise ValueError('Unexpected inference parent file roles')
  for filename,digest in pins['files'].items():
   if terminal['artifact_sha256'].get(filename)!=digest or sha(root/filename)!=digest:raise ValueError('Parent file hash mismatch: '+filename)
  directory=root/f'run-{run}';metadata=json.loads((directory/'run_config.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
  if (metadata['torch']!=manifest['torch'] or metadata['cuda']!=manifest['cuda'] or metadata['config']!=pins['config'] or
      metrics['completed_steps']!=completed or pins.get('completed_steps',completed)!=completed or
      metrics['selected_step']!=pins['selected_step'] or not 0<pins['selected_step']<=completed):raise ValueError('Parent software/config/steps mismatch')
  paths[run]=directory
 return paths


def finish_children(children):
 for child in children:
  if child.poll() is not None:continue
  try:os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   child.wait(timeout=5)
