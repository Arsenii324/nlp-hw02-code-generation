"""Actual cooldown selected weights CPU tensor audit only; protocol integrity is separate."""
import hashlib,json,sys,tempfile,zipfile
from pathlib import Path
from datetime import datetime,timezone
import torch
ROOT=Path(__file__).resolve().parents[1]
WT=ROOT/'.worktrees/strong-inference-fallback-20261022'
sys.path.insert(0,str(WT/'scripts'))
from cooldown_sampling_contract_b import EXPECTED_CONFIG,MODEL_SOURCE_SHA,RUNTIME,TRAINING_SOURCE_SHA

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def require(c,m):
 if not c:raise ValueError(m)

def main():
 torch.set_num_threads(2)
 diagnostics=ROOT/'artifacts/kaggle/cooldown-prefix-v1-b-diagnostics'
 models=ROOT/'artifacts/kaggle/cooldown-prefix-v1-b-selected-models'
 archive=ROOT/'.worktrees/gated-cooldown-prefix-20261023/artifacts/kaggle/cooldown-prefix-tools-20261023b/cooldown_continuation_source.bin'
 require(sha(archive)==TRAINING_SOURCE_SHA,'source archive changed')
 terminal=json.loads((diagnostics/'session_result.json').read_text())
 require(terminal['training_complete'] and terminal['complete'] and terminal['artifact_hashes_complete'],'incomplete actual terminal')
 results={}
 with tempfile.TemporaryDirectory(prefix='cooldown-cpu-integrity-') as temp:
  temp=Path(temp)
  with zipfile.ZipFile(archive) as source:
   for name,digest in RUNTIME.items():
    data=source.read(name);require(hashlib.sha256(data).hexdigest()==digest,'source member changed')
    target=temp/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  sys.path.insert(0,str(temp/'src'))
  from nlp_hw02.model import Seq2Seq,ModelConfig
  for run in ('cooldown-full','cooldown-half'):
   pointer_path=diagnostics/('run-'+run)/'latest_checkpoint.json';pointer=json.loads(pointer_path.read_text())
   prefix='run-'+run+'/'+pointer['slot']+'/'
   model_path=models/(prefix+'model.pt');digest=sha(model_path)
   require(digest==terminal['artifact_sha256'][prefix+'model.pt']==pointer['files_sha256']['model.pt'],'selected model SHA differs')
   checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
   require(set(checkpoint)=={'config','model','step','validation_loss','seed','dataset_revision'},'six fields required')
   require(checkpoint['config']==EXPECTED_CONFIG and checkpoint['seed']==42 and checkpoint['dataset_revision']=='b2724689ff079e9c22efd548fe134bfb29036dbb','config/data changed')
   metrics=pointer['metrics'];require(checkpoint['step']==metrics['selected_step'] and checkpoint['validation_loss']==metrics['best_validation_loss'],'CEbest metadata differs')
   model=Seq2Seq(ModelConfig(**checkpoint['config']))
   expected=model.state_dict();state=checkpoint['model']
   require(set(expected)==set(state),'state keys differ')
   require(all(v.shape==expected[k].shape and v.dtype==expected[k].dtype and bool(torch.isfinite(v).all()) for k,v in state.items()),'shape/dtype/finite failed')
   groups={}
   for name,param in model.named_parameters(remove_duplicate=False):groups.setdefault(id(param),[]).append(name)
   alias_groups=[names for names in groups.values() if len(names)>1]
   require(all(all(torch.equal(state[names[0]],state[n]) for n in names[1:]) for names in alias_groups),'alias state mismatch')
   model.load_state_dict(state,strict=True)
   require(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()),'strict load equality failed')
   count=sum(p.numel() for p in model.parameters());xsa={n:p.detach().tolist() for n,p in model.named_parameters() if n.endswith('.xsa_alpha')}
   require(count==17524256 and sum(len(v) for v in xsa.values())==32,'parameter/XSA count differs')
   require(len(model.encoder)==len(model.decoder)==4 and model.encoder_repeat_gates is None and model.decoder_repeat_gates is None,'topology differs')
   record={'schema':'cooldown_selected_weights_cpu_tensor_audit_v1','complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),'training_terminal_sha256':sha(diagnostics/'session_result.json'),'pointer_sha256':sha(pointer_path),'last_training_sha256':pointer['files_sha256']['last_training.pt'],'model_sha256':digest,'model_source_sha256':MODEL_SOURCE_SHA,'run_id':run,'slot':pointer['slot'],'config':checkpoint['config'],'step':checkpoint['step'],'validation_loss':checkpoint['validation_loss'],'seed':checkpoint['seed'],'dataset_revision':checkpoint['dataset_revision'],'parameters':count,'checkpoint_fields':sorted(checkpoint),'finite_state_tensors':True,'strict_alias_check':True,'unique_xsa_scalars':32,'state_tensor_count':len(state),'alias_groups':alias_groups,'xsa_alpha':xsa,'model_path':str(model_path),'scope':'Actual weights-only CPU tensor inspection/strict loading; no forward, synthetic fixture, CUDA, training, inference, or optimizer read. Fullstate hash is remote terminal/pointer provenance only.'}
   destination=ROOT/'reports/evidence/cooldown_prefix_launch_20261023b'/('selected_'+run+'_cpu_tensor_audit.json')
   require(not destination.exists(),'preserve existing integrity receipt')
   destination.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
   results[run]={'path':str(destination.relative_to(ROOT)),'sha256':sha(destination),'model_sha256':digest,'step':checkpoint['step'],'validation_loss':checkpoint['validation_loss'],'parameters':count}
   del model,checkpoint,state,expected
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
