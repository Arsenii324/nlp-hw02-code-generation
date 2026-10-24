"""Build a local-only source package for the three-model 1169 mixture proof.

This helper performs no checkpoint deserialization, inference, provider call, or upload.
The built notebook is explicitly source-only pending independent review and root binding.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/kaggle/antigravity-mixture1169-proof-stage-20261024a"
TOOLS = STAGE / "tools"
MODELS = TOOLS / "models"
SOURCE_SHA = "07e58176c39eed8ed03134074282d74589be97120e6d6569a15f27eee1538b26"
TOK_SHA = "a255d879979714a88f50e64dac1d462705a935acc1bb01f5f055f02bbad66d6f"
ARCH = {
    "own768_gelu": {"vocab_size":2048,"d_model":768,"heads":12,"encoder_layers":4,"decoder_layers":4,"ff":3072,"dropout":0.1,"max_source":256,"max_target":192,"exclusive_self_attention":False,"learnable_xsa":True,"activation":"gelu","loop_repeats":1,"share_loop_weights":True,"residual_scale":1.0,"loop_order":"stack","loop_share_scope":"block"},
    "own512_gelu": {"vocab_size":2048,"d_model":512,"heads":8,"encoder_layers":4,"decoder_layers":4,"ff":2048,"dropout":0.1,"max_source":256,"max_target":192,"exclusive_self_attention":False,"learnable_xsa":True,"activation":"gelu","loop_repeats":1,"share_loop_weights":True,"residual_scale":1.0,"loop_order":"stack","loop_share_scope":"block"},
}
WEIGHTS = [
    ("768_fp_n8", "artifacts/paid-native-backups-57422123/gelu768/20261024T091350-79019/model.pt", "ea4f5bc5e72d418987bc8fe1439e0afc736a7ae51447aab5363a70d154bd6972", "own768_gelu", 531072, "wide_gelu768_fullpopulation_continuation_a_v1", 8),
    ("512_400k_n4", "artifacts/native-gelu512-400k-parent-20261024a/roles/model.pt", "81fd42627751284c7ed61d23fa3aec76f8a2c34f857ae9f78d5bd12155068306", "own512_gelu", 400000, "wide_scratch_training_a_v1", 4),
    ("512_new_n2", "artifacts/paid-native-backups-57422123/gelu/20261024T103443-79019/model.pt", "fd5bbeeadc8f451a2d47c740d73a38e75d3e5cdb65b72194e5065a138c161462", "own512_gelu", 662144, "wide_fullpopulation_continuation_a_v1", 2),
]
PROOF_ZIP = ROOT / "artifacts/kaggle/own-fullpop512-n16-proof-20261024a/tools/candidate_bundle.bin"
TEST_PRED = ROOT / "artifacts/mixture_768fp_n8_512old_n4_512new_n2_test_predictions.jsonl"
POOL_FILES = [
    ROOT / "artifacts/antigravity_gelu768_fullpop531k_eval/test_n8_candidates.jsonl",
    ROOT / "artifacts/kaggle/wide-t4-budget-v1-a-diagnostics/run-gelu512/test_n8_candidates.jsonl",
    ROOT / "artifacts/antigravity_gelu512_long262k_eval/test_n8_candidates.jsonl",
]
VAL_FILES = [
    ROOT / "artifacts/antigravity_gelu768_fullpop531k_eval/validation_n8_candidates.jsonl",
    ROOT / "artifacts/kaggle/wide-t4-budget-v1-a-diagnostics/run-gelu512/validation_n8_candidates.jsonl",
    ROOT / "artifacts/antigravity_gelu512_long262k_eval/validation_n8_candidates.jsonl",
]


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def source_only_notebook() -> dict:
    md = """# Three-model plurality code-generation proof — source only

This package freezes 768_FP N=8 + 512_400k N=4 + 512_new N=2. It contains no test labels or extra training data. It runs only after an independent review and root binding. No score or T4 qualification is implied by this notebook artifact.

The execution path loads three independently identified checkpoints from the same reviewed 4+4 GELU model source, samples each model at its assigned count, and applies the frozen numeric safe-AST plurality selector across the 14 candidates per question. The expected output digest is a post-run reproduction check only; it is not read during selection.
"""
    setup = '''import os,sys,time,json,hashlib,zipfile,shutil,tempfile
from pathlib import Path,PurePosixPath
NOTEBOOK_START=time.perf_counter()
ROOT=Path.cwd(); PACKAGE=ROOT/"candidate_bundle_v5.zip"
EXPECTED_PACKAGE_SHA="SOURCE_ONLY_UNBOUND"
ROOT_BOUND=False
if not ROOT_BOUND or EXPECTED_PACKAGE_SHA == "SOURCE_ONLY_UNBOUND": raise RuntimeError("Unbound source-only package; wait for independent review and root binding")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(PACKAGE)!=EXPECTED_PACKAGE_SHA: raise RuntimeError("Exact root-bound bundle SHA required")
WORK=Path(tempfile.mkdtemp(prefix="mixture-proof-")); names=[]
with zipfile.ZipFile(PACKAGE) as z:
 names=z.namelist()
 if len(names)!=len(set(names)): raise RuntimeError("Duplicate package members")
 MANIFEST=json.loads(z.read("stage_manifest_v5.json"))
 for info in z.infolist():
  q=PurePosixPath(info.filename)
  if q.is_absolute() or ".." in q.parts or info.is_dir() or (info.external_attr>>16)&0o170000==0o120000: raise RuntimeError("Unsafe package path")
  dst=WORK.joinpath(*q.parts);dst.parent.mkdir(parents=True,exist_ok=True)
  with z.open(info) as src,dst.open('xb') as out:shutil.copyfileobj(src,out,1048576)
  expected=MANIFEST.get("payload_sha256",{}).get(info.filename)
  if expected and sha(dst)!=expected: raise RuntimeError("Payload SHA mismatch: "+info.filename)
if not MANIFEST.get("verified_actual_values") or MANIFEST.get("status")!="ROOT_BOUND_READY": raise RuntimeError("Root binding absent")
if sha(WORK/"test_questions.json")!=MANIFEST["test_questions_sha256"]: raise RuntimeError("Question projection hash changed")
sys.path.insert(0,str(WORK/"src"))
'''
    load = '''import torch
from tokenizers import Tokenizer
from nlp_hw02.model import load_model
if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or "T4" not in torch.cuda.get_device_name(0): raise RuntimeError("Exactly one visible NVIDIA T4 required")
torch.set_num_threads(2)
MODEL_LOAD_START=time.perf_counter(); models=[]; specs=[]
if sha(WORK/"src/nlp_hw02/model.py")!=MANIFEST["model_source_sha256"]: raise RuntimeError("Frozen shared model source differs")
for spec in MANIFEST["models"]:
 path=WORK/spec["file"]
 if sha(path)!=spec["sha256"]: raise RuntimeError("Model weight SHA differs: "+spec["name"])
 model,checkpoint=load_model(path)
 if checkpoint.get("config")!=spec["config"] or checkpoint.get("step")!=spec["step"] or checkpoint.get("seed")!=42 or checkpoint.get("dataset_revision")!="b2724689ff079e9c22efd548fe134bfb29036dbb": raise RuntimeError("Checkpoint metadata differs: "+spec["name"])
 if model.cfg.max_source!=256 or model.cfg.max_target!=192: raise RuntimeError("Generation limits differ")
 models.append(model); specs.append(spec); del checkpoint
TOKENIZER=Tokenizer.from_file(str(WORK/"tokenizer.json")); QUESTIONS=json.loads((WORK/"test_questions.json").read_text())
if len(QUESTIONS)!=1919 or any(not isinstance(q,str) for q in QUESTIONS): raise RuntimeError("Expected ordered question-only 1919 rows")
torch.cuda.synchronize(); MODEL_LOAD_SECONDS=time.perf_counter()-MODEL_LOAD_START
'''
    finish = '''from nlp_hw02.mixture_runtime import run_mixture
OUT=Path.cwd()/"candidate_output"
if OUT.exists(): raise RuntimeError("Fresh output directory required; preserve previous proof output")
RESULT=run_mixture(models,TOKENIZER,QUESTIONS,OUT,specs,MANIFEST)
NOTEBOOK_SECONDS=time.perf_counter()-NOTEBOOK_START
RESULT["clocks"].update({"model_tokenizer_input_load_seconds":MODEL_LOAD_SECONDS,"notebook_start_through_output_readback_hashes":NOTEBOOK_SECONDS})
RESULT["notebook_within180"]=NOTEBOOK_SECONDS<=180
RESULT["core_within180"]=RESULT["clocks"]["generation_selection_serialization_seconds"]<=180
RESULT["four_clock_worker_measurements_required"]=True
RESULT["proof_qualified"]=False
RESULT["reproduction_passed"]=RESULT["output_reproduced"] and RESULT["notebook_within180"] and RESULT["core_within180"]
(OUT/"candidate_timing.json").write_text(json.dumps(RESULT,indent=2,allow_nan=False)+"\\n")
print(json.dumps({"rows":RESULT["rows"],"output_reproduced":RESULT["output_reproduced"],"clocks":RESULT["clocks"],"proof_qualified":False},indent=2))
if not RESULT["reproduction_passed"]: raise RuntimeError("Output reproduction or notebook/core clock gate failed; preserve outputs")
'''
    cells=[{"cell_type":"markdown","metadata":{},"source":md.splitlines(True)}]
    for source in (setup,load,finish):
        compile(source,"<notebook-cell>","exec")
        cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(True)})
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"},"proof_status":"SOURCE_ONLY_UNBOUND"},"nbformat":4,"nbformat_minor":5}


def validation_replay() -> dict:
    """Replay the frozen safe numeric selector on labeled validation-only pools."""
    sys.path.insert(0,str(ROOT/"scripts"))
    from antigravity_build_champion_mixture import select_plurality, equal
    from nlp_hw02.arithmetic import evaluate_code
    pools=[[json.loads(line) for line in p.read_text().splitlines()] for p in VAL_FILES]
    refs=json.loads((ROOT/"reports/evidence/primary_reference_consensus.json").read_text())["groups_by_question_sha256"]
    if any(len(p)!=1792 for p in pools): raise ValueError("Expected all three 1792-row validation candidate pools")
    solvable=correct=0
    for i in range(1792):
        q=pools[0][i]["question_sha256"]
        if any(pool[i]["question_sha256"]!=q for pool in pools): raise ValueError("Validation candidate row orders differ")
        ref=refs[q]
        if ref["status"]!="supported_consensus": continue
        codes=pools[0][i]["candidates"][:8]+pools[1][i]["candidates"][:4]+pools[2][i]["candidates"][:2]
        scores=pools[0][i]["scores"][:8]+pools[1][i]["scores"][:4]+pools[2][i]["scores"][:2]
        idx,_=select_plurality(codes,scores);pred=evaluate_code(codes[idx]);solvable+=1
        correct+=bool(pred.supported and equal(pred.value,ref["consensus_value"]))
    if (correct,solvable)!=(1169,1529): raise ValueError(f"Frozen validation replay differs: {correct}/{solvable}")
    return {"schema":"antigravity_mixture1169_validation_replay_a_v1","config":[8,4,2],"rows":1792,"supported_consensus_denominator":solvable,"correct":correct,"score":correct/solvable,"candidate_pool_sha256":[sha(p) for p in VAL_FILES],"consensus_sha256":sha(ROOT/"reports/evidence/primary_reference_consensus.json"),"selector_sha256":sha(TOOLS/"src/nlp_hw02/sampling_candidate.py"),"uses_test_labels":False}


def main() -> None:
    if not PROOF_ZIP.is_file() or not TEST_PRED.is_file(): raise FileNotFoundError("Reviewed donor runtime or selected mixture output absent")
    free=shutil.disk_usage(ROOT).free
    need=sum((ROOT/p).stat().st_size for _,p,_,_,_,_,_ in WEIGHTS)*2+1024**3
    if free < need: raise OSError(f"Insufficient disk before model copies: have {free}, need {need}")
    for name,rel,digest,*_ in WEIGHTS:
        p=ROOT/rel
        if sha(p)!=digest: raise ValueError(f"Current source weight SHA changed: {name}")
    STAGE.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    model_meta=[]
    for name,rel,digest,architecture,step,family,count in WEIGHTS:
        dst=MODELS/(name+".pt")
        if dst.exists() and sha(dst)!=digest: raise FileExistsError(f"Preserving changed existing weight: {dst}")
        if not dst.exists(): shutil.copyfile(ROOT/rel,dst)
        if sha(dst)!=digest: raise ValueError(f"Copied weight SHA mismatch: {name}")
        model_meta.append({"name":name,"file":str(dst.relative_to(TOOLS)),"sha256":digest,"architecture":architecture,"config":ARCH[architecture],"model_source_sha256":SOURCE_SHA,"step":step,"family":family,"samples_per_question":count,"source_path":rel})
    with zipfile.ZipFile(PROOF_ZIP) as src:
        for name in ("src/nlp_hw02/__init__.py","src/nlp_hw02/arithmetic.py","src/nlp_hw02/model.py","src/nlp_hw02/sampling_candidate.py","src/nlp_hw02/tokenizer.py","src/nlp_hw02/own_selected_sampling_d.py","tokenizer.json","test_questions.json"):
            out=TOOLS/name;out.parent.mkdir(parents=True,exist_ok=True)
            data=src.read(name)
            if out.exists() and sha(out)!=hashlib.sha256(data).hexdigest(): raise FileExistsError(f"Preserving changed existing payload: {out}")
            if not out.exists(): out.write_bytes(data)
    excluded={"candidate_bundle.zip","candidate_bundle_v2.zip","candidate_bundle_v3.zip","candidate_bundle_v4.zip","candidate_bundle_v5.zip","inference.ipynb","inference_v2.ipynb","inference_v3.ipynb","inference_v4.ipynb","inference_v5.ipynb","stage_manifest.json","stage_manifest_v2.json","stage_manifest_v3.json","stage_manifest_v4.json","stage_manifest_v5.json"}
    payload={p.relative_to(TOOLS).as_posix():sha(p) for p in sorted(TOOLS.rglob("*")) if p.is_file() and p.name not in excluded}
    test_questions=json.loads((TOOLS/"test_questions.json").read_text())
    if len(test_questions)!=1919 or any(not isinstance(q,str) for q in test_questions): raise ValueError("Donor question projection is not exact 1919 question-only rows")
    validation=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in VAL_FILES]
    test_pools=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in POOL_FILES]
    replay=validation_replay()
    evidence_dir=ROOT/"reports/evidence/antigravity_mixture1169_stage_20261024a"
    evidence_dir.mkdir(parents=True,exist_ok=True)
    replay_path=evidence_dir/"validation_replay_v2.json"
    if replay_path.exists(): raise FileExistsError(f"Preserving prior validation replay: {replay_path}")
    replay_path.write_text(json.dumps(replay,indent=2)+"\n")
    if any(not p.is_file() for p in VAL_FILES+POOL_FILES): raise FileNotFoundError("A candidate pool input is missing")
    manifest={"schema":"antigravity_mixture1169_source_stage_a_v1","status":"LOCAL_STAGE_AWAITING_INDEPENDENT_REVIEW_AND_ROOT_BINDING","verified_actual_values":False,"no_inference_or_model_deserialization":True,"no_provider_actions":True,"models":model_meta,"model_source_sha256":SOURCE_SHA,"tokenizer_sha256":TOK_SHA,"counts":[8,4,2],"candidates_per_question":14,"test_rows":1919,"test_questions_sha256":sha(TOOLS/"test_questions.json"),"expected_test_output_sha256":sha(TEST_PRED),"expected_test_output_rows":1919,"validation_candidate_pools":validation,"test_candidate_pools_for_source_review_only":test_pools,"reference_validation_consensus_sha256":sha(ROOT/"reports/evidence/primary_reference_consensus.json"),"payload_sha256":payload,"t4_runtime_estimate_seconds":163.2,"four_clock_requirement_seconds":180,"fallback_candidate":{"counts":[8,1,3],"validation_legacy":1168,"estimated_t4_seconds":150.2,"use_only_if_material_timing_margin_required":True}}
    manifest.update({"selector_sha256":payload["src/nlp_hw02/sampling_candidate.py"],"ordered_test_questions_sha256":hashlib.sha256(json.dumps(test_questions,ensure_ascii=False,separators=(",",":")).encode()).hexdigest(),"selected_validation_legacy":replay["correct"],"selected_validation_denominator":replay["supported_consensus_denominator"],"validation_replay_receipt_sha256":sha(replay_path),"test_label_access":False,"validation_reference_is_used_for_selection_only":True})
    (TOOLS/"stage_manifest_v5.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(TOOLS/"candidate_bundle_v5.zip","x",compression=zipfile.ZIP_STORED) as z:
        for p in sorted(TOOLS.rglob("*")):
            if p.is_file() and (p.name not in excluded or p.name=="stage_manifest_v5.json"): z.write(p,p.relative_to(TOOLS).as_posix())
    nb=source_only_notebook();(TOOLS/"inference_v5.ipynb").write_text(json.dumps(nb,indent=1)+"\n")
    build={"schema":"antigravity_mixture1169_local_build_a_v1","complete":True,"stage":str(STAGE.relative_to(ROOT)),"payload_sha256":payload,"package_sha256":sha(TOOLS/"candidate_bundle_v5.zip"),"notebook_sha256":sha(TOOLS/"inference_v5.ipynb"),"validation_replay_receipt_sha256":sha(replay_path),"weighted_copy_bytes":sum((MODELS/(m["name"]+".pt")).stat().st_size for m in model_meta),"free_before_bytes":free,"free_after_bytes":shutil.disk_usage(ROOT).free,"proof_qualified":False,"model_or_notebook_execution":False,"provider_action":False,"promotion":False}
    (STAGE/"build_receipt_v5.json").write_text(json.dumps(build,indent=2)+"\n")
    print(json.dumps({k:build[k] for k in ("complete","stage","package_sha256","notebook_sha256","weighted_copy_bytes","free_before_bytes","free_after_bytes","proof_qualified")},indent=2))


if __name__=="__main__": main()
