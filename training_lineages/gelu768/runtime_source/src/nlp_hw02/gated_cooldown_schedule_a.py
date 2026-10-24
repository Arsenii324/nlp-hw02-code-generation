"""Versioned prospective cooldown plan and exact saved-LR oracle; stdlib only."""
from copy import deepcopy
import hashlib
import json
import math

ORIGIN=345000;PREFIX=1024;ROOT_CAP=81920
ARMS=('full-remaining','half-remaining')

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def valid_sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def prospective_plan(arm):
    require(arm in ARMS,'Explicit cooldown arm required')
    return {'schema':'gated_common_prefix_cooldown_plan_a_v1','arm':arm,'origin_absolute_step':ORIGIN,'prefix_NEW_iterations':PREFIX,'root_cap_NEW_iterations':ROOT_CAP,
        'base_schedule':'6e-4*min(absolute/200,sqrt(200/max(absolute,200)))','envelope':'(R-k)/(R-1)' if arm==ARMS[0] else '1 if k<=R/2 else (R-k)/(R-R/2)',
        'indexing':'j=absolute-origin;k=j-1024;R=boundH-1024;firstpostprefix1,last0','admission':'common_actual1024_checkpoint_inclusive_min64','schedule_binding_separate':True}

def validate_plan(plan):
    require(isinstance(plan,dict) and plan==prospective_plan(plan['arm']),'Immutable prospective schedule plan differs');return plan['arm']

def validate_binding(binding,allow_pending_release=False):
    require(isinstance(binding,dict) and binding['schema']=='gated_cooldown_schedule_binding_a_v1','Explicit schedule binding required')
    horizon=binding['common_NEW_horizon'];require(type(horizon) is int and PREFIX+64<=horizon<=ROOT_CAP and horizon%64==0,'Actual common64-aligned horizon required')
    require(binding['origin_absolute_step']==ORIGIN and binding['prefix_NEW_iterations']==PREFIX and binding['remaining_iterations']==horizon-PREFIX and binding['remaining_midpoint']==(horizon-PREFIX)//2,'Bound phase/index frontier differs')
    require(valid_sha(binding['decision_sha256']),'Missing actual common decision SHA')
    require(valid_sha(binding['release_sha256']) or (allow_pending_release and binding['release_sha256'] is None),'Missing actual common release SHA')
    return horizon

def base_lr(absolute):return 6e-4*min(absolute/200,math.sqrt(200/max(absolute,200)))

def scheduled_lr(absolute,plan,binding):
    arm=validate_plan(plan);require(type(absolute) is int and absolute>=ORIGIN,'Absolute LR index required');j=absolute-ORIGIN
    if binding is None:
        require(j<=PREFIX,'Never advance past1024 before common decision/release binding');return base_lr(absolute)
    horizon=validate_binding(binding,allow_pending_release=j<=PREFIX);require(j<=horizon,'Step exceeds immutable admitted horizon')
    if j<=PREFIX:return base_lr(absolute)
    remaining=horizon-PREFIX;k=j-PREFIX
    multiplier=(remaining-k)/(remaining-1) if arm==ARMS[0] else (1. if k<=remaining//2 else (remaining-k)/(remaining-remaining//2))
    return base_lr(absolute)*multiplier

def checkpoint_schedule(absolute,plan,binding):
    rate=scheduled_lr(absolute,plan,binding);j=absolute-ORIGIN
    phase=('waiting_admission' if j==PREFIX else 'prefix') if binding is None else (('decision_bound_prefix' if binding['release_sha256'] is None else 'bound_prefix') if j<=PREFIX else ('finished' if j==binding['common_NEW_horizon'] else 'cooldown'))
    base=base_lr(absolute)
    return {'schema':'gated_cooldown_checkpoint_a_v1','plan':deepcopy(plan),'plan_sha256':digest(plan),'binding':deepcopy(binding),'phase':phase,
        'origin_absolute_step':ORIGIN,'absolute_lr_step':absolute,'NEW_offset':j,'last_base_lr':base,'last_envelope':rate/base,'last_scheduled_lr':rate}

def saved_lr_oracle(state,metadata):
    saved=state.get('cooldown_checkpoint')
    if saved is None:
        require('cooldown_plan' not in metadata,'Cooldown parent lacks explicit saved schedule state');return base_lr(state['step'])
    plan=metadata['cooldown_plan'];require(metadata['cooldown_plan_sha256']==digest(plan),'Saved prospective plan digest differs')
    require(saved==checkpoint_schedule(state['step'],plan,saved['binding']),'Saved schedule phase/index/LR differs')
    require(state['sampler_checkpoint']['cooldown_checkpoint']==saved,'Sampler/fullstate schedule differs')
    return saved['last_scheduled_lr']
