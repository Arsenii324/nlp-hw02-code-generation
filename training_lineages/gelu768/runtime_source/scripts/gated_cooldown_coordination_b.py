"""Bounded common-prefix admission over local parent-owned atomic JSON files."""
import hashlib
import json
import math
from pathlib import Path
import time
from nlp_hw02.gated_cooldown_schedule_a import ORIGIN,PREFIX,ROOT_CAP,ARMS,require,digest,valid_sha,validate_binding

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read_json(path):
    require(Path(path).stat().st_size<=1048576,'Bounded coordinator JSON required');return json.loads(Path(path).read_text())

def request_record(run_id,arm,contract_sha,plan_sha,pointer_sha,elapsed,rate,save_cost,raw_sha,applied_sha):
    require(math.isfinite(elapsed) and elapsed>0 and math.isfinite(rate) and rate>0 and math.isfinite(save_cost) and save_cost>0,'Measured positive checkpoint-inclusive prefix timing required')
    return {'schema':'gated_cooldown_prefix_request_a_v1','run_id':run_id,'arm':arm,'contract_sha256':contract_sha,'plan_sha256':plan_sha,'pointer_sha256':pointer_sha,
        'prefix_NEW_iterations':PREFIX,'absolute_step':ORIGIN+PREFIX,'ready_monotonic':time.monotonic(),'checkpoint_inclusive_seconds':elapsed,'measured_steps_per_second':rate,'fullsave_cost_seconds':save_cost,'raw_segment_sha256':raw_sha,'applied_segment_sha256':applied_sha}

def check_request(record,spec,contract_sha,plan_sha):
    require(record['schema']=='gated_cooldown_prefix_request_a_v1' and record['run_id']==spec['run_id'] and record['arm']==spec['cooldown_arm'] and record['contract_sha256']==contract_sha and record['plan_sha256']==plan_sha,'Actual prefix request role/source/plan differs')
    require(record['prefix_NEW_iterations']==PREFIX and record['absolute_step']==ORIGIN+PREFIX,'Exact actual1024 frontier required')
    for k in ('pointer_sha256','raw_segment_sha256','applied_segment_sha256'):require(valid_sha(record[k]),'Prefix provenance missing:'+k)
    for k in ('ready_monotonic','checkpoint_inclusive_seconds','measured_steps_per_second','fullsave_cost_seconds'):require(math.isfinite(record[k]) and record[k]>0,'Prefix measured rate/cost differs')
    require(record['measured_steps_per_second']==PREFIX/record['checkpoint_inclusive_seconds'] and record['ready_monotonic']<=time.monotonic(),'Prefix rate/actual ready clock differs')

def make_decision(requests,request_sha,worker_deadline,now,barrier_cutoff):
    require(len(requests)==2 and {v['arm'] for v in requests.values()}==set(ARMS),'Both explicit arms required')
    require(len({v['raw_segment_sha256'] for v in requests.values()})==len({v['applied_segment_sha256'] for v in requests.values()})==1,'Common prefix raw/applied draws differ')
    fits={};rows={}
    for name,r in requests.items():
        # Includes barrier and two bound saves; existing final-save/uncertainty reserve remains.
        reserve=max(180.,1.5*r['fullsave_cost_seconds']);binding_reserve=max(60.,2.5*r['fullsave_cost_seconds']);remaining=worker_deadline-now
        fit=PREFIX+max(0,int(max(0.,remaining-reserve-binding_reserve)*r['measured_steps_per_second']/1.25)//64*64)
        fits[name]=min(ROOT_CAP,fit);rows[name]={'fit_NEW_horizon':fits[name],'remaining_seconds':remaining,'final_save_reserve_seconds':reserve,'binding_save_reserve_seconds':binding_reserve,'uncertainty_multiplier':1.25,'request':r,'request_sha256':request_sha[name]}
    horizon=min(fits.values());admitted=horizon>=PREFIX+64
    return {'schema':'gated_cooldown_common_decision_a_v1','admitted':admitted,'common_NEW_horizon':horizon if admitted else None,'origin_absolute_step':ORIGIN,'prefix_NEW_iterations':PREFIX,'root_cap_NEW_iterations':ROOT_CAP,'worker_deadline_monotonic':worker_deadline,'decision_monotonic':now,'barrier_cutoff_monotonic':barrier_cutoff,'runs':rows}

def await_binding(coord,run_id,request,atomic_json,deadline,stopping,checkpoint_bound):
    coord=Path(coord);request_path=coord/(run_id+'-prefix-request.json');atomic_json(request_path,request)
    end=min(deadline-180,time.monotonic()+120)
    def guard():
        if stopping() or time.monotonic()>=end:raise InterruptedError('Bounded shared-prefix barrier/release ended; retain durable1024 prefix')
    def wait(path):
        guard()
        while not path.is_file():
            guard();time.sleep(.1)
        guard();value=read_json(path);guard();return value
    decision_path=coord/'common_decision.json';decision=wait(decision_path);require(decision['schema']=='gated_cooldown_common_decision_a_v1' and decision['admitted'] is True,'Shared horizon not admitted; preserve prefix')
    cutoff=decision['barrier_cutoff_monotonic'];require(math.isfinite(cutoff) and cutoff<=deadline-180 and cutoff>decision['decision_monotonic'],'Explicit common barrier cutoff differs')
    end=min(end,cutoff);guard()
    name=decision['runs'][run_id];require(name['request_sha256']==sha(request_path) and name['request']==request and decision['origin_absolute_step']==ORIGIN and decision['prefix_NEW_iterations']==PREFIX and decision['root_cap_NEW_iterations']==ROOT_CAP,'Shared decision/request differs')
    # Release SHA cannot be circular with an acked fullstate: commit decision-only at prefix,
    # then bind actual release into another durable checkpoint BEFORE first postprefix draw.
    provisional={'schema':'gated_cooldown_schedule_binding_a_v1','common_NEW_horizon':decision['common_NEW_horizon'],'origin_absolute_step':ORIGIN,'prefix_NEW_iterations':PREFIX,'remaining_iterations':decision['common_NEW_horizon']-PREFIX,'remaining_midpoint':(decision['common_NEW_horizon']-PREFIX)//2,'decision_sha256':sha(decision_path),'release_sha256':None}
    guard();ack=checkpoint_bound(provisional,False);guard();atomic_json(coord/(run_id+'-decision-ack.json'),{'schema':'gated_cooldown_decision_ack_a_v1','run_id':run_id,'decision_sha256':sha(decision_path),'committed_pointer_sha256':ack})
    guard();release_path=coord/'common_release.json';release=wait(release_path)
    require(release['schema']=='gated_cooldown_common_release_a_v1' and release['decision_sha256']==sha(decision_path) and release['acks'][run_id]==sha(coord/(run_id+'-decision-ack.json')),'Shared release/ack differs')
    binding={**provisional,'release_sha256':sha(release_path)};validate_binding(binding);guard();checkpoint_bound(binding,True);guard();return binding,decision


def coordinate(lanes,coord,worker_deadline,state,atomic_json,emit,stop_peer):
    """Called each parent poll; no waits or new phase deadline resets."""
    coord=Path(coord);now=time.monotonic()
    if state.get('released'):
        rejected=[l['spec']['run_id'] for l in lanes if (coord/(l['spec']['run_id']+'-postprefix_reject.json')).is_file() or (l.get('finish_failure') and not l.get('horizon_complete',False)) or ('exited' in l and not l.get('horizon_complete',False))]
        if rejected and not state.get('peer_stop_requested'):
            state['pair_incomplete']=True;state['peer_stop_requested']=rejected;state['peer_stop_decision']=stop_peer('Shared immutableH experiment lost a training arm: '+','.join(rejected),rejected)
        return
    if any(l['child'].poll() is not None for l in lanes):raise RuntimeError('A trainer exited before shared cooldown release; preserve peer prefix')
    for lane in lanes:
        name=lane['spec']['run_id'];path=coord/(name+'-prefix-request.json')
        if path.is_file() and name not in state.setdefault('requests',{}):
            record=read_json(path);check_request(record,lane['spec'],lane['contract_sha256'],lane['plan_sha256'])
            pointer=lane['run']/'latest_checkpoint.json';require(sha(pointer)==record['pointer_sha256'],'Prefix committed pointer differs')
            committed=read_json(pointer);metrics=committed['metrics'];sampler=metrics['sampler_checkpoint'];schedule=metrics['cooldown_checkpoint'];segment=sampler['segments'][-1]
            require(committed['schema']=='gated_window_durable_pointer_v1' and committed['slot'] in ('durable-slot0','durable-slot1') and committed['absolute_step']==metrics['absolute_completed_step']==ORIGIN+PREFIX and committed['since_resume_iterations']==metrics['since_resume_iterations']==PREFIX,'Actual durable prefix pointer/frontier differs')
            require(committed['contract_sha256']==record['contract_sha256'] and sampler['at_consumed_window_boundary'] and sampler['pending_rows']==0 and sampler['absolute_step']==ORIGIN+PREFIX,'Prefix must be exact committed consumed-window state')
            require(schedule==sampler['cooldown_checkpoint'] and schedule['plan_sha256']==record['plan_sha256'] and schedule['binding'] is None and schedule['phase']=='waiting_admission','Prefix prospective schedule must remain unbound')
            require(segment['start_step']==ORIGIN and segment['end_step']==ORIGIN+PREFIX and segment['raw_segment_sha256']==record['raw_segment_sha256'] and segment['applied_segment_sha256']==record['applied_segment_sha256'],'Actual durable prefix raw/applied segment differs')
            state['requests'][name]=record;state.setdefault('request_sha256',{})[name]=sha(path);state['barrier_started']=min(state.get('barrier_started',record['ready_monotonic']),record['ready_monotonic']);state['barrier_cutoff']=min(worker_deadline-180,state['barrier_started']+120);emit('cooldown_prefix_ready',run_id=name,request=record)
    if state.get('barrier_cutoff') is not None and now>=state['barrier_cutoff']:raise RuntimeError('Shared1024 decision/ack/release barrier exceeded120s')
    if len(state.get('requests',{}))!=2:return
    decision_path=coord/'common_decision.json'
    if not decision_path.exists():
        decision=make_decision(state['requests'],state['request_sha256'],worker_deadline,now,state['barrier_cutoff']);atomic_json(decision_path,decision);state['decision']=decision;emit('cooldown_common_admission',decision=decision)
        require(decision['admitted'],'Neither arm may leave1024 without a positive shared64-aligned tail')
    decision=read_json(decision_path);acks={}
    for lane in lanes:
        name=lane['spec']['run_id'];path=coord/(name+'-decision-ack.json')
        if not path.is_file():return
        ack=read_json(path);require(ack['schema']=='gated_cooldown_decision_ack_a_v1' and ack['run_id']==name and ack['decision_sha256']==sha(decision_path) and sha(lane['run']/'latest_checkpoint.json')==ack['committed_pointer_sha256'],'Schedule-bound prefix ack/pointer differs');acks[name]=sha(path)
    require(time.monotonic()<decision['barrier_cutoff_monotonic'],'Common barrier cutoff reached before release')
    atomic_json(coord/'common_release.json',{'schema':'gated_cooldown_common_release_a_v1','decision_sha256':sha(decision_path),'acks':acks});state['released']=True;emit('cooldown_common_release',common_NEW_horizon=decision['common_NEW_horizon'])
