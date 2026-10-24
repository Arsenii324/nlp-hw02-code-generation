"""Observational pinned2.10 engine diagnostics, never optimizer/admission truth."""
import math,time


def require(ok,message):
    if not ok:raise ValueError(message)


def module_identity(model):
    return {'parameters':[(n,id(p),list(p.shape),str(p.dtype)) for n,p in model.named_parameters()],
        'state_keys':list(model.state_dict()),'module_class':type(model).__module__+'.'+type(model).__qualname__}


def register_engine(model,engine):
    require(engine in ('eager','inplace-compile'),'Explicit engine required')
    before=module_identity(model);began=time.monotonic()
    if engine=='inplace-compile':model.compile(backend='inductor',mode='default',dynamic=True)
    seconds=time.monotonic()-began;after=module_identity(model)
    require(before==after,'In-place engine registration changed parameter/order/alias/state keys')
    return {'engine':engine,'backend':'inductor' if engine=='inplace-compile' else None,
        'mode':'default' if engine=='inplace-compile' else None,'dynamic':True if engine=='inplace-compile' else None,
        'registration_seconds':seconds,'registration_is_not_total_compile_time':True,
        'underlying_module_identity_unchanged':True,'parameter_count':sum(p.numel() for p in model.parameters()),
        'state_keys':after['state_keys'],'dropout_numerical_trajectory_equivalence_claimed':False}


def observe_engine(torch):
    record={'source':'torch._dynamo.utils pinned2.10 internal observational interface',
        'available':False,'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
        'peak_reserved_bytes':int(torch.cuda.max_memory_reserved())}
    try:
        from torch._dynamo import utils
        record['counters']={str(k)[:160]:{str(n)[:256]:int(v) for n,v in x.items() if isinstance(v,int)} for k,x in utils.counters.items()}
        phases={}
        for name,values in utils.compilation_time_metrics.items():
            clean=[float(v) for v in values if isinstance(v,(int,float)) and math.isfinite(v) and v>=0]
            if clean:phases[str(name)[:160]]={'count':len(clean),'seconds_sum':sum(clean),'seconds_max':max(clean)}
        record['compile_seconds_by_phase']=phases
        record['phase_times_may_overlap_do_not_sum_across_phases']=True
        records=utils.get_compilation_metrics();record['retained_compile_record_count']=len(records)
        keys=('compile_id','frame_key','co_name','cache_size','accumulated_cache_size','guard_count','graph_op_count',
            'entire_frame_compile_time_s','backend_compile_time_s','inductor_compile_time_s','code_gen_time_s','fail_type')
        record['compile_records']=[{k:v for k in keys if (v:=getattr(row,k,None)) is None or isinstance(v,(str,int,bool)) or isinstance(v,float) and math.isfinite(v)} for row in records[-64:]]
        record['retained_records_are_not_complete_lifetime_trace']=True;record['available']=True
    except Exception as error:record['diagnostic_error_type']=type(error).__name__
    return record
