"""Fixed-H schedule for a typed wide origin0 parent; no optimizer reset or H shortening."""
import math
from .wide_scratch_recipe_a import absolute_lr

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_schedule(schedule,parent_step,horizon):
    require(set(schedule)=={'schema','mode','parent_absolute_step','new_horizon'},'Exact continuation schedule fields required')
    require(schedule['schema']=='wide_continuation_schedule_a_v1' and schedule['mode'] in ('absolute_isqrt','linear_cooldown'),'Explicit supported schedule required')
    require(type(parent_step) is int and parent_step>=1024 and parent_step%64==0 and schedule['parent_absolute_step']==parent_step,'Actual parent schedule origin required')
    require(type(horizon) is int and 1024<=horizon<=2000000 and horizon%64==0 and schedule['new_horizon']==horizon,'Fixed root64-aligned NEW horizon required')
    return schedule

def scheduled_lr(schedule,new_iterations):
    start=schedule['parent_absolute_step'];horizon=schedule['new_horizon'];validate_schedule(schedule,start,horizon)
    require(type(new_iterations) is int and 0<=new_iterations<=horizon,'Attempted NEW iteration outside fixed horizon')
    value=absolute_lr(start+new_iterations) if schedule['mode']=='absolute_isqrt' else absolute_lr(start)*max(0.,1.-new_iterations/horizon)
    require(math.isfinite(value) and value>=0,'Nonfinite scheduled LR');return value
