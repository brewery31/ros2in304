"""Local stopping-budget governor for continuous capped velocity tracking."""
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree
from .physical_execution import track,braking_distance


def clip(x,limit):return x*np.minimum(1.,limit/np.maximum(np.linalg.norm(x,axis=1),1e-30))[:,None]

@lru_cache(maxsize=64)
def speed_limit(v_max,r_vis,safe,dt,a_max,tau):
    if r_vis<=safe:raise ValueError('visibility must exceed safety threshold')
    lo=0.;hi=v_max
    for _ in range(48):
        mid=(lo+hi)/2
        if safe+2*braking_distance(mid,a_max,tau)+2*mid*dt<=r_vis:lo=mid
        else:hi=mid
    return lo


def physical_command(pm,vm,v,selected,*,dt,a_max,tau,v_max,r_vis,d_safe,margin=.02,normal=None,priority=False):
    safe=d_safe+margin;limit=speed_limit(v_max,r_vis,safe,dt,a_max,tau)
    n=len(v);s=np.linalg.norm(v,axis=1);b=braking_distance(s,a_max,tau);bm=braking_distance(np.linalg.norm(vm,axis=1),a_max,tau)
    budget=np.full(n,np.inf);outside=s>limit+1e-10
    pairs=cKDTree(pm).query_pairs(r_vis,output_type='ndarray')
    if len(pairs):
        i,j=pairs.T;gap=np.linalg.norm(pm[i]-pm[j],axis=1)-safe
        np.minimum.at(budget,i,.5*(gap+b[i]-bm[j]));np.minimum.at(budget,j,.5*(gap+b[j]-bm[i]))
    outside|=budget < b-1e-10
    target=clip(selected,limit);zero=np.zeros_like(v)
    def cost(command):
        _,end,length=track(zero,v,command,dt,a_max=a_max,tau=tau)
        return length+braking_distance(np.linalg.norm(end,axis=1),a_max,tau)
    brake_cost=cost(zero);infeasible=brake_cost>budget+1e-9
    def feasible_segment(base,direction):
        candidate=base+direction;full_ok=cost(candidate)<=budget+1e-12
        if np.all(full_ok):return candidate
        lo=np.zeros(n);hi=np.ones(n)
        # Monotonicity is not assumed: retain an explicitly feasible endpoint.
        for _ in range(32):
            t=(lo+hi)/2;ok=cost(base+t[:,None]*direction)<=budget
            lo=np.where(ok,t,lo);hi=np.where(ok,hi,t)
        t=np.where(full_ok,1.,lo)
        return base+t[:,None]*direction
    if priority and normal is not None:
        nn=normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
        normal_part=np.sum(target*nn,axis=1)[:,None]*nn
        retained=feasible_segment(zero,normal_part)
        result=feasible_segment(retained,target-normal_part)
    else:result=feasible_segment(zero,target)
    result[outside|infeasible]=0.
    residual=cost(result)-budget
    if np.any((~outside)&(~infeasible)&(residual>1e-9)):raise RuntimeError('physical stopping budget acceptance failed')
    return result,dict(speed_limit=limit,outside_agents=int(outside.sum()),infeasible_agents=int(infeasible.sum()),
        intervened_agents=int(np.sum(np.linalg.norm(result-target,axis=1)>1e-9)))
