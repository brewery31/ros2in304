"""Cooperative local stopping-budget governor for the specified discrete lag model.

Guarantee scope: synchronous participants, exact instantaneous observations,
known constant command lag, bounded speed and initial stopping-budget membership.
No explicit robot-to-robot messages. Outside-set fallback is physical braking,
not a claimed guarantee or a zero-velocity teleport.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree


def stopping_distance(speed, dt, a_max, alpha):
    """Exact future travel under command braking max(s-a_max*dt,0).

    Actual next speed is s-alpha*min(s,a_max*dt); the final low-speed
    geometric tail is summed to infinity, rather than truncated as zero.
    """
    s=np.asarray(speed,dtype=float)
    if dt<=0 or a_max<=0 or not 0<alpha<=1 or np.any(s<0):
        raise ValueError('positive dt/acceleration, nonnegative speed and 0<alpha<=1 required')
    c=a_max*dt;delta=alpha*c
    count=np.maximum(np.ceil((s-c)/delta),0)
    remainder=np.maximum(s-count*delta,0)
    return dt*(count*s-delta*count*(count+1)/2+remainder*(1-alpha)/alpha)


@lru_cache(maxsize=64)
def visibility_speed_limit(v_max, r_vis, safe, dt, a_max, alpha):
    if r_vis<=safe:raise ValueError('visibility range must exceed safe distance plus margin')
    lo=0.;hi=float(v_max)
    for _ in range(50):
        mid=(lo+hi)/2
        if safe+2*stopping_distance(mid,dt,a_max,alpha)+2*mid*dt<=r_vis:lo=mid
        else:hi=mid
    return lo


def _clip(x,limit):
    return x*np.minimum(1.,limit/np.maximum(np.linalg.norm(x,axis=1),1e-15))[:,None]


def stopping_command(p_meas,v_meas,own_v,selected,*,dt,a_max,v_max,r_vis,d_safe,alpha,margin=.02):
    p=np.asarray(p_meas,float);vm=np.asarray(v_meas,float);v=np.asarray(own_v,float)
    safe=d_safe+margin;n=len(p)
    if not (p.shape==vm.shape==v.shape==np.shape(selected)) or p.shape!=(n,3):raise ValueError('matching N x 3 states required')
    limit=visibility_speed_limit(v_max,r_vis,safe,dt,a_max,alpha)
    speed=np.linalg.norm(v,axis=1);measured_speed=np.linalg.norm(vm,axis=1)
    b=stopping_distance(speed,dt,a_max,alpha);bm=stopping_distance(measured_speed,dt,a_max,alpha)
    budget=np.full(n,np.inf);outside=speed>limit+1e-10
    pairs=cKDTree(p).query_pairs(r_vis,output_type='ndarray')
    minimum_slack=np.inf
    if len(pairs):
        i,j=pairs.T;gap=np.linalg.norm(p[i]-p[j],axis=1)-safe
        ei=gap-b[i]-bm[j];ej=gap-b[j]-bm[i]
        minimum_slack=float(min(ei.min(),ej.min()))
        np.logical_or.at(outside,i,ei < -1e-10);np.logical_or.at(outside,j,ej < -1e-10)
        np.minimum.at(budget,i,.5*(gap+b[i]-bm[j]))
        np.minimum.at(budget,j,.5*(gap+b[j]-bm[i]))
    # Invert L(s)=dt*s+B(s); maximum braking is feasible inside the set.
    lo=np.zeros(n);hi=np.full(n,limit)
    for _ in range(42):
        mid=(lo+hi)/2;ok=dt*mid+stopping_distance(mid,dt,a_max,alpha)<=budget
        lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    cap=lo
    brake_command=v-_clip(v,a_max*dt)
    command=v+_clip(_clip(np.asarray(selected,float),limit)-v,a_max*dt)
    w0=v+alpha*(command-v);wb=v+alpha*(brake_command-v)
    # Convex interpolation preserves both original command constraints.
    # Select the first intersection with the convex speed ball analytically.
    constrained=np.linalg.norm(w0,axis=1)>cap+1e-12
    infeasible=np.linalg.norm(wb,axis=1)>cap+1e-9
    delta=wb-w0;aa=np.sum(delta*delta,axis=1);bb=2*np.sum(w0*delta,axis=1)
    cc=np.sum(w0*w0,axis=1)-cap*cap
    root=np.sqrt(np.maximum(bb*bb-4*aa*cc,0))
    t=np.divide(2*cc,-bb+root,out=np.ones(n),where=(-bb+root)>1e-30)
    t=np.where(constrained,np.clip(t,0,1),0.)
    t=np.where(outside|infeasible,1.,t)
    result=(1-t[:,None])*command+t[:,None]*brake_command
    return result,dict(speed_limit=limit,outside_agents=int(outside.sum()),
        infeasible_agents=int(infeasible.sum()),intervened_agents=int(np.sum(t>1e-9)),
        minimum_stopping_slack=minimum_slack)
