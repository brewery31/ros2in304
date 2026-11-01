"""Exact zero-order-held velocity setpoint tracking with a physical acceleration cap.

v_dot = sat_norm((u-v)/tau, a_max), p_dot=v. tau=0 is rate-limited
velocity tracking. Bounds are on actual acceleration, not command change.
"""
import numpy as np
from .metrics import trajectory_step_min
from scipy.spatial import cKDTree


def track(p,v,u,t,*,a_max,tau):
    p=np.asarray(p,float);v=np.asarray(v,float);u=np.asarray(u,float)
    error=u-v;size=np.linalg.norm(error,axis=-1);direction=error/np.maximum(size[...,None],1e-30)
    if tau<0 or a_max<=0 or t<0:raise ValueError('nonnegative time/lag and positive acceleration required')
    if tau==0:
        linear=np.minimum(t,size/a_max);reduction=np.minimum(a_max*t,size)
        integral=.5*a_max*linear**2+size*(t-linear)
    else:
        linear=np.minimum(t,np.maximum(size/a_max-tau,0.))
        rest=t-linear;remaining=np.maximum(size-a_max*linear,0.)
        fraction=-np.expm1(-rest/tau)
        reduction=a_max*linear+remaining*fraction
        integral=.5*a_max*linear**2+a_max*linear*rest+remaining*(rest-tau*fraction)
    pn=p+v*t+direction*integral[...,None]
    vn=v+direction*reduction[...,None]
    # Convexity of velocity norm supplies an upper bound on actual path length.
    mixing=np.divide(integral,size,out=np.zeros_like(size),where=size>1e-30)
    length_bound=t*np.linalg.norm(v,axis=-1)+mixing*(np.linalg.norm(u,axis=-1)-np.linalg.norm(v,axis=-1))
    return pn,vn,np.maximum(length_bound,0.)


def braking_distance(speed,a_max,tau):
    s=np.asarray(speed,float)
    if np.any(s<0) or a_max<=0 or tau<0:raise ValueError('invalid braking model')
    if tau==0:return s*s/(2*a_max)
    return np.where(s<=a_max*tau,tau*s,s*s/(2*a_max)+a_max*tau*tau/2)


def audit_interval(p,v,u,dt,*,a_max,tau,d_safe):
    """Certified distance bracket for actual curved trajectories, not endpoint chords.

    Relative deviation from a chord is at most (2*a_max)*h^2/8. Refine
    only distance-threshold ambiguities; bounds never certify an ambiguous pass.
    """
    parts=1
    while True:
        left=p;lower=np.inf;upper=float(cKDTree(p).query(p,k=2)[0][:,1].min())
        for k in range(1,parts+1):
            right=track(p,v,u,k*dt/parts,a_max=a_max,tau=tau)[0]
            chord=trajectory_step_min(left,right)
            lower=min(lower,max(0.,chord-a_max*(dt/parts)**2/4))
            upper=min(upper,float(cKDTree(right).query(right,k=2)[0][:,1].min()))
            left=right
        if lower>=d_safe or upper<d_safe or parts>=64:break
        parts*=2
    return lower,upper,parts
