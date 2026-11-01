"""Persistent local separating-plane stopping certificates.

Perfect synchronous observations and common known execution model required.
Each ego candidate is checked only against its own contributions; no neighbor
candidate commands are read or modified. Pair planes can be independently
reconstructed by both robots from the same observations and tracked IDs.
"""
import numpy as np
from scipy.spatial import cKDTree
from .physical_execution import track,braking_distance
from .physical_governor import speed_limit,clip


def projected_braking(v,normal,a,tau):
    s=np.linalg.norm(v,axis=1)
    return np.maximum(-np.sum(v*normal,axis=1),0)*np.divide(braking_distance(s,a,tau),s,out=np.zeros_like(s),where=s>1e-30)


class DirectionalGovernor:
    def __init__(self):self.planes={}

    def command(self,p,vm,v,selected,*,dt,a_max,tau,v_max,r_vis,d_safe,margin=.02,surface_normal=None,tangent_priority=False):
        safe=d_safe+margin;limit=speed_limit(v_max,r_vis,safe,dt,a_max,tau)
        pairs=cKDTree(p).query_pairs(r_vis,output_type='ndarray');n=len(v)
        target=clip(selected,limit);outside=np.linalg.norm(v,axis=1)>limit+1e-10
        if not len(pairs):
            self.planes={};target[outside]=0.
            return target,dict(speed_limit=limit,outside_agents=int(outside.sum()),infeasible_agents=0,intervened_agents=int(outside.sum()),plane_resets=0)
        i,j=pairs.T;relative=p[i]-p[j];distance=np.linalg.norm(relative,axis=1)
        fresh=relative/np.maximum(distance[:,None],1e-30)
        proposed=projected_braking(vm[i],fresh,a_max,tau)+projected_braking(vm[j],-fresh,a_max,tau)
        reset=distance-safe-proposed>=-1e-10
        plane=np.array([fresh[k] if reset[k] else self.planes.get((int(ii),int(jj)),fresh[k]) for k,(ii,jj) in enumerate(pairs)])
        self.planes={(int(ii),int(jj)):plane[k].copy() for k,(ii,jj) in enumerate(pairs)}
        owner=np.r_[i,j];normal=np.concatenate([plane,-plane])
        sigma=projected_braking(v[owner],normal,a_max,tau)
        measured_sigma=projected_braking(vm[owner],normal,a_max,tau)
        signed=np.sum(relative*plane,axis=1)-safe
        # Each owner uses its own actual velocity and the neighbor observation.
        # Canonical subtraction order preserves the exact-observation arithmetic.
        slack_i=signed-sigma[:len(i)]-measured_sigma[len(i):]
        slack_j=signed-measured_sigma[:len(i)]-sigma[len(i):]
        budgets=np.r_[sigma[:len(i)]+slack_i/2,sigma[len(i):]+slack_j/2]
        np.logical_or.at(outside,i,slack_i < -1e-9);np.logical_or.at(outside,j,slack_j < -1e-9)
        zero=np.zeros_like(v)
        def per_edge_cost(u):
            delta,end,_=track(zero,v,u,dt,a_max=a_max,tau=tau)
            endpoint=-np.sum(delta[owner]*normal,axis=1)+projected_braking(end[owner],normal,a_max,tau)
            start_y=np.sum(v[owner]*normal,axis=1);end_y=np.sum(end[owner]*normal,axis=1)
            error=u-v;length=np.linalg.norm(error,axis=1);direction=error/np.maximum(length[:,None],1e-30)
            slope=np.sum(direction[owner]*normal,axis=1)
            crossing=(start_y<0)&(end_y>0)&(slope>1e-30)
            reduction=np.divide(-start_y,slope,out=np.zeros_like(slope),where=crossing)
            size=length[owner]
            if tau==0:
                t=np.clip(reduction/a_max,0,dt);linear=np.minimum(t,size/a_max)
                integral=.5*a_max*linear**2+size*(t-linear)
            else:
                switch=np.maximum(size/a_max-tau,0);linear_reduction=a_max*switch
                rest_size=np.minimum(size,a_max*tau)
                extra=np.divide(reduction-linear_reduction,rest_size,out=np.zeros_like(size),where=rest_size>1e-30)
                t=np.where(reduction<=linear_reduction,reduction/a_max,switch-tau*np.log(np.maximum(1-extra,1e-30)))
                t=np.clip(t,0,dt);linear=np.minimum(t,switch);rest=t-linear;remaining=np.maximum(size-a_max*linear,0)
                integral=.5*a_max*linear**2+a_max*linear*rest+remaining*(rest-tau*(-np.expm1(-rest/tau)))
            internal=np.where(crossing,-start_y*t-slope*integral,0.)
            return np.maximum(np.maximum(endpoint,internal),0.)
        def accept(u,tolerance=0.):
            fail=per_edge_cost(u)>budgets+tolerance
            result=np.ones(n,dtype=bool);np.logical_and.at(result,owner,~fail)
            return result
        feasible_brake=accept(zero,1e-9);infeasible=~feasible_brake
        full=accept(target,1e-12)
        def segment(base,direction):
            candidate=base+direction;ok_full=accept(candidate,1e-12)
            if np.all(ok_full):return candidate
            lo=np.zeros(n);hi=np.ones(n)
            for _ in range(32):
                fraction=(lo+hi)/2;ok=accept(base+fraction[:,None]*direction)
                lo=np.where(ok,fraction,lo);hi=np.where(ok,hi,fraction)
            return np.where(ok_full[:,None],candidate,base+lo[:,None]*direction)
        if np.all(full)&(~np.any(outside)):
            result=target
        elif tangent_priority and surface_normal is not None:
            nn=surface_normal/np.maximum(np.linalg.norm(surface_normal,axis=1)[:,None],1e-30)
            normal_part=np.sum(target*nn,axis=1)[:,None]*nn
            tangential=segment(zero,target-normal_part)
            trial=segment(tangential,normal_part)
            result=np.where(full[:,None],target,trial)
        else:
            result=segment(zero,target)
        result[outside|infeasible]=0.
        accepted=accept(result,1e-8)
        if np.any((~outside)&(~infeasible)&(~accepted)):raise RuntimeError('directional stopping certificate acceptance failed')
        return result,dict(speed_limit=limit,outside_agents=int(outside.sum()),infeasible_agents=int(infeasible.sum()),
            intervened_agents=int(np.sum(np.linalg.norm(result-target,axis=1)>1e-9)),plane_resets=int(reset.sum()))
