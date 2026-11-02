"""Experimental, platform-neutral ego velocity interface; no hardware I/O.

Only one ego's fresh local velocity is supplied. Other velocities are measured
neighbor estimates. The output is NOT a certified real-flight command. Invalid
frames raise rather than inventing a safe zero velocity or an aircraft action.
"""
import numpy as np
from .curve_catalog import parse
from .curve_tasks import curve_acceleration
from .energy import Params
from .observation_filter import CausalCVFilter
from .directional_governor import DirectionalGovernor
from .orca_adapter import orca_velocity
from .sim_protocol import Settings,geometry,clip_norm


class CurveEgoInterface:
    def __init__(self,task,*,ego_id,agent_ids,max_observation_age=.1,max_local_velocity_age=.005):
        self.ids=tuple(agent_ids)
        if len(set(self.ids))!=len(self.ids) or ego_id not in self.ids:
            raise ValueError('unique tracked IDs and a valid ego ID required')
        self.ego=self.ids.index(ego_id);self.cfg=Settings(**task.get('sim',{}));self.time=None
        kind,n=parse(task['shape'])
        if kind not in ('ring','tilted_ring','helix','helix_flat','digit2','digit0','digit6','letterL','letterR') or n!=len(self.ids) or n!=task['n_agents'] or n>8:
            raise ValueError('interface is restricted to selected <=8-agent curve tasks')
        if task['method']!='curve_ours' or task.get('research_mode')!='observation_stress':
            raise ValueError('explicit empirical curve task required')
        c=self.cfg
        for value in [c.dt,c.a_max,c.v_max,c.a,c.d_safe,c.r_vis_factor,c.horizon]:
            if not np.isfinite(value) or value<=0:raise ValueError('positive finite physical settings required')
        if not np.isfinite(c.actuator_tau) or c.actuator_tau<0 or not np.isfinite(c.margin) or c.margin<0:
            raise ValueError('invalid execution lag/reserve')
        if not (c.execution_model=='physical_zoh' and c.safety and c.stopping_governor and c.governor_geometry=='directional'):
            raise ValueError('selected physical ORCA and directional-governor interface required')
        if not np.isfinite(c.tangential_damping) or not 0<=c.tangential_damping<=1:
            raise ValueError('tangential damping must be within [0,1]')
        if not np.isfinite(c.command_filter_tau) or not 0<=c.command_filter_tau<=2:
            raise ValueError('command filter time constant must be within [0,2] s')
        if c.normal_priority or c.tangent_priority or c.uncertain_tangent_speed or task.get('params'):
            raise ValueError('unsupported optional nominal/projection modes')
        for age in [max_observation_age,max_local_velocity_age]:
            if not np.isfinite(age) or age<0:raise ValueError('finite nonnegative age limits required')
        self.max_age=float(max_observation_age);self.max_local_age=float(max_local_velocity_age)
        self.cloud=geometry(task['shape'],1,task.get('control_points',4096),task.get('eval_points',32768),c.a)[0]
        self.prm=Params(a=c.a,mu=c.a/c.d_safe,r_vis=c.r_vis_factor)
        processor=task.get('observation_processor')
        if processor!=dict(smoothing_seconds=0.,predict_to_now=True):
            raise ValueError('only the frozen timestamp-extrapolation candidate is supported')
        self.observer=CausalCVFilter(v_max=c.v_max,**processor)
        self.command_filtered=None
        self.governor=DirectionalGovernor();self.endpoint=task.get('curve_controller',{}).get('endpoint_cone',False)

    def step(self,*,agent_ids,position,velocity,measurement_time,decision_time,
             own_velocity,own_velocity_time,coordinate_frame='world_z_up_m'):
        c=self.cfg
        if tuple(agent_ids)!=self.ids:raise ValueError('missing/reordered/changed tracked IDs')
        if coordinate_frame!='world_z_up_m':raise ValueError('explicit coordinate conversion is required')
        p=np.asarray(position,float);v=np.asarray(velocity,float);local=np.asarray(own_velocity,float)
        if p.shape!=(len(self.ids),3) or v.shape!=p.shape or local.shape!=(3,):
            raise ValueError('invalid state shape')
        if not all(np.isfinite(x).all() for x in [p,v,local]):raise ValueError('nonfinite observation')
        if not all(np.isfinite(x) for x in [measurement_time,decision_time,own_velocity_time]):raise ValueError('invalid timestamps')
        age=decision_time-measurement_time;local_age=decision_time-own_velocity_time
        if age < -1e-12 or age>self.max_age+1e-12:raise ValueError('future/stale observation')
        if local_age < -1e-12 or local_age>self.max_local_age+1e-12:raise ValueError('future/stale own velocity')
        if self.time is not None and abs(decision_time-self.time-c.dt)>1e-6:
            raise ValueError('control cadence differs from the validated nominal dt')
        if np.linalg.norm(local)>c.v_max+1e-9:raise ValueError('own velocity is outside validated speed bound')
        pm,vm=self.observer.update(p,v,measurement_time=measurement_time,decision_time=decision_time)
        actual_proxy=vm.copy();actual_proxy[self.ego]=local
        limit=min(c.a_max,c.nominal_a_max) if c.nominal_a_max is not None else c.a_max
        acc,normal=curve_acceleration(pm,vm,self.cloud,self.prm,limit,endpoint_cone=self.endpoint)
        gain=c.dt if not c.actuator_tau else c.dt/(-np.expm1(-c.dt/c.actuator_tau))
        preferred=clip_norm(actual_proxy+gain*acc,c.v_max)
        # Experimental shaping belongs BEFORE ORCA/governor. Never alter an
        # already checked action afterwards. Defaults preserve frozen replay.
        if c.tangential_damping>0:
            tangent=self.cloud.meta['tangents'][self.cloud.tree.query(pm)[1]]
            preferred-=c.tangential_damping*np.sum(preferred*tangent,axis=1)[:,None]*tangent
        if c.command_filter_tau>0:
            if self.command_filtered is None:self.command_filtered=preferred.copy()
            else:
                alpha=-np.expm1(-c.dt/c.command_filter_tau)
                self.command_filtered+=alpha*(preferred-self.command_filtered)
            preferred=self.command_filtered.copy()
        selected,residual=orca_velocity(pm,vm,preferred,dt=c.dt,r_vis=self.prm.r_vis_abs,d_safe=c.d_safe,
            v_max=c.v_max,horizon=c.horizon,margin=c.margin,nonfinite_policy=c.orca_nonfinite_policy)
        command,info=self.governor.command(pm,vm,actual_proxy,selected,dt=c.dt,a_max=c.a_max,tau=c.actuator_tau,
            v_max=c.v_max,r_vis=self.prm.r_vis_abs,d_safe=c.d_safe,margin=c.margin,surface_normal=normal,tangent_priority=False)
        result=command[self.ego].copy()
        if not np.isfinite(result).all():raise RuntimeError('invalid ego output')
        self.time=float(decision_time)
        return result,dict(ego_id=self.ids[self.ego],measurement_age_s=float(age),
            orca_nonfinite=bool(not np.isfinite(residual[self.ego])),speed_limit=info['speed_limit'],
            safety_certificate_applicable=False,scope='experimental ego command only; no hardware send or fallback action')
