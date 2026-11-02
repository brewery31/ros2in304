"""Common execution and measurement protocol for the 2026-10-29 study.

Historical batches and their runner are deliberately untouched. All dynamic
methods here share observations, physical clipping, timing and evaluation.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (PointCloud, make_sphere, make_torus, make_box, make_dome,
                       read_ply, sample_mesh)
from .energy import Params, composite_forces
from .baselines import BoidsTarget, cvt_surface, fp_surface
from .metrics import trajectory_step_min
from .orca_adapter import orca_velocity

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ('sphere', 'cylinder', 'cone', 'box', 'prism', 'torus', 'bunny')


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest()


def clip_norm(x, limit):
    return x * np.minimum(1., limit / np.maximum(np.linalg.norm(x, axis=1), 1e-15))[:, None]


def _unit(pc):
    return PointCloud.from_points(pc.points / np.sqrt(pc.area), area=1., name=pc.name)


def uniform_surface(shape, m, seed):
    """Area sampling; correct cap proportions and prism cap barycentrics.

    These fixes are isolated from historical geometry to preserve old batches.
    """
    rng = np.random.default_rng(seed)
    if shape == 'bunny':
        mesh = read_ply(ROOT / 'data/bun_zipper.ply')
        return _unit(sample_mesh(mesh['vertices'], mesh['faces'], m, rng))
    if shape in ('sphere', 'torus', 'box', 'dome'):
        # Independent pool seeds via rigid sample-index permutation for sphere;
        # random area sampling below avoids identical control/evaluation points.
        if shape == 'sphere':
            p = rng.normal(size=(m, 3)); p /= np.linalg.norm(p, axis=1)[:, None]
            return _unit(PointCloud.from_points(p, area=4*np.pi))
        if shape == 'torus':
            chunks=[]; need=m
            while need:
                u,v=rng.uniform(0,2*np.pi,(2,4*need))
                keep=rng.random(4*need)<(.6+.25*np.cos(v))/.85
                u,v=u[keep][:need],v[keep][:need]
                chunks.append(np.column_stack([(.6+.25*np.cos(v))*np.cos(u),
                    (.6+.25*np.cos(v))*np.sin(u),.25*np.sin(v)]));need-=len(u)
            return _unit(PointCloud.from_points(np.concatenate(chunks),area=4*np.pi**2*.6*.25))
        if shape == 'dome':
            z=rng.random(m); th=rng.uniform(0,2*np.pi,m); r=np.sqrt(1-z*z)
            return _unit(PointCloud.from_points(np.column_stack([r*np.cos(th),r*np.sin(th),z]),area=2*np.pi))
        f=rng.integers(6,size=m); p=rng.uniform(-.5,.5,(m,3))
        p[np.arange(m),f//2]=np.where(f%2,.5,-.5)
        return _unit(PointCloud.from_points(p,area=6.))
    r,h=.5,1.
    if shape in ('cylinder','cone'):
        lateral=2*np.pi*r*h if shape=='cylinder' else np.pi*r*np.sqrt(r*r+h*h)
        caps=2 if shape=='cylinder' else 1; area=lateral+caps*np.pi*r*r
        side=rng.random(m)<lateral/area
        th=rng.uniform(0,2*np.pi,m); u=rng.random(m)
        if shape=='cylinder':
            rad=np.where(side,r,r*np.sqrt(u))
            z=np.where(side,rng.uniform(-h/2,h/2,m),rng.choice([-h/2,h/2],m))
        else:
            rad=np.where(side,r*np.sqrt(u),r*np.sqrt(rng.random(m)))
            z=np.where(side,h/2-h*np.sqrt(u),-h/2)
        return _unit(PointCloud.from_points(np.column_stack([rad*np.cos(th),rad*np.sin(th),z]),area=area))
    if shape=='prism':
        n=6; angle=np.arange(n)*2*np.pi/n; vertices=r*np.column_stack([np.cos(angle),np.sin(angle)])
        side_area=n*2*r*np.sin(np.pi/n)*h; cap_area=.5*n*r*r*np.sin(2*np.pi/n)
        side=rng.random(m)<side_area/(side_area+2*cap_area)
        face=rng.integers(n,size=m); a=vertices[face]; b=vertices[(face+1)%n]
        u=rng.random(m); v=rng.random(m)
        xy=np.sqrt(u)[:,None]*((1-v)[:,None]*a+v[:,None]*b)
        xy[side]=(1-u[side,None])*a[side]+u[side,None]*b[side]
        z=np.where(side,rng.uniform(-h/2,h/2,m),rng.choice([-h/2,h/2],m))
        return _unit(PointCloud.from_points(np.column_stack([xy,z]),area=side_area+2*cap_area))
    raise ValueError(shape)


@lru_cache(maxsize=16)
def geometry(shape, n_scale, control_points=6000, eval_points=60000, a=.48):
    if shape.startswith('curve_'):
        from .curve_tasks import curve_geometry
        return curve_geometry(shape,control_points,eval_points)
    # Scale around a single fixed origin: never independently recenter priors.
    pool=uniform_surface(shape,max(60000,control_points),700)
    center=pool.centroid.copy()
    area=n_scale*np.sqrt(3)/2*a*a
    scale=np.sqrt(area)
    order=np.random.default_rng(701).permutation(pool.m)
    control=PointCloud.from_points((pool.points[order[:control_points]]-center)*scale,area=area,name=shape)
    ep=uniform_surface(shape,eval_points,1100)
    ev=PointCloud.from_points((ep.points-center)*scale,area=area,name=shape)
    return control,ev


def initial_state(n, cloud, kind, seed, d_safe=.3):
    """Safe array-valued initial conditions, paired across methods."""
    rng=np.random.default_rng(seed)
    spacing=1.4*d_safe; jitter=.025*d_safe
    width=max(int(np.ceil(n**(1/3)))+1, int(np.ceil(2.2*cloud.radius/spacing)))
    g=(np.arange(-width,width+1))*spacing
    x,y,z=np.meshgrid(g,g,g,indexing='ij'); sites=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    order=np.argsort(np.linalg.norm(sites,axis=1),kind='stable')
    center=cloud.centroid; radius=cloud.radius
    if kind=='surround':
        # Uniformly choose from safe lattice sites in the original enclosing ball.
        candidates=sites[np.linalg.norm(sites,axis=1)<=2.2*radius]
        if len(candidates)<n: candidates=sites[order[:max(n,2*n)]]
        p=candidates[rng.choice(len(candidates),n,replace=False)].copy()
    else:
        p=sites[order[:n]].copy()
        if kind=='one_side': p[:,0]+=radius-p[:,0].min()+.5
        elif kind=='cluster': p[:,0]+=radius-p[:,0].min()+.1
        elif kind=='multi_cluster':
            # Three separated cohorts, each internally on a safe lattice.
            for i,inds in enumerate(np.array_split(np.arange(n),3)):
                theta=i*2*np.pi/3
                local=sites[order[:len(inds)]]
                offset=(radius+np.linalg.norm(local,axis=1).max()+.8)*np.array([np.cos(theta),np.sin(theta),0])
                p[inds]=local+offset
        else: raise ValueError(kind)
    p+=rng.uniform(-jitter,jitter,p.shape)+center
    d=float(cKDTree(p).query(p,k=2)[0][:,1].min())
    if d<1.2*d_safe: raise ValueError('unsafe initialization')
    return p,np.zeros_like(p)


def distances(p, q):
    return cKDTree(p).query(q,k=1)[0]


def first_hold(t, mask, hold=1.):
    """Earliest start of a contiguous observed interval lasting hold seconds."""
    start=None
    for k,ok in enumerate(mask):
        if not ok: start=None
        elif start is None: start=k
        if start is not None and t[k]-t[start]>=hold-1e-9: return float(t[start])
    return None


def observed_settle(t,x,band=.05):
    ok=np.abs(x-x[-1])<=band*max(abs(x[-1]),1e-12)
    bad=np.flatnonzero(~ok)
    return float(t[bad[-1]+1]) if len(bad) else float(t[0])


def _neighbors(p,r,k):
    d,idx=cKDTree(p).query(p,k=min(len(p),k+1),distance_upper_bound=r)
    mask=np.isfinite(d)&(idx<len(p))&(idx!=np.arange(len(p))[:,None])
    return np.where(mask,idx,0),mask,d


def lloyd_velocity(p,v,cloud,prm,dt,k_samples=128,tangent_gain=4.,k_neighbors=32):
    """Finite-range sampled surface Lloyd, explicitly an approximation.

    Only target samples inside sensing range and observed neighbors compete.
    Normal approach uses the shared prior; local centroid direction is tangent.
    """
    idx,mask,_=_neighbors(p,prm.r_vis_abs,k_neighbors)
    ds,si=cloud.tree.query(p,k=min(k_samples,cloud.m))
    samples=cloud.points[si]
    candidates=np.concatenate([p[:,None],p[idx]],axis=1)
    cmask=np.concatenate([np.ones((len(p),1),bool),mask],axis=1)
    dd=((samples[:,:,None]-candidates[:,None])**2).sum(-1)
    dd=np.where(cmask[:,None],dd,np.inf)
    mine=(dd[:,:,0]<=dd.min(2))&(ds<=prm.r_vis_abs)
    count=mine.sum(1); centroid=(samples*mine[:,:,None]).sum(1)/np.maximum(count,1)[:,None]
    d,qi=cloud.tree.query(p); n=(p-cloud.points[qi])/np.maximum(d,1e-12)[:,None]
    tangent=centroid-p; tangent-=(tangent*n).sum(1)[:,None]*n; tangent[count==0]=0
    acc=-prm.k_s*d[:,None]*n+tangent_gain*tangent-prm.c*v
    return v+dt*acc


def sun_surface_velocity(p,v,cloud,prm,dt):
    """Sun2023-inspired surface adaptation, NOT native published algorithm.

    Source cosine kernel / fill+unoccupied exploration / repulsion+alignment.
    Grid grayscale entering is replaced by nearest-surface approach; target pose
    is common prior, so negotiation omitted. Gains are developer-set below.
    """
    idx,mask,nbd=_neighbors(p,prm.r_vis_abs,32)
    ds,si=cloud.tree.query(p,k=min(128,cloud.m)); qs=cloud.points[si]
    valid=ds<prm.r_vis_abs
    dd=np.linalg.norm(qs[:,:,None]-p[idx][:,None],axis=-1)
    dd=np.where(mask[:,None],dd,np.inf)
    free=dd.min(2)>prm.d_safe/2
    w=.5*(1+np.cos(np.pi*np.minimum(ds/prm.r_vis_abs,1)))*valid
    def mean(ww):
        return ((qs-p[:,None])*ww[:,:,None]).sum(1)/np.maximum(ww.sum(1),1e-12)[:,None]
    d,qi=cloud.tree.query(p); entering=cloud.points[qi]-p
    rep=(prm.d_safe/np.maximum(nbd,1e-12)-1).clip(0)
    unit=(p[:,None]-p[idx])/np.maximum(nbd,1e-12)[:,:,None]
    rep=np.where(mask,rep,0)
    avoid=(rep[:,:,None]*unit).sum(1)
    alignment=(np.where(mask[:,:,None],v[idx]-v[:,None],0)).sum(1)/np.maximum(mask.sum(1),1)[:,None]
    return 2*entering+mean(w)+3*mean(w*free)+5*avoid+alignment


@dataclass
class Settings:
    dt:float=.02
    t_end:float=60.
    v_max:float=1.
    a_max:float=3.
    a:float=.48
    d_safe:float=.30
    r_vis_factor:float=2.5
    position_sigma:float=0.
    velocity_sigma:float=0.
    delay_steps:int=0
    actuator_tau:float=0.
    horizon:float=.4
    margin:float=.02
    monitor_points:int=3000
    monitor_every:int=5
    quality_coverage:float=.90
    quality_fit:float=.10
    hold_seconds:float=1.
    stop_speed:float=.05
    safety:bool=True
    stopping_governor:bool=False
    execution_model:str="legacy"
    normal_priority:bool=False
    governor_geometry:str="norm"
    uncertain_tangent_speed:bool=False
    tangent_priority:bool=False
    nominal_a_max:float|None=None
    native_acceleration_adapter:bool=False
    orca_nonfinite_policy:str='raise'
    # Damp the preferred along-curve command BEFORE ORCA and the governor.
    # 0 keeps the frozen behaviour; >0 damps the undamped sliding mode that
    # ORCA kicks excite on the curve. Not yet flight-validated.
    tangential_damping:float=0.
    # First-order filter on the preferred command, BEFORE safety processing.
    # Preserves steady (DC) motion along
    # the curve -- the thing tangential damping destroys -- while attenuating
    # the ~1 Hz sliding limit cycle. Not yet flight-validated.
    command_filter_tau:float=0.


def execute(task, artifact_dir=None):
    """Return JSON-safe row. Scalar audit is run at EVERY executed step."""
    start=time.perf_counter(); cfg=Settings(**task.get('sim',{})); n=task['n_agents']
    if task.get('audit_sample_trace') and (artifact_dir is None or cfg.execution_model!='physical_zoh'):
        raise ValueError('audit sample trace requires physical execution and an artifact directory')
    exact_observation=cfg.position_sigma==0 and cfg.velocity_sigma==0 and cfg.delay_steps==0
    research=task.get('research_mode','')
    if research not in ('','observation_stress','uncertified_ablation'):raise ValueError('unknown research mode')
    if research and cfg.execution_model!='physical_zoh':raise ValueError('research exceptions require physical execution')
    if research=='observation_stress' and not (cfg.stopping_governor and cfg.governor_geometry=='directional'):
        raise ValueError('observation stress must keep the directional governor')
    if research=='uncertified_ablation' and (cfg.stopping_governor or not exact_observation):
        raise ValueError('uncertified ablation must disable the governor and use exact observations')
    admissible_observation=exact_observation or research=='observation_stress'
    observer=None
    if task.get('observation_processor'):
        if research!='observation_stress' or task.get('method')!='curve_ours':
            raise ValueError('observation processing is an explicit empirical curve research option')
        from .observation_filter import CausalCVFilter
        observer=CausalCVFilter(v_max=cfg.v_max,**task['observation_processor'])
    observation_errors=dict(raw_squared=0.,processed_squared=0.,raw_max=0.,processed_max=0.,samples=0)
    if (task.get('params',{}).get('tangent_all_pair') or task.get('params',{}).get('soft_pair_only')) and not (cfg.execution_model=='physical_zoh' and ((cfg.stopping_governor and admissible_observation) or research=='uncertified_ablation')):
        raise ValueError('tangent_all_pair requires the physical governor and exact instantaneous observations')
    if cfg.uncertain_tangent_speed and not (cfg.execution_model=='physical_zoh' and cfg.stopping_governor and task.get('params',{}).get('surface_geometry') and task.get('method','ours')=='ours'):
        raise ValueError('geometry-dependent tangent speed requires physical execution and independent safety governor')
    if cfg.nominal_a_max is not None and cfg.nominal_a_max<=0:raise ValueError('nominal acceleration limit must be positive')
    if cfg.native_acceleration_adapter and (cfg.execution_model!='physical_zoh' or task.get('method') not in ('orig_orca','local_lloyd')):
        raise ValueError('native acceleration adaptation supports physical original-STFC/local-Lloyd only')
    if cfg.orca_nonfinite_policy not in ('raise','ego_brake'):raise ValueError('unknown ORCA nonfinite policy')
    if cfg.orca_nonfinite_policy=='ego_brake' and not (cfg.execution_model=='physical_zoh' and ((cfg.stopping_governor and cfg.governor_geometry=='directional' and admissible_observation) or research=='uncertified_ablation')):
        raise ValueError('per-ego ORCA failure braking requires the exact-observation directional stopping certificate')
    if cfg.execution_model not in ('legacy','physical_zoh'):raise ValueError('unknown execution model')
    ns=task.get('n_scale',n); method=task.get('method','ours'); seed=task['seed']
    cloud,ev=geometry(task['shape'],ns,task.get('control_points',6000),task.get('eval_points',60000),cfg.a)
    init_cloud=cloud
    if task.get('initial_reference_points'):
        init_cloud=geometry(task['shape'],ns,task['initial_reference_points'],task.get('eval_points',60000),cfg.a)[0]
    if 'hover_positions' in task:
        if task.get('scenario'):raise ValueError('hover positions cannot be combined with a scenario')
        p=np.zeros((n,3));v=np.zeros_like(p)  # Validated explicit state is assigned below.
    else:
        p,v=initial_state(n,init_cloud,task.get('init','surround'),seed,cfg.d_safe)
        if task['shape'].startswith('curve_'):
            from .curve_tasks import curve_initial_state
            p,v=curve_initial_state(n,task.get('init','surround'),seed,cfg.d_safe)
    if task.get('scenario'):
        gap=float(task['initial_gap']);speed=float(task['initial_speed'])
        scenario=task['scenario']
        if scenario=='head_on':
            p=np.array([[-gap/2,0,0],[gap/2,0,0]])
            v=np.array([[speed,0,0],[-speed,0,0]])
        elif scenario=='crossing':
            p=np.array([[-gap/np.sqrt(2),0,0],[0,-gap/np.sqrt(2),0]])
            v=np.array([[speed,0,0],[0,speed,0]])
        elif scenario=='three_way':
            theta=np.arange(3)*2*np.pi/3;direction=np.column_stack([np.cos(theta),np.sin(theta),np.zeros(3)])
            p=gap/np.sqrt(3)*direction;v=-speed*direction
        else:raise ValueError(scenario)
        if len(p)!=n:raise ValueError('scenario population mismatch')
        rotation=np.linalg.qr(np.random.default_rng(seed).normal(size=(3,3)))[0]
        p=p@rotation+cloud.centroid;v=v@rotation
    if 'hover_positions' in task:
        if not (cfg.execution_model=='physical_zoh' and research=='observation_stress' and method=='curve_ours' and task['shape'].startswith('curve_demo_')) or task.get('scenario'):
            raise ValueError('explicit settled-hover positions require the empirical physical curve task')
        hover=np.asarray(task['hover_positions'],float)
        if hover.shape!=(n,3) or not np.isfinite(hover).all() or np.any(hover[:,2]<=0):
            raise ValueError('finite positive-height N by3 hover positions required')
        if np.ptp(hover[:,2])>1e-12:raise ValueError('this rehearsal uses a common-height settled hover')
        if cKDTree(hover).query(hover,k=2)[0][:,1].min()<cfg.d_safe+cfg.margin:
            raise ValueError('hover placement violates the physical threshold plus reserve')
        p=hover.copy();v=np.zeros_like(p)
    p0=p.copy(); v0=v.copy(); p_sha=array_sha(p0)
    initial_surface_distances=ev.tree.query(p0)[0]
    initial_neighbors=cKDTree(p0).query_ball_point(p0, cfg.r_vis_factor*cfg.a, return_length=True)-1
    prm=Params(a=cfg.a,mu=cfg.a/cfg.d_safe,r_vis=cfg.r_vis_factor,**task.get('params',{}))
    directional_governor=None
    if cfg.tangent_priority and not (cfg.governor_geometry=='directional' and task.get('method','ours')=='ours'):
        raise ValueError('tangent-priority selection requires the directional governor and surface nominal controller')
    if cfg.governor_geometry=='directional':
        if cfg.normal_priority:raise ValueError('directional governor does not implement normal-priority command selection')
        if not (cfg.execution_model=='physical_zoh' and cfg.stopping_governor and admissible_observation):
            raise ValueError('directional governor requires the physical model and exact synchronous observations')
        from .directional_governor import DirectionalGovernor
        directional_governor=DirectionalGovernor()
    elif cfg.governor_geometry!='norm':raise ValueError('unknown governor geometry')
    source='current_controller' 
    orig=None
    assigned_goals=None;assigned_preprocessing=0.
    if method=='assigned_fps':
        from scipy.optimize import linear_sum_assignment
        tick=time.perf_counter()
        goals=fp_surface(cloud,n,np.random.default_rng(seed))
        ii,jj=linear_sum_assignment(np.linalg.norm(p[:,None]-goals[None],axis=2)**2)
        assigned_goals=np.empty_like(goals);assigned_goals[ii]=goals[jj]
        assigned_preprocessing=time.perf_counter()-tick
        source='Centralized FPS + initial global Hungarian assignment + bounded PD tracking; reference pipeline, not distributed FPS'
    if method=='orig_orca':
        # Frozen previously tuned law; do not retune against test seeds.
        original_parameters=dict(alpha_c=.08,alpha_s=.1,beta_a=30.,beta_r=30.,gamma_ms=2.)
        original_parameters.update(task.get('controller',{}))
        orig=BoidsTarget(cloud,cfg.a,cfg.d_safe,max_speed=cfg.v_max,
            r_vis=prm.r_vis_abs,**original_parameters)
    if method=='sun_surface_adapted':source='Sun2023-inspired surface adaptation; not native reproduction'
    tree_controller=None;preprocessing_seconds=assigned_preprocessing
    if method=='yue_tree':
        from .tree_baseline import TreeShape
        tick=time.perf_counter();tree_controller=TreeShape(cloud.points,task.get('tree_depth',6),task.get('solid_sphere',False))
        preprocessing_seconds=time.perf_counter()-tick
        source='Independent Yue2026 tree-controller implementation on a voxelized shell; shared execution layer'
    monitor=ev.points[np.random.default_rng(222).choice(ev.m,min(cfg.monitor_points,ev.m),replace=False)]
    audit=np.empty(int(round(cfg.t_end/cfg.dt))+1); audit[0]=cKDTree(p).query(p,k=2)[0][:,1].min()
    audit_upper=audit.copy();ambiguous_steps=0;refined_audit_steps=0
    times=[]; ps=[]; vs=[]; curves=[]; history=[]; mean_age=[]
    timing=dict(nominal=0.,orca=0.,audit=0.,measurement=0.)
    orca_failure_steps=0;orca_failure_agents=0;orca_first_failure_time=None
    first_orca_failure=None
    saturation=0; violations=0; path=0.; effort=0.; orca_change=0.; clip_change=0.; tangent_speed_change=0.
    governor_outside=0;governor_infeasible=0;governor_interventions=0;governor_limit=cfg.v_max;governor_initial_outside=0
    nsteps=len(audit)-1
    native_enter_time=None
    # Full trace only for prespecified representative seeds and modest N.
    full=artifact_dir is not None and (seed==100 or task.get("full_trace",False)) and n<=200
    fullp=[p.copy()] if full else None; fullv=[v.copy()] if full else None
    fullcommands=[] if full else None
    probe_indices=set(np.linspace(0,nsteps-1,min(32,nsteps),dtype=int)) if task.get('audit_sample_trace') else set()
    probes={};worst_probe=None;worst_probe_lower=np.inf
    def measure(t):
        tick=time.perf_counter(); d=distances(p,monitor); fit=ev.tree.query(p)[0]
        times.append(t);ps.append(p.astype(np.float32));vs.append(v.astype(np.float32))
        curves.append([d.mean()/cfg.a,np.percentile(d,90)/cfg.a,d.max()/cfg.a,
            np.mean(d<=.6*cfg.a),np.percentile(fit,95)/cfg.a,np.percentile(np.linalg.norm(v,axis=1),95)])
        timing['measurement']+=time.perf_counter()-tick
    measure(0.)
    static=method in ('fps','cvt','random')
    if static:
        rng=np.random.default_rng(seed)
        if method=='fps':p=fp_surface(cloud,n,rng)
        elif method=='cvt':p=cvt_surface(cloud,n,iters=60,rng=rng)[0]
        else:p=cloud.points[rng.choice(cloud.m,n,replace=False)]
        audit=audit[:1];audit[0]=cKDTree(p).query(p,k=2)[0][:,1].min()
        audit_upper=audit.copy()
        times.clear();ps.clear();vs.clear();curves.clear();measure(0.)
    else:
        for k in range(nsteps):
            history.append((p.copy(),v.copy()))
            if len(history)>cfg.delay_steps+1:history.pop(0)
            hi=max(0,len(history)-1-cfg.delay_steps)
            pm,vm=history[hi];pm=pm.copy();vm=vm.copy()
            mean_age.append(min(k,cfg.delay_steps)*cfg.dt)
            # Paired exogenous noise: fixed seed and draws independent of method.
            rng=np.random.default_rng(np.random.SeedSequence([seed,9000,k]))
            if cfg.position_sigma:pm+=rng.normal(0,cfg.position_sigma,pm.shape)
            if cfg.velocity_sigma:vm+=rng.normal(0,cfg.velocity_sigma,vm.shape)
            if observer is not None or task.get('log_observation_error',False):
                # Truth is used ONLY for diagnostics after raw observations are formed.
                # These scalars never enter observer, nominal, ORCA or governor inputs.
                raw_error=np.linalg.norm(pm-p,axis=1)
                if observer is not None:
                    pm,vm=observer.update(pm,vm,measurement_time=max(0,k-cfg.delay_steps)*cfg.dt,decision_time=k*cfg.dt)
                processed_error=np.linalg.norm(pm-p,axis=1)
                observation_errors['raw_squared']+=float(raw_error@raw_error)
                observation_errors['processed_squared']+=float(processed_error@processed_error)
                observation_errors['raw_max']=max(observation_errors['raw_max'],float(raw_error.max()))
                observation_errors['processed_max']=max(observation_errors['processed_max'],float(processed_error.max()))
                observation_errors['samples']+=len(p)
            tick=time.perf_counter()
            normal_hint=None
            if method=='ours':
                nominal_limit=min(cfg.a_max,cfg.nominal_a_max) if cfg.nominal_a_max is not None else cfg.a_max
                components=task.get('force_components',{})
                if set(components)-{'use_surface','use_pair','use_entropy','use_alignment'}:raise ValueError('unknown force component')
                nominal=composite_forces(pm,vm,cloud,prm,k_max=task.get('nominal_neighbor_cap',32),a_max=nominal_limit,**(dict(use_alignment=True)|components))
                gain=cfg.dt
                if cfg.execution_model=='physical_zoh' and cfg.actuator_tau:
                    gain=cfg.dt/(-np.expm1(-cfg.dt/cfg.actuator_tau))
                pref=v+gain*nominal.total;normal_hint=nominal.n
            elif method in ('curve_ours','assigned_fps'):
                nominal_limit=min(cfg.a_max,cfg.nominal_a_max) if cfg.nominal_a_max is not None else cfg.a_max
                if method=='curve_ours':
                    from .curve_tasks import curve_acceleration
                    acc,normal_hint=curve_acceleration(pm,vm,cloud,prm,nominal_limit,**task.get('curve_controller',{}))
                    source='One-dimensional curve adaptation: nearest-curve attraction and tangent-projected local repulsion; not the surface controller'
                else:
                    acc=clip_norm(prm.k_s*(assigned_goals-pm)-2*np.sqrt(prm.k_s)*vm,nominal_limit)
                gain=cfg.dt if not cfg.actuator_tau else cfg.dt/(-np.expm1(-cfg.dt/cfg.actuator_tau))
                pref=v+gain*acc
            elif method=='orig_orca':pref=orig.step(pm,vm,cfg.dt)[1]
            elif method=='local_lloyd':pref=lloyd_velocity(pm,vm,cloud,prm,cfg.dt,**task.get('controller',{}))
            elif method=='sun_surface_adapted':pref=sun_surface_velocity(pm,vm,cloud,prm,cfg.dt)
            elif method=='yue_tree':pref=tree_controller.velocity(pm,prm.r_vis_abs,cfg.d_safe,**task.get('controller',{}))
            else:raise ValueError(method)
            if cfg.native_acceleration_adapter:
                from .baseline_execution import acceleration_setpoint
                nominal_limit=min(cfg.a_max,cfg.nominal_a_max) if cfg.nominal_a_max is not None else cfg.a_max
                pref=acceleration_setpoint(pref,vm,v,dt=cfg.dt,tau=cfg.actuator_tau,nominal_limit=nominal_limit)
            timing['nominal']+=time.perf_counter()-tick
            pref=clip_norm(pref,cfg.v_max)
            tick=time.perf_counter()
            if cfg.safety:
                selected,orca_residual=orca_velocity(pm,vm,pref,dt=cfg.dt,r_vis=prm.r_vis_abs,d_safe=cfg.d_safe,
                    v_max=cfg.v_max,horizon=cfg.horizon,margin=cfg.margin,nonfinite_policy=cfg.orca_nonfinite_policy)
                failed=~np.isfinite(orca_residual)
                if np.any(failed):
                    if cfg.orca_nonfinite_policy!='ego_brake':raise RuntimeError('unhandled ORCA nonfinite residual')
                    orca_failure_steps+=1;orca_failure_agents+=int(failed.sum())
                    if first_orca_failure is None:
                        orca_first_failure_time=k*cfg.dt
                        first_orca_failure=(pm.copy(),vm.copy(),pref.copy(),np.flatnonzero(failed))
            else:selected=pref
            timing['orca']+=time.perf_counter()-tick
            orca_change+=float(np.linalg.norm(selected-pref,axis=1).sum())
            if cfg.uncertain_tangent_speed:
                before_tangent_cap=selected.copy()
                from .local_surface import local_surface
                _,_,reliability=local_surface(cloud,pm,cfg.a,multiscale=prm.surface_multiscale)
                nn=normal_hint/np.maximum(np.linalg.norm(normal_hint,axis=1)[:,None],1e-30)
                normal_part=np.sum(selected*nn,axis=1)[:,None]*nn
                # Half of the existing 0.1a fitting tolerance times the ideal
                # normal frequency; no shape/seed-dependent gain selection.
                slow=.5*np.sqrt(prm.k_s)*(.1*cfg.a)
                local_limit=reliability*cfg.v_max+(1-reliability)*min(cfg.v_max,slow)
                selected=normal_part+clip_norm(selected-normal_part,local_limit)
                tangent_speed_change+=float(np.linalg.norm(selected-before_tangent_cap,axis=1).sum())
            if cfg.execution_model=='physical_zoh':
                from .physical_execution import track
                command=clip_norm(selected,cfg.v_max)
                if cfg.stopping_governor:
                    if directional_governor is not None:
                        command,governor_info=directional_governor.command(pm,vm,v,selected,dt=cfg.dt,a_max=cfg.a_max,
                            tau=cfg.actuator_tau,v_max=cfg.v_max,r_vis=prm.r_vis_abs,d_safe=cfg.d_safe,margin=cfg.margin,surface_normal=normal_hint,tangent_priority=cfg.tangent_priority)
                    else:
                        from .physical_governor import physical_command
                        command,governor_info=physical_command(pm,vm,v,selected,dt=cfg.dt,a_max=cfg.a_max,
                            tau=cfg.actuator_tau,v_max=cfg.v_max,r_vis=prm.r_vis_abs,d_safe=cfg.d_safe,
                            margin=cfg.margin,normal=normal_hint,priority=cfg.normal_priority)
                pn,vn,_=track(p,v,command,cfg.dt,a_max=cfg.a_max,tau=cfg.actuator_tau)
            else:
                command=v+clip_norm(clip_norm(selected,cfg.v_max)-v,cfg.a_max*cfg.dt)
                if cfg.stopping_governor:
                    from .stopping_governor import stopping_command
                    alpha=1.-np.exp(-cfg.dt/cfg.actuator_tau) if cfg.actuator_tau else 1.
                    command,governor_info=stopping_command(pm,vm,v,selected,dt=cfg.dt,a_max=cfg.a_max,
                        v_max=cfg.v_max,r_vis=prm.r_vis_abs,d_safe=cfg.d_safe,alpha=alpha,margin=cfg.margin)
                # Lag is downstream of the final command, followed by actual bounds.
                alpha=1.-np.exp(-cfg.dt/cfg.actuator_tau) if cfg.actuator_tau else 1.
                vn=v+alpha*(command-v)
            clip_change+=float(np.linalg.norm(command-selected,axis=1).sum())
            if cfg.stopping_governor:
                if k==0:governor_initial_outside=governor_info['outside_agents']
                governor_outside+=governor_info['outside_agents'];governor_infeasible+=governor_info['infeasible_agents']
                governor_interventions+=governor_info['intervened_agents'];governor_limit=governor_info['speed_limit']
            saturation+=int(np.sum(np.linalg.norm(vn-v,axis=1)>=.999*cfg.a_max*cfg.dt))
            violations+=int(np.sum((np.linalg.norm(vn,axis=1)>cfg.v_max+1e-9)|
                (np.linalg.norm(vn-v,axis=1)>cfg.a_max*cfg.dt+1e-9)))
            if cfg.execution_model=='legacy':pn=p+cfg.dt*vn
            if not np.isfinite(pn).all():raise FloatingPointError('nonfinite executed state')
            tick=time.perf_counter()
            if cfg.execution_model=='physical_zoh':
                from .physical_execution import audit_interval
                audit[k+1],audit_upper[k+1],parts=audit_interval(p,v,command,cfg.dt,a_max=cfg.a_max,tau=cfg.actuator_tau,d_safe=cfg.d_safe)
                ambiguous_steps+=int(audit[k+1]<cfg.d_safe<=audit_upper[k+1]);refined_audit_steps+=int(parts>1)
            else:
                audit[k+1]=trajectory_step_min(p,pn);audit_upper[k+1]=audit[k+1]
            timing['audit']+=time.perf_counter()-tick
            if probe_indices:
                if k in probe_indices or audit[k+1]<worst_probe_lower:
                    probe=(p.copy(),v.copy(),command.copy(),pn.copy(),vn.copy())
                    if k in probe_indices:probes[k+1]=probe
                    if audit[k+1]<worst_probe_lower:worst_probe=(k+1,probe);worst_probe_lower=audit[k+1]
            path+=float(np.linalg.norm(pn-p,axis=1).sum())/n
            effort+=float(np.mean(((vn-v)/cfg.dt)**2))*cfg.dt
            p,v=pn,vn
            if tree_controller is not None and native_enter_time is None and tree_controller.contains(p).all():
                native_enter_time=(k+1)*cfg.dt
            if full:fullp.append(p.copy());fullv.append(v.copy());fullcommands.append(command.copy())
            if (k+1)%cfg.monitor_every==0 or k+1==nsteps:measure((k+1)*cfg.dt)
    d=distances(p,ev.points);fit=ev.tree.query(p)[0];nn=cKDTree(p).query(p,k=2)[0][:,1]
    t=np.array(times);c=np.array(curves)
    tq=None if static else first_hold(t,(c[:,3]>=cfg.quality_coverage)&(c[:,4]<=cfg.quality_fit),cfg.hold_seconds)
    ts=None if static else first_hold(t,c[:,5]<=cfg.stop_speed,cfg.hold_seconds)
    row=dict(task_id=task['task_id'],suite=task['suite'],shape=task['shape'],n_agents=n,n_scale=ns,
        method=method,seed=seed,init=task.get('init','surround'),control_points=cloud.m,
        scenario=task.get('scenario','default'),
        eval_points=ev.m,initial_sha256=p_sha,control_sha256=array_sha(cloud.points),eval_sha256=array_sha(ev.points),
        e_mean_norm=float(d.mean()/cfg.a),r90_norm=float(np.percentile(d,90)/cfg.a),r_cov_norm=float(d.max()/cfg.a),
        c06=float(np.mean(d<=.6*cfg.a)),c08=float(np.mean(d<=.8*cfg.a)),
        fit_mean_norm=float(fit.mean()/cfg.a),fit_p95_norm=float(np.percentile(fit,95)/cfg.a),
        nn_cv=float(nn.std()/max(nn.mean(),1e-12)),d_min_ratio=float(audit.min()/cfg.d_safe),
        target_diameter=cloud.diameter,
        initial_surface_mean_norm=float(initial_surface_distances.mean()/cfg.a),
        initial_surface_p95_norm=float(np.percentile(initial_surface_distances,95)/cfg.a),
        initial_neighbor_mean=float(initial_neighbors.mean()),initial_neighbor_max=int(initial_neighbors.max()),
        path_measure='endpoint_chord_sum_lower' if cfg.execution_model=='physical_zoh' else 'linear_exact',
        audit_model='physical_zoh_certified' if cfg.execution_model=='physical_zoh' else 'linear_exact',
        d_min_upper_ratio=float(audit_upper.min()/cfg.d_safe),audit_ambiguous_steps=ambiguous_steps,audit_refined_steps=refined_audit_steps,
        collision_confirmed=bool(audit_upper.min()<cfg.d_safe),
        collision=bool(audit.min()<cfg.d_safe),collision_steps=int(np.sum(audit[1:]<cfg.d_safe)),
        collision_step_exposure=float(np.sum(audit[1:]<cfg.d_safe)*cfg.dt),
        t_quality=tq,quality_reached=tq is not None,t_stop=ts,
        t_coverage90=None if static else first_hold(t,c[:,3]>=.90,cfg.hold_seconds),
        t_coverage95=None if static else first_hold(t,c[:,3]>=.95,cfg.hold_seconds),
        t_fit=None if static else first_hold(t,c[:,4]<=cfg.quality_fit,cfg.hold_seconds),
        t_settle=None if static else observed_settle(t,c[:,1]),path_mean=path,acceleration_effort=effort,
        execution_violations=violations,saturation_agent_steps=saturation,
        orca_nonfinite_steps=orca_failure_steps,orca_nonfinite_agent_steps=orca_failure_agents,orca_first_nonfinite_time=orca_first_failure_time,
        orca_mean_correction=orca_change/max(n*nsteps,1),tangent_speed_mean_correction=tangent_speed_change/max(n*nsteps,1),clip_mean_correction=clip_change/max(n*nsteps,1),
        governor_initial_outside_agents=governor_initial_outside,governor_outside_agent_steps=governor_outside,governor_infeasible_agent_steps=governor_infeasible,
        governor_interventions=governor_interventions,governor_speed_limit=governor_limit,
        governor_proof_observation_scope=(exact_observation and observer is None),
        wall_seconds=time.perf_counter()-start,timing=timing,source=source,
        settings=asdict(cfg),task=task,status='ok',trajectory_stride=1 if full else cfg.monitor_every,
        research_mode=research,certificate_applicable=bool(cfg.stopping_governor and exact_observation and not research and observer is None),
        observation_processor=task.get('observation_processor'),
        observation_error_raw_rmse_m=float(np.sqrt(observation_errors['raw_squared']/observation_errors['samples'])) if observation_errors['samples'] else None,
        observation_error_processed_rmse_m=float(np.sqrt(observation_errors['processed_squared']/observation_errors['samples'])) if observation_errors['samples'] else None,
        observation_error_raw_max_m=observation_errors['raw_max'] if observation_errors['samples'] else None,
        observation_error_processed_max_m=observation_errors['processed_max'] if observation_errors['samples'] else None,
        observation_repeated_frames_ignored=observer.ignored_frames if observer is not None else 0,
        target_dimension=1 if task['shape'].startswith('curve_') else 2,
        preprocessing_seconds=preprocessing_seconds,
        voxel_quantization_bound=tree_controller.quantization_bound if tree_controller else None,
        native_entering_rate=float(tree_controller.contains(p).mean()) if tree_controller else None,
        native_entering_time=native_enter_time,
        monitor_point_sha256=array_sha(monitor),mean_observation_age=float(np.mean(mean_age)) if mean_age else 0.,
        coverage_profile={str(round(r,2)):float(np.mean(d<=r*cfg.a)) for r in np.linspace(.3,1.2,19)})
    if artifact_dir is not None:
        folder=Path(artifact_dir);folder.mkdir(parents=True,exist_ok=True)
        target=folder/(task['task_id']+'.npz');temp=folder/(task['task_id']+'.tmp.npz')
        storage_stride=int(task.get('position_record_every',1))
        if storage_stride<1:raise ValueError('position_record_every must be positive')
        record=np.unique(np.r_[np.arange(0,len(t),storage_stride),len(t)-1])
        row['position_storage_every_monitor_samples']=storage_stride
        row['trajectory_stride']=1 if full else cfg.monitor_every*storage_stride
        if worst_probe is not None:probes[worst_probe[0]]=worst_probe[1]
        probe_steps=sorted(probes);probe_values=[probes[k] for k in probe_steps]
        row['audit_probe_count']=len(probe_steps)
        np.savez_compressed(temp,time=t,position_time=t[record],pos=np.array(ps)[record],vel=np.array(vs)[record],curves=c,
            audit_min=audit,audit_upper=audit_upper,p0=p0,v0=v0,final_p=p,final_v=v,
            orca_failure_p=first_orca_failure[0] if first_orca_failure is not None else np.empty((0,3)),
            orca_failure_v=first_orca_failure[1] if first_orca_failure is not None else np.empty((0,3)),
            orca_failure_preferred=first_orca_failure[2] if first_orca_failure is not None else np.empty((0,3)),
            orca_failure_agents=first_orca_failure[3] if first_orca_failure is not None else np.empty(0,dtype=int),
            full_pos=np.array(fullp) if full else np.empty(0),full_vel=np.array(fullv) if full else np.empty(0),full_command=np.array(fullcommands) if full else np.empty(0),
            audit_probe_steps=np.array(probe_steps,dtype=int),
            audit_probe_pos=np.array([x[0] for x in probe_values]),audit_probe_vel=np.array([x[1] for x in probe_values]),
            audit_probe_command=np.array([x[2] for x in probe_values]),audit_probe_end_pos=np.array([x[3] for x in probe_values]),audit_probe_end_vel=np.array([x[4] for x in probe_values]))
        temp.replace(target)
        row['artifact']=str(target.name)
    return row
