"""Simulation mission and explicit velocity-to-FullState reference conversion.

Not a flight driver. The reference model does not certify rigid-body tracking.
"""
import numpy as np
from .scenarios import SceneController, LandmarkController, hover_grid, scene_metrics
from .research.physical_execution import track
from .research.sim_protocol import clip_norm


def polynomial_reference(start, end, elapsed, duration=10.):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('positive duration required')
    u = np.clip(elapsed/duration,0.,1.)
    f = 35*u**4-84*u**5+70*u**6-20*u**7
    df = (140*u**3-420*u**4+420*u**5-140*u**6)/duration
    ddf = (420*u**2-1680*u**3+2100*u**4-840*u**5)/duration**2
    delta = np.asarray(end)-np.asarray(start)
    return dict(position=np.asarray(start)+f*delta,velocity=df*delta,acceleration=ddf*delta)


def full_state_reference(p,v,command,dt=.02):
    # A 0.27m/s reference reserve reduces nominal tracking overshoot against the
    # controller's 0.3m/s observation guard. Actual rigid-body speed is measured.
    u = clip_norm(command,.27)
    pn,vn,_ = track(p,v,u,dt,a_max=1.,tau=.1)
    return dict(position=pn,velocity=vn,acceleration=(vn-v)/dt)


class SimulationMission:
    def __init__(self, shape, agents, *, seed=0, formation_seconds=60.):
        if not np.isfinite(formation_seconds) or formation_seconds <= 0:
            raise ValueError('positive formation duration required')
        self.controller = SceneController(shape,agents)
        self.shape,self.agents = shape,agents
        self.home = hover_grid(agents,seed)
        self.ground = self.home.copy();self.ground[:,2] = 0.
        self.phase = 'takeoff';self.phase_start = 0.
        self.hold_start = None;self.last_time = None
        self.duration = formation_seconds
        self.formation_summary = None
        self.formation_hold_start = None
        self.return_controller = LandmarkController(self.home,assign=False)

    def _transition(self, phase, now):
        self.phase,self.phase_start,self.hold_start = phase,now,None

    def _held(self, good, now, duration=1.):
        if not good:self.hold_start = None
        elif self.hold_start is None:self.hold_start = now
        return self.hold_start is not None and now-self.hold_start >= duration-1e-9

    def step(self, p,v,now):
        if not np.isfinite(np.r_[p.ravel(),v.ravel(),now]).all():
            raise ValueError('nonfinite simulated state')
        if self.last_time is not None and abs(now-self.last_time-.02)>1e-6:
            raise ValueError('mission requires 20ms simulated decisions')
        self.last_time=now
        elapsed = now-self.phase_start
        if self.phase == 'takeoff':
            ref = polynomial_reference(self.ground,self.home,elapsed)
            if elapsed >= 10.:
                good = np.max(np.linalg.norm(p-self.home,axis=1)) < .05 and np.linalg.norm(v,axis=1).max() < .07
                if self._held(good,now):self._transition('formation',now)
                elif elapsed > 20.:raise RuntimeError('takeoff/settled-hover timeout')
            return ref
        if self.phase == 'formation':
            command = self.controller.command(p,v,now)
            if round(elapsed/.02)%5 == 0 or elapsed >= self.duration:
                metric=scene_metrics(self.shape,p,self.controller)
                speed=float(np.linalg.norm(v,axis=1).max())
                good=(metric.get('max_landmark_error_m',metric.get('max_curve_error_m',1.))<=.05
                      and metric.get('coverage_ratio_048m',1.)>=.9 and speed<=.05)
                if not good:self.formation_hold_start=None
                elif self.formation_hold_start is None:self.formation_hold_start=now
            if elapsed >= self.duration:
                self.formation_summary = dict(**metric,max_actual_speed_mps=speed,
                    terminal_held_success=self.formation_hold_start is not None and now-self.formation_hold_start>=1.-1e-9)
                self._transition('return',now)
            return full_state_reference(p,v,command)
        if self.phase == 'return':
            command = self.return_controller.command(p,v,now)
            good = np.linalg.norm(p-self.home,axis=1).max()<.04 and np.linalg.norm(v,axis=1).max()<.04
            if self._held(good,now):
                self.landing_start = p.copy()
                self._transition('land',now)
            elif elapsed > 40.:raise RuntimeError('return-to-separated-columns timeout')
            return full_state_reference(p,v,command)
        if self.phase == 'land':
            ref = polynomial_reference(self.landing_start,self.ground,elapsed)
            if elapsed >= 10. and np.max(p[:,2])<.1:
                self._transition('done',now)
            elif elapsed > 20.:raise RuntimeError('landing timeout')
            return ref
        return dict(position=p.copy(),velocity=np.zeros_like(v),acceleration=np.zeros_like(v))
