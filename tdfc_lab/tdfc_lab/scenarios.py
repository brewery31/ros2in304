"""Shared local curve/stroke controller; no formation target assignment."""
import numpy as np
from functools import lru_cache
from .research.sim_protocol import clip_norm
from .research.orca_adapter import orca_velocity
from .research.directional_governor import DirectionalGovernor

SCENES = ('ring', 'helix', 'R', 'A', 'L')
ROBOT_NAMES = ('cf5', 'cf6', 'cf7', 'cf8', 'cf9', 'cfA')


def validate_scene(shape, agents):
    if shape in ('ring', 'helix', 'tilted_ring') and agents in (4, 6):
        return
    if shape in ('R', 'A', 'L') and agents == 5:
        return
    raise ValueError('ring/helix: --agents 4 or 6; R/A/L: --agents 5')


def hover_grid(agents, seed=None, height=1.2):
    if agents not in (4, 5, 6):
        raise ValueError('supported populations: 4/5/6')
    cols = 2 if agents == 4 else 3
    xy = np.array([(i % cols, i // cols) for i in range(agents)], float)*.9
    xy -= xy.mean(0)
    if seed is not None:
        xy += np.random.default_rng(seed).uniform(-.01, .01, xy.shape)
    return np.column_stack([xy, np.full(agents, height)])


def letter_targets(shape):
    """Display-only graph vertices, NOT one goal per robot."""
    from .research.stroke_geometry import stroke_graph
    return stroke_graph(shape)[0]


def letter_edges(shape):
    from .research.stroke_geometry import stroke_graph
    return stroke_graph(shape)[1]


class ReturnToHomeController:
    """Mission only: each ID returns to its own recorded home; no matching."""
    def __init__(self, homes):
        self.goals = np.array(homes,float)
        if self.goals.ndim!=2 or self.goals.shape[1]!=3 or not np.isfinite(self.goals).all():
            raise ValueError('finite per-ID homes required')
        self.governor = DirectionalGovernor()

    def command(self,p,v,t):
        preferred = clip_norm(1.2*(self.goals-p),.3)
        selected,_ = orca_velocity(p,v,preferred,dt=.02,r_vis=2.4,d_safe=.5,v_max=.3,
                                   horizon=.4,margin=.08,nonfinite_policy='ego_brake')
        command,_ = self.governor.command(p,v,v,selected,dt=.02,a_max=1.,tau=.1,
                           v_max=.3,r_vis=2.4,d_safe=.5,margin=.08,tangent_priority=False)
        return command


class SceneController:
    def __init__(self,shape,agents):
        validate_scene(shape,agents)
        self.shape,self.agents=shape,agents
        self.names=ROBOT_NAMES[:agents]
        from .adapter import CandidateController,load_task
        self.task=load_task(shape,agents=agents)
        self.curve=CandidateController(self.task,self.names)

    def command(self,p,v,t):
        frame=dict(position=p,velocity=v,measurement_time=t,own_velocities=v,
                   own_times=[t]*self.agents)
        return self.curve.step(frame,t)['velocity']

    @property
    def method(self):return 'curve_ours'


@lru_cache(maxsize=8)
def _evaluation_curve(shape, agents):
    from .research.curve_tasks import curve_geometry
    name=f'curve_letter_{shape}_n{agents}' if shape in ('R','A','L') else f'curve_demo_{shape}_n{agents}'
    return curve_geometry(name)[1]


def scene_metrics(shape, p, controller):
    from scipy.spatial import cKDTree
    ev = _evaluation_curve(shape,len(p))
    fit = ev.tree.query(p)[0]
    coverage = cKDTree(p).query(ev.points)[0]
    weights=ev.meta.get('quadrature_weights',np.ones(ev.m))
    return dict(mean_curve_error_m=float(fit.mean()), max_curve_error_m=float(fit.max()),
                coverage_ratio_048m=float(np.average(coverage <= .48,weights=weights)),
                covering_radius_m=float(coverage.max()))
