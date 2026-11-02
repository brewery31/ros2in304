"""Explicit experiment definitions: curve formation vs assigned five-point letters."""
import numpy as np
from functools import lru_cache
from scipy.optimize import linear_sum_assignment
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
    # Upright X-Z plane, viewed from -Y. Five LANDMARKS, not full stroke coverage.
    points = {
        'R': [(-.7,-.7),(-.7,.7),(.7,.7),(-.7,0.),(.7,-.7)],
        'A': [(-.7,-.7),(0.,.7),(.7,-.7),(-.35,0.),(.35,0.)],
        'L': [(-.7,.7),(-.7,0.),(-.7,-.7),(0.,-.7),(.7,-.7)],
    }
    p = np.array(points[shape], float)
    return np.column_stack([p[:,0], np.zeros(5), 1.3+p[:,1]])


def letter_edges(shape):
    return {'R': [(0,1),(1,2),(2,3),(3,4)],
            'A': [(0,1),(1,2),(3,4)],
            'L': [(0,1),(1,2),(2,3),(3,4)]}[shape]


class LandmarkController:
    """Fixed goal tracking plus ORCA/governor, with one initial assignment.

    This is a separate demonstration controller, not continuous curve coverage.
    Assignment uses centralized initial positions and is explicitly reported.
    """
    def __init__(self, goals, *, assign=True):
        self.goals = np.array(goals, float)
        if self.goals.ndim != 2 or self.goals.shape[1] != 3 or not np.isfinite(self.goals).all():
            raise ValueError('finite N x 3 goals required')
        self.assign = assign
        self.assigned = None
        self.governor = DirectionalGovernor()

    def command(self, p, v, t):
        if self.assigned is None:
            if self.assign:
                rows, cols = linear_sum_assignment(np.linalg.norm(p[:,None]-self.goals[None],axis=2))
                self.assigned = self.goals[cols[np.argsort(rows)]].copy()
            else:
                self.assigned = self.goals.copy()
        preferred = clip_norm(1.2*(self.assigned-p), .3)
        selected, _ = orca_velocity(p,v,preferred,dt=.02,r_vis=2.4,d_safe=.5,v_max=.3,
                                   horizon=.4,margin=.08,nonfinite_policy='ego_brake')
        command, _ = self.governor.command(p,v,v,selected,dt=.02,a_max=1.,tau=.1,
                           v_max=.3,r_vis=2.4,d_safe=.5,margin=.08,tangent_priority=False)
        return command


class SceneController:
    def __init__(self, shape, agents):
        validate_scene(shape, agents)
        self.shape, self.agents = shape, agents
        self.names = ROBOT_NAMES[:agents]
        self.landmark = None
        self.task = None
        if shape in ('R', 'A', 'L'):
            self.landmark = LandmarkController(letter_targets(shape))
        else:
            from .adapter import CandidateController, load_task
            self.task = load_task(shape, agents=agents)
            self.curve = CandidateController(self.task, self.names)

    def command(self, p, v, t):
        if self.landmark is not None:
            return self.landmark.command(p,v,t)
        frame = dict(position=p,velocity=v,measurement_time=t,own_velocities=v,
                     own_times=[t]*self.agents)
        return self.curve.step(frame,t)['velocity']

    @property
    def method(self):
        return 'assigned_five_landmarks' if self.landmark is not None else 'curve_ours'


@lru_cache(maxsize=8)
def _evaluation_curve(shape, agents):
    from .research.curve_tasks import curve_geometry
    return curve_geometry(f'curve_demo_{shape}_n{agents}')[1]


def scene_metrics(shape, p, controller):
    if shape in ('R','A','L'):
        error = np.linalg.norm(p-controller.landmark.assigned,axis=1)
        return dict(max_landmark_error_m=float(error.max()), rms_landmark_error_m=float(np.sqrt((error**2).mean())))
    from scipy.spatial import cKDTree
    ev = _evaluation_curve(shape,len(p))
    fit = ev.tree.query(p)[0]
    coverage = cKDTree(p).query(ev.points)[0]
    return dict(mean_curve_error_m=float(fit.mean()), max_curve_error_m=float(fit.max()),
                coverage_ratio_048m=float((coverage <= .48).mean()),
                covering_radius_m=float(coverage.max()))
