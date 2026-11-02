"""ROS-independent alignment and candidate generation. Never sends to hardware."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .research.curve_ego_interface import CurveEgoInterface


def load_task(shape, config=None, agents=4):
    path = Path(config) if config else Path(__file__).parents[1] / 'config/tasks.json'
    if not path.exists():
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory('tdfc_lab')) / 'config/tasks.json'
    import copy
    letter=shape in ('R','A','L')
    if (letter and agents!=5) or (not letter and agents not in (4,6)):
        raise ValueError('letters require five agents; ring/helix require four or six')
    task = copy.deepcopy(json.loads(path.read_text())['ring' if letter else shape])
    if letter or agents != 4:
        from .scenarios import hover_grid
        task['shape'] = f'curve_letter_{shape}_n{agents}' if letter else f'curve_demo_{shape}_n{agents}'
        task['n_agents'] = agents
        task['hover_positions'] = hover_grid(agents).tolist()
        task['catalog_kind'] = 'stroke_'+shape if letter else shape
    return task


class StateBuffer:
    """Align asynchronous logs at a common time by interpolation, never future data.

    Input stamps are ROS receive-time stamps until the server is instrumented.
    Reported age therefore excludes unknown sensor/radio latency.
    """
    def __init__(self, names, max_age=.1, local_max_age=.04):
        self.names = tuple(names)
        if len(self.names) not in (4, 5, 6) or len(set(self.names)) != len(self.names):
            raise ValueError('four, five or six unique configured robot names required')
        if not all(np.isfinite(a) and a >= 0 for a in (max_age, local_max_age)):
            raise ValueError('invalid age limits')
        self.max_age, self.local_max_age = max_age, local_max_age
        self.history = {name: deque(maxlen=64) for name in self.names}

    def add(self, name, stamp, position, velocity):
        if name not in self.history:
            raise ValueError('unknown robot name')
        p, v = np.asarray(position, float), np.asarray(velocity, float)
        if p.shape != (3,) or v.shape != (3,) or not np.isfinite(np.r_[stamp, p, v]).all():
            raise ValueError('invalid odometry')
        h = self.history[name]
        if h and stamp <= h[-1][0]:
            raise ValueError('duplicate or backwards odometry timestamp')
        h.append((float(stamp), p.copy(), v.copy()))

    def aligned(self, decision_time):
        if not np.isfinite(decision_time):
            raise ValueError('nonfinite decision time')
        eligible = [[s for s in self.history[n] if s[0] <= decision_time] for n in self.names]
        if any(not h for h in eligible):
            raise ValueError('waiting for all configured nonfuture states')
        latest = [h[-1] for h in eligible]
        common = min(s[0] for s in latest)
        if decision_time - common > self.max_age:
            raise ValueError('stale aligned observation')
        if any(decision_time - s[0] > self.local_max_age for s in latest):
            raise ValueError('stale local velocity')
        positions, velocities = [], []
        for h in eligible:
            if h[0][0] > common:
                raise ValueError('insufficient history for interpolation')
            before = max((s for s in h if s[0] <= common), key=lambda s: s[0])
            after = min((s for s in h if s[0] >= common), key=lambda s: s[0])
            weight = 0. if after[0] == before[0] else (common-before[0])/(after[0]-before[0])
            positions.append(before[1] + weight*(after[1]-before[1]))
            velocities.append(before[2] + weight*(after[2]-before[2]))
        return dict(position=np.array(positions), velocity=np.array(velocities), measurement_time=common,
                    own_velocities=np.array([s[2] for s in latest]), own_times=[s[0] for s in latest])


class CandidateController:
    def __init__(self, task, names, local_max_age=.04):
        self.names = tuple(names)
        if len(self.names) not in (4,5,6) or len(set(self.names)) != len(self.names) or task['n_agents'] != len(self.names):
            raise ValueError('four/five/six unique robot names matching the curve task required')
        self.dt = .02
        self.controllers = [CurveEgoInterface(task, ego_id=n, agent_ids=self.names,
                            max_local_velocity_age=local_max_age) for n in self.names]
        self.latched = False

    def step(self, frame, decision_time):
        if self.latched:
            raise RuntimeError('candidate controller fault latched; recreate after inspection')
        try:
            outputs, info = [], []
            for i, c in enumerate(self.controllers):
                u, diagnostic = c.step(agent_ids=self.names, position=frame['position'],
                    velocity=frame['velocity'], measurement_time=frame['measurement_time'],
                    decision_time=decision_time, own_velocity=frame['own_velocities'][i],
                    own_velocity_time=frame['own_times'][i])
                outputs.append(u); info.append(diagnostic)
            # One-step full-state preview only. Reanchored to predicted measured position;
            # this conversion changes the plant interface and is NOT flight-validated.
            age = decision_time-frame['measurement_time']
            position = frame['position'] + age*frame['velocity']
            velocity = np.array(outputs)
            return dict(position=position + self.dt*velocity, velocity=velocity,
                        acceleration=(velocity-frame['own_velocities'])/self.dt, diagnostics=info)
        except Exception:
            self.latched = True
            raise
