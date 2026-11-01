"""Portable replay CLI; has no ROS or aircraft imports."""
import argparse
import json
import numpy as np
from .adapter import CandidateController, load_task
from .research.physical_execution import track, audit_interval


def smoke(shape, steps):
    task = load_task(shape)
    names = ('cf5', 'cf6', 'cf7', 'cf8')
    controller = CandidateController(task, names)
    p = np.array(task['hover_positions'], float)
    v = np.zeros_like(p)
    minimum, max_speed = np.inf, 0.
    for k in range(steps):
        t = k*.02
        frame = dict(position=p, velocity=v, measurement_time=t, own_velocities=v,
                     own_times=[t]*4)
        out = controller.step(frame, t)
        u = out['velocity']
        minimum = min(minimum, audit_interval(p, v, u, .02, a_max=1., tau=.1, d_safe=.5)[0])
        p, v, _ = track(p, v, u, .02, a_max=1., tau=.1)
        max_speed = max(max_speed, float(np.linalg.norm(v, axis=1).max()))
    if not np.isfinite(p).all() or minimum < .5 or max_speed > .3+1e-9:
        raise AssertionError('smoke regression failed')
    return dict(shape=shape, steps=steps, min_distance_lower_m=minimum, max_actual_speed_mps=max_speed,
                model='original velocity-setpoint plant, NOT FullState hardware', hardware_flights=0)


def replay(path, task):
    # Measurements already aligned on one shared timestamp per decision.
    # Never infer a sampling period or timestamp from row order.
    with np.load(path, allow_pickle=False) as data:
        required = ('position', 'velocity', 'measurement_time', 'decision_time', 'own_velocities', 'own_times')
        arrays = {key: np.array(data[key]) for key in required}
        names = tuple(str(n) for n in data['agent_ids'])
        length = len(arrays['decision_time'])
        if length == 0 or any(len(a) != length for a in arrays.values()):
            raise ValueError('nonempty equal-length replay arrays required')
        expected = np.array(data['expected_command']) if 'expected_command' in data else None
        if expected is not None and expected.shape != (length, 4, 3):
            raise ValueError('expected command must be K x 4 x 3')
    controller = CandidateController(task, names)
    error = 0.
    for k in range(length):
        frame = {key: arrays[key][k] for key in required if key != 'decision_time'}
        out = controller.step(frame, float(arrays['decision_time'][k]))
        if expected is not None:
            error = max(error, float(np.max(np.abs(out['velocity']-expected[k]))))
    if expected is not None and error > 1e-9:
        raise AssertionError(f'command mismatch: {error}')
    return dict(steps=length, expected_compared=expected is not None, maximum_command_error=error,
                scope='offline aligned-observation replay only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', choices=['ring', 'tilted_ring', 'helix'], default='ring')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--replay', help='portable aligned-observation NPZ; schema in README')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    result = replay(args.replay, load_task(args.shape)) if args.replay else smoke(args.shape, args.steps)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
