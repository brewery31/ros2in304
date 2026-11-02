"""Portable replay CLI; has no ROS or aircraft imports."""
import argparse
import json
import numpy as np
from .adapter import CandidateController, load_task
from .research.physical_execution import track, audit_interval


def smoke(shape, steps, plot_path=None, csv_path=None):
    task = load_task(shape)
    names = ('cf5', 'cf6', 'cf7', 'cf8')
    controller = CandidateController(task, names)
    p = np.array(task['hover_positions'], float)
    v = np.zeros_like(p)
    minimum, max_speed = np.inf, 0.
    trace = [p.copy()]
    for k in range(steps):
        t = k*.02
        frame = dict(position=p, velocity=v, measurement_time=t, own_velocities=v,
                     own_times=[t]*4)
        out = controller.step(frame, t)
        u = out['velocity']
        minimum = min(minimum, audit_interval(p, v, u, .02, a_max=1., tau=.1, d_safe=.5)[0])
        p, v, _ = track(p, v, u, .02, a_max=1., tau=.1)
        trace.append(p.copy())
        max_speed = max(max_speed, float(np.linalg.norm(v, axis=1).max()))
    if not np.isfinite(p).all() or minimum < .5 or max_speed > .3+1e-9:
        raise AssertionError('smoke regression failed')
    trace = np.array(trace)
    if csv_path:
        np.savetxt(csv_path, np.column_stack([np.arange(len(trace))*.02,
                    trace.reshape(len(trace), -1)]), delimiter=',',
                    header='t,'+','.join(f'{n}_{a}' for n in names for a in 'xyz'), comments='')
    if plot_path:
        plot_trajectories(shape, trace, plot_path)
    return dict(shape=shape, steps=steps, min_distance_lower_m=minimum, max_actual_speed_mps=max_speed,
                model='original velocity-setpoint plant, NOT FullState hardware', hardware_flights=0,
                final_positions=trace[-1].tolist())


def plot_trajectories(shape, trace, path):
    """Save a 3D view of the closed-loop positions; matplotlib is only needed here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .research.curve_catalog import catalog_parametric
    colors = ['#e8453c', '#2f8fd8', '#3bb54a', '#e0a020']
    names = ('cf5', 'cf6', 'cf7', 'cf8')
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    try:
        curve = catalog_parametric(load_task(shape)['shape'], np.linspace(0, 1, 400))
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '--', color='#9aa0a6', lw=2,
                label='nominal curve')
    except (KeyError, ValueError):
        pass
    for i, name in enumerate(names):
        ax.plot(trace[:, i, 0], trace[:, i, 1], trace[:, i, 2], color=colors[i], lw=2, label=name)
        ax.scatter(*trace[-1, i], color=colors[i], s=40, depthshade=False)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(f'{shape}: closed-loop positions, {len(trace)-1} steps')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


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
    parser.add_argument('--plot', help='save a 3D PNG of the closed-loop positions (smoke mode)')
    parser.add_argument('--csv', help='save per-step positions as CSV (smoke mode)')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    result = (replay(args.replay, load_task(args.shape)) if args.replay
              else smoke(args.shape, args.steps, plot_path=args.plot, csv_path=args.csv))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
