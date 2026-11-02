"""Portable replay CLI; has no ROS or aircraft imports."""
import argparse
import json
from pathlib import Path
import numpy as np
from .adapter import CandidateController, load_task
from .research.physical_execution import track, audit_interval


def smoke(shape, steps, plot_path=None, csv_path=None, agents=4, seed=None, trace_path=None):
    from .scenarios import SceneController, hover_grid, scene_metrics
    controller = SceneController(shape, agents)
    names = controller.names
    p = (np.array(controller.task['hover_positions'], float) if controller.task is not None and seed is None
         else hover_grid(agents, seed))
    v = np.zeros_like(p)
    minimum, max_speed = np.inf, 0.
    trace, velocities, success = [p.copy()], [v.copy()], []
    first_success = None
    # Evaluate success every 0.1s; require it to persist for one full second.
    hold_start = None
    for k in range(steps):
        t = k*.02
        u = controller.command(p,v,t)
        minimum = min(minimum, audit_interval(p,v,u,.02,a_max=1.,tau=.1,d_safe=.5)[0])
        p,v,_ = track(p,v,u,.02,a_max=1.,tau=.1)
        trace.append(p.copy()); velocities.append(v.copy())
        max_speed = max(max_speed,float(np.linalg.norm(v,axis=1).max()))
        if (k+1)%5 == 0:
            metric = scene_metrics(shape,p,controller)
            good = (metric['max_curve_error_m'] <= .05
                    and metric.get('coverage_ratio_048m',1.) >= .9
                    and np.linalg.norm(v,axis=1).max() <= .05)
            now = (k+1)*.02
            if not good: hold_start = None
            elif hold_start is None: hold_start = now
            if hold_start is not None and now-hold_start >= 1.-1e-9 and first_success is None:
                first_success = hold_start
            success.append(good)
    if not np.isfinite(p).all() or minimum < .5 or max_speed > .3+1e-9:
        raise AssertionError('physical-model safety/speed regression failed')
    trace, velocities = np.array(trace), np.array(velocities)
    for path in (plot_path,csv_path,trace_path):
        if path: Path(path).parent.mkdir(parents=True,exist_ok=True)
    if csv_path:
        np.savetxt(csv_path,np.column_stack([np.arange(len(trace))*.02,trace.reshape(len(trace),-1)]),delimiter=',',
                   header='t,'+','.join(f'{n}_{a}' for n in names for a in 'xyz'),comments='')
    if trace_path:
        np.savez_compressed(trace_path,pos=trace,vel=velocities,time=np.arange(len(trace))*.02)
    if plot_path: plot_trajectories(shape,trace,plot_path)
    metrics = scene_metrics(shape,p,controller)
    terminal_success = bool(success and all(success[-11:]) and len(success)>=11)
    return dict(shape=shape,agents=agents,seed=seed,steps=steps,method=controller.method,
                min_distance_lower_m=minimum,max_actual_speed_mps=max_speed,metrics=metrics,
                first_held_success_s=first_success,terminal_held_success=terminal_success,
                model='physical velocity-setpoint model, NOT firmware SIL or hardware',hardware_flights=0,
                final_positions=trace[-1].tolist())


def plot_trajectories(shape,trace,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .scenarios import letter_targets,letter_edges,ROBOT_NAMES
    from .research.curve_catalog import catalog_parametric
    n=trace.shape[1];letter=shape in ('R','A','L')
    goals=letter_targets(shape) if letter else catalog_parametric(f'curve_demo_{shape}_n{n}',np.linspace(0,1,400))
    fig=plt.figure(figsize=(11,5));ax=fig.add_subplot(121,projection='3d');front=fig.add_subplot(122)
    axes=(0,1) if shape=='ring' else (0,2)
    if letter:
        for a,b in letter_edges(shape):
            ax.plot(*goals[[a,b]].T,'--',color='.7',lw=1)
            front.plot(goals[[a,b],axes[0]],goals[[a,b],axes[1]],'--',color='.7',lw=1)
        # Graph corners are not assigned robot goals; draw strokes only.
    else:
        ax.plot(*goals.T,'--',color='.65')
        front.plot(goals[:,axes[0]],goals[:,axes[1]],'--',color='.65',label='target curve')
    for i in range(n):
        color=plt.get_cmap('tab10')(i)
        ax.plot(*trace[:,i].T,color=color,alpha=.65,lw=1)
        ax.scatter(*trace[-1,i],color=color,s=45);ax.scatter(*trace[0,i],color=color,marker='x',s=30)
        front.scatter(trace[-1,i,axes[0]],trace[-1,i,axes[1]],color=color,s=60,label=ROBOT_NAMES[i],zorder=4)
    allp=np.vstack([trace.reshape(-1,3),goals]);low=allp.min(0);high=allp.max(0)
    span=max(np.ptp(allp,axis=0).max(),1.)*1.15;center=(low+high)/2
    ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_zlim(center[2]-span/2,center[2]+span/2);ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-60)
    ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title='Feedback paths and final positions')
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(MaxNLocator(4))
    front.set(xlabel='x [m]',ylabel='y [m]' if shape=='ring' else 'z [m]',title='Final points: top view' if shape=='ring' else 'Final points: front view')
    front.set_aspect('equal',adjustable='box');front.margins(.2);front.grid(alpha=.25)
    front.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    fig.suptitle(f'{shape}, N={n}: numerical model; dashed strokes are reference only')
    fig.tight_layout();fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)


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
        if expected is not None and expected.shape != (length, len(names), 3):
            raise ValueError('expected command must be K x N x 3')
    if any(not np.isfinite(a).all() for a in arrays.values()) or (expected is not None and not np.isfinite(expected).all()):
        raise ValueError('nonfinite replay arrays')
    controller = CandidateController(task, names)
    error = 0.
    for k in range(length):
        frame = {key: arrays[key][k] for key in required if key != 'decision_time'}
        out = controller.step(frame, float(arrays['decision_time'][k]))
        if expected is not None:
            error = max(error, float(np.max(np.abs(out['velocity']-expected[k]))))
    if expected is not None and error > 1e-7:
        raise AssertionError(f'command mismatch: {error}')
    return dict(steps=length, expected_compared=expected is not None, maximum_command_error=error,
                scope='offline aligned-observation replay only')


def run_suite(output, steps=3000):
    import hashlib
    import csv
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    matrix=[('ring',4),('ring',6),('helix',4),('helix',6)]
    plan=dict(matrix=matrix,seeds=[0,1,2],steps=steps,dt=.02,initialization='common-height grid + <=1cm XY jitter',
              observations='exact synchronous numerical states',model='velocity-loop model, NOT firmware SIL',
              success='last >=1s: curve max error<=5cm, C(0.48m)>=0.9; max speed<=0.05m/s')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for shape,n in matrix:
        for seed in plan['seeds']:
            stem=f'{shape}_n{n}_seed{seed}'
            try:
                row=smoke(shape,steps,agents=n,seed=seed,
                          plot_path=output/(stem+'.png') if seed==0 else None,
                          trace_path=output/(stem+'.npz') if seed==0 else None)
                row['status']='completed'
            except Exception as error:
                row=dict(shape=shape,agents=n,seed=seed,status='failed',error=str(error))
            rows.append(row)
            (output/(stem+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            print(stem,row['status'],row.get('terminal_held_success'),flush=True)
    fields=['shape','agents','seed','status','method','min_distance_lower_m','max_actual_speed_mps','terminal_held_success','first_held_success_s']
    with open(output/'summary.csv','w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    files={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in Path(__file__).parent.rglob('*.py')}
    result=dict(runs=len(rows),completed=sum(r['status']=='completed' for r in rows),
                terminal_successes=sum(r.get('terminal_held_success',False) for r in rows),
                hardware_flights=0,firmware_sil_runs=0,source_sha256=files)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', choices=['ring', 'helix'], default='ring')
    parser.add_argument('--agents', type=int, choices=[4,6],default=4,help='four or six agents')
    parser.add_argument('--seed', type=int, help='jittered common-height grid seed')
    parser.add_argument('--trace', help='save pos/vel/time NPZ')
    parser.add_argument('--suite',action='store_true',help='four curve scenes x three fixed seeds')
    parser.add_argument('--out',help='new result directory for --suite')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--replay', help='portable aligned-observation NPZ; schema in README')
    parser.add_argument('--plot', help='save a 3D PNG of the closed-loop positions (smoke mode)')
    parser.add_argument('--csv', help='save per-step positions as CSV (smoke mode)')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.suite:
        if not args.out:parser.error('--suite requires --out (new directory)')
        print(json.dumps(run_suite(args.out,args.steps),indent=2));return
    agents = args.agents
    from .scenarios import validate_scene
    validate_scene(args.shape, agents)
    result = (replay(args.replay, load_task(args.shape, agents=agents)) if args.replay
              else smoke(args.shape,args.steps,plot_path=args.plot,csv_path=args.csv,agents=agents,seed=args.seed,trace_path=args.trace))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
