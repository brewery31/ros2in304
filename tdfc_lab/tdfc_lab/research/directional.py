"""Local velocity projection with explicit conditional step certificates.

The QP-like convex problem contains linear inward-speed constraints and two
balls. Solver success is not trusted; actual returned velocity is checked.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize


def directional_velocity(p, current, preferred, fallback, *, d_safe, a_max,
                         v_max, dt, error, r_vis, margin=.02):
    tree=cKDTree(p)
    result=preferred.copy()
    failures=0; solver_failures=0; max_residual=0.; precondition_failures=0
    for i, ids in enumerate(tree.query_ball_point(p,r_vis)):
        js=np.asarray([j for j in ids if j!=i],dtype=int)
        if not len(js): continue
        delta=p[js]-p[i]; distance=np.linalg.norm(delta,axis=1)
        precondition_failures += int(np.any(distance < d_safe+2*error))
        normals=delta/np.maximum(distance[:,None],1e-15)
        residual_gap=distance-d_safe-margin-2*error-2*v_max*dt
        braking_cap=np.sqrt(a_max*np.maximum(residual_gap,0))
        # Pairwise sum of these halfspaces implies projected separation >= ds.
        step_cap=(distance-d_safe-2*error)/(2*dt)
        cap=np.minimum(braking_cap,step_cap)
        def violation(w):
            return max(float(np.max(normals@w-cap)),
                       np.linalg.norm(w)-v_max,
                       np.linalg.norm(w-current[i])-a_max*dt)
        target=preferred[i]
        if violation(target)>1e-8:
            cons=[dict(type='ineq',fun=lambda w:cap-normals@w,jac=lambda w:-normals),
                  dict(type='ineq',fun=lambda w:v_max**2-w@w,jac=lambda w:-2*w),
                  dict(type='ineq',fun=lambda w:(a_max*dt)**2-np.sum((w-current[i])**2),
                       jac=lambda w:-2*(w-current[i]))]
            opt=minimize(lambda w:.5*np.sum((w-target)**2),fallback[i],
                         jac=lambda w:w-target,constraints=cons,method='SLSQP',
                         options=dict(maxiter=40,ftol=1e-10))
            solver_failures+=int(not opt.success)
            if np.all(np.isfinite(opt.x)) and violation(opt.x)<=1e-8:
                result[i]=opt.x
            else:
                result[i]=fallback[i]
        # Enforce exact actuator bounds even when the optimizer accepts a tiny
        # numerical violation; then re-check the halfspaces on the actual action.
        dv=result[i]-current[i]
        result[i]=current[i]+dv*min(1.,a_max*dt/max(np.linalg.norm(dv),1e-15))
        result[i]*=min(1.,v_max/max(np.linalg.norm(result[i]),1e-15))
        residual=violation(result[i])
        failures+=int(residual>1e-8)
        max_residual=max(max_residual,residual)
    return result,dict(failures=failures,solver_failures=solver_failures,max_residual=max_residual,precondition_failures=precondition_failures)
