"""Acceleration-Velocity Obstacles (AVO) for distributed 3-D shape formation.

This module implements a *buffered velocity-obstacle* avoidance layer for
velocity-controlled quadrotors with hard actuator limits

    |v| <= v_max,        |v_{k+1} - v_k| <= a_max * dt,        dt = 0.02 s.

It is a pure-Python (NumPy/SciPy) replacement for the "solve, then clip"
pattern used around ORCA.  The acceleration limit is part of the obstacle: a
pair is kept outside a *buffered* velocity obstacle whose radius contains the
distance needed to brake the closing motion.

Model
-----
For a pair ``(i, j)`` let

    p_ij = p_j - p_i,   d = |p_ij|,   n = p_ij / d,
    s_ij = max(0, n . (v_i - v_j))          (closing speed),

and let ``r_pair = d_safe + margin`` be the combined body radius.  The pair's
buffered radius is

    R_ij = r_pair + s_ij^2 / (2 * (a_i + a_j)),                  (1)

i.e. the base radius plus the distance needed to remove the current closing
speed at the combined braking capability ``a_i + a_j`` (here ``2 a_max``).
The radius therefore grows with the pair's closing speed, which a single
scalar ORCA radius cannot express.

Both robots run the same reciprocal rule and take half of the required change
of relative closing speed.  For a desired post-step closing cap ``C`` this
gives the half-space, for robot ``i``,

    n . u_i  <=  (n . v_i + n . v_j) / 2  +  C / 2.              (2)

Adding (2) for the two robots gives ``n . (u_i - u_j) <= C``.  The step-safe
cap used here is the largest closing speed that still leaves a full stopping
distance inside the gap *after* the step:

    C(g) = -a_pair tau + sqrt((a_pair tau)^2 + 2 a_pair g),
    a_pair = a_i + a_j,   tau = _REACTION_STEPS * dt,   g = d - r_pair.  (3)

``C(g)`` is the non-negative root of
``C^2 / (2 a_pair) + tau C = g``.  The ``tau C`` term reserves the distance
travelled while the pair reacts; it is why the cap is not merely
``sqrt(2 a_pair g)``.  The implementation default is
``_REACTION_STEPS = 1.5`` (one step is the minimum for a self-consistent
discrete rule; 1.5 steps adds a small allowance).  If the current closing
speed is above ``C(g)`` the half-space asks the pair to brake.  If even full
braking by both robots cannot reach ``C(g)`` this step
(``s - C > (a_i+a_j) dt``), the requested cap is unreachable; the cap is
relaxed to the reachable value ``s - (a_i+a_j) dt`` and the step is *reported*
as infeasible (``info['n_pair_infeasible']``).

The half-space (2) is a plane cut through the buffered velocity obstacle.  The
exact buffered VO is a curved cone/disk set because ``R`` depends on the
relative velocity; the plane cut is the tangent cut along the line of centres
and is an approximation (see ``docs/18_AVO_DESIGN.md``).  It is conservative
for the radial closing motion and ignores lateral motion, which does not need
radial braking; it is not the full non-convex VO.

What is guaranteed
------------------
G1 (unconditional, by construction, for valid input with ``|v_i| <= v_max``):
the returned ``v_new`` satisfies

    |v_new - v| <= a_max * dt        and        |v_new| <= v_max,

including for infeasible agents and for the best-effort fallback.  This is
checked on every call and reported in ``info['bounds_ok']``.  No post-hoc
clipping is applied to the AVO action: every action is built as
``v + alpha * dhat`` with ``alpha <= a_max*dt`` and ``|v + alpha*dhat| <=
v_max``, where the speed cap is solved analytically.

G2 (conditional, ideal sensing and synchronous reciprocal execution): for a
pair with ``d >= r_pair``, suppose the unrelaxed cap ``C(g)`` is reachable
(``s - C <= (a_i+a_j) dt``; otherwise the module reports the pair/agent as
infeasible), both robots execute the same half-space (2) built from the same
state, and every pair that could otherwise threaten them is sensed.  Then,
under the piecewise-constant velocity step model, the pair's separation is at
least ``r_pair`` at the end of the step, and in fact at every instant of the
step:

    n.(p_j' - p_i') >= d - dt C  and  dt C <= g  =>  >= r_pair.

The key inequality ``dt C <= g`` follows from
``C^2/(2 a_pair) + tau C = g`` with ``tau >= dt``.  This is a one-step
statement.  It is not a recursive safety theorem: a feasible invariant set,
neighbour truncation by ``r_vis``, asynchronous updates, inconsistent
neighbour estimates and the multi-neighbour feasibility of the half-space
system remain separate conditions.  The design doc states them explicitly.
Numerically, the solver accepts a half-space residual of at most 1e-5 m/s,
which degrades the bound by 2e-5*dt = 4e-7 m; ``info['max_violation']``
reports the actual residual on every call.

What is *not* guaranteed
------------------------
* No global collision-avoidance theorem.  With noisy or delayed state, with
  asynchronous updates, with robots not running the same reciprocal rule, or
  when the acceleration-bounded feasible set is empty, G2 does not apply.
  Infeasible steps are reported, never silently hidden.
* A best-effort action is not a safety certificate.  It only respects the
  actuator bounds and minimises a constraint-violation score.
* Neighbour velocities are the *sensed/estimated* ones.  In the batch API
  they are supplied as the ``v`` argument (or ``v_neighbour`` when the caller
  keeps its own velocity exact and only neighbour velocities are sensed).
  The one-step result is conditional on those estimates.
* No recursive feasibility or liveness claim for the surface-formation task;
  coverage quality is an empirical question answered by the comparison script.

Interface
---------
``avoid(p, v, preferred, *, dt, r_vis, d_safe, v_max, a_max, margin, horizon)``
returns ``(v_new, info)`` with the same shape/semantics as
``stfc.orca_adapter.orca_velocity``: ``p`` positions, ``v`` current
velocities and ``preferred`` desired velocities are all ``N x 3``.  Only
neighbours within ``r_vis`` are used.  Each agent solves its own half-spaces
using neighbour positions and neighbour velocities; no candidate action or
intention is exchanged.

``v_neighbour`` is an optional extension used by the simulation harness when
the robot knows its own velocity exactly (``v``) but senses neighbours with a
delay/noise model (``v_neighbour``).  If omitted, ``v`` is used for both, which
is the strictly local semantics of ``orca_velocity``.  The required signature
is unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["avoid"]

# Numerical tolerances.  They are deliberately far below the actuator scales
# (a_max*dt = 0.06 m/s, d_safe = 0.3 m) so that they cannot hide a physically
# meaningful violation.
_TOL = 1e-5            # half-space feasibility tolerance (m/s);
                       # 1e-5 m/s over dt is 2e-7 m, far below any
                       # physical scale.  The iterative solver must be
                       # allowed this round-off band, otherwise feasible
                       # sets are mislabelled infeasible.
_BOUND_EPS = 1e-9      # tolerance when *checking* the actuator balls
_BUF_TOL = 1e-8        # tolerance for the buffered-radius diagnostic
_DEGENERATE = 1e-9     # pair distance below which n is undefined
# Integration-time reserve in the discrete braking cap:
#     C^2 / (2 a_pair) + _REACTION_STEPS * dt * C = g.
# One step of reserve is the minimum for a self-consistent discrete rule;
# 1.5 steps gives a small extra allowance without the coverage cost of a full
# two-step reserve.  It is a module-level constant so the design can be
# ablated without changing the API.
_REACTION_STEPS = 1.5


def _check_inputs(p, v, preferred):
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    preferred = np.asarray(preferred, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must be an N x 3 array")
    if v.shape != p.shape or preferred.shape != p.shape:
        raise ValueError("p, v and preferred must have the same N x 3 shape")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))
            and np.all(np.isfinite(preferred))):
        raise ValueError("p, v and preferred must be finite")
    return p, v, preferred


def _ray_cap(u, v, a_dt, v_max):
    """Project ``u`` onto the ray from ``v`` into the two actuator balls.

    Returns ``(capped_point, step_length)``.  For ``|v| <= v_max`` every
    returned point satisfies ``|out - v| <= a_dt`` and ``|out| <= v_max``.
    The construction is exact up to a 1e-12 relative round-off margin.
    """
    delta = u - v
    length = np.linalg.norm(delta, axis=1)
    dhat = np.zeros_like(delta)
    nz = length > 1e-15
    dhat[nz] = delta[nz] / length[nz, None]
    vd = np.einsum("nd,nd->n", v, dhat)
    disc = np.maximum(vd * vd + v_max * v_max - np.einsum("nd,nd->n", v, v),
                      0.0)
    alpha_speed = np.maximum(-vd + np.sqrt(disc), 0.0)
    alpha = np.minimum(np.minimum(length, a_dt), alpha_speed)
    # Relative margin against a last-bit violation of either ball.
    alpha = np.minimum(alpha, a_dt * (1.0 - 1e-12))
    out = v + alpha[:, None] * dhat
    # Guard against a last-bit speed overshoot.  This changes the action by
    # O(1e-16) and cannot create a meaningful acceleration violation.
    sp = np.linalg.norm(out, axis=1)
    over = sp > v_max
    if np.any(over):
        out[over] *= (v_max * (1.0 - 1e-12) / sp[over])[:, None]
    return out, alpha


def _pair_arrays(p, v_ego, v_nb, *, dt, r_vis, d_safe, margin, v_max, a_max,
                 horizon):
    """Build padded per-agent pair constraints.

    Returns a dict with arrays of shape ``(N, K)`` (K = max neighbour count):

    ``n``, ``b``, ``mask``
        Plane-cut normal, right-hand side and validity mask.  Inactive entries
        have ``b = +inf`` and zero normal.
    ``dist``, ``cap``, ``cap_eff``, ``s``
        Pair distance, requested cap (0 if inside the buffered radius (1),
        otherwise (3)), relaxed cap actually used, and current closing speed.
    ``pair_bad``
        ``s - cap > (a_i+a_j) dt``: the requested cap is unreachable even with
        both robots at full braking.
    ``buffer_bad``
        ``d < r_pair + s^2/(2(a_i+a_j))``: the current state is already inside
        the buffered radius (1).
    """
    n_agents = len(p)
    a_pair = 2.0 * a_max
    r_pair = d_safe + margin
    # A pair farther than this cannot reach the buffered region within the
    # horizon even at the maximum closing speed 2 v_max.  With the study
    # parameters the activation radius exceeds r_vis, so the horizon does not
    # truncate any sensed neighbour; it is still computed and reported.
    r_act = min(r_vis, r_pair + (2.0 * v_max) ** 2 / (2.0 * a_pair)
                + 2.0 * v_max * horizon)
    tree = cKDTree(p)
    lists = tree.query_ball_point(p, r_act)
    counts = np.array([max(len(ids) - 1, 0) for ids in lists], dtype=int)
    kmax = int(counts.max()) if len(counts) and counts.max() > 0 else 0
    n = np.zeros((n_agents, kmax, 3))
    b = np.full((n_agents, kmax), np.inf)
    dist = np.full((n_agents, kmax), np.inf)
    cap = np.zeros((n_agents, kmax))
    cap_eff = np.zeros((n_agents, kmax))
    s_arr = np.zeros((n_agents, kmax))
    mask = np.zeros((n_agents, kmax), dtype=bool)
    pair_bad = np.zeros((n_agents, kmax), dtype=bool)
    buffer_bad = np.zeros((n_agents, kmax), dtype=bool)
    if kmax == 0:
        return dict(n=n, b=b, mask=mask, dist=dist, cap=cap,
                    cap_eff=cap_eff, s=s_arr, pair_bad=pair_bad,
                    buffer_bad=buffer_bad, r_act=r_act)
    for i, ids in enumerate(lists):
        js = np.array([j for j in ids if j != i], dtype=int)
        if len(js) == 0:
            continue
        k = len(js)
        rel = p[js] - p[i]
        di = np.linalg.norm(rel, axis=1)
        good = di > _DEGENERATE
        if np.any(~good):
            # Coincident pair: no direction is defined.  A zero normal with a
            # very negative bound forces the solver to report infeasibility.
            nhat = np.zeros((k, 3))
            di = np.maximum(di, _DEGENERATE)
            nhat[good] = rel[good] / di[good, None]
            si = np.zeros(k)
            gi = np.full(k, -np.inf)
            ci = np.zeros(k)
            bi = np.full(k, -1e6)
        else:
            nhat = rel / di[:, None]
            si = np.maximum(
                np.einsum("kd,kd->k", nhat, v_ego[i] - v_nb[js]), 0.0)
            gi = di - r_pair
            # Reaction reserve: C solves
            # C^2/(2 a_pair) + react * C = g with react = 1.5 dt.  The
            # reserve makes the discrete rule self-consistent: the pair
            # brakes before the continuous stopping-distance boundary, so the
            # relaxation used when the cap is unreachable does not walk into
            # contact.  For the one-step proof only react >= dt matters.
            react = _REACTION_STEPS * dt
            ci_gap = np.maximum(
                -a_pair * react
                + np.sqrt((a_pair * react) ** 2 + 2.0 * a_pair
                          * np.maximum(gi, 0.0)), 0.0)
            # If the pair is already inside the buffered radius (1), the
            # stopping distance no longer fits in the gap: the requested cap
            # is zero (brake as hard as the reciprocal rule allows), not the
            # gap-only cap.  This is the AVO test itself, not an extra
            # heuristic.
            R_now = r_pair + si * si / (2.0 * a_pair)
            ci = np.where(di < R_now - _BUF_TOL, 0.0, ci_gap)
            bi = (0.5 * (np.einsum("kd,d->k", nhat, v_ego[i])
                         + np.einsum("kd,kd->k", nhat, v_nb[js]))
                  + 0.5 * np.maximum(ci, si - a_pair * dt))
        n[i, :k] = nhat
        dist[i, :k] = di
        s_arr[i, :k] = si
        cap[i, :k] = ci
        cap_eff[i, :k] = np.maximum(ci, si - a_pair * dt)
        # Requested (unrelaxed) cap unreachable even with both robots braking
        # fully.  For a pair inside the buffered radius the requested cap is
        # zero, so this is the hard-brake case.
        R = r_pair + si * si / (2.0 * a_pair)
        buffer_bad[i, :k] = di < R - _BUF_TOL
        pair_bad[i, :k] = (si - ci) > (a_pair * dt + _TOL)
        if np.any(~good):
            pair_bad[i, :k] |= ~good
            buffer_bad[i, :k] |= ~good
            b[i, :k] = bi
        else:
            b[i, :k] = bi
        mask[i, :k] = True
    # A plane cut can only matter if its bound is below the largest projection
    # reachable inside the acceleration ball.  Inactive cuts are dropped
    # exactly (not heuristically): n.u <= n.v + a_max*dt <= b for every
    # admissible u.  They are represented by b=+inf.
    if kmax:
        nv = np.einsum("nkd,nd->nk", n, v_ego)
        inactive = mask & (b >= nv + a_max * dt + _TOL)
        b = np.where(inactive, np.inf, b)
    return dict(n=n, b=b, mask=mask, dist=dist, cap=cap, cap_eff=cap_eff,
                s=s_arr, pair_bad=pair_bad, buffer_bad=buffer_bad,
                r_act=r_act)


def _violations(n, b, u):
    """Per-agent maximum half-space residual ``n.u - b`` (inactive -> -inf)."""
    if n.shape[1] == 0:
        return np.zeros(len(u))
    val = np.einsum("nkd,nd->nk", n, u) - b
    return val.max(axis=1)


def _project_halfspaces_cimmino(u, n, b, v, *, a_dt, v_max, iters):
    """Cimmino projections onto the half-spaces, with the actuator-ray cap.

    Not a certified QP solver.  The residual is checked by the caller; when it
    is not below tolerance the agent is reported infeasible and the best-effort
    branch is used.  The actuator bounds hold after every operation.
    """
    u = u.copy()
    if n.shape[1] == 0:
        return u, 0
    used = 0
    for _ in range(iters):
        pos = np.maximum(
            np.einsum("nkd,nd->nk", n, u) - b, 0.0)
        if pos.max() <= _TOL:
            break
        used += 1
        cnt = (pos > 0).sum(axis=1)
        delta = -np.einsum("nkd,nk->nd", n, pos) / np.maximum(cnt, 1)[:, None]
        u, _ = _ray_cap(u + delta, v, a_dt, v_max)
    return u, used


def _best_effort(v, preferred, n, b, mask, *, a_dt, v_max):
    """Actuator-bounded action minimising the worst half-space violation.

    Every candidate is a point on the ray from ``v`` capped by both actuator
    balls, so G1 holds for every candidate (and for the returned action).
    This is a heuristic, not a certificate; the residual is reported.
    """
    n_agents = len(v)
    out = _ray_cap(preferred, v, a_dt, v_max)[0]
    worst = _violations(n, b, out)
    best = np.maximum(worst, 0.0).copy()
    best_u = out.copy()
    if n.shape[1] == 0:
        return best_u, best

    speed = np.linalg.norm(v, axis=1)
    directions = []
    dv = np.zeros_like(v)
    nz = speed > 1e-12
    dv[nz] = -v[nz] / speed[nz, None]
    directions.append(dv)

    # Away from constraints that are violated, or would bind this step.
    slack = np.einsum("nkd,nd->nk", n, v) - b
    urgency = np.where(mask & (slack > -a_dt), slack + a_dt, 0.0)
    urgency = np.maximum(urgency, 0.0)
    wsum = np.einsum("nkd,nk->nd", n, urgency)
    wn = np.linalg.norm(wsum, axis=1)
    dw = np.zeros_like(v)
    nz = wn > 1e-12
    dw[nz] = -wsum[nz] / wn[nz, None]
    directions.append(dw)

    # Away from the single most violated normal.
    val = np.einsum("nkd,nd->nk", n, out) - b
    kstar = np.argmax(val, axis=1)
    dk = -n[np.arange(n_agents), kstar]
    dk[~np.isfinite(b[np.arange(n_agents), kstar])] = 0.0
    directions.append(dk)

    # Two deterministic lateral escapes orthogonal to the urgency direction.
    axis = np.eye(3)[np.argmin(np.abs(dw), axis=1)]
    lat = np.cross(dw, axis)
    ln = np.linalg.norm(lat, axis=1)
    nz = ln > 1e-12
    lat[nz] /= ln[nz, None]
    directions.append(lat)
    directions.append(-lat)

    for d in directions:
        dn = np.linalg.norm(d, axis=1)
        ok = dn > 1e-12
        if not np.any(ok):
            continue
        dhat = np.zeros_like(d)
        dhat[ok] = d[ok] / dn[ok, None]
        cand, _ = _ray_cap(v + (a_dt * 0.999999) * dhat, v, a_dt, v_max)
        w = np.maximum(_violations(n, b, cand), 0.0)
        better = w < best - 1e-12
        best = np.where(better, w, best)
        best_u = np.where(better[:, None], cand, best_u)
    return best_u, best


def avoid(p, v, preferred, *, dt, r_vis, d_safe, v_max, a_max, margin,
          horizon, v_neighbour=None):
    """Buffered AVO velocity selection for every agent.

    Parameters
    ----------
    p, v, preferred : (N, 3) float arrays
        Positions, current velocities and desired velocities.  ``v`` must
        already satisfy ``|v| <= v_max`` (the previous action of the same
        filter does); ``preferred`` may violate the actuator limits and is
        projected onto the reachable actuator ray first.
    dt, r_vis, d_safe, v_max, a_max, margin, horizon : float
        Integration step, sensing radius, required separation, actuator
        limits, extra safety margin and AVO look-ahead horizon.
    v_neighbour : (N, 3), optional
        Sensed velocities of the agents as seen by their neighbours.  When
        omitted, ``v`` is used for both ego and neighbour velocities (the
        batch semantics of ``orca_velocity``).  A physical robot knows its
        own velocity exactly but only estimates its neighbours'; the
        simulation harness uses this keyword to express that.

    Returns
    -------
    v_new : (N, 3) float array
        Executed velocity.  ``|v_new - v| <= a_max*dt`` and
        ``|v_new| <= v_max`` hold by construction (G1).
    info : dict
        ``infeasible`` (bool, any agent), ``infeasible_agents`` (bool array),
        ``n_infeasible`` (agents), ``n_solver_infeasible`` (agents for which
        the half-space solver found no admissible velocity),
        ``n_pair_infeasible`` (agents involved in an unreachable pair cap),
        ``n_pair_infeasible_pairs`` (directed pairs), ``n_buffer_violation_pairs``
        and ``buffer_violations`` (directed pairs already inside R_ij),
        ``n_buffer_violation_agents``, ``max_violation`` (max half-space
        residual, m/s), ``violation`` (per-agent max residual), ``bounds_ok``,
        ``max_accel``, ``max_speed``, ``min_buffer_margin``, ``n_constraints``,
        ``n_active_pairs``, ``solver_iterations``, ``horizon``,
        ``r_activation``, ``r_vis_covers_step``, ``reaction_steps``.

    Notes
    -----
    G1 is unconditional for valid input (finite arrays, ``|v|<=v_max``).
    G2 (one-step buffered separation) is conditional on ideal synchronous
    sensing/execution and on both robots using the same reciprocal rule; it is
    not a recursive or global safety theorem.  See the module docstring and
    ``docs/18_AVO_DESIGN.md``.
    """
    p, v, preferred = _check_inputs(p, v, preferred)
    if v_neighbour is None:
        v_nb = v
    else:
        v_nb = np.asarray(v_neighbour, dtype=float)
        if v_nb.shape != v.shape or not np.all(np.isfinite(v_nb)):
            raise ValueError("v_neighbour must be a finite N x 3 array")
    for name, val in (("dt", dt), ("r_vis", r_vis), ("d_safe", d_safe),
                      ("v_max", v_max), ("a_max", a_max), ("horizon", horizon)):
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be a positive finite scalar")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be a finite non-negative scalar")
    speed = np.linalg.norm(v, axis=1)
    if np.any(speed > v_max + _BOUND_EPS):
        raise ValueError(
            "initial speed exceeds v_max; the actuator guarantee is defined "
            "for |v| <= v_max")
    a_dt = a_max * dt

    pairs = _pair_arrays(p, v, v_nb, dt=dt, r_vis=r_vis, d_safe=d_safe,
                         margin=margin, v_max=v_max, a_max=a_max,
                         horizon=horizon)
    n, b, mask = pairs["n"], pairs["b"], pairs["mask"]
    n_agents = len(p)

    # Nominal action projected onto the reachable actuator ray.
    u0, _ = _ray_cap(preferred, v, a_dt, v_max)
    u = u0.copy()
    solver_bad = np.zeros(n_agents, dtype=bool)
    solver_iters = 0
    violation = np.zeros(n_agents)

    if n.shape[1]:
        val0 = np.einsum("nkd,nd->nk", n, u0) - b
        worst0 = np.maximum(val0.max(axis=1), 0.0)
        violation = worst0.copy()
        need = worst0 > _TOL
        if np.any(need):
            idx = np.nonzero(need)[0]
            sub, used = _project_halfspaces_cimmino(
                u0[idx], n[idx], b[idx], v[idx], a_dt=a_dt, v_max=v_max,
                iters=60)
            u[idx] = sub
            solver_iters = used
            worst = np.maximum(
                np.einsum("nkd,nd->nk", n[idx], sub) - b[idx], 0.0).max(axis=1)
            solver_bad[idx] = worst > _TOL
            violation[idx] = worst
    # Pair-level infeasibility: the unrelaxed cap cannot be reached this step.
    pair_bad_agents = pairs["pair_bad"].any(axis=1)
    infeasible = solver_bad | pair_bad_agents
    if np.any(solver_bad):
        eff, eff_viol = _best_effort(v[solver_bad], preferred[solver_bad],
                                     n[solver_bad], b[solver_bad],
                                     mask[solver_bad], a_dt=a_dt, v_max=v_max)
        u[solver_bad] = eff
        violation[solver_bad] = eff_viol

    # Final diagnostics.  The returned action is always built through
    # _ray_cap, so the actuator bounds are structural; the check below makes a
    # numerical regression impossible to hide.
    if n.shape[1]:
        violation = np.maximum(
            np.einsum("nkd,nd->nk", n, u) - b, 0.0).max(axis=1)
    delta = u - v
    max_accel = float(np.linalg.norm(delta, axis=1).max()) if n_agents else 0.0
    max_speed = float(np.linalg.norm(u, axis=1).max()) if n_agents else 0.0
    bounds_ok = bool(max_accel <= a_dt + _BOUND_EPS
                     and max_speed <= v_max + _BOUND_EPS)
    if not bounds_ok:  # pragma: no cover - construction guarantees this
        raise RuntimeError("internal error: AVO violated the actuator bounds")

    buffer_bad = pairs["buffer_bad"]
    buffer_violations = int(buffer_bad.sum())
    if mask.any():
        r_pair = d_safe + margin
        R = r_pair + pairs["s"] ** 2 / (4.0 * a_max)
        margin_pairs = np.where(mask, pairs["dist"] - R, np.inf)
        min_buffer_margin = float(margin_pairs.min())
    else:
        min_buffer_margin = float("inf")
    active_pairs = int((mask & np.isfinite(b)).sum())

    info = dict(
        infeasible=bool(infeasible.any()),
        infeasible_agents=infeasible,
        n_infeasible=int(infeasible.sum()),
        n_solver_infeasible=int(solver_bad.sum()),
        n_pair_infeasible=int(pair_bad_agents.sum()),
        n_pair_infeasible_pairs=int(pairs["pair_bad"].sum()),
        n_buffer_violation_pairs=int(buffer_violations),
        buffer_violations=int(buffer_violations),
        n_buffer_violation_agents=int(buffer_bad.any(axis=1).sum()),
        n_agents=int(n_agents),
        r_vis_covers_step=bool(r_vis >= d_safe + margin + 2.0 * v_max * dt),
        reaction_steps=float(_REACTION_STEPS),
        max_violation=float(np.max(violation)) if n_agents else 0.0,
        violation=violation,
        bounds_ok=bounds_ok,
        max_accel=max_accel,
        max_speed=max_speed,
        min_buffer_margin=min_buffer_margin,
        n_constraints=int(mask.sum()),
        n_active_pairs=active_pairs,
        solver_iterations=int(solver_iters),
        horizon=float(horizon),
        r_activation=float(pairs["r_act"]),
    )
    return u, info
