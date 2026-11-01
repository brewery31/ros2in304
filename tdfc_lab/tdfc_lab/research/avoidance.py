"""Predict-and-select avoidance (frozen config: 6 candidates, H = 0.4 s).

Semantics fixed by the author's spec (2026-10-21):

* neighbours are described by **locally estimated** positions/velocities (no
  candidate actions, no true velocities, no broadcast data);
* the prediction is constant-velocity -- an ASSUMPTION that must be tested
  against neighbours turning / accelerating simultaneously;
* candidates: nominal (follow the surface controller), limited braking, and
  four veers in a frame transverse to nominal motion, oriented by the target
  normal, with a nondegenerate axis fallback; frame continuity is not guaranteed;
* every candidate is rolled out with the actuator model
  ``|v_{k+1} - v_k| <= a_max dt`` and ``|v| <= v_max`` for the whole horizon
  (braking = limited deceleration, never an instantaneous zero);
* collision check: synchronous relative segment between adjacent prediction
  points (exact for piecewise-linear motion), not just sampled points;
* if every candidate fails, execute the limited brake and record the failure --
  it is a degraded strategy, NOT a safe fallback, and a finite candidate set
  failing says nothing about physical recoverability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AvoidConfig:
    n_candidates: int = 6          # frozen
    horizon: float = 0.40          # frozen
    dt_pred: float = 0.05          # legacy setting; rollout uses execution dt
    dt: float = 0.02
    d_safe: float = 0.30
    margin: float = 0.02
    veer_angle: float = 0.5        # rad
    w_progress: float = 1.0
    w_smooth: float = 0.3
    w_off: float = 0.5


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def candidates(v_des, v_prev, n_i, d_closest, cfg: AvoidConfig, a_max, v_max):
    """Six POLICIES (each maps the current velocity to a target velocity).

    * nominal  : follow the surface controller (v_des)
    * brake    : keep decelerating at a_max (NOT a one-off target) -- this is the
                 difference that made the previous version judge braking infeasible
    * four veers around +-t1 / +-t2 (target frame from the target normal)
    """
    # Four distinct lateral directions even when normal and motion align.
    bn = _unit(v_des) if np.linalg.norm(v_des) > 1e-9 else (
        _unit(v_prev) if np.linalg.norm(v_prev) > 1e-9 else np.array([1., 0., 0.]))
    t1 = np.cross(n_i, bn)
    if np.linalg.norm(t1) < 1e-6:
        axis = np.eye(3)[np.argmin(np.abs(bn))]
        t1 = np.cross(axis, bn)
    t1 = _unit(t1)
    t2 = _unit(np.cross(bn, t1))
    base = np.array(v_des, float)

    def nominal(v):
        return base

    def brake(v):
        sp = np.linalg.norm(v)
        f = max(0.0, 1.0 - a_max * cfg.dt / sp) if sp > 1e-9 else 0.0
        return f * v

    out = [("nominal", nominal), ("brake", brake)]
    bn = _unit(base) if np.linalg.norm(base) > 1e-9 else np.array([1.0, 0.0, 0.0])
    for name, t in (("+t1", t1), ("-t1", -t1), ("+t2", t2), ("-t2", -t2)):
        # orthogonalise the veer direction against the nominal direction: without
        # this a "veer" can become a pure speed-up (|v| > v_max).
        tp = t - float(t @ bn) * bn
        ntp = np.linalg.norm(tp)
        if ntp < 1e-6:
            continue
        tp = tp / ntp
        def pol(v, tp=tp):
            return (base * np.cos(cfg.veer_angle)
                    + np.linalg.norm(base) * tp * np.sin(cfg.veer_angle))
        out.append((name, pol))
    return out[: cfg.n_candidates]


def _rollout(policy, v_prev, cfg, a_max, v_max, n_steps):
    """Roll a *policy* out under the actuator model (|dv| <= a_max dt each step)."""
    vs = [np.array(v_prev, float)]
    for _ in range(n_steps):
        target = policy(vs[-1])
        dv = target - vs[-1]
        nd = np.linalg.norm(dv)
        step = min(1.0, a_max * cfg.dt / max(nd, 1e-12))
        v = vs[-1] + step * dv
        sp = np.linalg.norm(v)
        if sp > v_max:
            v *= v_max / sp
        vs.append(v)
    return np.array(vs)


def _rel_seg_min(ri, rj):
    """min |r(t)| on [0,1] for the relative segment r(t)=ri+t(rj-ri)."""
    d = rj - ri
    nn = d @ d
    t = 0.0 if nn < 1e-18 else float(np.clip(-(ri @ d) / nn, 0.0, 1.0))
    return float(np.linalg.norm(ri + t * d))


def select_action(p_i, v_prev, v_des, p_nb, v_nb, n_i, cfg: AvoidConfig,
                  a_max, v_max, d_off=0.0):
    """Return executable first velocity and prediction diagnostics.

    fail means no candidate passed the buffered prediction threshold, not collision.
    Constant-velocity neighbour prediction is not a joint safety guarantee.
    d_off / w_off are reserved: no surface-distance evaluator is available here.
    """
    if cfg.dt <= 0 or cfg.horizon <= 0 or a_max <= 0 or v_max <= 0:
        raise ValueError("positive time and actuator limits required")
    p_i, v_prev, v_des = map(lambda x: np.asarray(x, float), (p_i, v_prev, v_des))
    p_nb = np.asarray(p_nb, float).reshape(-1, 3)
    v_nb = np.asarray(v_nb, float).reshape(-1, 3)
    if p_nb.shape != v_nb.shape:
        raise ValueError("neighbour position/velocity shapes differ")
    if np.linalg.norm(v_prev) > v_max + 1e-9:
        raise ValueError("initial speed exceeds actuator model limit")
    # Predict at the execution integration rate. dt_pred is not an integration step.
    n_steps = max(1, int(np.ceil(cfg.horizon / cfg.dt - 1e-12)))
    d_closest = (p_nb[np.argmin(np.linalg.norm(p_nb - p_i, axis=1))] - p_i
                 if len(p_nb) else np.array([1., 0., 0.]))
    options = []
    times = np.arange(n_steps + 1) * cfg.dt
    for k, (name, policy) in enumerate(candidates(v_des, v_prev, n_i, d_closest,
                                                  cfg, a_max, v_max)):
        traj = _rollout(policy, v_prev, cfg, a_max, v_max, n_steps)
        positions = np.vstack([p_i, p_i + np.cumsum(traj[1:] * cfg.dt, axis=0)])
        clear = np.inf
        if len(p_nb):
            relative = positions[:, None, :] - (p_nb[None] + times[:, None, None] * v_nb[None])
            r0, delta = relative[:-1], np.diff(relative, axis=0)
            denom = np.sum(delta * delta, axis=-1)
            fraction = np.clip(-np.sum(r0 * delta, axis=-1) / np.maximum(denom, 1e-18), 0, 1)
            # Last segment is truncated if the horizon is not a multiple of dt.
            fraction[-1] = np.minimum(fraction[-1], (cfg.horizon - times[-2]) / cfg.dt)
            clear = float(np.min(np.linalg.norm(r0 + fraction[..., None] * delta, axis=-1)))
        first = traj[1]
        score = (cfg.w_progress * float(first @ v_des) / max(np.linalg.norm(v_des), 1e-12)
                 - cfg.w_smooth * float(np.linalg.norm(first - v_prev)))
        options.append((k, name, first, clear, score))
    feasible = [x for x in options if x[3] >= cfg.d_safe + cfg.margin]
    chosen = max(feasible, key=lambda x: x[4]) if feasible else options[1]
    k, name, first, clear, score = chosen
    return first.copy(), dict(
        fail=not bool(feasible), no_margin_candidate=not bool(feasible),
        cand=k, selected=name, clear=clear,
        pred_clear_phys=clear - cfg.d_safe,
        pred_clear_margin=clear - cfg.d_safe - cfg.margin,
        reasons=[(x[1], x[3]) for x in options],
        prediction_horizon=cfg.horizon,
    )


def assert_consistent(problem, cfg: AvoidConfig) -> None:
    """Consistency assertion required by the author: a == mu * d_safe."""
    assert problem.mu > 0, "mu must be set explicitly"
    assert abs(problem.a - problem.mu * problem.d_safe) < 1e-9, (
        f"inconsistent design: a={problem.a:.6f} != mu*d_safe={problem.mu*problem.d_safe:.6f}")
