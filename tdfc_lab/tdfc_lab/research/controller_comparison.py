"""Exploratory three-controller comparison, separate from historical experiments.

This module deliberately makes no invariance claim. The archived CBF algorithm
is called unchanged; its actuator violations are measured rather than repaired.
All controllers share the same nominal forces, sensing and integration model.
"""
from __future__ import annotations

from time import process_time
import numpy as np
from scipy.spatial import cKDTree

from .dynamics import Simulator, _min_pair_distance
from .energy import composite_forces
from .avoidance import AvoidConfig, select_action
from .orca_adapter import orca_velocity
from .directional import directional_velocity


def limit_norm(x, limit):
    return x * np.minimum(1.0, limit / np.maximum(np.linalg.norm(x, axis=-1), 1e-15))[..., None]


def bounded_errors(rng, shape, bound):
    """Independent uniform-volume errors in Euclidean 3-balls."""
    directions = rng.normal(size=shape)
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-15)
    return directions * (bound * rng.random(shape[:-1]) ** (1 / 3))[..., None]


def early_braking(p, v, nominal, *, d_safe, r_vis, a_max, age,
                  eps_p, eps_v, reaction_time=0.20, clearance=0.03):
    """Ego-only radial deceleration toward the most urgent observed neighbour.

    Activation uses relative closing speed, a stopping-distance heuristic and
    an uncertainty allowance. It is NOT a robust barrier or a feasibility test.
    Each output depends only on that agent and observed neighbours in r_vis;
    no candidate control or intention of another agent is read.
    """
    out = nominal.copy()
    weights = np.zeros(len(p))
    tree = cKDTree(p)
    for i, neighbours in enumerate(tree.query_ball_point(p, r_vis)):
        js = np.asarray([j for j in neighbours if j != i], dtype=int)
        if not len(js):
            continue
        delta = p[i] - p[js]
        distance = np.linalg.norm(delta, axis=1)
        away = delta / np.maximum(distance, 1e-12)[:, None]
        closing = np.maximum(-np.sum((v[i] - v[js]) * away, axis=1), 0.0)
        uncertainty = 2 * eps_p + 2 * eps_v * (age + reaction_time)
        trigger = (d_safe + clearance + uncertainty
                   + closing * (reaction_time + age)
                   + closing ** 2 / (2 * a_max))
        risk = np.clip((trigger - distance) / np.maximum(trigger - d_safe, 1e-12), 0, 1)
        j = int(np.argmax(risk))
        weights[i] = risk[j]
        out[i] = (1 - risk[j]) * nominal[i] + risk[j] * a_max * away[j]
    return limit_norm(out, a_max), weights


def separation_speed_governor(p_observed, current_velocity, proposal, *,
                             d_safe, a_max, v_max, dt, position_error,
                             r_vis, margin=.02):
    """Cooperative distance-dependent speed cap; no neighbour action access.

    Allocate half of the residual pair clearance to each stopping distance:
    speed**2/(2*a_max) <= clearance/2. This engineering supervisor is audited,
    not a general invariance proof.
    position_error bounds each extrapolated position. Reserve includes two
    errors and one integration interval's worst relative travel.
    """
    distances, _ = cKDTree(p_observed).query(p_observed, k=2)
    nearest = distances[:, 1]
    reserve = margin + 2 * position_error + 2 * v_max * dt
    cap = np.minimum(v_max, np.sqrt(a_max * np.maximum(nearest - d_safe - reserve, 0)))
    cap = np.where(nearest <= r_vis, cap, v_max)
    result = proposal.copy()
    need = np.linalg.norm(result, axis=1) > cap + 1e-12
    # Try clipping the desired vector; if its turn + braking exceeds the
    # acceleration budget, brake along current velocity instead.
    clipped = limit_norm(result, cap)
    reachable = np.linalg.norm(clipped-current_velocity, axis=1) <= a_max*dt+1e-12
    result[need & reachable] = clipped[need & reachable]
    fallback = need & ~reachable
    speed = np.linalg.norm(current_velocity, axis=1)
    radial_speed = np.maximum(cap, speed-a_max*dt)
    radial_speed = np.minimum(radial_speed, speed)
    radial = current_velocity * (radial_speed / np.maximum(speed, 1e-15))[:, None]
    result[fallback] = radial[fallback]
    return result, cap


class ComparisonSimulator(Simulator):
    """Shared bounded observation harness; legacy CBF is an audited comparator.

    Noise is generated for every agent before any controller call, using a
    separate RNG, so paired runs see identical errors at identical steps.
    Agent state errors are shared across observers (not independent sensor
    errors per edge). This is an explicit simulation simplification.
    """
    def __init__(self, *args, method, eps_v=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        if method not in ("base", "light", "cbf", "predict", "governed", "orca_raw", "orca_limited", "directional"):
            raise ValueError(method)
        self.method = method
        self.eps_v = eps_v
        self.obs_rng = np.random.default_rng(self.cfg.seed + 100000)
        self.states = []
        self.last_measurement = None
        self.controller_cpu = 0.0
        self.interventions = 0
        self.agent_updates = 0
        self.precondition_failures = 0
        self.constraint_failures = 0
        self.solver_failures = 0
        self.max_constraint_residual = 0.0
        self.governor_interventions = 0
        self.governor_unreachable = 0
        self.prediction_failures = 0
        self.prediction_phys_rejections = 0

    def observe(self):
        self.states.append((self.p.copy(), self.v.copy()))
        delay = self.cfg.latency
        # At startup the simulator has a stationary prehistory at the initial
        # state; at k>=delay use exactly k-delay, not k-delay+1.
        if len(self.states) > delay + 1:
            self.states.pop(0)
        p, v = self.states[0]
        pm = p + bounded_errors(self.obs_rng, p.shape, float(self.cfg.eps_p or 0))
        vm = v + bounded_errors(self.obs_rng, v.shape, self.eps_v)
        self.last_measurement = (pm.copy(), vm.copy())
        return pm, vm

    def step(self, dt):
        pm, vm = self.observe()
        cfg = self.cfg
        started = process_time()
        force = composite_forces(pm, vm, self.cloud, self.params,
                                 k_max=cfg.k_max, a_max=cfg.a_max).total
        nominal = force.copy()
        if self.method in ("light", "governed"):
            force, _ = early_braking(
                pm, vm, force, d_safe=self.params.d_safe,
                r_vis=self.params.r_vis_abs, a_max=cfg.a_max,
                age=cfg.latency * dt, eps_p=float(cfg.eps_p or 0),
                eps_v=self.eps_v)
        if self.method == "predict":
            # Extrapolate aged observations, without access to neighbour controls.
            age = cfg.latency * dt
            sensed_p = pm + age * vm
            sensed_v = limit_norm(vm, cfg.v_max)
            normals = self.cloud.outward_normal(sensed_p)
            neighbours = cKDTree(sensed_p).query_ball_point(sensed_p, self.params.r_vis_abs)
            avoid_cfg = AvoidConfig(dt=dt, d_safe=self.params.d_safe)
            desired = limit_norm(sensed_v + dt * nominal, cfg.v_max)
            forces = []
            for i, near in enumerate(neighbours):
                js = np.asarray([j for j in near if j != i], dtype=int)
                action, info = select_action(sensed_p[i], sensed_v[i], desired[i],
                    sensed_p[js], sensed_v[js], normals[i], avoid_cfg, cfg.a_max, cfg.v_max)
                # Apply requested acceleration through the plant, not estimated velocity reset.
                forces.append((action - sensed_v[i]) / dt)
                self.prediction_failures += int(info['fail'])
                self.prediction_phys_rejections += int(info['pred_clear_phys'] < 0)
            force = limit_norm(np.asarray(forces), cfg.a_max)
        # Plant integration uses actual state; controllers do not receive it.
        # Projection to the speed ball preserves acceleration bounds if the
        # previous velocity is inside that ball.
        proposal = limit_norm(self.v + dt * force, cfg.v_max)
        if self.method == "cbf":
            updated = self._cbf_filter(proposal, dt, pm, vm)
            changed = np.linalg.norm(updated - proposal, axis=1) > 1e-9
        elif self.method in ("light", "predict", "governed"):
            changed = np.linalg.norm(force - nominal, axis=1) > 1e-9
            updated = proposal
        else:
            changed = np.zeros(self.n, dtype=bool)
            updated = proposal
        if self.method in ("governed", "directional"):
            age = cfg.latency * dt
            # Semi-implicit Euler contributes an additional O(a*age*dt) term.
            error = float(cfg.eps_p or 0) + self.eps_v*age + .5*cfg.a_max*age*(age+dt)
            guarded, cap = separation_speed_governor(pm+age*vm, self.v, updated,
                d_safe=self.params.d_safe, a_max=cfg.a_max, v_max=cfg.v_max,
                dt=dt, position_error=error, r_vis=self.params.r_vis_abs)
            if self.method == "directional":
                guarded, info = directional_velocity(pm+age*vm, self.v, proposal, guarded,
                    d_safe=self.params.d_safe, a_max=cfg.a_max, v_max=cfg.v_max,
                    dt=dt, error=error, r_vis=self.params.r_vis_abs)
                self.precondition_failures += info['precondition_failures']
                self.constraint_failures += info['failures']
                self.solver_failures += info['solver_failures']
                self.max_constraint_residual = max(self.max_constraint_residual,info['max_residual'])
            changed = np.linalg.norm(guarded-proposal, axis=1) > 1e-9
            self.governor_interventions += int(changed.sum())
            self.governor_unreachable += int(np.sum(np.linalg.norm(guarded,axis=1)>cap+1e-9))
            updated = guarded
        if self.method in ("orca_raw", "orca_limited"):
            action, residual = orca_velocity(pm, vm, proposal, dt=dt,
                r_vis=self.params.r_vis_abs, d_safe=self.params.d_safe, v_max=cfg.v_max)
            self.constraint_failures += int(np.sum(residual > 1e-5))
            self.max_constraint_residual = max(self.max_constraint_residual,float(np.max(residual)))
            updated = (action if self.method == "orca_raw" else
                       limit_norm(self.v + limit_norm(action-self.v,cfg.a_max*dt),cfg.v_max))
            changed = np.linalg.norm(updated-proposal,axis=1)>1e-9
        self.controller_cpu += process_time() - started
        self.interventions += int(changed.sum())
        self.agent_updates += self.n
        self.v = updated
        self.p = self.p + dt * self.v
        self.t += dt
        return {"d_min": _min_pair_distance(self.p)}


class TrajectoryAudit:
    """Independent exhaustive pair audit of every simulator step, including t=0.

    Intra-step interpolation matches this semi-implicit Euler simulator; it is
    not a proof about an accelerating physical robot between control samples.
    """
    def __init__(self, p, v, *, d_safe, dt, a_max, v_max):
        self.p, self.v = p.copy(), v.copy()
        self.ii, self.jj = np.triu_indices(len(p), 1)
        self.ds, self.dt, self.amax, self.vmax = d_safe, dt, a_max, v_max
        self.minimum = _min_pair_distance(p)
        self.initial = self.minimum
        self.bad_steps = self.accel_bad_steps = self.speed_bad_steps = 0
        self.first_bad = 0.0 if self.minimum < d_safe - 1e-9 else None
        self.max_accel = self.max_speed = 0.0
        self.steps = 0

    def update(self, p, v):
        r0 = self.p[self.ii] - self.p[self.jj]
        r1 = p[self.ii] - p[self.jj]
        delta = r1 - r0
        fraction = np.clip(-np.sum(r0 * delta, axis=1) /
                           np.maximum(np.sum(delta * delta, axis=1), 1e-30), 0, 1)
        distance = np.linalg.norm(r0 + fraction[:, None] * delta, axis=1)
        minimum = float(distance.min()) if len(distance) else float("inf")
        self.minimum = min(self.minimum, minimum)
        if minimum < self.ds - 1e-9:
            self.bad_steps += 1
            if self.first_bad is None:
                j = int(np.argmin(distance))
                self.first_bad = (self.steps + float(fraction[j])) * self.dt
        acceleration = float(np.linalg.norm(v - self.v, axis=1).max() / self.dt)
        speed = float(np.linalg.norm(v, axis=1).max())
        self.max_accel = max(self.max_accel, acceleration)
        self.max_speed = max(self.max_speed, speed)
        self.accel_bad_steps += int(acceleration > self.amax + 1e-8)
        self.speed_bad_steps += int(speed > self.vmax + 1e-8)
        self.steps += 1
        self.p, self.v = p.copy(), v.copy()

    def result(self):
        return dict(initial_min_ratio=self.initial / self.ds,
                    intra_min_ratio=self.minimum / self.ds,
                    safe_all=self.minimum >= self.ds - 1e-9,
                    violation_step_fraction=self.bad_steps / max(self.steps, 1),
                    first_violation_s=self.first_bad,
                    max_accel=self.max_accel, max_speed=self.max_speed,
                    accel_violation_steps=self.accel_bad_steps,
                    speed_violation_steps=self.speed_bad_steps)
