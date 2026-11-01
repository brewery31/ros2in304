"""Experiment engine: problems, methods, metrics, and (parallel) execution.

A ``Problem`` fixes the geometry (shape, scale, area), the swarm size, the safety
distance and therefore -- through the design rule of :mod:`stfc.capacity` --
the lattice spacing ``a`` and the reference covering radius (centralised CVT).
A ``Run`` is one (problem, method, seed) triple.

Everything is deterministic given the seed, and every reported number can be
regenerated with ``python experiments/run_all.py``.  Results are written as CSV
(one row per run) so that the figures and the statistics in the paper can be
regenerated from the raw data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

from .baselines import BoidsTarget, covering_radius, cvt_surface, fp_surface, random_surface
from .capacity import Design, design_at_scale, design_scale
from .dynamics import SimConfig, simulate
from .energy import Params
from .geometry import (PointCloud, make_box, make_cone, make_cylinder, make_dome, make_helix,
                       make_prism, make_ring, make_sphere, make_torus, read_ply)
from .metrics import chamfer, min_pair_distance, nn_stats, quantization_energy, riesz_energy

DATA = Path(__file__).resolve().parent.parent / "data"

__all__ = ["Problem", "make_problem", "run", "SHAPE_BUILDERS"]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def _letter(name: str) -> PointCloud:
    pc = PointCloud.from_file(DATA / f"{name}_10000.ply", name=name)
    return pc.normalized_area(1.0)


def _bunny() -> PointCloud:
    """Stanford bunny resampled uniformly by area from the original mesh."""
    from .geometry import sample_mesh
    d = read_ply(DATA / "bun_zipper.ply")
    pc = sample_mesh(d["vertices"], d["faces"], 60000, rng=np.random.default_rng(7))
    return pc.normalized_area(1.0)


MODEL_DIR = DATA / "models"
EXTERNAL_MODELS = ["armadillo", "dragon", "buddha", "drill", "lucy", "statuette",
                   "asdragon", "letterA3d", "butterfly", "hangyodon", "mushroom"]


def _load_model(name: str) -> tuple[PointCloud, PointCloud]:
    """(control prior, evaluation cloud) for an external model (unit area)."""
    d = np.load(MODEL_DIR / f"{name}.npz")
    ctrl = PointCloud.from_points(d["points"], area=float(d["area"]), name=name)
    ev = PointCloud.from_points(d["eval"], area=float(d["area"]), name=name)
    return ctrl, ev


SHAPE_BUILDERS: dict[str, Callable[[], PointCloud]] = {
    "sphere": lambda: make_sphere(8000).normalized_area(1.0),
    "cylinder": lambda: make_cylinder(8000).normalized_area(1.0),
    "cone": lambda: make_cone(8000).normalized_area(1.0),
    "torus": lambda: make_torus(8000).normalized_area(1.0),
    "box": lambda: make_box(8000).normalized_area(1.0),
    "prism": lambda: make_prism(8000, n=6).normalized_area(1.0),
    "bunny": _bunny,
    "letterA": lambda: _letter("a"),
    "letterC": lambda: _letter("c"),
    "letterR": lambda: _letter("r"),
    "letterI": lambda: _letter("i"),
    "ring": lambda: make_ring(6000, r=1.0),
    "helix": lambda: make_helix(6000, r=0.65, height=0.8, turns=1.25),
    "dome": lambda: make_dome(6000, r=0.55),
}
for _m in EXTERNAL_MODELS:
    SHAPE_BUILDERS[_m] = (lambda m=_m: _load_model(m)[0])

#: intrinsic dimension of every target (1 = curve, 2 = surface)
SHAPE_DIM: dict[str, int] = {"ring": 1, "helix": 1, "letterA": 2, "letterC": 2,
                             "letterR": 2, "letterI": 2}


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

DENSE_EVAL = {  # evaluation clouds (much denser than the control prior)
    "sphere": lambda: make_sphere(60000).normalized_area(1.0),
    "cylinder": lambda: make_cylinder(60000).normalized_area(1.0),
    "cone": lambda: make_cone(60000).normalized_area(1.0),
    "torus": lambda: make_torus(60000).normalized_area(1.0),
    "box": lambda: make_box(60000).normalized_area(1.0),
    "prism": lambda: make_prism(60000, n=6).normalized_area(1.0),
    "ring": lambda: make_ring(60000, r=1.0),
    "helix": lambda: make_helix(60000, r=0.65, height=0.8, turns=1.25),
    "dome": lambda: make_dome(60000, r=0.55),
}


@dataclass
class Problem:
    shape: str
    n_agents: int
    d_safe: float = 0.3
    mu: float = 1.6
    control_points: int = 6000
    seed_target: int = 0
    intrinsic_dim: Optional[int] = None

    # derived
    design: Design = field(init=False)
    cloud: PointCloud = field(init=False)      # control prior (scaled)
    eval_cloud: PointCloud = field(init=False)  # dense evaluation cloud (scaled)
    params: Params = field(init=False)
    ref: dict = field(init=False)              # reference configurations/metrics

    def __post_init__(self) -> None:
        base = SHAPE_BUILDERS[self.shape]()
        if base.m > self.control_points:
            base = base.subsample(self.control_points, rng=np.random.default_rng(self.seed_target))
        self.dim = int(self.intrinsic_dim if self.intrinsic_dim is not None
                       else SHAPE_DIM.get(self.shape, 2))
        if self.dim == 1:
            from .capacity import design_scale_1d
            self.design = design_scale_1d(base.area, self.n_agents, self.d_safe, mu=self.mu)
        else:
            self.design = design_scale(base.area, self.n_agents, self.d_safe, mu=self.mu)
        self.cloud = base.scaled(self.design.scale)
        if self.shape in EXTERNAL_MODELS:
            ev = _load_model(self.shape)[1]
            ev = PointCloud.from_points(ev.points * self.design.scale,
                                        area=ev.area * self.design.scale ** 2,
                                        name=self.shape)
        elif self.shape in DENSE_EVAL:
            ev = DENSE_EVAL[self.shape]().scaled(self.design.scale)
        elif self.shape == "bunny":
            from .geometry import sample_mesh
            d = read_ply(DATA / "bun_zipper.ply")
            ev = sample_mesh(d["vertices"], d["faces"], 120000,
                             rng=np.random.default_rng(11)).normalized_area(1.0)
            ev = ev.scaled(self.design.scale)
        else:  # letters: use every available sample for evaluation
            ev = SHAPE_BUILDERS[self.shape]().scaled(self.design.scale)
        self.eval_cloud = ev
        self.params = Params(a=self.design.lattice_spacing, F0=1.0, mu=self.mu)
        self.ref = self._references()

    @property
    def a(self) -> float:
        return float(self.design.lattice_spacing)

    def _references(self) -> dict:
        rng = np.random.default_rng(12345)
        p_cvt, _ = cvt_surface(self.cloud, self.n_agents, iters=60, rng=rng)
        p_fps = fp_surface(self.cloud, self.n_agents, np.random.default_rng(12345))
        p_rnd = random_surface(self.cloud, self.n_agents, np.random.default_rng(12345))
        ev = self.eval_cloud
        out = {}
        for name, p in (("cvt", p_cvt), ("fps", p_fps), ("random", p_rnd)):
            d = cKDTree(p).query(ev.points, k=1)[0]
            out[name] = {
                "pos": p,
                "r_cov": float(d.max()),
                "r90": float(np.percentile(d, 90)),
                "r75": float(np.percentile(d, 75)),
                "cover_frac": {f"{k:g}": float(np.mean(d <= k * self.a)) for k in (0.6, 0.8, 1.0)},
            }
        out["a"] = self.a
        return out

    def describe(self) -> dict:
        d = self.design.to_dict()
        d.update({"shape": self.shape, "dim": self.dim, "n_agents": self.n_agents,
                  "d_safe": self.d_safe, "mu": self.mu,
                  "ref_r_cov_cvt": self.ref["cvt"]["r_cov"],
                  "ref_r_cov_fps": self.ref["fps"]["r_cov"],
                  "target_area": self.cloud.area})
        return d


def make_problem(shape: str, n_agents: int, d_safe: float = 0.3,
                 mu: float = 1.6, control_points: int = 6000) -> Problem:
    return Problem(shape=shape, n_agents=n_agents, d_safe=d_safe, mu=mu,
                   control_points=control_points)


# ---------------------------------------------------------------------------
# Running one method
# ---------------------------------------------------------------------------

def _coverage_series(res, ev: PointCloud, r_ref: float, n_eval: int = 3000,
                     rng: Optional[np.random.Generator] = None) -> dict:
    rng = np.random.default_rng(rng)
    q = ev.points
    if len(q) > n_eval:
        q = q[rng.choice(len(q), n_eval, replace=False)]
    r90, rcov, cce = [], [], []
    for p in res.pos:
        d = cKDTree(p).query(q, k=1)[0]
        r90.append(float(np.percentile(d, 90)))
        rcov.append(float(d.max()))
        cce.append(float(np.mean(d <= r_ref)))
    return {"t": res.time, "r90": np.array(r90), "r_cov": np.array(rcov), "cce": np.array(cce)}


def _settle(t: np.ndarray, x: np.ndarray, band: float = 0.05) -> float:
    """First time after which ``x`` stays within ``band`` of its final value."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float(t[-1]) if len(t) else float("nan")
    final = x[-1]
    ok = np.abs(x - final) <= band * max(abs(final), 1e-12) + 1e-12
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            return float(t[min(i + 1, len(t) - 1)])
    return float(t[0])


def run(problem: Problem, method: str = "ours", seed: int = 0,
        t_end: float = 60.0, dt: float = 0.02,
        params: Optional[dict] = None, sim: Optional[dict] = None,
        record_state: bool = True, traj_path: Optional[str] = None) -> dict:
    """Run one method on one problem; returns a flat metrics dict."""
    t0 = time.time()
    a = problem.a
    d_safe = problem.d_safe
    ev = problem.eval_cloud
    # primary reference: centralised CVT (coverage-optimal but *unconstrained*,
    # it may place agents closer than d_safe); secondary: FPS (safe).
    r_ref = problem.ref["cvt"]["r_cov"]
    r_ref_fps = problem.ref["fps"]["r_cov"]
    over = dict(params or {})
    cfg_over = dict(sim or {})

    if method in ("cvt", "fps", "random"):
        p = problem.ref[method]["pos"]
        d = cKDTree(p).query(ev.points, k=1)[0]
        return _row(problem, method, seed, p, d, r_ref, t_wall=0.0,
                    n_filter_rejects=0, v_mono=np.nan, e_final=np.nan,
                    t_settle=0.0, effort=0.0, state="static")

    if method in ("cvt_safe", "local_cvt"):
        from .baselines import cvt_safe, local_cvt
        if method == "cvt_safe":
            p, _ = cvt_safe(problem.cloud, problem.n_agents, problem.d_safe,
                            iters=60, rng=np.random.default_rng(seed),
                            init=problem.ref["cvt"]["pos"].copy())
        else:
            p, _ = local_cvt(problem.cloud, problem.n_agents, problem.a,
                             iters=20, rng=np.random.default_rng(seed),
                             init=problem.ref["fps"]["pos"].copy())
        d = cKDTree(p).query(ev.points, k=1)[0]
        return _row(problem, method, seed, p, d, r_ref, t_wall=time.time() - t0,
                    n_filter_rejects=0, v_mono=np.nan, e_final=np.nan,
                    t_settle=0.0, effort=0.0, state="static")

    if method == "ours":
        prm = Params(a=a, F0=1.0, mu=problem.mu, **over)
        cfg = SimConfig(n_agents=problem.n_agents, t_end=t_end, dt=dt,
                        log_every=5,
                        a_max=cfg_over.pop("a_max", 3.0),
                        v_max=cfg_over.pop("v_max", 1.0),
                        seed=seed, init_radius=cfg_over.pop("init_radius", 2.2 * problem.cloud.radius),
                        record_state=record_state, **cfg_over)
        res = simulate(problem.cloud, prm, cfg, traj_path=traj_path)
        p = res.p
        d = cKDTree(p).query(ev.points, k=1)[0]
        ser = _coverage_series(res, ev, r_ref, rng=seed)
        vmono = float(np.mean(np.diff(np.asarray(res.energy["V"])) <= 1e-6 * abs(res.energy["V"][0])))
        effort = float(np.trapz(np.asarray(res.energy["KE"]), res.time))
        return _row(problem, method, seed, p, d, r_ref, t_wall=time.time() - t0,
                    n_filter_rejects=int(res.final["n_filter_rejects"]),
                    v_mono=vmono, e_final=float(np.asarray(res.energy["V"])[-1]),
                    t_settle=_settle(ser["t"], ser["r90"]), effort=effort,
                    state="sim", series=ser, res=res, prm=prm,
                    d_min_traj=float(res.final.get("d_min_traj", np.nan))
                    if np.isfinite(res.final.get("d_min_traj", np.nan)) else None)

    if method == "stfc_orig":
        v_max_b = float(cfg_over.pop("v_max", 1.0))
        a_max_b = float(cfg_over.pop("a_max", 3.0))
        use_orca = bool(cfg_over.pop("orca_layer", False))
        bt = BoidsTarget(problem.cloud, a_lattice=a, d_safe=d_safe,
                         max_speed=v_max_b,
                         alpha_c=over.get("alpha_c", 0.008),
                         alpha_a=over.get("alpha_a", 0.05),
                         alpha_s=over.get("alpha_s", 0.05),
                         beta_a=over.get("beta_a", 0.02),
                         beta_r=over.get("beta_r", 0.02),
                         gamma_ms=over.get("gamma_ms", 0.05),
                         kappa_damp=over.get("kappa_damp", 0.9),
                         d_h=over.get("d_h", 0.0))
        rng = np.random.default_rng(seed)
        # same enclosing initialisation as ours (jittered lattice in a ball)
        from .dynamics import SimConfig as _SC, initial_positions
        cfg = _SC(n_agents=problem.n_agents, t_end=t_end, dt=dt, seed=seed,
                  init_radius=2.2 * problem.cloud.radius)
        p = initial_positions(problem.n_agents, problem.cloud, cfg, rng, d_safe)
        v = np.zeros_like(p)
        trace_p, trace_v, trace_t = [p.copy()], [v.copy()], [0.0]
        n_steps = int(t_end / dt)
        log_every = 5
        ts, r90s, rcovs, dmin_log = [], [], [], [float(_min_dist(p))]
        use_filter = bool(cfg_over.pop("filter_baseline", False))
        for k in range(n_steps):
            p_old = p
            p_prev, v_prev = p.copy(), v.copy()
            p, v = bt.step(p, v, dt)
            # SAME execution constraints in both branches (review item 4):
            # clip the bare law exactly like the +ORCA branch, otherwise the
            # comparison mixes the safety layer with the actuator limit.
            v = np.asarray(v, float)
            sp0 = np.linalg.norm(v, axis=1)
            v = v * np.minimum(1.0, v_max_b / np.maximum(sp0, 1e-12))[:, None]
            dvb0 = v - v_prev
            nd0 = np.linalg.norm(dvb0, axis=1)
            v = v_prev + dvb0 * np.minimum(1.0, a_max_b * dt / np.maximum(nd0, 1e-12))[:, None]
            p = p_prev + dt * v
            if use_orca:
                from .orca_adapter import orca_velocity
                v_o, _ = orca_velocity(p_prev, v_prev, v, dt=dt,
                                       r_vis=problem.params.r_vis_abs,
                                       d_safe=d_safe, v_max=v_max_b,
                                       margin=0.02)
                v_o = np.asarray(v_o, float)
                sp = np.linalg.norm(v_o, axis=1)
                v_o *= np.minimum(1.0, v_max_b / np.maximum(sp, 1e-12))[:, None]
                dvb = v_o - v_prev
                nd = np.linalg.norm(dvb, axis=1)
                v_o = v_prev + dvb * np.minimum(1.0, a_max_b * dt / np.maximum(nd, 1e-12))[:, None]
                v = v_o
                p = p_prev + dt * v
            if use_filter and _min_dist(p) <= d_safe:
                for j in range(8):
                    h = 0.5 ** j
                    p_try = p_old + h * (p - p_old)
                    if _min_dist(p_try) > d_safe:
                        p = p_try
                        v = v * h
                        break
                else:
                    p, v = p_old, np.zeros_like(v)
            from .metrics import trajectory_step_min
            dmin_log.append(trajectory_step_min(p_prev, p))
            if traj_path is not None:
                trace_p.append(p.copy()); trace_v.append(v.copy()); trace_t.append((k + 1) * dt)
            if k % log_every == 0:
                dd = cKDTree(p).query(ev.points, k=1)[0]
                ts.append(k * dt); r90s.append(float(np.percentile(dd, 90))); rcovs.append(float(dd.max()))
        d = cKDTree(p).query(ev.points, k=1)[0]
        if traj_path is not None:
            with open(traj_path, "xb") as stream:
                np.savez_compressed(stream, pos=np.asarray(trace_p), vel=np.asarray(trace_v), time=np.asarray(trace_t))
        tconv = _settle(np.array(ts), np.array(r90s))
        row = _row(problem, method, seed, p, d, r_ref, t_wall=time.time() - t0,
                    n_filter_rejects=0, v_mono=np.nan, e_final=np.nan,
                    t_settle=tconv, effort=float(np.trapz(np.asarray(r90s) * 0 + np.nan, ts)) if False else np.nan,
                    state="sim",
                    d_min_traj=float(np.min(np.asarray(dmin_log))) if dmin_log else None)
        config = dict(v_max=v_max_b, a_max=a_max_b, dt=dt, t_end=t_end,
                      executed_steps=n_steps, executed_t_end=n_steps * dt,
                      mu=problem.mu, a=a, d_safe=d_safe, seed=seed,
                      shape=problem.shape, n_agents=problem.n_agents,
                      init_radius=cfg.init_radius, control_points=problem.control_points,
                      latency=0, meas_noise=0.0, observation="ideal_state",
                      safety_filter="orca_limited" if use_orca else "none",
                      filter_baseline=use_filter, audit="all_pairs_linear_segment",
                      r_vis=problem.params.r_vis_abs, orca_margin=0.02,
                      nominal_params={k: getattr(bt, k) for k in
                          ("alpha_c", "alpha_a", "alpha_s", "beta_a", "beta_r",
                           "gamma_ms", "kappa_damp", "d_h", "r_vis", "h")})
        row.update(dt=float(dt), t_end=float(t_end),
                   config_json=json.dumps(config, sort_keys=True, allow_nan=False))
        return row


    raise ValueError(f"unknown method {method!r}")


def _row(problem: Problem, method: str, seed: int, p: np.ndarray, d: np.ndarray,
         r_ref: float, *, t_wall: float, n_filter_rejects: int, v_mono: float,
         e_final: float, t_settle: float, effort: float, state: str,
         series: Optional[dict] = None, res=None, prm=None,
         d_min_traj: Optional[float] = None) -> dict:
    a = problem.a
    ev = problem.eval_cloud
    nn = nn_stats(p)
    row = {
        "shape": problem.shape, "n_agents": problem.n_agents, "method": method,
        "seed": seed, "a": a, "d_safe": problem.d_safe, "mu": problem.mu,
        "area": problem.cloud.area,
        # --- coverage (target -> agent) ---
        "r_cov": float(d.max()),
        "r90": float(np.percentile(d, 90)),
        "r75": float(np.percentile(d, 75)),
        "r_cov_norm": float(d.max() / a),
        "r90_norm": float(np.percentile(d, 90) / a),
        "r_cov_ratio": float(d.max() / r_ref),
        "r90_ratio": float(np.percentile(d, 90) / problem.ref["cvt"]["r90"]),
        "r_cov_ratio_fps": float(d.max() / problem.ref["fps"]["r_cov"]),
        "r90_ratio_fps": float(np.percentile(d, 90) / problem.ref["fps"]["r90"]),
        "cce_at_ref": float(np.mean(d <= r_ref)),
        "cce06": float(np.mean(d <= 0.6 * a)),
        "cce08": float(np.mean(d <= 0.8 * a)),
        "cce10": float(np.mean(d <= 1.0 * a)),
        # --- shape / distribution ---
        "chamfer": chamfer(p, ev),
        "quant": quantization_energy(p, ev) / a ** 2,
        "riesz": riesz_energy(p, s=1.0) * a,
        "nn_mean": nn["mean"] / a, "nn_cv": nn["cv"],
        # --- safety ---
        "d_mean": float(ev.distance(p).mean()),
        "d_min": float(_min_dist(p)), "d_min_ratio": float(_min_dist(p) / problem.d_safe),
        # final-state feasibility and *whole-trajectory* safety are different
        # things: the first can hold while the run violated the constraint on the
        # way (review finding #5).  Report both, plus the trajectory minimum.
        "safe": bool(_min_dist(p) > problem.d_safe),
        "d_min_traj": float(d_min_traj) if d_min_traj is not None else float(_min_dist(p)),
        "d_min_traj_ratio": (float(d_min_traj / problem.d_safe) if d_min_traj is not None
                             else float(_min_dist(p) / problem.d_safe)),
        "safe_all": bool((d_min_traj if d_min_traj is not None else _min_dist(p))
                         > problem.d_safe),
        # --- convergence ---
        "t_settle": float(t_settle), "t_wall": float(t_wall),
        "n_filter_rejects": int(n_filter_rejects),
        "v_monotone": float(v_mono), "V_final": float(e_final), "effort": float(effort),
        "dt": float(getattr(res, "config", None).dt) if res is not None else np.nan,
        "t_end": float(getattr(res, "config", None).t_end) if res is not None else np.nan,
        "success": bool((np.percentile(d, 90) / problem.ref["cvt"]["r90"] <= 1.10)
                        and (_min_dist(p) > problem.d_safe)),
        "success_all": bool((np.percentile(d, 90) / problem.ref["cvt"]["r90"] <= 1.10)
                            and (d_min_traj if d_min_traj is not None else _min_dist(p))
                            > problem.d_safe),
        "state": state,
        "n_nom_sat": int(res.final.get("n_nom_sat", 0)) if res is not None else np.nan,
        "n_exec_sat": int(res.final.get("n_exec_sat", 0)) if res is not None else np.nan,
        "n_exec_viol": int(res.final.get("n_exec_viol", 0)) if res is not None else np.nan,
        "agent_steps": int(res.final.get("agent_steps", 0)) if res is not None else np.nan,
        # --- per-task provenance (review: do not label a scan as "defaults")
        "obs_age_ms": float(getattr(getattr(res, "config", None), "latency", 0) or 0)
                      * float(getattr(getattr(res, "config", None), "dt", 0.02) or 0.02) * 1000.0,
        "n_orca_fail": int(getattr(res.final, "get", lambda *a: 0)("n_orca_fail", 0)) if res else 0,
        "status": ("orca_fail" if res and res.final.get("n_orca_fail", 0) else "ok"),
        "config_json": json.dumps({
            "v_max": float(getattr(getattr(res, "config", None), "v_max", 1.0) or 1.0),
            "a_max": float(getattr(getattr(res, "config", None), "a_max", 3.0) or 3.0),
            "latency": int(getattr(getattr(res, "config", None), "latency", 0) or 0),
            "meas_noise": float(getattr(getattr(res, "config", None), "meas_noise", 0.0) or 0.0),
            "safety_filter": str(getattr(getattr(res, "config", None), "safety_filter", "")) or "unknown(non-sim path)",
            "config_source": "res" if res is not None else "problem+defaults",
            "mu": float(problem.mu), "a": float(a), "dt": float(getattr(getattr(res, "config", None), "dt", np.nan) or np.nan),
        }, sort_keys=True),
    }
    if series is not None:
        row["t_series"] = json.dumps(np.asarray(series["t"]).tolist())
        row["r90_series"] = json.dumps(np.asarray(series["r90"]).tolist())
    return row


def _min_dist(p: np.ndarray) -> float:
    if len(p) < 2:
        return float("inf")
    return float(cKDTree(p).query(p, k=2)[0][:, 1].min())
