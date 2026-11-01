"""Closed-loop simulation of STFC-2 (semi-implicit Euler, optional safety filter).

The simulator is deliberately small and explicit: agents are unit-mass
double integrators ``p' = v, v' = u`` with

* acceleration and speed saturation (``a_max``, ``v_max``),
* optional exponential command smoothing with rate ``sat_rate`` (first-order
  actuator lag, which is how the Crazyflie velocity loop is modelled),
* optional per-step safety filter (backtracking on the velocity increment) that
  makes every accepted configuration ``d_safe``-separated by construction.

Neighbour and target queries are the only "perception" the controller receives.
No agent ever sees another agent's index-dependent state beyond its position
and velocity inside ``r_vis`` -- this is the formal counterpart of the paper's
"no inter-agent communication" assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from .energy import Params, composite_forces, total_energy
from .geometry import PointCloud

__all__ = ["SimConfig", "SimResult", "Simulator", "initial_positions"]


@dataclass
class SimConfig:
    n_agents: int = 500
    t_end: float = 30.0
    dt: float = 0.02
    a_max: Optional[float] = None      # acceleration saturation (world units)
    v_max: Optional[float] = None      # speed saturation
    safety_filter: object = "orca"     # "orca" (adopted) | "cbf" (ablation) | "shrink" | False
    safety_backtracks: int = 8
    log_every: int = 5                 # steps between logged samples
    nb_every: int = 1                  # steps between neighbour-list rebuilds
    k_max: int = 32
    sat_rate: Optional[float] = None   # actuator lag (1/time); None = ideal
    meas_noise: float = 0.0            # std of the position noise seen by the controller
    eps_p: Optional[float] = None      # DETERMINISTIC position-error bound used by the FILTER
                                       # (separate from meas_noise: a std is not a hard bound)
    latency: int = 0                   # control delay in steps (zero-order hold)
    est_tau: Optional[float] = None    # state-estimator low-pass time constant (s)
    plane_z: Optional[float] = None
    forces_off: bool = False           # test hook: zero the nominal force (measurement audit)
    audit_k: object = None             # trajectory audit: None = FULL pairwise (no size fallback)
    seed: int = 0
    init: str = "lattice"              # lattice | ball | shell | box
    init_radius: Optional[float] = None
    init_spacing: Optional[float] = None   # lattice spacing (default 1.25 d_safe)
    stop_on_stable: bool = False
    stable_v: float = 5e-3
    stable_steps: int = 200
    min_steps: int = 200
    record_state: bool = True


@dataclass
class SimResult:
    time: np.ndarray
    pos: np.ndarray                # (T, n, 3) logged positions
    vel: np.ndarray                # (T, n, 3)
    energy: dict                   # logged scalars
    final: dict
    final_p: np.ndarray = None
    final_v: np.ndarray = None
    config: SimConfig = None
    params: Params = None

    @property
    def v(self) -> np.ndarray:
        """FINAL velocities (not the last logged sample)."""
        return self.final_v if getattr(self, "final_v", None) is not None else self.vel[-1]

    @property
    def p(self) -> np.ndarray:
        """FINAL positions (not the last logged sample)."""
        return self.final_p if getattr(self, "final_p", None) is not None else self.pos[-1]

    @property
    def v(self) -> np.ndarray:
        return self.vel[-1]


def initial_positions(n: int, cloud: PointCloud, cfg: SimConfig,
                      rng: np.random.Generator, d_safe: float = 0.0) -> np.ndarray:
    """Initial configuration.

    ``lattice`` (default): jittered cubic lattice inside a ball whose radius is
    the smallest one containing ``n`` sites at the requested spacing.  The
    lattice guarantees a pairwise distance ``>= spacing - 2*sqrt(3)*jitter``,
    so with the default ``spacing = 1.25 d_safe`` and ``jitter = 0.05 d_safe``
    the initial configuration is *safe* (every pair strictly farther than
    ``d_safe``).  This matters: the paper's uniform-random initialisation is
    almost surely unsafe for realistic ``d_safe`` (see docs/02), so safety
    cannot be claimed from it.

    ``ball``  : uniform in a ball of radius ``init_radius`` (the paper's).
    ``shell`` : uniform on a sphere of radius ``init_radius``.
    ``box``   : uniform in the inflated bounding box.
    """
    c = cloud.centroid
    R = cfg.init_radius if cfg.init_radius is not None else 2.5 * max(cloud.radius, 1e-9)
    if cfg.plane_z is not None:
        # planar mode: initialise inside the plane as well (a 3-D initialisation
        # projected onto the plane can be unsafe, which would freeze the filter)
        sp = cfg.init_spacing if cfg.init_spacing is not None else 1.25 * max(d_safe, 1e-9)
        jit = 0.05 * max(d_safe, 1e-9)
        Rb = R if cfg.init_radius is not None else (n * sp ** 2 / np.pi) ** 0.5 * 1.08
        while True:
            m = int(np.ceil(Rb / sp)) + 1
            g = np.arange(-m, m + 1) * sp
            X, Y = np.meshgrid(g, g, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel()], axis=1)
            pts = pts[np.linalg.norm(pts, axis=1) <= Rb]
            if len(pts) >= n:
                break
            Rb *= 1.15
        sel = rng.choice(len(pts), size=n, replace=False)
        xy = pts[sel] + rng.uniform(-jit, jit, size=(n, 2))
        p = np.concatenate([xy, np.full((n, 1), cfg.plane_z)], axis=1)
        return p + np.array([c[0], c[1], 0.0])
    if cfg.init == "lattice":
        sp = cfg.init_spacing if cfg.init_spacing is not None else 1.25 * max(d_safe, 1e-9)
        jit = 0.05 * max(d_safe, 1e-9)
        # radius that contains n cubic-lattice sites at spacing sp
        Rb = R if cfg.init_radius is not None else (3.0 * n * sp ** 3 / (4.0 * np.pi)) ** (1.0 / 3.0) * 1.08
        while True:
            m = int(np.ceil(Rb / sp)) + 1
            g = np.arange(-m, m + 1) * sp
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            pts = pts[np.linalg.norm(pts, axis=1) <= Rb]
            if len(pts) >= n:
                break
            Rb *= 1.15
        sel = rng.choice(len(pts), size=n, replace=False)
        pts = pts[sel] + rng.uniform(-jit, jit, size=(n, 3))
        return pts + c
    if cfg.init == "ball":
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        p *= R * rng.random(n)[:, None] ** (1.0 / 3.0)
        return p + c
    if cfg.init == "shell":
        p = rng.normal(size=(n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p * R + c
    if cfg.init == "box":
        lo, hi = cloud.bbox()
        return rng.uniform(lo - R, hi + R, size=(n, 3))
    raise ValueError(cfg.init)


class Simulator:
    """Stateful simulator; one instance per run."""

    def __init__(self, cloud: PointCloud, params: Params, cfg: SimConfig,
                 p0: Optional[np.ndarray] = None, v0: Optional[np.ndarray] = None):
        self.cloud = cloud
        self.params = params
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.p = (initial_positions(cfg.n_agents, cloud, cfg, self.rng, params.d_safe)
                  if p0 is None else np.array(p0, float))
        self.v = np.zeros_like(self.p) if v0 is None else np.array(v0, float)
        self.t = 0.0
        self.n = self.p.shape[0]
        self._norm_hint = None
        self._prev_total = np.zeros_like(self.p)
        self.n_filter_rejects = 0
        self.n_orca_fail = 0
        self.n_nom_sat = 0        # nominal command hit a bound (before the layer)
        self.n_exec_sat = 0       # executed command sits at a bound (after the layer)
        self.n_exec_viol = 0      # executed command VIOLATES a bound (must stay 0)
        self.n_steps_recorded = 0
        self.d_min_traj = _min_pair_distance(self.p)
        self._hist = []
        self._p_filt = self.p.copy()
        self._v_filt = self.v.copy()

    # -- one integration step -------------------------------------------
    def _perceived_state(self):
        """State available to the controller: delayed, noisy (mocap model)."""
        cfg = self.cfg
        # _hist already contains the state at the START of this step (appended in
        # step()), so a delay of `latency` steps is _hist[-(latency+1)].
        if cfg.latency and len(self._hist) >= cfg.latency + 1:
            p_meas, v_meas = self._hist[-(cfg.latency + 1)]
        else:
            p_meas, v_meas = self.p, self.v
        if cfg.meas_noise:
            p_meas = p_meas + self.rng.normal(scale=cfg.meas_noise, size=p_meas.shape)
            v_meas = v_meas + self.rng.normal(scale=cfg.meas_noise, size=v_meas.shape)
        if cfg.est_tau:
            alpha = 1.0 - np.exp(-self._dt / cfg.est_tau)
            self._p_filt = self._p_filt + alpha * (p_meas - self._p_filt)
            self._v_filt = self._v_filt + alpha * (v_meas - self._v_filt)
            return self._p_filt, self._v_filt
        return p_meas, v_meas

    def step(self, dt: float) -> dict:
        prm, cfg = self.params, self.cfg
        self._dt = dt
        p_old = self.p.copy()
        self._hist.append((self.p.copy(), self.v.copy()))
        if len(self._hist) > max(cfg.latency, 1) + 1:
            self._hist.pop(0)
        p_meas, v_meas = self._perceived_state()
        f = composite_forces(p_meas, v_meas, self.cloud, prm,
                             k_max=cfg.k_max, a_max=cfg.a_max,
                             use_alignment=True)
        self._norm_hint = f.n
        acc = np.zeros(3 * self.n).reshape(self.n, 3) if cfg.forces_off else f.total
        if cfg.sat_rate is not None:
            # first-order actuator lag on the commanded acceleration
            alpha = 1.0 - np.exp(-cfg.sat_rate * dt)
            self._prev_total = self._prev_total + alpha * (acc - self._prev_total)
            acc = self._prev_total
        v_new = self.v + dt * acc
        if cfg.v_max is not None:
            sp = np.linalg.norm(v_new, axis=1)
            over = sp > cfg.v_max
            v_new[over] *= (cfg.v_max / sp[over])[:, None]
        p_new = self.p + dt * v_new

        if cfg.plane_z is not None:
            # planar-formation mode: the altitude loop holds z (hard constraint)
            p_new[:, 2] = cfg.plane_z
            v_new[:, 2] = 0.0
        if cfg.v_max is not None and cfg.a_max is not None:
            _sp = np.linalg.norm(v_new, axis=1)
            _dv = np.linalg.norm(v_new - self.v, axis=1)
            self.n_nom_sat += int(np.sum((_sp >= 0.999 * cfg.v_max) |
                                         (_dv >= 0.999 * cfg.a_max * dt)))
        if cfg.safety_filter == "orca":
            # adopted safety layer (docs/18): upstream RVO2-3D, then clipped onto
            # the acceleration ball ("orca_limited").  The library has no
            # acceleration bound, so the clip is explicit and the standard ORCA
            # guarantee does NOT carry over -- this is empirical safety only.
            from .orca_adapter import orca_velocity
            try:
                v_orca, res = orca_velocity(p_meas, v_meas, v_new, dt=dt,
                                            r_vis=prm.r_vis_abs,
                                            d_safe=prm.d_safe, v_max=cfg.v_max or 1e9,
                                            margin=0.02)
            except Exception as exc:                # never fail silently
                self.n_orca_fail = getattr(self, "n_orca_fail", 0) + 1
                raise RuntimeError(f"ORCA layer unavailable/failed: {exc}") from exc
            v_new = np.array(v_orca, dtype=float)
            if cfg.v_max is not None:
                sp = np.linalg.norm(v_new, axis=1)
                v_new = v_new * np.minimum(1.0, cfg.v_max / np.maximum(sp, 1e-12))[:, None]
            if cfg.a_max is not None:
                dvb = v_new - self.v
                nd = np.linalg.norm(dvb, axis=1)
                v_new = self.v + dvb * np.minimum(1.0, cfg.a_max * dt / np.maximum(nd, 1e-12))[:, None]
            p_new = self.p + dt * v_new
        elif cfg.safety_filter == "shrink":
            p_new, v_new = self._safety_backtrack(p_new, v_new, dt)
        elif cfg.safety_filter:
            v_new = self._cbf_filter(v_new, dt, p_meas, v_meas)
            p_new = self.p + dt * v_new
        if cfg.v_max is not None and cfg.a_max is not None:
            _sp = np.linalg.norm(v_new, axis=1)
            _dv = np.linalg.norm(v_new - self.v, axis=1)
            self.n_exec_sat += int(np.sum((_sp >= 0.999 * cfg.v_max) |
                                          (_dv >= 0.999 * cfg.a_max * dt)))
            self.n_exec_viol += int(np.sum((_sp > 1.0001 * cfg.v_max) |
                                           (_dv > 1.0001 * cfg.a_max * dt)))
        self.n_steps_recorded += 1
        self.p, self.v = p_new, v_new
        self.t += dt
        from .metrics import trajectory_step_min
        self.d_min_traj = min(self.d_min_traj,
                              trajectory_step_min(p_old, self.p))
        return self._diagnostics()

    def _cbf_filter(self, v_des: np.ndarray, dt: float,
                    p_meas: np.ndarray | None = None,
                    v_meas: np.ndarray | None = None) -> np.ndarray:
        """无通信分布式鲁棒安全滤波（docs/12 的约束 (C) + 可行性 (F)）。

        每个机器人只用**自己的观测**（$p^m$）与**邻居速度的观测值** $v^m_j$，
        对邻居的**可达球** $\mathcal B(\hat c_j,r_j)$ 施加"端点必须在球外"的约束：

            || p_i^m + dt v_i - \hat c_j || >= rho_j = d_safe + r_j + eps_p .

        不需要通信、不读写邻居的候选速度；对任意满足 H2（|Δv|<=a_max dt）的
        邻居动作都成立。若 (F) 不成立 —— 即允许的端点球完全落在 rho 球内 ——
        该步是**不可安全恢复状态（ICS）**，执行**最大制动**（而非冻结，冻结违反 H2）。
        """
        p = self.p if p_meas is None else p_meas
        v_prev = self.v if v_meas is None else v_meas
        n = self.n
        if n < 2:
            return v_des
        prm, cfg = self.params, self.cfg
        eps_p = float(getattr(cfg, "eps_p", None) or 0.0)          # deterministic bound
        if eps_p <= 0.0:                                          # not declared: be explicit
            eps_p = 3.0 * float(getattr(cfg, "meas_noise", 0.0) or 0.0)
            self.eps_p_assumed = eps_p
        eps_v = 2.0 * eps_p / 0.02                                # velocity from position diff
        tau = float(getattr(cfg, "latency", 0) or 0) * dt         # information age
        vmax = float(getattr(cfg, "v_max", None) or 1e9)
        amax = float(getattr(cfg, "a_max", None) or 1e9)
        k = min(int(getattr(cfg, "k_max", 32)), n - 1)
        r_vis = float(getattr(cfg, "r_vis", None) or prm.r_vis_abs)
        d, idx = cKDTree(p).query(p, k=k + 1)
        d, idx = d[:, 1:], idx[:, 1:]
        mask = d <= r_vis
        r_j = vmax * tau + dt * eps_v + amax * (tau + dt) * dt + eps_p
        # ego side: its own measurement is stale by tau as well, so its true
        # current position differs by up to v_max*tau from the one it uses.
        rho = prm.d_safe + r_j + 2.0 * eps_p + vmax * tau
        c_hat = p[idx] + dt * v_prev[idx]                          # (n,k,3)

        v = np.array(v_des, dtype=float, copy=True)
        # ---- CONVEX sufficient condition (docs/12 §11):
        #   n_ij = (c_hat_j - p_i)/|.|,   n_ij^T v_i >= b_ij,  b_ij = (rho_j - |c_hat_j - p_i|)/dt
        # The half-space is a sufficient condition for the ball constraint. The
        # intersection is convex, so cyclic projection converges to a feasible
        # point whenever one exists (unlike the previous ball projection).
        nvec = p[:, None, :] - c_hat          # points AWAY from the ball centre
        dist_c = np.linalg.norm(nvec, axis=-1)
        nhat = nvec / np.maximum(dist_c, 1e-9)[..., None]
        b_req = (rho - dist_c) / dt
        # joint alternating projection onto
        #   {n_ij^T v >= b_ij} ∩ {|v| <= vmax} ∩ {|v - v_prev| <= amax dt}
        # (all convex: cyclic projections converge to a point of the intersection
        #  whenever it is non-empty).  No separate clipping afterwards.
        # ---- P1: velocity-level CBF (monotone) + reachable-set robustness.
        #  2 dp.(v_i - v_j^m) >= -h/dt - dt|v_i - v_j^m|^2,  h=|dp|^2-rho^2
        #  linearised with |v_i - v_j^m|^2 <= 4 vmax^2
        dp = p[:, None, :] - c_hat
        h = (dp * dp).sum(-1) - rho ** 2
        a_vec = 2.0 * dp
        b_req = -h / dt - 4.0 * dt * vmax ** 2 + 2.0 * (dp * v_prev[idx]).sum(-1)
        for _ in range(60):
            moved = False
            for sl in range(idx.shape[1]):
                m = mask[:, sl]
                if not m.any():
                    continue
                av = a_vec[:, sl]
                cur = (av * v).sum(1)
                need = np.where(m & (cur < b_req[:, sl]), b_req[:, sl] - cur, 0.0)
                if np.any(need > 0):
                    moved = True
                    v = v + (need / np.maximum((av * av).sum(1), 1e-12))[:, None] * av
            sp = np.linalg.norm(v, axis=1)
            if np.any(sp > vmax):
                moved = True
                v = v * np.minimum(1.0, vmax / np.maximum(sp, 1e-12))[:, None]
            dvb = v - v_prev
            nd = np.linalg.norm(dvb, axis=1)
            if np.any(nd > amax * dt):
                moved = True
                v = v_prev + dvb * np.minimum(1.0, amax * dt / np.maximum(nd, 1e-12))[:, None]
            if not moved:
                break
        # ICS: allowed endpoint ball entirely inside a rho ball  (F fails)
        e0 = p + dt * v_prev
        ics = np.zeros(n, dtype=bool)
        for sl in range(idx.shape[1]):
            m = mask[:, sl]
            if not m.any():
                continue
            dist = np.linalg.norm(e0 - c_hat[:, sl], axis=1)
            ics |= m & (dist + amax * dt * dt < rho)
        if ics.any():
            # NOTE: this is *robust infeasibility* of the conservative bound, NOT a
            # physical inevitable-collision state (docs/12 §4).
            self.n_filter_rejects += int(ics.sum())
            self.n_robust_infeas = getattr(self, "n_robust_infeas", 0) + int(ics.sum())
            spv = np.linalg.norm(v_prev, axis=1)
            factor = np.maximum(0.0, 1.0 - amax * dt / np.maximum(spv, 1e-12))
            brake = factor[:, None] * v_prev          # stops at zero, never reverses
            v = np.where(ics[:, None], brake, v)
        # 3) FINAL acceptance: the returned action must satisfy every constraint;
        # clipping and the multi-neighbour projections can undo each other.
        pn = p + dt * v
        viol_any = False
        for sl in range(idx.shape[1]):
            m = mask[:, sl]
            if not m.any():
                continue
            r = np.linalg.norm(pn - c_hat[:, sl], axis=1)
            if np.any(m & (r < rho)):
                viol_any = True
        # unrecoverable test: a single half-space already needs more than the
        # acceleration ball allows  (Delta = b - n.v_prev > a_max dt)
        active = mask & (b_req > 0)
        delta = np.where(active, b_req - (a_vec * v_prev[:, None, :]).sum(-1), -np.inf)
        self.n_unrecoverable = getattr(self, "n_unrecoverable", 0) + int(
            np.sum(np.any(delta > amax * dt, axis=1)))
        if viol_any:
            # PER-AGENT: only the agents that failed their own acceptance brake
            self.n_return_viol = getattr(self, "n_return_viol", 0) + 1
            self.n_filter_rejects += 1
            bad = np.zeros(n, dtype=bool)
            for sl in range(idx.shape[1]):
                m = mask[:, sl]
                if m.any():
                    bad |= m & (np.linalg.norm(pn - c_hat[:, sl], axis=1) < rho)
            spv = np.linalg.norm(v_prev, axis=1)
            factor = np.maximum(0.0, 1.0 - amax * dt / np.maximum(spv, 1e-12))
            v = np.where(bad[:, None], factor[:, None] * v_prev, v)
            # re-verify after the (local) brake
            pn2 = p + dt * v
            for sl in range(idx.shape[1]):
                m = mask[:, sl]
                if m.any() and np.any(m & (np.linalg.norm(pn2 - c_hat[:, sl], axis=1) < rho)):
                    self.n_brake_fail = getattr(self, "n_brake_fail", 0) + 1
                    break
        return v

    def _safety_backtrack(self, p_prop: np.ndarray, v_prop: np.ndarray, dt: float):
        """Shrink the step until the configuration is d_safe-separated.

        Guarantees (by construction) that every accepted configuration keeps a
        pairwise distance strictly larger than ``d_safe``; if no fraction of the
        step is safe the previous (safe) state is kept and the velocity is
        zeroed.  See ``docs/03_THEORY.md``, Proposition 3 (discrete safety).
        """
        d_safe = self.params.d_safe
        if self.n < 2:
            return p_prop, v_prop
        p0, v0 = self.p, self.v
        for j in range(self.cfg.safety_backtracks):
            h = 0.5 ** j
            p_try = p0 + h * (p_prop - p0)
            v_try = v0 + h * (v_prop - v0)
            if _min_pair_distance(p_try) > d_safe:
                return p_try, v_try
        self.n_filter_rejects += 1
        return p0.copy(), np.zeros_like(v0)

    # -- diagnostics ------------------------------------------------------
    def _diagnostics(self) -> dict:
        sp = np.linalg.norm(self.v, axis=1)
        return {"t": self.t, "v_rms": float(np.sqrt((sp ** 2).mean())),
                "v_max": float(sp.max()), "d_min": _min_pair_distance(self.p)}

    def energy(self) -> dict:
        return total_energy(self.p, self.cloud, self.params, k_max=self.cfg.k_max,
                            v=self.v)


def _min_pair_distance(p: np.ndarray) -> float:
    """Exact minimum pairwise distance (O(n log n)); inf for n < 2."""
    n = p.shape[0]
    if n < 2:
        return float("inf")
    
    tree = cKDTree(p)
    d, _ = tree.query(p, k=2)
    return float(d[:, 1].min())


def simulate(cloud: PointCloud, params: Params, cfg: SimConfig,
             p0: Optional[np.ndarray] = None, v0: Optional[np.ndarray] = None,
             callback: Optional[Callable[[Simulator], None]] = None,
             traj_path: Optional[str] = None) -> SimResult:
    """Run one closed-loop simulation and log positions, energy and diagnostics."""
    sim = Simulator(cloud, params, cfg, p0=p0, v0=v0)
    trace_p, trace_v, trace_t = [sim.p.copy()], [sim.v.copy()], [0.0]
    n_steps = int(round(cfg.t_end / cfg.dt))
    log_t, log_p, log_v = [], [], []
    keys = ["E", "E_surf", "E_pair", "E_rep", "KE", "V", "v_rms", "v_max",
            "d_min", "d_mean", "d_max"]
    logs: dict[str, list[float]] = {k: [] for k in keys}
    stable = 0
    for s in range(n_steps):
        diag = sim.step(cfg.dt)
        if traj_path is not None:
            trace_p.append(sim.p.copy()); trace_v.append(sim.v.copy()); trace_t.append(sim.t)
        if s % cfg.log_every == 0:
            e = sim.energy()
            log_t.append(sim.t)
            if cfg.record_state:
                log_p.append(sim.p.copy())
                log_v.append(sim.v.copy())
            for k in keys:
                logs[k].append(float({**e, **diag}[k]))
        if callback is not None:
            callback(sim)
        if cfg.stop_on_stable and s > cfg.min_steps:
            stable = stable + 1 if diag["v_max"] < cfg.stable_v else 0
            if stable >= cfg.stable_steps:
                break
    if traj_path is not None:
        with open(traj_path, "xb") as stream:
            np.savez_compressed(stream, pos=np.asarray(trace_p), vel=np.asarray(trace_v), time=np.asarray(trace_t))
    e_end = sim.energy()
    final = {"t_end": sim.t, **e_end, **_min_pair_diag(sim),
             "n_filter_rejects": sim.n_filter_rejects,
             "n_nom_sat": sim.n_nom_sat, "n_exec_sat": sim.n_exec_sat,
             "n_exec_viol": sim.n_exec_viol,
             "agent_steps": sim.n_steps_recorded * sim.n}
    pos = np.stack(log_p) if log_p else np.zeros((0, sim.n, 3))
    vel = np.stack(log_v) if log_v else np.zeros((0, sim.n, 3))
    return SimResult(time=np.asarray(log_t), pos=pos, vel=vel,
                     energy=logs, final=final, final_p=sim.p.copy(), final_v=sim.v.copy(),
                     config=cfg, params=params)


def _min_pair_diag(sim: Simulator) -> dict:
    sp = np.linalg.norm(sim.v, axis=1)
    return {"d_min": _min_pair_distance(sim.p),
            "d_min_traj": float(getattr(sim, "d_min_traj", _min_pair_distance(sim.p))),
            "n_robust_infeas": int(getattr(sim, "n_robust_infeas", 0)),
            "n_return_viol": int(getattr(sim, "n_return_viol", 0)),
            "n_unrecoverable": int(getattr(sim, "n_unrecoverable", 0)),
            "n_brake_fail": int(getattr(sim, "n_brake_fail", 0)),
            "v_rms": float(np.sqrt((sp ** 2).mean())),
            "v_max": float(sp.max()),
            "p_final": sim.p.copy()}
