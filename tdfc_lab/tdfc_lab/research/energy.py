"""Energy, forces and the composite control law of STFC-2.

The original STFC paper superposed five heuristic force terms.  This module
writes the *same physics* as the negative gradient of a single scalar
functional plus dissipative velocity terms, which is what makes the closed loop
analysable.

Scalar functional (all terms are state functions of the configuration ``p``)::

    E(p) = sum_i phi_s(d_i)                     surface conformity ("surface tension")
         + sum_{i<j} U_pair(r_ij)               cohesion + separation (Boids)
         + sum_{i<j} K_h(r_ij)                  density equalisation (mean shift)

    phi_s(d)      = (k_s/2) (d - d_h)^2                       quadratic well at d = d_h
    U_pair(r)     = LJ well in the shifted variable (r - d_s)  barrier at d_s, well at r_0
    K_h(r)        = (1 - (r/h)^2)^3  for r < h, else 0          polynomial bump

Control law::

    u_i = -k_s (d_i - d_h) n_i                          surface tension
        - sum_j U_pair'(r_ij) e_ij                      Boids pair interaction
        - g_i  sum_j K_h'(r_ij) e_ij                    density equalisation (mean shift)
    a_i =  k_al sum_j (v_j - v_i)                       velocity alignment (unnormalised Laplacian)
    vdot_i = u_i + a_i - c v_i

with ``e_ij = (p_i - p_j)/r_ij`` and ``g_i in (0, g_max]`` a bounded positive
*density-dependent gain*::

    g_i = clip(gamma / (S_i + eps), 0, g_max),     S_i = sum_j K_h(r_ij)

Two facts are used by the theory (see ``docs/03_THEORY.md``):

* every force term is the negative gradient of a scalar in ``E`` (a.e.); the
  switching of the nearest target point and of the neighbour set changes the
  gradient but not ``E`` itself, and ``E`` is semialgebraic (continuous,
  piecewise analytic), so the Kurdyka-Lojasiewicz framework applies;
* the mean-shift term is a *positively preconditioned* gradient of the bump
  energy (``g_i > 0``), so it is a descent direction of ``E`` even though it is
  not literally ``-dE/dp_i``.

All gains are expressed in *dimensionless* form relative to the lattice spacing
``a`` and a reference acceleration ``F0``.  As a result the closed loop is
exactly scale invariant: multiplying every length by ``lambda`` and time by
``sqrt(lambda)`` maps trajectories onto trajectories.  This is what makes the
same dimensionless parameter set usable from a 50 m simulation to a 1.5 m room.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import PointCloud

#: value of the extended pair potential on the unsafe set (r <= d_s)
WALL_FACTOR = 1e6
#: stiffness of the hard-wall repulsion on the unsafe set
WALL_STIFFNESS = 1e3

__all__ = [
    "Params",
    "bump_kernel",
    "bump_kernel_dr",
    "pair_potential",
    "pair_force_scalar",
    "surface_energy",
    "Forces",
    "composite_forces",
    "total_energy",
]


# ---------------------------------------------------------------------------
# Elementary potentials
# ---------------------------------------------------------------------------

def bump_kernel(r: np.ndarray | float, h: float) -> np.ndarray:
    """C^2 compactly supported bump ``(1-(r/h)^2)^3`` (Epanechnikov-like)."""
    r = np.asarray(r, dtype=np.float64)
    u = np.clip(r / h, 0.0, 1.0)
    return (1.0 - u * u) ** 3


def bump_kernel_dr(r: np.ndarray | float, h: float) -> np.ndarray:
    """d/dr of :func:`bump_kernel`."""
    r = np.asarray(r, dtype=np.float64)
    u = np.clip(r / h, 0.0, 1.0)
    inside = r < h
    return np.where(inside, -6.0 * (1.0 - u * u) ** 2 * u / h, 0.0)


def pair_potential(r, d_s, r0, r_c, kappa_p, k_b, lam_b) -> np.ndarray:
    """Pair potential: soft-sphere repulsion + thin boundary layer + hard wall.

    ``U(r) = kappa_p/2 (r_c - r)^2 + k_b (lam_b/(r-d_s) - 1)^4``  for d_s < r < d_s+lam_b

    The second term is a **divergent** boundary layer: it vanishes at
    ``r = d_s + lam_b`` and tends to +inf as ``r -> d_s^+``, so every sublevel set
    of the total energy is closed and bounded away from the unsafe set (this is
    what the invariance proof T1a needs; a finite barrier cannot give it).

    * the soft-sphere term is purely repulsive with range ``r_c`` and vanishing
      force at ``r_c``; it is the *separation* behaviour of Boids.  There is no
      attractive (cohesive) term: on a closed target surface the confinement is
      provided by the surface potential, and cohesion only pulls agents into
      patches (this is verified by the ablation in the paper);
    * the rational term is a *divergent* boundary layer of width ``lam_b``
      (zero at ``r = d_s + lam_b``, ``+inf`` at ``r = d_s``).  At spacings of
      interest ``r >= a = 1.6 d_s`` it is exactly zero, so it does not shift the
      equilibrium spacing; the divergence is what makes the safe set invariant
      in continuous time;
    * on the unsafe set ``r <= d_s`` the extended value ``WALL_FACTOR*k_b`` is
      returned: the idealised potential of the theory is ``+infinity`` there.

    All terms are polynomial/rational and compactly supported, hence
    semialgebraic: this is what lets the Kurdyka-Lojasiewicz argument apply to
    the closed loop.
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.full_like(r, WALL_FACTOR * k_b)
    m = (r > d_s) & (r < r_c)
    if np.any(m):
        rm = r[m]
        u = np.maximum(rm - d_s, 1e-12)
        bar = k_b * np.maximum(lam_b / u - 1.0, 0.0) ** 4  # divergent at u -> 0
        out[m] = 0.5 * kappa_p * (r_c - rm) ** 2 + bar
    out[r >= r_c] = 0.0
    return out


def pair_force_scalar(r, d_s, r0, r_c, kappa_p, k_b, lam_b) -> np.ndarray:
    """Scalar ``-dU/dr`` (positive = repulsive, along ``+e_ij``)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    wall = r <= d_s
    if np.any(wall):
        out[wall] = WALL_STIFFNESS * k_b / lam_b
    m = (r > d_s) & (r < r_c)
    if np.any(m):
        rm = r[m]
        u = np.maximum(rm - d_s, 1e-12)
        # d/dr of k_b (lam_b/u - 1)^4  =  -4 k_b lam_b (lam_b/u - 1)^3 / u^2
        bar = np.where(u < lam_b,
                       -4.0 * k_b * lam_b * np.maximum(lam_b / u - 1.0, 0.0) ** 3 / u ** 2,
                       0.0)
        out[m] = kappa_p * (r_c - rm) - bar      # -dU/dr  (sign: U_b increases as r decreases)
    return out


def _switch(r, r1, r_c):
    """C^2 switch: 1 for r <= r1, 0 with zero slope/curvature at r_c."""
    xi = np.clip((r - r1) / (r_c - r1), 0.0, 1.0)
    return 1.0 - xi ** 3 * (10.0 - 15.0 * xi + 6.0 * xi * xi)


def _switch_dr(r, r1, r_c):
    xi = np.clip((r - r1) / (r_c - r1), 0.0, 1.0)
    return -30.0 * xi ** 2 * (1.0 - xi) ** 2 / (r_c - r1)


def surface_energy(d: np.ndarray, d_h: float, k_s: float) -> np.ndarray:
    """Per-agent potential ``(k_s/2)(d-d_h)^2``."""
    return 0.5 * k_s * (d - d_h) ** 2


# ---------------------------------------------------------------------------
# Parameters (dimensionless)
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """Dimensionless control parameters plus the world-scale calibration.

    Dimensionless knobs (default values are used in every simulation unless
    stated otherwise; they satisfy the stability conditions of the theory):

    ``kappa_s``   surface stiffness            (force per unit length, /F0/a)
    ``eps_hat``   pair well depth              (energy, /F0*a)
    ``gamma_hat`` density equalisation gain    (energy, /F0*a)
    ``c_hat``     linear drag                  (1/time, *T0)
    ``k_al_hat``  alignment gain               (1/time, *T0)
    ``mu``        safety factor a/d_safe (>= 1 + margin)
    ``c_pair``    pair cutoff / a
    ``h_kernel``  mean-shift bandwidth / a
    ``r_vis``     neighbour sensing radius / a
    ``g_max_hat`` upper bound on the density gain (multiple of gamma)
    """

    # --- dimensionless ---
    kappa_s: float = 10.0
    kappa_p: float = 1.0
    kb_hat: float = 0.25
    lam_b: float = 0.1
    gamma_hat: float = 4.0
    c_hat: float = 1.0
    k_al_hat: float = 0.0     # alignment off by default (ablation: it hurts)
    mu: float = 1.6
    c_pair: float = 1.6
    h_kernel: float = 1.5
    r_vis: float = 2.5
    g_max_hat: float = 3.0
    d_h_factor: float = 0.0       # holding distance / a
    surface_multiscale: bool = False  # experimental reliable smaller-support fallback
    surface_geometry: bool = False  # experimental local-plane geometry
    soft_pair_only: bool = False  # certified guardian takes safety responsibility
    uncertain_tangent_gate: bool = False  # approach first when local geometry is unreliable
    critical_normal_damping: bool = False  # analytic flat-normal critical damping
    tangent_all_pair: bool = False  # experimental: requires independent certified governor
    tangent_soft_pair: bool = False  # experimental split: tangential soft, 3D near repulsion
    tangential_entropy: bool = True   # tangential projection of the density term
    use_lloyd: bool = False           # OPTIONAL: local Lloyd instead of mean shift (worse, see ablation)
    kappa_l: float = 2.0              # Lloyd gain (/T0)
    k_samples: int = 32               # target samples per agent for the local cell
    sat_rate: float = 0.0         # optional exponential smoothing (1/time, *T0)

    # --- world calibration ---
    a: float = 1.0                # lattice spacing (length unit)
    F0: float = 1.0               # reference acceleration (force unit, unit mass)
    mass: float = 1.0

    # --- derived (filled by __post_init__) ---
    d_safe: float = field(init=False)
    r0: float = field(init=False)
    r_c: float = field(init=False)
    h: float = field(init=False)
    r_vis_abs: float = field(init=False)
    k_s: float = field(init=False)
    k_p: float = field(init=False)
    k_b: float = field(init=False)
    lam_b_abs: float = field(init=False)
    gamma: float = field(init=False)
    c: float = field(init=False)
    k_al: float = field(init=False)
    k_l: float = field(init=False)
    g_max: float = field(init=False)
    d_h: float = field(init=False)
    T0: float = field(init=False)
    v_scale: float = field(init=False)
    sat_rate_abs: float = field(init=False)

    def __post_init__(self) -> None:
        a, F0, m = float(self.a), float(self.F0), float(self.mass)
        self.d_safe = a / self.mu
        self.r0 = a
        self.r_c = self.c_pair * a
        self.h = self.h_kernel * a
        self.r_vis_abs = self.r_vis * a
        self.d_h = self.d_h_factor * a
        self.T0 = np.sqrt(m * a / F0)
        self.v_scale = a / self.T0
        # force = kappa_s * F0/a * (d - d_h)
        self.k_s = self.kappa_s * F0 / a
        # energies scale as F0*a  (so that -dE/dp has units of F0)
        self.k_p = self.kappa_p * F0 / a
        self.k_b = self.kb_hat * F0 * a
        self.lam_b_abs = self.lam_b * a
        self.gamma = self.gamma_hat * F0 * a
        self.c = self.c_hat / self.T0
        self.k_al = self.k_al_hat / self.T0
        self.k_l = self.kappa_l / self.T0
        self.g_max = self.g_max_hat * self.gamma
        self.sat_rate_abs = self.sat_rate / self.T0

    # -- helpers ---------------------------------------------------------
    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()}
        d.update({k: getattr(self, k) for k in
                  ("d_safe", "r0", "r_c", "h", "k_s", "k_p", "k_b", "lam_b_abs",
                   "gamma", "c", "k_al", "g_max", "d_h")})
        return d

    def with_a(self, a: float) -> "Params":
        """Same dimensionless parameters at a different physical scale."""
        d = self.as_dict()
        keep = {k: d[k] for k in ("kappa_s", "kappa_p", "kb_hat", "lam_b", "gamma_hat", "c_hat",
                                  "k_al_hat", "mu", "c_pair", "h_kernel", "r_vis",
                                  "g_max_hat", "d_h_factor", "sat_rate", "surface_geometry", "tangent_soft_pair", "surface_multiscale", "tangent_all_pair", "critical_normal_damping", "uncertain_tangent_gate", "soft_pair_only")}
        return Params(a=float(a), F0=self.F0, mass=self.mass, **keep)


# ---------------------------------------------------------------------------
# Force evaluation
# ---------------------------------------------------------------------------

@dataclass
class Forces:
    total: np.ndarray
    surface: np.ndarray
    pair: np.ndarray
    entropy: np.ndarray
    align: np.ndarray
    drag: np.ndarray
    d: np.ndarray
    n: np.ndarray
    nn_dist: np.ndarray          # nearest-neighbour distance among agents
    # diagnostics
    d_min: float = float("inf")
    saturation_frac: float = 0.0


def _neighbour_lists(p: np.ndarray, r_vis: float, k_max: int):
    """K nearest neighbours within ``r_vis`` (padded arrays + mask)."""
    n = p.shape[0]
    tree = cKDTree(p)
    k = int(min(k_max + 1, n))
    dist, idx = tree.query(p, k=k, distance_upper_bound=r_vis)
    if k == 1:
        dist = dist[:, None]; idx = idx[:, None]
    mask = (np.isfinite(dist)) & (idx < n) & (idx != np.arange(n)[:, None])
    idx = np.where(mask, idx, 0)
    dist = np.where(mask, dist, np.inf)
    return dist, idx, mask


def composite_forces(p: np.ndarray, v: np.ndarray, cloud: PointCloud,
                     params: Params, k_max: int = 32,
                     target_tree: Optional[cKDTree] = None,
                     a_max: Optional[float] = None,
                     v_max: Optional[float] = None,
                     adaptive_gain: bool = False,
                     use_alignment: bool = True,
                     use_entropy: bool = True,
                     use_surface: bool = True,
                     use_pair: bool = True,
                     q_hint: Optional[np.ndarray] = None) -> Forces:
    """Evaluate all force terms at configuration ``(p, v)`` (vectorised)."""
    n = p.shape[0]
    tree = target_tree or cloud.tree
    d, qi = tree.query(p, k=1)
    q = cloud.points[qi]
    diff = p - q
    dn = np.linalg.norm(diff, axis=1)
    nrm = diff / np.maximum(dn[:, None], 1e-12)
    # at d = 0 the normal is undefined; use the previous normal if given
    if q_hint is not None:
        bad = dn < 1e-9
        nrm[bad] = q_hint[bad]

    base_nrm = nrm
    plane_base = None
    geometry_weight = None
    if params.surface_geometry:
        from .local_surface import local_surface
        plane_q, plane_n, geometry_weight = local_surface(cloud, p, params.a, multiscale=params.surface_multiscale)
        phi = np.sum((p-plane_q)*plane_n, axis=1)
        # For nonzero offsets an oriented surface is required; reject ambiguous use.
        if params.d_h != 0:
            raise ValueError('local-plane geometry currently supports zero surface offset only')
        plane_base = -params.k_s*phi[:,None]*plane_n
        aligned = plane_n*np.where(np.sum(plane_n*nrm,axis=1)<0,-1.,1.)[:,None]
        mixed = (1-geometry_weight[:,None])*nrm+geometry_weight[:,None]*aligned
        nrm = mixed/np.maximum(np.linalg.norm(mixed,axis=1)[:,None],1e-12)

    # ---- surface tension ------------------------------------------------
    if use_surface:
        # adaptive gain: positive schedule in (0, 1], zero derivative issues
        base = -params.k_s * (d - params.d_h)[:, None] * base_nrm
        if plane_base is not None:
            base = (1-geometry_weight[:,None])*base+geometry_weight[:,None]*plane_base
    else:
        base = np.zeros_like(p)

    # ---- neighbours -----------------------------------------------------
    dist, idx, mask = _neighbour_lists(p, params.r_vis_abs, k_max)
    r = np.where(mask, dist, 1.0)
    pj = p[idx]                                            # (n, k, 3)
    eij = (p[:, None, :] - pj) / np.maximum(r, 1e-12)[:, :, None]             # unit vectors
    eij = np.where(mask[:, :, None], eij, 0.0)
    r = np.where(mask, dist, 0.0)

    # pairwise force: sum_j (-dU/dr)(r_ij) e_ij
    if use_pair:
        if params.soft_pair_only:
            fscalar=np.where((r>0)&(r<params.r_c),params.k_p*(params.r_c-r),0.)
        else:
            fscalar = pair_force_scalar(r, params.d_safe, params.r0, params.r_c,
                                        params.k_p, params.k_b, params.lam_b_abs)
        fscalar = np.where(mask, fscalar, 0.0)
        pair_f = (fscalar[:, :, None] * eij).sum(axis=1)
        if params.tangent_soft_pair:
            # Match the exact soft term used by pair_force_scalar; retain its
            # full three-dimensional wall/boundary force, including r<=d_safe.
            soft = np.where(mask & (r>params.d_safe) & (r<params.r_c),
                            params.k_p*(params.r_c-r),0.)
            soft_f = (soft[:,:,None]*eij).sum(axis=1)
            pair_f = pair_f - np.sum(soft_f*nrm,axis=1)[:,None]*nrm
        if params.tangent_all_pair:
            pair_f = pair_f - np.sum(pair_f*nrm,axis=1)[:,None]*nrm
    else:
        pair_f = np.zeros_like(p)

    # ---- density equalisation ------------------------------------------
    S = np.zeros(n)
    ent_f = np.zeros_like(p)
    if use_entropy and params.use_lloyd:
        # local Lloyd / quantisation flow: each agent moves to the centroid of
        # its own local Voronoi cell, computed from the k nearest target
        # samples and the agents inside the sensing radius only.
        ks = int(min(params.k_samples, cloud.m))
        ds, si = tree.query(p, k=ks)
        qs = cloud.points[si]                                   # (n, ks, 3)
        cand = np.concatenate([p[:, None, :], p[idx]], axis=1)   # (n, 1+k, 3)
        cmask = np.concatenate([np.ones((n, 1), bool), mask], axis=1)
        diff2 = ((qs[:, :, None, :] - cand[:, None, :, :]) ** 2).sum(-1)
        diff2 = np.where(cmask[:, None, :], diff2, np.inf)
        mine = diff2[:, :, 0] <= diff2.min(axis=2)               # (n, ks)
        w = mine.sum(axis=1)
        cent = (qs * mine[:, :, None]).sum(axis=1) / np.maximum(w, 1)[:, None]
        f_l = params.k_l * (cent - p)
        f_l *= (w >= 3)[:, None]
        if params.tangential_entropy:
            f_l = f_l - (f_l * nrm).sum(axis=1, keepdims=True) * nrm
        ent_f = f_l
    elif use_entropy:
        K = np.where(mask, bump_kernel(r, params.h), 0.0)
        S = K.sum(axis=1)
        dK = np.where(mask, bump_kernel_dr(r, params.h), 0.0)
        if adaptive_gain:
            # state-dependent gain: kept for the ablation only.  With this flag
            # the density force is NOT -grad E_rep (the gain depends on p), so
            # the Lyapunov argument of docs/03 only applies with a constant gain.
            g = np.clip(params.gamma / (S + 1e-12), 0.0, params.g_max)
        else:
            g = np.full(n, params.gamma)
        # repulsion = -dE_rep/dp_i with E_rep = sum_{i<j} K_h(r_ij);
        # optionally projected on the target tangent plane so that the
        # density-equalisation flow redistributes agents *along* the surface
        # instead of pushing them off it (P = I - n n^T is PSD, so the term
        # remains a descent direction of E_rep).
        ent_f = (-(g[:, None] * dK)[:, :, None] * eij).sum(axis=1)
        if params.tangential_entropy:
            ent_f = ent_f - (ent_f * nrm).sum(axis=1, keepdims=True) * nrm

    if params.uncertain_tangent_gate:
        if geometry_weight is None or not params.tangent_all_pair:
            raise ValueError('uncertain tangent gating requires local geometry and guarded all-pair projection')
        # Half the fixed 0.1a fitting criterion; not shape/seed-specific tuning.
        approach_gain=geometry_weight+(1-geometry_weight)/(1+(d/(.05*params.a))**4)
        pair_f=pair_f*approach_gain[:,None]
        ent_f=ent_f*approach_gain[:,None]

    # ---- alignment (unnormalised graph Laplacian) -----------------------
    align = np.zeros_like(p)
    if use_alignment and params.k_al != 0.0:
        dv = np.where(mask[:, :, None], v[idx] - v[:, None, :], 0.0)
        align = params.k_al * dv.sum(axis=1)

    drag = -params.c * v
    if params.critical_normal_damping:
        normal_c=max(params.c,2*np.sqrt(params.k_s))
        drag -= (normal_c-params.c)*np.sum(v*nrm,axis=1)[:,None]*nrm
    f = base + pair_f + ent_f + align + drag

    sat = 0.0
    if a_max is not None:
        mag = np.linalg.norm(f, axis=1)
        over = mag > a_max
        sat = float(over.mean())
        f[over] *= (a_max / mag[over])[:, None]

    nn = np.where(mask[:, 1:], dist[:, 1:], np.inf).min(axis=1) if r.shape[1] > 1 else np.full(n, np.inf)
    return Forces(total=f, surface=base, pair=pair_f, entropy=ent_f, align=align,
                  drag=drag, d=d, n=nrm, nn_dist=nn,
                  d_min=float(nn.min()), saturation_frac=sat)


# ---------------------------------------------------------------------------
# Energy (for diagnostics / Lyapunov monitoring)
# ---------------------------------------------------------------------------

def total_energy(p: np.ndarray, cloud: PointCloud, params: Params,
                 k_max: int = 32, target_tree: Optional[cKDTree] = None,
                 v: Optional[np.ndarray] = None) -> dict:
    """Evaluate ``E = E_surf + E_pair + E_rep`` and its three components.

    ``E_rep`` is defined as ``sum_{i<j} K_h(r_ij)`` (density equalisation) and
    ``E_pair`` as ``sum_{i<j} U_pair(r_ij)``.  Edge sums are accumulated over
    the directed neighbour lists and halved; the lists are symmetric because
    the cutoff ``r_c`` is well below the ``k_max``-th neighbour distance.
    """
    tree = target_tree or cloud.tree
    d, _ = tree.query(p, k=1)
    e_surf = float(surface_energy(d, params.d_h, params.k_s).sum())
    dist, idx, mask = _neighbour_lists(p, params.r_c, k_max)
    r = np.where(mask, dist, 0.0)
    if params.soft_pair_only:
        up=np.where(mask,.5*params.k_p*np.maximum(params.r_c-r,0.)**2,0.)
    else:
        up = np.where(mask, pair_potential(r, params.d_safe, params.r0, params.r_c,
                                           params.k_p, params.k_b, params.lam_b_abs), 0.0)
    uk = np.where(mask, bump_kernel(r, params.h), 0.0)
    e_pair = float(0.5 * up.sum())
    e_ent = float(0.5 * uk.sum())
    out = {"E_surf": e_surf, "E_pair": e_pair, "E_rep": e_ent,
           "E": e_surf + e_pair + e_ent, "d_mean": float(d.mean()),
           "d_max": float(d.max())}
    if v is not None:
        ke = 0.5 * float((v ** 2).sum())
        out["KE"] = ke
        out["V"] = out["E"] + ke
    return out
