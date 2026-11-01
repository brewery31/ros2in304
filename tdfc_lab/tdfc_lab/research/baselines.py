"""Baselines and reference solutions.

The paper compares against ablation-only baselines, which the reviewers
correctly rejected as unconvincing.  This module provides the baselines that a
reviewer of a shape-formation paper expects:

``fp_surface``   farthest-point sampling of the target (a centralised blue-noise
                 reference); paired with deadbeat assignment it is the
                 "perfect assignment" upper bound.
``cvt_surface``  centroidal Voronoi tessellation on the surface (Lloyd's
                 algorithm).  This is the *performance ceiling* of the whole
                 coverage family: it needs global state (all agents' positions)
                 and is therefore centralised, but it is method-agnostic and is
                 the standard reference for "how well can N points cover a
                 surface".
``boids_target`` the original STFC force superposition (cohesion + separation +
                 alignment + nearest-target attraction + mean shift + damping)
                 reimplemented as a plain force law, i.e. the paper's method.
``random_surface``  uniform random placement on the target samples.

Every routine returns positions on (or very close to) the target surface so
that the comparison is about the *distribution*, not about hovering height.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import PointCloud, _fps

__all__ = ["fp_surface", "cvt_surface", "cvt_safe", "local_cvt", "random_surface",
           "covering_radius", "BoidsTarget"]


def covering_radius(p: np.ndarray, cloud: PointCloud,
                    max_target_samples: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> float:
    """max over target samples of the distance to the nearest agent."""
    q = cloud.points
    if max_target_samples is not None and cloud.m > max_target_samples:
        rng = np.random.default_rng(rng)
        q = q[rng.choice(cloud.m, max_target_samples, replace=False)]
    return float(cKDTree(p).query(q, k=1)[0].max())


def fp_surface(cloud: PointCloud, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Farthest-point sample of the target point cloud (centralised)."""
    rng = np.random.default_rng(rng)
    idx = _fps(cloud.points, int(n), rng)
    return cloud.points[idx].copy()


def random_surface(cloud: PointCloud, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    idx = rng.choice(cloud.m, int(n), replace=False)
    return cloud.points[idx].copy()


def cvt_surface(cloud: PointCloud, n: int, iters: int = 60,
                rng: Optional[np.random.Generator] = None,
                init: Optional[np.ndarray] = None,
                tol: float = 1e-9) -> tuple[np.ndarray, dict]:
    """Lloyd / CVT on the surface, discretised with the target samples.

    Assignment is Euclidean nearest (a good approximation to the geodesic
    Voronoi diagram for dense samples and radii small compared with the
    curvature radius).  Each generator is moved to the centroid of its assigned
    target samples and re-projected on the surface, which is the standard
    "CVT on a point cloud" construction.  Returns the generators and a small
    history dict.
    """
    rng = np.random.default_rng(rng)
    p = fp_surface(cloud, n, rng) if init is None else np.array(init, dtype=np.float64)
    q = cloud.points
    hist = {"cover": [], "quant": []}
    for it in range(iters):
        d, idx = cKDTree(p).query(q, k=1)
        # centroid of each cell; empty cells keep their generator
        cnt = np.bincount(idx, minlength=len(p)).astype(np.float64)
        new = np.zeros_like(p)
        for k in range(3):
            new[:, k] = np.bincount(idx, weights=q[:, k], minlength=len(p))
        nonempty = cnt > 0
        new[nonempty] /= cnt[nonempty, None]
        # re-project on the surface
        d2, j = cloud.tree.query(new, k=1)
        new[nonempty] = cloud.points[j[nonempty]]
        shift = np.linalg.norm(new - p, axis=1).max()
        p = new
        hist["cover"].append(covering_radius(p, cloud))
        hist["quant"].append(float(np.mean(d ** 2)))
        if shift < tol:
            break
    return p, hist


def _separate(p: np.ndarray, cloud: PointCloud, d_min: float, iters: int = 60) -> np.ndarray:
    """Push apart every pair closer than ``d_min`` and re-project on the surface."""
    p = p.copy()
    for _outer in range(6):
        for _ in range(max(1, iters // 6)):
            pairs = cKDTree(p).query_pairs(d_min, output_type="ndarray")
            if len(pairs) == 0:
                break
            i, j = pairs[:, 0], pairs[:, 1]
            d = p[i] - p[j]
            r = np.linalg.norm(d, axis=1, keepdims=True)
            push = 0.5 * (d_min - r + 1e-9) * d / np.maximum(r, 1e-12)
            np.add.at(p, i, push)
            np.add.at(p, j, -push)
        if len(cKDTree(p).query_pairs(d_min)) == 0:
            break
    return p


def cvt_safe(cloud: PointCloud, n: int, d_safe: float, iters: int = 60,
             margin: float = 0.05, rng: Optional[np.random.Generator] = None,
             init: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """Centroidal Voronoi tessellation *with a minimum-distance constraint*.

    Lloyd iterations interleaved with a pairwise separation projection, so the
    resulting configuration satisfies ``min_{i!=j} |p_i - p_j| >= d_safe`` while
    staying as close as possible to the CVT.  This is the "safe centralised
    ceiling": what a global optimiser can do once it is held to the same safety
    requirement as the distributed methods.
    """
    p, hist = cvt_surface(cloud, n, iters=iters // 2, rng=rng, init=init)
    p = _separate(p, cloud, d_safe * (1 + margin))
    p, _ = cvt_surface(cloud, n, iters=iters // 2, rng=rng, init=p, tol=1e-12)
    p = _separate(p, cloud, d_safe * (1 + margin))
    hist["d_min"] = float(cKDTree(p).query(p, k=2)[0][:, 1].min())
    return p, hist


def local_cvt(cloud: PointCloud, n: int, a: float, k: int = 9,
              r_cell: float = 1.6, iters: int = 25,
              rng: Optional[np.random.Generator] = None,
              init: Optional[np.ndarray] = None, merge: float = 0.6) -> tuple[np.ndarray, dict]:
    """Distributed (one-hop) approximation of Lloyd's algorithm.

    Each generator may only use (i) the positions of its ``k`` nearest
    neighbours and (ii) the target samples within ``r_cell * a`` of itself.
    Its cell is the set of those samples for which it is the closest among
    {itself} U neighbours; it then moves to the cell centroid (``merge`` is the
    step size).  This is the *same information class* as our method and is the
    natural strong distributed baseline.
    """
    rng = np.random.default_rng(rng)
    p = fp_surface(cloud, n, rng) if init is None else np.array(init, dtype=float)
    R = r_cell * a
    for _ in range(iters):
        tree_p = cKDTree(p)
        _, nb = tree_p.query(p, k=min(k + 1, n))
        new = p.copy()
        for i in range(n):
            q_idx = cloud.tree.query_ball_point(p[i], R)
            if not q_idx:
                continue
            q = cloud.points[q_idx]
            cand = np.vstack([p[i][None, :], p[nb[i, 1:]]])
            d2 = ((q[:, None, :] - cand[None, :, :]) ** 2).sum(-1)
            mine = d2[:, 0] <= d2.min(axis=1)
            if mine.sum() >= 3:
                new[i] = p[i] + merge * (q[mine].mean(0) - p[i])
        _, kk = cloud.tree.query(new, k=1)
        p = cloud.points[kk]
    return p, {"iters": iters}


class BoidsTarget:
    """The original STFC force law, reimplemented as in the ICRA submission.

    * cohesion      ``a_c * mean_j (p_j - p_i)``
    * separation    ``a_s * sum_{|p_i-p_j|<r_safe} (p_i-p_j)/|p_i-p_j|^2``
    * alignment     ``a_a * mean_j (v_j - v_i)``
    * surface       ``beta * (q* - p) * (|q*-p| - d_h)`` (attraction / repulsion)
    * mean shift    inverse mean shift with an Epanechnikov kernel
    * damping       conditional: ``k (v + dt F)`` when ``|v| < mean_j |v_j|``

    This is deliberately *not* our algorithm: it keeps the original
    position-dependent gains, the normalised alignment, the hard separation
    cutoff and the conditional damping, so the comparison isolates the effect of
    the reformulation (energy-based forces, tangent-space equalisation,
    unnormalised Laplacian, safety filter).
    """

    def __init__(self, cloud: PointCloud, a_lattice: float, d_safe: float,
                 d_h: float = 0.0, alpha_c: float = 0.008, alpha_a: float = 0.05,
                 alpha_s: float = 0.05, beta_a: float = 0.02, beta_r: float = 0.02,
                 gamma_ms: float = 0.05, kappa_damp: float = 0.9,
                 r_vis: Optional[float] = None, bandwidth: Optional[float] = None,
                 max_speed: float = 2.0):
        self.cloud = cloud
        self.a = a_lattice
        self.d_safe = d_safe
        self.d_h = d_h
        self.alpha_c, self.alpha_a, self.alpha_s = alpha_c, alpha_a, alpha_s
        self.beta_a, self.beta_r = beta_a, beta_r
        self.gamma_ms = gamma_ms
        self.kappa_damp = kappa_damp
        self.r_vis = 2.5 * a_lattice if r_vis is None else r_vis
        self.h = 1.5 * a_lattice if bandwidth is None else bandwidth
        self.v_max = max_speed

    def forces(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = len(p)
        tree = cKDTree(p)
        dist, idx = tree.query(p, k=min(n, 24), distance_upper_bound=self.r_vis)
        mask = np.isfinite(dist) & (idx < n) & (idx != np.arange(n)[:, None])
        idx = np.where(mask, idx, 0)
        dd = np.where(mask, dist, np.inf)

        with np.errstate(invalid="ignore"):
            coh = self.alpha_c * np.where(mask[:, :, None],
                                          (p[idx] - p[:, None, :]), 0.0).sum(1)
            coh /= np.maximum(mask.sum(1, keepdims=True), 1)
            al = self.alpha_a * np.where(mask[:, :, None], (v[idx] - v[:, None, :]), 0.0).sum(1)
            al /= np.maximum(mask.sum(1, keepdims=True), 1)
            sep_close = mask & (dd < self.d_safe)
            sep = np.where(sep_close[:, :, None],
                           (p[:, None, :] - p[idx]) / np.maximum(dd[:, :, None] ** 2, 1e-12), 0.0).sum(1)
            sep *= self.alpha_s

        d, qi = self.cloud.tree.query(p, k=1)
        q = self.cloud.points[qi]
        vec = q - p                                    # paper: (q* - p)
        dn = np.linalg.norm(vec, axis=1, keepdims=True)
        unit = vec / np.maximum(dn, 1e-12)
        surf = np.where(dn > self.d_h, self.beta_a, self.beta_r) * vec * (dn - self.d_h)
        surf = np.where(np.abs(dn - self.d_h) > 1e-9, surf, 0.0)

        K = np.where(mask & (dd < self.h), np.maximum(1.0 - dd ** 2 / self.h ** 2, 0.0), 0.0)
        denom = K.sum(1, keepdims=True)
        # With no kernel-supported neighbors the mean-shift vector is zero.
        # Dividing a zero numerator by eps and subtracting p would create an
        # artificial, origin-dependent outward force on isolated agents.
        centroid = np.divide((K[:, :, None] * p[idx]).sum(1), denom,
                             out=p.copy(), where=denom > 0)
        mshift = centroid - p                                  # classical mean shift
        ms = -self.gamma_ms * mshift                             # inverse (repulsive)
        return coh + sep + al + surf + ms

    def step(self, p: np.ndarray, v: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        f = self.forces(p, v)
        v_new = v + dt * f
        sp = np.linalg.norm(v_new, axis=1)
        over = sp > self.v_max
        v_new[over] *= (self.v_max / sp[over])[:, None]
        # conditional damping exactly as in the paper
        speed = np.linalg.norm(v, axis=1)
        tree = cKDTree(p)
        dist, idx = tree.query(p, k=min(len(p), 12), distance_upper_bound=self.r_vis)
        mask = np.isfinite(dist) & (idx < len(p)) & (idx != np.arange(len(p))[:, None])
        vv = np.where(mask, np.linalg.norm(v[np.where(mask, idx, 0)], axis=2), 0.0)
        mean_nb = vv.sum(1) / np.maximum(mask.sum(1), 1)
        slow = speed < mean_nb
        v_new[slow] *= self.kappa_damp
        return p + dt * v_new, v_new
