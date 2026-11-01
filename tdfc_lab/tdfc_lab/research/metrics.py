"""Evaluation metrics for surface-covering swarms.

The original paper reported SCR / SUR / CCE with a hand-chosen ``r_cov``.  Here
the radius is tied to the design rule (``r_cov = a/sqrt(3)``) and we add the
scale-free and distribution-shape metrics that the point-sampling literature
uses (Chamfer, quantisation energy, nearest-neighbour spacing statistics, the
pair-correlation function).  See ``docs/04_EVALUATION.md`` for the rationale and
``docs/research/C_EVAL_AND_SIM2REAL.md`` for the literature the metric set is
based on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .geometry import PointCloud

__all__ = [
    "coverage_metrics",
    "chamfer",
    "hausdorff",
    "quantization_energy",
    "riesz_energy",
    "nn_stats",
    "pair_correlation",
    "min_pair_distance",
    "coverage_profile",
    "settling_time",
]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_metrics(p: np.ndarray, cloud: PointCloud, r: float,
                     target_tree: Optional[cKDTree] = None,
                     max_target_samples: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None) -> dict:
    """SCR / SUR / CCE at a single radius (definitions as in the paper).

    * ``scr`` -- fraction of target samples with an agent within ``r``
    * ``sur`` -- fraction of agents with a target sample within ``r``
    * ``cce`` -- ``scr * sur``
    """
    tree = target_tree or cloud.tree
    q = cloud.points
    if max_target_samples is not None and cloud.m > max_target_samples:
        rng = np.random.default_rng(rng)
        idx = rng.choice(cloud.m, max_target_samples, replace=False)
        q = q[idx]
    d_pq, _ = tree.query(p, k=1)
    d_qp, _ = cKDTree(p).query(q, k=1)
    scr = float(np.mean(d_qp <= r))
    sur = float(np.mean(d_pq <= r))
    return {"scr": scr, "sur": sur, "cce": scr * sur, "r_cov": float(r),
            "sur_util": float(np.mean(d_pq <= r))}


def coverage_profile(p: np.ndarray, cloud: PointCloud, radii: Sequence[float],
                     target_tree: Optional[cKDTree] = None,
                     max_target_samples: int = 4000,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """SCR at several radii (the scale-free coverage curve)."""
    tree = target_tree or cloud.tree
    rng = np.random.default_rng(rng)
    q = cloud.points
    if cloud.m > max_target_samples:
        q = q[rng.choice(cloud.m, max_target_samples, replace=False)]
    d_qp, _ = cKDTree(p).query(q, k=1)
    return np.array([float(np.mean(d_qp <= r)) for r in radii])


# ---------------------------------------------------------------------------
# Shape agreement
# ---------------------------------------------------------------------------

def chamfer(p: np.ndarray, cloud: PointCloud, target_tree: Optional[cKDTree] = None,
            max_target_samples: int = 4000, rng: Optional[np.random.Generator] = None) -> float:
    """Symmetric Chamfer distance (mean nearest-neighbour distance, both ways)."""
    tree = target_tree or cloud.tree
    rng = np.random.default_rng(rng)
    q = cloud.points
    if cloud.m > max_target_samples:
        q = q[rng.choice(cloud.m, max_target_samples, replace=False)]
    d_pq, _ = tree.query(p, k=1)
    d_qp, _ = cKDTree(p).query(q, k=1)
    return float(0.5 * (d_pq.mean() + d_qp.mean()))


def hausdorff(p: np.ndarray, cloud: PointCloud, target_tree: Optional[cKDTree] = None,
              max_target_samples: int = 4000, rng: Optional[np.random.Generator] = None) -> float:
    """Symmetric Hausdorff distance."""
    tree = target_tree or cloud.tree
    rng = np.random.default_rng(rng)
    q = cloud.points
    if cloud.m > max_target_samples:
        q = q[rng.choice(cloud.m, max_target_samples, replace=False)]
    d_pq, _ = tree.query(p, k=1)
    d_qp, _ = cKDTree(p).query(q, k=1)
    return float(max(d_pq.max(), d_qp.max()))


def quantization_energy(p: np.ndarray, cloud: PointCloud,
                        target_tree: Optional[cKDTree] = None,
                        max_target_samples: int = 4000,
                        rng: Optional[np.random.Generator] = None) -> float:
    """Mean squared distance from a target sample to its nearest agent.

    This is the (sampled) quantisation / CVT energy; it is the natural
    objective that a well-spread covering minimises and is *not* normalised by
    the arbitrary coverage radius.
    """
    tree = target_tree or cloud.tree
    rng = np.random.default_rng(rng)
    q = cloud.points
    if cloud.m > max_target_samples:
        q = q[rng.choice(cloud.m, max_target_samples, replace=False)]
    d_qp, _ = cKDTree(p).query(q, k=1)
    return float(np.mean(d_qp ** 2))


def riesz_energy(p: np.ndarray, s: float = 1.0, r_max: Optional[float] = None) -> float:
    """Riesz s-energy per agent; low values mean a well-spread (blue-noise) set."""
    tree = cKDTree(p)
    dd, _ = tree.query(p, k=min(len(p), 16))
    dd = dd[:, 1:]
    dd = np.where(dd > 1e-12, dd, np.inf)
    if r_max is not None:
        dd = np.where(dd < r_max, dd, np.inf)
    with np.errstate(divide="ignore"):
        e = np.where(np.isfinite(dd), dd ** (-s), 0.0)
    return float(e.sum() / len(p))


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------

def nn_stats(p: np.ndarray, k: int = 1) -> dict:
    """Nearest-neighbour distance statistics among agents (uniformity)."""
    tree = cKDTree(p)
    d, _ = tree.query(p, k=k + 1)
    dn = d[:, k]
    return {"mean": float(dn.mean()), "std": float(dn.std()),
            "cv": float(dn.std() / max(dn.mean(), 1e-12)),
            "p05": float(np.percentile(dn, 5)), "p95": float(np.percentile(dn, 95))}


def pair_correlation(p: np.ndarray, r_max: float, n_bins: int = 40,
                     area: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic pair-correlation function ``g(r)`` of the agent positions.

    For a homogeneous 2-D set of density ``rho = n/area``, ``g(r) = 1`` for a
    Poisson process and ``g(r) ~ 0`` for ``r`` below the typical spacing with a
    first peak at the lattice spacing for a blue-noise / hexagonal set.
    """
    n = len(p)
    tree = cKDTree(p)
    dd, _ = tree.query(p, k=min(n, 64))
    dd = dd[:, 1:].ravel()
    dd = dd[np.isfinite(dd) & (dd < r_max)]
    hist, edges = np.histogram(dd, bins=n_bins, range=(0.0, r_max))
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 2 * np.pi * centres * (edges[1] - edges[0])
    rho = n / area if area else 1.0
    # each unordered pair counted twice
    g = hist / (n * rho * shell * 0.5)
    return centres, g


def min_pair_distance(p: np.ndarray) -> float:
    if len(p) < 2:
        return float("inf")
    d, _ = cKDTree(p).query(p, k=2)
    return float(d[:, 1].min())


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def settling_time(t: np.ndarray, series: np.ndarray, tol: float,
                  window: int = 5) -> float:
    """First time after which ``series`` stays within ``tol`` of its final value."""
    series = np.asarray(series)
    final = series[-1]
    ok = np.abs(series - final) <= tol
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            return float(t[min(i + 1, len(t) - 1)])
    return float(t[0])


@dataclass
class RunMetrics:
    """Bundle of every scalar reported for one simulation run."""

    scr: float
    sur: float
    cce: float
    r_cov: float
    chamfer: float
    hausdorff: float
    quant: float
    riesz: float
    nn_mean: float
    nn_cv: float
    d_min: float
    d_safe: float
    safe: bool
    d_mean: float
    t_conv: float
    t_end: float
    energy_final: float
    lyapunov_monotone_frac: float
    v_end: float
    n_agents: int
    area: float
    a_lattice: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_run(res, cloud: PointCloud, *, r_cov: float, a_lattice: float,
                 d_safe: float, conv_tol: float = 2e-3) -> RunMetrics:
    """Compute the full metric bundle for a finished :class:`SimResult`."""
    p = res.p if res.pos.size else res.final["p_final"]
    tree = cloud.tree
    cm = coverage_metrics(p, cloud, r_cov, target_tree=tree)
    t = res.time
    scr_t = np.array([coverage_metrics(pp, cloud, r_cov, target_tree=tree,
                                       max_target_samples=1500,
                                       rng=np.random.default_rng(0))["scr"]
                      for pp in res.pos[:: max(1, len(res.pos) // 60)]])
    if len(t) > 1:
        idx = np.linspace(0, len(res.pos) - 1, len(scr_t)).astype(int)
        t_scr = t[idx]
        t_conv = settling_time(t_scr, scr_t, tol=conv_tol)
    else:
        t_conv = float(t[-1]) if len(t) else float("nan")
    V = np.array(res.energy["V"]) if "V" in res.energy else np.array(res.energy["E"])
    mono = float(np.mean(np.diff(V) <= 1e-6 * max(1.0, abs(V[0])))) if len(V) > 1 else float("nan")
    nn = nn_stats(p)
    return RunMetrics(
        scr=cm["scr"], sur=cm["sur"], cce=cm["cce"], r_cov=float(r_cov),
        chamfer=chamfer(p, cloud, target_tree=tree),
        hausdorff=hausdorff(p, cloud, target_tree=tree),
        quant=quantization_energy(p, cloud, target_tree=tree),
        riesz=riesz_energy(p, s=1.0),
        nn_mean=nn["mean"], nn_cv=nn["cv"],
        d_min=min_pair_distance(p), d_safe=float(d_safe),
        safe=bool(min_pair_distance(p) > d_safe),
        d_mean=float(np.asarray(res.energy["d_mean"])[-1]),
        t_conv=float(t_conv), t_end=float(res.final.get("t_end", t[-1] if len(t) else np.nan)),
        energy_final=float(np.asarray(res.energy["E"])[-1]),
        lyapunov_monotone_frac=mono,
        v_end=float(np.asarray(res.energy["v_max"])[-1]),
        n_agents=int(p.shape[0]), area=float(cloud.area), a_lattice=float(a_lattice),
    )


def trajectory_step_min(p0: np.ndarray, p1: np.ndarray, cut: float | None = None) -> float:
    """Exact synchronous linear-step minimum, with certified spatial pruning.

    ``cut`` is a legacy search-radius hint, never a truncation threshold.
    Let U be the minimum midpoint distance and D the largest single-agent
    endpoint displacement. Every pair satisfies min_t distance >= mid_distance-D.
    A pair attaining U has a step minimum <= U, so pairs beyond U+D cannot
    improve the global minimum. We query that radius (with rounding slack).
    """
    if p0.shape != p1.shape or not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("trajectory endpoints must have matching finite coordinates")
    n_all = len(p0)
    if n_all < 2:
        return float("inf")
    if n_all > 400:
        from scipy.spatial import cKDTree
        pm = 0.5 * (p0 + p1)
        tree = cKDTree(pm)
        upper = float(tree.query(pm, k=2)[0][:, 1].min())
        displacement = float(np.linalg.norm(p1-p0, axis=1).max())
        radius = upper + displacement
        # Absolute coordinate scale covers subtraction/averaging roundoff.
        slack = 64*np.finfo(float).eps*max(1., radius, float(np.abs(p0).max()), float(np.abs(p1).max()))
        pairs = tree.query_pairs(radius + slack, output_type="ndarray")
        if len(pairs) == 0:
            raise RuntimeError("Certified pair search unexpectedly excluded its midpoint witness")
        a, b = pairs[:, 0], pairs[:, 1]
        r0 = p0[a] - p0[b]
        r1 = p1[a] - p1[b]
        dd = r1 - r0
        nn = np.einsum("ij,ij->i", dd, dd)
        t = np.zeros_like(nn)
        np.divide(-np.einsum("ij,ij->i", r0, dd), nn, out=t, where=nn > 0)
        np.clip(t, 0.0, 1.0, out=t)
        return float(np.linalg.norm(r0 + t[:, None] * dd, axis=-1).min())
    best = float("inf")
    # Row blocks bound memory without a radius or nearest-neighbour truncation.
    n = len(p0)
    for start in range(0, n, 32):
        end = min(start + 32, n)
        r = p0[start:end, None, :] - p0[None, :, :]
        delta = (p1[start:end, None, :] - p1[None, :, :]) - r
        den = np.einsum("ijk,ijk->ij", delta, delta)
        t = np.zeros_like(den)
        np.divide(-np.einsum("ijk,ijk->ij", r, delta), den, out=t, where=den > 0)
        np.clip(t, 0, 1, out=t)
        near = r + t[..., None] * delta
        sq = np.einsum("ijk,ijk->ij", near, near)
        sq[np.arange(n)[None, :] <= np.arange(start,end)[:,None]] = np.inf
        best = min(best, float(sq.min()))
    return float(np.sqrt(best))
