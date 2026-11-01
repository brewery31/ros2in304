"""Design theory: capacity, scale rule and feasibility certificate.

The original STFC paper scaled the target point cloud by "volumetric
equivalence" with the convex-hull volume of the cloud (its Eq. (8)) and then
chose the coverage radius ``r_cov`` and the safety radius ``r_safe`` by hand.
That rule is wrong for hollow or shell-like targets (the convex hull of a ring
is a disk) and it does not relate the two radii, which is exactly where the
reviewer saw *conflicting constraints*.

This module replaces it with a 2-measure (surface-area) argument.  Agents cover
a 2-manifold ``Sigma`` of area ``A``; the relevant geometric facts are the
classical covering/packing densities of the Euclidean plane:

    thinnest covering density      Theta_cov = 2*pi/(3*sqrt(3)) = 1.209199...  (Kershner 1939)
    densest packing density        Theta_pack = pi/(2*sqrt(3)) = 0.906900...   (Thue / Fejes Toth)

Both are attained by the triangular lattice, which is therefore the optimal
*design target*: agents sitting on a triangular lattice of spacing ``a`` cover
the surface with geodesic radius ``a/sqrt(3)`` while keeping pairwise distance
``a``.  This yields a parameter-free scale rule and an explicit feasible window.

Everything here is pure geometry and is used once, offline, to turn
``(A, N, d_safe)`` into ``(scale s, lattice spacing a, coverage radius r_cov)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "THETA_COV",
    "THETA_PACK",
    "LATTICE_CELL",
    "Design",
    "covering_capacity",
    "packing_capacity",
    "lattice_spacing",
    "lattice_covering_radius",
    "feasibility",
    "design_scale",
    "design_from_coverage_radius",
]

#: thinnest covering density of the plane by equal disks
THETA_COV = 2.0 * np.pi / (3.0 * np.sqrt(3.0))
#: densest packing density of equal disks in the plane
THETA_PACK = np.pi / (2.0 * np.sqrt(3.0))
#: area of the Voronoi cell of a triangular lattice with unit spacing
LATTICE_CELL = np.sqrt(3.0) / 2.0


# ---------------------------------------------------------------------------
# Basic lattice relations
# ---------------------------------------------------------------------------

def lattice_spacing(area: float, n: int) -> float:
    """Spacing of the triangular lattice that gives ``n`` sites on ``area``.

    ``n * LATTICE_CELL * a^2 = area``  =>  ``a = sqrt(2*area/(sqrt(3)*n))``.
    """
    if area <= 0 or n <= 0:
        raise ValueError("area and n must be positive")
    return float(np.sqrt(2.0 * area / (np.sqrt(3.0) * n)))


def lattice_covering_radius(a: float) -> float:
    """Geodesic covering radius of a triangular lattice of spacing ``a``."""
    return float(a / np.sqrt(3.0))


def lattice_packing_radius(a: float) -> float:
    """Largest guaranteed empty disk radius around each site (``a/2``)."""
    return float(a / 2.0)


def covering_capacity(area: float, r_cov: float) -> float:
    """Minimum number of agents that can *possibly* ``r_cov``-cover ``area``.

    Necessity: a covering by ``n`` disks of radius ``r`` has density
    ``n*pi*r^2/area >= THETA_COV``.  Hence ``n >= THETA_COV*area/(pi*r^2)``.
    For a planar region this is sharp (triangular lattice).
    """
    return float(THETA_COV * area / (np.pi * r_cov ** 2))


def packing_capacity(area: float, d_safe: float) -> float:
    """Maximum number of agents that can be ``d_safe``-separated on ``area``.

    Necessity: disks of radius ``d_safe/2`` around the agents are disjoint and
    contained in the surface, so ``n*pi*(d_safe/2)^2 <= area`` (up to a
    boundary/curvature correction of order ``d_safe/R``).
    """
    return float(area / (np.pi * (d_safe / 2.0) ** 2))


# ---------------------------------------------------------------------------
# Feasibility certificate
# ---------------------------------------------------------------------------

@dataclass
class Design:
    """Result of the design rule (all lengths in world units)."""

    feasible: bool
    scale: float                 # s: target cloud is scaled by this factor
    area: float                  # A of the scaled target (2-measure)
    n_agents: int
    lattice_spacing: float       # a: nominal nearest-neighbour spacing
    r_cov: float                 # coverage radius that the lattice achieves
    d_safe: float                # required safety distance
    d_hold: float = 0.0          # holding distance above the surface
    n_min_cov: float = float("nan")   # covering lower bound on n
    n_max_safe: float = float("nan")  # packing upper bound on n
    slack_cov: float = float("nan")   # a/sqrt(3) / r_cov  (<=1 ok)
    slack_safe: float = float("nan")  # a / d_safe       (>=1 ok)
    ratio_req: float = float("nan")   # required r_cov/d_safe lower bound
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (f"Design(feasible={self.feasible}, s={self.scale:.4g}, "
                f"a={self.lattice_spacing:.4g}, r_cov={self.r_cov:.4g}, "
                f"d_safe={self.d_safe:.4g}, N={self.n_agents})")


def feasibility(area: float, n: int, d_safe: float, r_cov: Optional[float] = None,
                margin: float = 0.05) -> dict:
    """Check the necessary conditions for a safe, fully covering configuration.

    ``margin`` inflates ``d_safe`` by ``(1+margin)`` so that the certificate is
    robust to the discretisation of the actual (non-lattice) configuration.
    """
    d = d_safe * (1.0 + margin)
    n_min = covering_capacity(area, r_cov) if r_cov is not None else float("nan")
    n_max = packing_capacity(area, d)
    out = {
        "n_min_cov": float(n_min),
        "n_max_safe": float(n_max),
        "coverage_ok": bool(n >= n_min) if r_cov is not None else True,
        "safety_ok": bool(n <= n_max),
        "ratio_req": 1.0 / np.sqrt(3.0),
        "ratio_ok": bool(r_cov / d >= 1.0 / np.sqrt(3.0)) if r_cov is not None else True,
    }
    out["feasible"] = bool(out["coverage_ok"] and out["safety_ok"] and out["ratio_ok"])
    return out


def design_at_scale(area0: float, n: int, scale: float, d_safe: float,
                    margin: float = 0.05, d_hold: float = 0.0,
                    curvature: float = 0.0) -> Design:
    """Derive the geometry at a *fixed* world scale (e.g. a room).

    The coverage radius is no longer a free parameter: for a swarm of ``n``
    agents on a surface of area ``A = scale^2 * area0`` the optimal-lattice
    covering radius is ``r_cov = a/sqrt(3)`` with
    ``a = sqrt(2A/(sqrt(3) n))``.  The design is feasible iff the lattice is
    also safe, ``a >= (1+margin) d_safe``.
    """
    area_eff = area0 * (1.0 + 2.0 * curvature * d_hold)
    area = area_eff * scale ** 2
    a = lattice_spacing(area, n)
    r_cov = a / np.sqrt(3.0)
    cert = feasibility(area, n, d_safe, r_cov, margin)
    return Design(
        feasible=bool(cert["feasible"] and a >= (1.0 + margin) * d_safe),
        scale=float(scale), area=float(area), n_agents=int(n),
        lattice_spacing=float(a), r_cov=float(r_cov), d_safe=float(d_safe),
        d_hold=float(d_hold),
        n_min_cov=float(covering_capacity(area, r_cov)),
        n_max_safe=float(packing_capacity(area, d_safe * (1.0 + margin))),
        slack_cov=1.0,
        slack_safe=float(a / (d_safe * (1.0 + margin))),
        ratio_req=float(1.0 / np.sqrt(3.0)),
        notes={"area0": float(area0), "margin": float(margin),
               "curvature": float(curvature), "certificate": cert},
    )


def design_scale(area0: float, n: int, d_safe: float, mu: float = 1.6,
                 margin: float = 0.05, d_hold: float = 0.0,
                 curvature: float = 0.0) -> Design:
    """Certified scale rule at a chosen *safety factor* ``mu = a / d_safe``.

    ``mu`` is the only free geometric knob.  It must exceed
    ``sqrt(4/(pi) / (sqrt(3)/2)) = 0.952`` for the lattice to be both safe and
    covering (pure geometry, see :func:`feasibility`); dynamically useful values
    are in ``[1.2, 2.5]`` (a large ``mu`` means a sparse, well-conditioned
    swarm, a small one means a dense, jamming-prone swarm).

    The scale follows from ``a = mu * d_safe`` and ``a = scale * a0``.
    """
    if mu <= 0:
        raise ValueError("mu must be positive")
    area_eff = area0 * (1.0 + 2.0 * curvature * d_hold)
    a0 = lattice_spacing(area_eff, n)
    scale = mu * d_safe / a0
    d = design_at_scale(area0, n, scale, d_safe, margin=margin,
                        d_hold=d_hold, curvature=curvature)
    d.notes.update({"mu": float(mu), "a0": float(a0)})
    return d


def design_scale_1d(length0: float, n: int, d_safe: float, mu: float = 1.6,
                    margin: float = 0.05) -> Design:
    """Scale rule for a 1-D target (a curve of length ``length0``).

    ``n`` agents equally spaced on a curve of length ``L`` have spacing
    ``L/n``; requiring that spacing to be the safety factor times ``d_safe``
    gives ``L = n * mu * d_safe`` and hence ``s = n*mu*d_safe/length0``.
    The "coverage radius" of an equal-spaced curve sample is half the spacing.
    """
    a = mu * d_safe
    scale = n * a / length0
    length = length0 * scale
    return Design(
        feasible=bool(a >= (1.0 + margin) * d_safe),
        scale=float(scale), area=float(length), n_agents=int(n),
        lattice_spacing=float(a), r_cov=float(a / 2.0), d_safe=float(d_safe),
        d_hold=0.0, n_min_cov=float(n), n_max_safe=float(n),
        slack_cov=1.0, slack_safe=float(a / ((1.0 + margin) * d_safe)),
        ratio_req=float(1.0 / np.sqrt(3.0)),
        notes={"length0": float(length0), "intrinsic_dim": 1, "mu": float(mu)},
    )


def design_from_coverage_radius(area0: float, n: int, r_cov_req: float,
                                d_safe: float, margin: float = 0.05,
                                d_hold: float = 0.0) -> Design:
    """Scale rule when the *metric* fixes the coverage radius in world units.

    Requires ``n >= THETA_COV*area/(pi*r_cov^2)`` at the chosen scale; the
    smallest safe scale that also meets the coverage radius is returned.  A
    negative ``slack_cov`` means the request is geometrically infeasible.
    """
    a_req_max = np.sqrt(3.0) * r_cov_req          # lattice spacing must be <= this
    a_safe_min = (1.0 + margin) * d_safe          # lattice spacing must be >= this
    a = max(a_req_max, a_safe_min)
    area_eff = area0 * (1.0 + 0.0 * d_hold)
    # a = s*a0  =>  s = a / a0
    a0 = lattice_spacing(area_eff, n)
    s = a / a0
    r_cov_lat = a / np.sqrt(3.0)
    area_s = area_eff * s ** 2
    cert = feasibility(area_s, n, d_safe, r_cov_req, margin)
    return Design(
        feasible=bool(cert["feasible"] and a_req_max >= a_safe_min),
        scale=float(s),
        area=float(area_s),
        n_agents=int(n),
        lattice_spacing=float(a),
        r_cov=float(r_cov_lat),
        d_safe=float(d_safe),
        d_hold=float(d_hold),
        n_min_cov=float(covering_capacity(area_s, r_cov_req)),
        n_max_safe=float(packing_capacity(area_s, d_safe * (1.0 + margin))),
        slack_cov=float(r_cov_req / r_cov_lat),
        slack_safe=float(a / a_safe_min),
        ratio_req=float(1.0 / np.sqrt(3.0)),
        notes={"area0": float(area0), "requested_r_cov": float(r_cov_req),
               "certificate": cert},
    )
