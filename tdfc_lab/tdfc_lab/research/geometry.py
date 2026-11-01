"""Target geometry: point-cloud I/O, area estimation, sampling and distance queries.

All geometry used by STFC-2 lives here.  A target shape is represented by a
finite point set ``P = {q_k}`` sampled from a compact 2-manifold ``Sigma``
embedded in R^3 (a "shell" target), or from a 2-dimensional region in the
plane (a "filled" 2-D target such as a handwritten letter).  The controller
only ever needs (i) nearest-point queries against ``P`` inside a bounded
radius and (ii) the total 2-measure ``A = area(Sigma)``, which is used once by
the design rule (see :mod:`stfc.capacity`).

Design notes
------------
* PLY reading supports ascii / binary_little_endian / binary_big_endian and
  the property layouts produced by Open3D and the Stanford scanner.
* Surface area is taken from the mesh when the file provides faces; otherwise
  it is estimated from the point set with a *calibrated local-patch*
  estimator (validated in ``tests/test_geometry.py`` against analytic areas).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

__all__ = [
    "PointCloud",
    "read_ply",
    "estimate_surface_area",
    "make_shape",
    "SHAPES",
]

# ---------------------------------------------------------------------------
# PLY reader
# ---------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_ply(path: str | Path) -> dict:
    """Read a PLY file.

    Returns a dict with keys ``vertices`` (m,3) float64, optional ``normals``
    and optional ``faces`` (f,3) int64.  Only vertex properties are parsed for
    binary files with face lists that use arbitrary properties; faces are read
    when their property list is ``uchar/int`` or ``uchar/uint``.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(b"ply"):
        raise ValueError(f"{path} is not a PLY file")

    # --- header ---------------------------------------------------------
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("PLY header not terminated")
    header_end = raw.find(b"\n", end) + 1
    header = raw[:header_end].decode("ascii", errors="replace").splitlines()

    fmt = None
    elements: list[dict] = []
    for line in header:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append({"name": tok[1], "count": int(tok[2]), "props": []})
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1]["props"].append(("list", tok[2], tok[3], tok[4]))
            else:
                elements[-1]["props"].append((tok[1], tok[2]))
    if fmt is None:
        raise ValueError("PLY format line missing")

    ascii_mode = fmt == "ascii"
    if not ascii_mode and fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt}")
    endian = "<" if fmt != "binary_big_endian" else ">"

    out: dict = {"vertices": None, "normals": None, "faces": None}

    if ascii_mode:
        body = raw[header_end:].decode("ascii", errors="replace").split()
        pos = 0
        for el in elements:
            n = el["count"]
            if el["name"] == "vertex":
                props = [p[1] for p in el["props"] if p[0] != "list"]
                width = len(props)
                vals = np.array(body[pos: pos + n * width], dtype=np.float64)
                vals = vals.reshape(n, width)
                pos += n * width
                idx = {name: j for j, name in enumerate(props)}
                out["vertices"] = vals[:, [idx["x"], idx["y"], idx["z"]]]
                if all(k in idx for k in ("nx", "ny", "nz")):
                    out["normals"] = vals[:, [idx["nx"], idx["ny"], idx["nz"]]]
            elif el["name"] == "face":
                faces = []
                for _ in range(n):
                    cnt = int(body[pos]); pos += 1
                    idx = [int(v) for v in body[pos: pos + cnt]]; pos += cnt
                    for k in range(1, cnt - 1):          # fan triangulation
                        faces.append([idx[0], idx[k], idx[k + 1]])
                out["faces"] = np.array(faces, dtype=np.int64) if faces else None
            else:
                # skip unknown element: assume scalar properties only
                width = sum(1 for p in el["props"] if p[0] != "list")
                pos += n * width
        return out

    # --- binary ---------------------------------------------------------
    pos = header_end
    for el in elements:
        n = el["count"]
        name = el["name"]
        if any(p[0] == "list" for p in el["props"]):
            # mixed record: parse sequentially (vertex elements are scalar in
            # every file we consume; face lists are the only lists we support)
            if name == "face":
                faces = []
                for _ in range(n):
                    (cnt,) = struct.unpack_from(endian + "B", raw, pos); pos += 1
                    idx = np.frombuffer(raw, dtype=endian + "i4", count=cnt, offset=pos)
                    pos += 4 * cnt
                    for k in range(1, cnt - 1):
                        faces.append([int(idx[0]), int(idx[k]), int(idx[k + 1])])
                out["faces"] = np.array(faces, dtype=np.int64) if faces else None
            else:
                raise ValueError(f"unsupported list property in element {name}")
            continue
        dtype = np.dtype([(f"f{j}", endian + _PLY_TYPES[p[0]]) for j, p in enumerate(el["props"])])
        arr = np.frombuffer(raw, dtype=dtype, count=n, offset=pos)
        pos += dtype.itemsize * n
        names = [p[1] for p in el["props"]]
        if name == "vertex":
            for axis in ("x", "y", "z"):
                if axis not in names:
                    raise ValueError(f"vertex property {axis} missing")
            out["vertices"] = np.stack([arr[f"f{names.index(a)}"] for a in ("x", "y", "z")], axis=1).astype(np.float64)
            if all(a in names for a in ("nx", "ny", "nz")):
                out["normals"] = np.stack([arr[f"f{names.index(a)}"] for a in ("nx", "ny", "nz")], axis=1).astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Area estimation
# ---------------------------------------------------------------------------

def _mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    if tri.shape[1] > 3:  # fan triangulation
        parts = [tri[:, [0, k, k + 1]] for k in range(1, tri.shape[1] - 1)]
        tri = np.concatenate(parts, axis=0)
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def _local_patch_area(points: np.ndarray, k: int = 8, seed: int = 0) -> float:
    """Deprecated fast estimator (kept for reference; biased on regular samples)."""
    m = points.shape[0]
    k = int(min(k, m - 1))
    if k < 3:
        return float("nan")
    tree = cKDTree(points)
    d, idx = tree.query(points, k=k + 1)
    nb = points[idx]
    rel = nb - points[:, None, :]
    cov = np.einsum("mki,mkj->mij", rel, rel)
    w, vec = np.linalg.eigh(cov)
    e1, e2 = vec[:, :, 2], vec[:, :, 1]
    xy = np.stack([np.einsum("mki,mi->mk", rel, e1), np.einsum("mki,mi->mk", rel, e2)], axis=2)
    areas = np.empty(m)
    for i in range(m):
        try:
            areas[i] = ConvexHull(xy[i]).volume
        except Exception:
            areas[i] = np.nan
    c = CAL_PATCH.get(k, np.nan)
    if not np.isfinite(c):
        c = _calibrate_patch(k)
    return float(m * np.nanmean(areas) / c)


def _clip_convex(poly: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """Clip a convex polygon by the half-plane {x : a . x <= b}."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        p = poly[i]
        q = poly[(i + 1) % n]
        fp = float(a @ p) - b
        fq = float(a @ q) - b
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0 < fq) or (fq < 0.0 < fp):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return float("nan")
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _voronoi_cell_area(points: np.ndarray, n_probe: int = 1200, k: int = 48,
                       rng: np.random.Generator | None = None) -> float:
    """Unbiased-in-the-limit surface area estimator based on local Voronoi cells.

    For a sample of a 2-manifold with ``m`` points and area ``A``, the average
    area of the (projected) Voronoi cell of a point equals ``A / m``.  The cell
    of the origin among its ``k`` nearest neighbours is the intersection of the
    half-planes ``x . q_j <= |q_j|^2 / 2``; it is computed exactly by polygon
    clipping in the local tangent plane.  The estimator is therefore insensitive
    to whether the sample is Poisson-like or lattice-like (unlike patch-hull or
    k-NN-density estimators).
    """
    rng = np.random.default_rng(rng)
    m = points.shape[0]
    if m < 8:
        return float("nan")
    k = int(min(k, m - 1))
    probe = np.arange(m) if m <= n_probe else rng.choice(m, n_probe, replace=False)
    tree = cKDTree(points)
    _, idx = tree.query(points[probe], k=k + 1)
    areas = np.empty(len(probe))
    for t, i in enumerate(probe):
        rel = points[idx[t, 1:]] - points[i]
        nrm = np.linalg.norm(rel, axis=1)
        keep = nrm > 1e-12
        rel, nrm = rel[keep], nrm[keep]
        if len(rel) < 6:
            areas[t] = np.nan
            continue
        cov = rel.T @ rel
        _, vec = np.linalg.eigh(cov)
        e1, e2 = vec[:, 2], vec[:, 1]
        xy = np.stack([rel @ e1, rel @ e2], axis=1)
        r2 = (xy ** 2).sum(1)
        L = 2.5 * float(np.sqrt(r2.max()))
        poly = np.array([[-L, -L], [L, -L], [L, L], [-L, L]], dtype=np.float64)
        for q, rr in zip(xy, r2):
            # bisector: x . q <= |q|^2 / 2   (Voronoi cell of the origin)
            poly = _clip_convex(poly, q, 0.5 * float(rr))
            if len(poly) < 3:
                break
        if len(poly) < 3 or np.abs(poly).max() > 0.995 * L:
            # unbounded cell: the neighbourhood does not surround the point
            # (crease, boundary, or high curvature) -> not a valid probe
            areas[t] = np.nan
            continue
        areas[t] = _polygon_area(poly)
    if np.all(np.isnan(areas)):
        return float("nan")
    return float(m * np.nanmean(areas))



# calibration constants c_k = E[hull area] * rho for a planar Poisson process
CAL_PATCH: dict[int, float] = {}


def _calibrate_patch(k: int, n: int = 20000, trials: int = 8, seed: int = 0) -> float:
    """Monte-Carlo calibration of the local-patch area estimator (cached)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        sq = rng.random((n, 2))
        d, idx = cKDTree(sq).query(sq, k=k + 1)
        nb = sq[idx]
        rel = nb - sq[:, None, :]
        areas = np.array([ConvexHull(rel[i]).volume for i in range(n)])
        # density of the unit square sample is n / 1
        vals.append(areas.mean() * n)
    c = float(np.mean(vals))
    CAL_PATCH[k] = c
    return c


def estimate_surface_area(points: np.ndarray, faces: Optional[np.ndarray] = None,
                          k: int = 48, area_hint: Optional[float] = None,
                          n_probe: int = 1200) -> float:
    """Total 2-measure of the target (mesh area if available, else estimated)."""
    if area_hint is not None:
        return float(area_hint)
    if faces is not None and len(faces) > 0:
        return _mesh_area(points, faces)
    return _voronoi_cell_area(points, n_probe=n_probe, k=k)


# ---------------------------------------------------------------------------
# Point Cloud
# ---------------------------------------------------------------------------

@dataclass
class PointCloud:
    """A target point set with cached spatial index and 2-measure."""

    points: np.ndarray
    normals: Optional[np.ndarray] = None
    area: float = float("nan")
    faces: Optional[np.ndarray] = None
    name: str = "target"
    meta: dict = field(default_factory=dict)
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_points(cls, points: np.ndarray, normals: Optional[np.ndarray] = None,
                    area: Optional[float] = None, name: str = "target",
                    faces: Optional[np.ndarray] = None) -> "PointCloud":
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (m,3)")
        a = estimate_surface_area(pts, faces=faces, area_hint=area)
        return cls(points=pts, normals=normals, area=float(a), faces=faces, name=name)

    @classmethod
    def from_file(cls, path: str | Path, name: Optional[str] = None,
                  area: Optional[float] = None) -> "PointCloud":
        d = read_ply(path)
        if d["vertices"] is None:
            raise ValueError(f"no vertices in {path}")
        return cls.from_points(d["vertices"], normals=d["normals"],
                               area=area, faces=d["faces"],
                               name=name or Path(path).stem)

    # -- basic properties -----------------------------------------------
    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    @property
    def m(self) -> int:
        return int(self.points.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def radius(self) -> float:
        """Radius of the smallest enclosing sphere around the centroid."""
        return float(np.linalg.norm(self.points - self.centroid, axis=1).max())

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.points.max(0) - self.points.min(0)))

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(0), self.points.max(0)

    # -- queries ---------------------------------------------------------
    def nearest(self, x: np.ndarray, k: int = 1, r_max: Optional[float] = None):
        """Nearest sample(s).  Returns (distance, index); distance=inf if none."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        d, i = self.tree.query(x, k=k, distance_upper_bound=np.inf if r_max is None else r_max)
        if k == 1:
            d = np.atleast_1d(d); i = np.atleast_1d(i)
        i = np.where(np.isfinite(d), i, 0).astype(np.int64)
        return d, i

    def distance(self, x: np.ndarray, r_max: Optional[float] = None) -> np.ndarray:
        d, _ = self.nearest(x, r_max=r_max)
        return d

    def nearest_point(self, x: np.ndarray) -> np.ndarray:
        d, i = self.nearest(x)
        return self.points[i]

    def outward_normal(self, x: np.ndarray) -> np.ndarray:
        """Unit vector pointing from the surface towards x (a.e. well defined)."""
        q = self.nearest_point(x)
        v = np.atleast_2d(x) - q
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    def normalized_area(self, area: float = 1.0) -> "PointCloud":
        """Rescale about the centroid so that the 2-measure equals ``area``."""
        s = float(np.sqrt(area / self.area))
        out = self.scaled(s)
        out.area = float(area)
        out.name = self.name
        return out

    # -- transformations -------------------------------------------------
    def scaled(self, s: float, center: Optional[np.ndarray] = None) -> "PointCloud":
        c = self.centroid if center is None else np.asarray(center, dtype=np.float64)
        pts = c + s * (self.points - c)
        return PointCloud(points=pts, normals=self.normals, area=float(self.area * s ** 2),
                          faces=None, name=f"{self.name}@s={s:.3g}",
                          meta=dict(self.meta, scale=float(s)))

    def subsample(self, m: int, rng: np.random.Generator | None = None,
                  method: str = "random") -> "PointCloud":
        rng = np.random.default_rng(rng)
        m = int(min(m, self.m))
        if method == "random":
            idx = rng.choice(self.m, size=m, replace=False)
        elif method == "fps":  # farthest point sampling, good for coarse priors
            idx = _fps(self.points, m, rng)
        else:
            raise ValueError(method)
        return PointCloud(points=self.points[idx],
                          normals=None if self.normals is None else self.normals[idx],
                          area=self.area, name=f"{self.name}@{m}", meta=dict(self.meta))

    def uniform_sample(self, m: int, rng: np.random.Generator | None = None) -> "PointCloud":
        """Resample the *surface* uniformly at random (bootstrap from samples)."""
        return self.subsample(m, rng=rng)

    # -- diagnostics ------------------------------------------------------
    def nn_spacing(self, k: int = 1) -> np.ndarray:
        d, _ = self.tree.query(self.points, k=k + 1)
        return d[:, k]

    def mean_spacing(self) -> float:
        """Mean nearest-neighbour distance of the target samples."""
        return float(self.nn_spacing(1).mean())

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PointCloud({self.name!r}, m={self.m}, area={self.area:.4g}, "
                f"radius={self.radius:.4g})")


def _fps(points: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    idx = np.empty(m, dtype=np.int64)
    idx[0] = rng.integers(n)
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for j in range(1, m):
        idx[j] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(points - points[idx[j]], axis=1))
    return idx


# ---------------------------------------------------------------------------
# Synthetic shapes (parametric, exactly area-parameterised)
# ---------------------------------------------------------------------------

def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _tri_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def make_sphere(m: int = 10000, r: float = 1.0) -> PointCloud:
    p = _fib_sphere(m) * r
    return PointCloud.from_points(p, normals=p / r, area=4 * np.pi * r ** 2, name="sphere")


def make_cylinder(m: int = 10000, r: float = 0.5, h: float = 1.0, n_ring: int = 6) -> PointCloud:
    lat, caps = 0.75, 0.25
    n_lat = int(m * lat)
    n_cap = (m - n_lat) // 2
    th = np.random.default_rng(0).random(n_lat) * 2 * np.pi
    z = np.random.default_rng(1).random(n_lat) * h - h / 2
    side = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    cr = r * np.sqrt(np.random.default_rng(2).random(n_cap))
    ca = np.random.default_rng(3).random(n_cap) * 2 * np.pi
    top = np.stack([cr * np.cos(ca), cr * np.sin(ca), np.full(n_cap, h / 2)], 1)
    bot = np.stack([cr * np.cos(ca), cr * np.sin(ca), np.full(n_cap, -h / 2)], 1)
    pts = np.concatenate([side, top, bot], 0)
    area = 2 * np.pi * r * h + 2 * np.pi * r ** 2
    return PointCloud.from_points(pts, area=area, name="cylinder")


def make_cone(m: int = 10000, r: float = 0.5, h: float = 1.0) -> PointCloud:
    rng = np.random.default_rng(0)
    n_side = int(m * 0.8)
    n_base = m - n_side
    # uniform on the lateral surface: sqrt parameterisation along height
    u = rng.random(n_side)
    z = h * (1 - np.sqrt(u)) - h / 2
    rad = r * (h / 2 - z) / h
    th = rng.random(n_side) * 2 * np.pi
    side = np.stack([rad * np.cos(th), rad * np.sin(th), z], 1)
    cr = r * np.sqrt(rng.random(n_base)); ca = rng.random(n_base) * 2 * np.pi
    base = np.stack([cr * np.cos(ca), cr * np.sin(ca), np.full(n_base, -h / 2)], 1)
    pts = np.concatenate([side, base], 0)
    area = np.pi * r * np.sqrt(r ** 2 + h ** 2) + np.pi * r ** 2
    return PointCloud.from_points(pts, area=area, name="cone")


def make_torus(m: int = 10000, R: float = 0.6, a: float = 0.25) -> PointCloud:
    rng = np.random.default_rng(0)
    # rejection sampling for uniform area (density ~ (R + a cos v))
    out = []
    need = m
    while need > 0:
        u = rng.random(4 * need) * 2 * np.pi
        v = rng.random(4 * need) * 2 * np.pi
        p = (R + a * np.cos(v)) / (R + a)
        keep = rng.random(4 * need) < p
        uu, vv = u[keep][:need], v[keep][:need]
        out.append(np.stack([(R + a * np.cos(vv)) * np.cos(uu),
                             (R + a * np.cos(vv)) * np.sin(uu),
                             a * np.sin(vv)], 1))
        need -= len(uu)
    pts = np.concatenate(out, 0)[:m]
    return PointCloud.from_points(pts, area=4 * np.pi ** 2 * R * a, name="torus")


def make_box(m: int = 10000, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> PointCloud:
    rng = np.random.default_rng(0)
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    counts = np.floor(areas / areas.sum() * m).astype(int)
    counts[-1] += m - counts.sum()
    faces = []
    for fidx, c in enumerate(counts):
        if c <= 0:
            continue
        u = rng.random(c); v = rng.random(c)
        if fidx == 0:   pts = np.stack([np.full(c, sx / 2), (u - .5) * sy, (v - .5) * sz], 1)
        elif fidx == 1: pts = np.stack([np.full(c, -sx / 2), (u - .5) * sy, (v - .5) * sz], 1)
        elif fidx == 2: pts = np.stack([(u - .5) * sx, np.full(c, sy / 2), (v - .5) * sz], 1)
        elif fidx == 3: pts = np.stack([(u - .5) * sx, np.full(c, -sy / 2), (v - .5) * sz], 1)
        elif fidx == 4: pts = np.stack([(u - .5) * sx, (v - .5) * sy, np.full(c, sz / 2)], 1)
        else:           pts = np.stack([(u - .5) * sx, (v - .5) * sy, np.full(c, -sz / 2)], 1)
        faces.append(pts)
    pts = np.concatenate(faces, 0)
    area = 2 * (sx * sy + sy * sz + sx * sz)
    return PointCloud.from_points(pts, area=area, name="box")


def make_prism(m: int = 10000, n: int = 6, r: float = 0.5, h: float = 1.0) -> PointCloud:
    rng = np.random.default_rng(0)
    ang = np.arange(n) * 2 * np.pi / n
    side_area = n * 2 * r * np.sin(np.pi / n) * h
    cap_area = 0.5 * n * r ** 2 * np.sin(2 * np.pi / n)
    frac = side_area / (side_area + 2 * cap_area)
    n_side = int(m * frac)
    n_cap = (m - n_side) // 2
    pts = []
    # lateral faces: pick a face uniformly by area, then sample the rectangle
    face_w = 2 * r * np.sin(np.pi / n)
    fid = rng.integers(n, size=n_side)
    s = rng.random(n_side) * face_w
    z = rng.random(n_side) * h - h / 2
    a0, a1 = ang[fid], ang[(fid + 1) % n]
    p0 = r * np.stack([np.cos(a0), np.sin(a0)], 1)
    p1 = r * np.stack([np.cos(a1), np.sin(a1)], 1)
    d = p1 - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    xy = p0 + s[:, None] * d
    pts.append(np.concatenate([xy, z[:, None]], 1))
    for zz in (h / 2, -h / 2):
        tri = rng.integers(n, size=n_cap)
        u = np.sqrt(rng.random(n_cap)); v = rng.random(n_cap)
        a = ang[tri] + u * 2 * np.pi / n
        rr = v * r
        pts.append(np.stack([rr * np.cos(a), rr * np.sin(a), np.full(n_cap, zz)], 1))
    pts = np.concatenate(pts, 0)
    area = side_area + 2 * cap_area
    return PointCloud.from_points(pts, area=area, name="prism")


def sample_mesh(points: np.ndarray, faces: np.ndarray, m: int,
                rng: np.random.Generator | None = None) -> PointCloud:
    """Uniform-by-area random sample of a triangle mesh.

    Mesh *vertices* are not a uniform-area sample (they concentrate where the
    scan is detailed), so using them directly as the control prior makes the
    "uniform agent density" objective ill-posed.  This routine samples the
    surface measure instead.
    """
    rng = np.random.default_rng(rng)
    tri = points[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    cdf = np.cumsum(area)
    total = cdf[-1]
    u = rng.random(m) * total
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, len(area) - 1)
    r1 = np.sqrt(rng.random(m))
    r2 = rng.random(m)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    pts = (w0[:, None] * tri[idx, 0] + w1[:, None] * tri[idx, 1] + w2[:, None] * tri[idx, 2])
    return PointCloud.from_points(pts, area=float(total), name="mesh_sample")


def make_ring(m: int = 4000, r: float = 1.0, n_turns: int = 1) -> PointCloud:
    """A 1-D circular curve (target on which ``length`` is the relevant measure)."""
    th = np.linspace(0.0, 2 * np.pi * n_turns, m, endpoint=False)
    pts = np.stack([r * np.cos(th), r * np.sin(th), 0.0 * th], axis=1)
    pc = PointCloud.from_points(pts, area=2 * np.pi * r * n_turns, name="ring")
    pc.meta["intrinsic_dim"] = 1
    return pc


def make_helix(m: int = 6000, r: float = 0.65, height: float = 0.8,
               turns: float = 1.25) -> PointCloud:
    """A 3-D helical curve (intrinsic dimension 1) -- a good hardware target."""
    t = np.linspace(0.0, 2 * np.pi * turns, m)
    p = np.stack([r * np.cos(t), r * np.sin(t), height * (t / t.max() - 0.5)], axis=1)
    length = turns * 2 * np.pi * np.sqrt(r ** 2 + (height / (2 * np.pi * turns)) ** 2)
    pc = PointCloud.from_points(p, area=length, name="helix")
    pc.meta["intrinsic_dim"] = 1
    return pc


def make_dome(m: int = 6000, r: float = 0.55) -> PointCloud:
    """Upper hemisphere (intrinsic dimension 2, open surface)."""
    rng = np.random.default_rng(0)
    z = rng.random(m)
    rr = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    th = 2 * np.pi * rng.random(m)
    p = np.stack([rr * np.cos(th), rr * np.sin(th), z], axis=1) * r
    pc = PointCloud.from_points(p, area=2 * np.pi * r * r, name="dome")
    pc.meta["intrinsic_dim"] = 2
    return pc


def make_letter(points: np.ndarray, name: str = "letter") -> PointCloud:
    """Filled 2-D letter (points are expected to be (m,3) with z~0)."""
    return PointCloud.from_points(points, name=name)


SHAPES = {
    "sphere": make_sphere,
    "cylinder": make_cylinder,
    "cone": make_cone,
    "torus": make_torus,
    "box": make_box,
    "prism": make_prism,
}


def make_shape(name: str, m: int = 10000, **kw) -> PointCloud:
    if name not in SHAPES:
        raise KeyError(f"unknown shape {name!r}; available: {sorted(SHAPES)}")
    return SHAPES[name](m=m, **kw)
