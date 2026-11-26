"""
strat_raytrace.py
==================

Ray-based density path-integral solver for `StratModel` (see
`stratmodel.py`). Traces straight rays from an origin (a muon
detector) along direction vectors (e.g. from `deg_to_vec` / a survey's
`dirs_vec`), finds where each ray crosses the model's horizon
surfaces, and integrates layer density along
the traversed path length. Rays are clipped to the topographic
surface (`model.order[-1]`).

This works on the analytic/interpolated horizon fields
directly, so intersection points are exact (to root-finding tolerance).

Sign convention
----------------
For a horizon with elevation field Z(x, y), define along a ray
P(t) = origin + t * direction (direction unit-length, t = distance):

    g_h(t) = Z_h(x(t), y(t)) - z(t)

g_h(t) > 0  ->  the ray point is *below* horizon h (in rock, if h is
                topography)
g_h(t) < 0  ->  the ray point is *above* horizon h (in air, if h is
                topography)

Root-finding on g_h(t) = 0 gives the ray's crossing points with h.

Typical use
-----------
    from stratmodel import StratModel
    from strat_raytrace import trace_rays_density

    origin = (detector.detector_x, detector.detector_y, detector.detector_z)
    results = trace_rays_density(model, origin, dirs_vec)

    opacities = np.array([r.density_integral for r in results])  # kg/m^2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------

@dataclass
class Crossing:
    t: float
    horizon: str
    point: np.ndarray  # (3,) xyz


@dataclass
class Segment:
    t0: float
    t1: float
    layer: Optional[str]        # None if the segment matched no layer
    density_integral: float     # kg/m^2 contribution (density * length)
    mean_density: float         # kg/m^3, density_integral / length


@dataclass
class RayResult:
    origin: np.ndarray
    direction: np.ndarray
    t_start: Optional[float]
    t_stop: Optional[float]
    stop_reason: str            # 'topo_exit' | 'max_dist_reached' | 'no_intersection'
    crossings: List[Crossing] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)

    @property
    def path_length(self) -> float:
        if self.t_start is None or self.t_stop is None:
            return 0.0
        return self.t_stop - self.t_start

    @property
    def density_integral(self) -> float:
        """Total column density (opacity) along the traversed path,
        kg/m^2. Zero (not NaN) if the ray never entered the model."""
        return float(sum(s.density_integral for s in self.segments))

    @property
    def mean_density(self) -> float:
        L = self.path_length
        return self.density_integral / L if L > 0 else 0.0


# ----------------------------------------------------------------------
# Horizon / layer evaluation helpers
# ----------------------------------------------------------------------

def build_interp_cache(model) -> Dict[str, object]:
    """Precompute a RegularGridInterpolator per horizon once, so many
    rays can be traced against the same model cheaply."""
    return {name: model.horizon_interpolator(name) for name in model.order}


def _horizon_z(interp, x, y) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    return interp(np.column_stack([y, x]))


def _make_g(interp, origin, direction):
    """Return g(t) = horizon_z(ray(t)) - ray_z(t), vectorised over t."""
    def g(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        pts = origin[None, :] + t[:, None] * direction[None, :]
        return _horizon_z(interp, pts[:, 0], pts[:, 1]) - pts[:, 2]
    return g


def _sign_change_indices(g_vals: np.ndarray):
    """Indices i such that g_vals[i], g_vals[i+1] straddle zero.
    Returns list of (i, kind) with kind='entry' for -/+ and
    kind='exit' for +/-."""
    s = np.sign(g_vals)
    s[s == 0] = 1.0
    idx = np.nonzero(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        kind = "entry" if g_vals[i] < 0 else "exit"
        out.append((i, kind))
    return out


def _layer_at_point(model, interp_cache, x: float, y: float, z: float) -> Optional[str]:
    """Which layer contains (x, y, z)? Mirrors build_treemesh's
    membership test (z_base < z <= z_top); if horizons overlap
    (e.g. near a fault) the topmost matching layer wins, matching
    build_treemesh's assignment-in-order behaviour."""
    hit = None
    for lname in model.layer_order:
        layer = model.layers[lname]
        z_top = _horizon_z(interp_cache[layer.top], x, y)[0]
        z_base = _horizon_z(interp_cache[layer.base], x, y)[0]
        if z_base < z <= z_top:
            hit = lname
    return hit


def _density_at(model, interp_cache, layer_name: Optional[str], x, y, z) -> float:
    if layer_name is None:
        return 0.0
    layer = model.layers[layer_name]
    if layer.density_model is not None:
        z_top = _horizon_z(interp_cache[layer.top], x, y)[0]
        z_base = _horizon_z(interp_cache[layer.base], x, y)[0]
        return float(layer.density_model(x, y, z, z_top, z_base)[0])
    return float(layer.density) if layer.density is not None else 0.0


# ----------------------------------------------------------------------
# Core single-ray tracer
# ----------------------------------------------------------------------

def trace_ray_density(model, origin, direction, max_dist: Optional[float] = None,
                       dt: Optional[float] = None, quad_points: int = 5,
                       interp_cache: Optional[Dict[str, object]] = None,
                       root_xtol: float = 1e-6) -> RayResult:
    """Trace one ray through `model` and integrate density along the
    part of its path that lies below topography.

    Parameters
    ----------
    model : StratModel
    origin : (3,) array-like
        Ray start point (x, y, z).
    direction : (3,) array-like
        Ray direction; normalised internally, need not be unit length.
    max_dist : float, optional
        Distance beyond which marching stops. Defaults to twice the
        model's bounding-box diagonal (generous but bounded).
    dt : float, optional
        Coarse marching step used to *bracket* horizon crossings before
        refining them with `brentq`. Defaults to `model.dx / 2`, i.e.
        roughly half a grid cell. Make this smaller if a horizon has
        structure finer than its own grid spacing along the ray direction.
    quad_points : int
        Gauss-Legendre points used per layer segment to integrate
        density (matters most for `vertical_gradient`/`lateral_gradient`
        density models; the correlated-noise term varies over a length
        scale of `corr_length * model.dx`, generally >> a segment, so a
        handful of points is already more than the noise field itself
        can resolve).
    interp_cache : dict, optional
        Precomputed `build_interp_cache(model)`; pass this in when
        tracing many rays through the same model to avoid rebuilding
        interpolators every call.

    Returns
    -------
    RayResult
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("direction must be non-zero")
    direction = direction / norm

    if interp_cache is None:
        interp_cache = build_interp_cache(model)

    top_name = model.order[-1]
    top_interp = interp_cache[top_name]

    # Bound the search distance from the model's own extent.
    x0, y0 = float(model.X.min()), float(model.Y.min())
    x1, y1 = float(model.X.max()), float(model.Y.max())
    z_lo = min(float(model.horizons[model.order[0]].min()), float(origin[2]))
    z_hi = max(float(model.horizons[top_name].max()), float(origin[2]))
    domain_diag = np.hypot(np.hypot(x1 - x0, y1 - y0), z_hi - z_lo)
    if max_dist is None:
        max_dist = 2.0 * domain_diag
    if dt is None:
        dt = model.dx / 2.0

    n = max(int(np.ceil(max_dist / dt)) + 1, 2)
    t_grid = np.linspace(0.0, max_dist, n)
    pts = origin[None, :] + t_grid[:, None] * direction[None, :]

    g_topo_vals = _horizon_z(top_interp, pts[:, 0], pts[:, 1]) - pts[:, 2]
    g_topo = _make_g(top_interp, origin, direction)

    def refine(g_scalar_fn, i):
        return brentq(lambda tt: float(g_scalar_fn(tt)[0]),
                      t_grid[i], t_grid[i + 1], xtol=root_xtol)

    changes = _sign_change_indices(g_topo_vals)

    t_start: Optional[float]
    t_stop: Optional[float]
    stop_reason: str

    if g_topo_vals[0] >= 0:
        # Origin starts below topography (underground) - typical for a
        # detector emplaced in rock.
        t_start = 0.0
        exits = [i for i, kind in changes if kind == "exit"]
        if exits:
            t_stop = refine(g_topo, exits[0])
            stop_reason = "topo_exit"
        else:
            t_stop = float(t_grid[-1])
            stop_reason = "max_dist_reached"
    else:
        # Origin starts above topography (in air) - find where the ray
        # first enters the ground, then where it next exits.
        entries = [i for i, kind in changes if kind == "entry"]
        if not entries:
            return RayResult(origin=origin, direction=direction,
                              t_start=None, t_stop=None,
                              stop_reason="no_intersection")
        t_start = refine(g_topo, entries[0])
        exits = [i for i, kind in changes if kind == "exit" and t_grid[i] > t_start]
        if exits:
            t_stop = refine(g_topo, exits[0])
            stop_reason = "topo_exit"
        else:
            t_stop = float(t_grid[-1])
            stop_reason = "max_dist_reached"

    # -- crossings of every horizon within (t_start, t_stop) ----------
    crossings: List[Crossing] = []
    eps = root_xtol * 10
    for name in model.order:
        interp = interp_cache[name]
        g_vals = _horizon_z(interp, pts[:, 0], pts[:, 1]) - pts[:, 2]
        g_fn = _make_g(interp, origin, direction)
        for i, _kind in _sign_change_indices(g_vals):
            if t_grid[i + 1] < t_start - eps or t_grid[i] > t_stop + eps:
                continue
            t_root = refine(g_fn, i)
            if t_start + eps < t_root < t_stop - eps:
                p = origin + t_root * direction
                crossings.append(Crossing(t=t_root, horizon=name, point=p))
    crossings.sort(key=lambda c: c.t)

    # -- integrate density over each layer segment ---------------------
    boundaries = [t_start] + [c.t for c in crossings] + [t_stop]
    nodes, weights = leggauss(quad_points)

    segments: List[Segment] = []
    for t_a, t_b in zip(boundaries[:-1], boundaries[1:]):
        length = t_b - t_a
        if length <= eps:
            continue
        t_mid = 0.5 * (t_a + t_b)
        p_mid = origin + t_mid * direction
        layer_name = _layer_at_point(model, interp_cache, p_mid[0], p_mid[1], p_mid[2])

        t_q = 0.5 * length * nodes + 0.5 * (t_a + t_b)
        w_q = 0.5 * length * weights
        integral = 0.0
        for tt, ww in zip(t_q, w_q):
            p = origin + tt * direction
            rho = _density_at(model, interp_cache, layer_name, p[0], p[1], p[2])
            integral += ww * rho

        segments.append(Segment(t0=t_a, t1=t_b, layer=layer_name,
                                 density_integral=integral,
                                 mean_density=integral / length))

    return RayResult(origin=origin, direction=direction,
                      t_start=t_start, t_stop=t_stop, stop_reason=stop_reason,
                      crossings=crossings, segments=segments)


# ----------------------------------------------------------------------
# Multi-ray convenience wrapper
# ----------------------------------------------------------------------

def trace_rays_density(model, origin, directions: Sequence[Sequence[float]],
                        max_dist: Optional[float] = None, dt: Optional[float] = None,
                        quad_points: int = 5) -> List[RayResult]:
    """Trace many rays from a single `origin` (one detector object).
    Builds the horizon-interpolator cache once and reuses it.

    Returns a list of `RayResult`, one per direction, in input order.
    For a quick opacity image:

        results = trace_rays_density(model, origin, dirs_vec)
        opacity = np.array([r.density_integral for r in results])  # kg/m^2
    """
    origin = np.asarray(origin, dtype=float)
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    interp_cache = build_interp_cache(model)

    results = []
    for d in directions:
        results.append(trace_ray_density(model, origin, d, max_dist=max_dist, dt=dt,
                                          quad_points=quad_points,
                                          interp_cache=interp_cache))
    return results
