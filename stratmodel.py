"""
stratmodel.py
=============

A small stratigraphic layer-cake modelling toolkit.

Workflow
--------
1. Build a `StratModel` on a regular XY grid.
2. Grow the stack upward: `add_basement()` then repeated `deposit()` calls.
   Each `deposit()` call defines a new horizon *and* the `Layer` sandwiched
   between it and the horizon below.
3. Apply structural events to the whole current stack of horizons:
   - `apply_fold()`      - sinusoidal folding / tilting
   - `apply_fault()`     - planar fault (dip, strike, throw)
   - `apply_unconformity()` - erosional truncation surface, beds below
                              are cut off, then deposition can resume
                              on top of it (angular unconformity if its
                              dip/strike differs from the beds it cuts)
4. Assign physical properties (currently density, kg/m^3) to each layer,
   either as a single uniform value or as a spatially varying field
   (`set_density`).
5. Discretize into a SimPEG `discretize` TreeMesh with `build_treemesh()`,
   which also returns a per-cell density array ready to hand to SimPEG
   forward/inverse gravity code.

Sign convention
----------------
All horizon arrays `Z` are **elevation** (m), positive up. Layers are
grown by adding *positive* thickness on top of the previous horizon, so
`model.order` is a bottom-to-top (oldest-to-youngest) list of horizon
names, and `model.order[-1]` is always the current ground surface /
topography.

Everything is vectorised over the model's regular (ny, nx) grid; the
`Basin`/`unit_dip`/`deposit`/`channel`/`fold`/`normal_fault`/`write_tsurf`
functions from the original notebook are kept (or generalised) below so
existing scripts keep working, but the recommended entry point going
forward is the `StratModel` class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def unit_dip(X, Y, dip, strike):
    """Elevation (m) of an infinite plane through the origin with the
    given dip (degrees, downward positive) and strike (degrees),
    evaluated at each (X, Y).

    Kept identical to the original notebook's convention so existing
    dip/strike values used in scripts don't need to change.
    """
    strike_rad = np.radians(-strike)
    slope = np.tan(np.radians(dip))

    ux = np.sin(strike_rad)
    uy = np.cos(strike_rad)

    return -(X * ux + Y * uy) * slope


def plane_z(X, Y, dip, strike, x0=0.0, y0=0.0, z0=0.0):
    """Elevation (m) of a dipping plane passing through (x0, y0, z0)."""
    return z0 + unit_dip(X - x0, Y - y0, dip, strike)


def fold_field(X, Y, amplitude, wavelength, azimuth=0.0, phase=0.0):
    """Sinusoidal fold/warp displacement field (m), added to elevation.

    `azimuth` (degrees) is the direction *across* which elevation
    oscillates (i.e. perpendicular to the fold axis). `wavelength` is in
    the same length units as X/Y. `phase` in radians shifts the fold
    laterally (use this to co-locate anticline/syncline hinges between
    calls, or to offset folds affecting different horizons).
    """
    az = np.radians(azimuth)
    u = X * np.cos(az) + Y * np.sin(az)
    return amplitude * np.sin(2 * np.pi * u / wavelength + phase)


# ----------------------------------------------------------------------
# Faults
# ----------------------------------------------------------------------

class Fault:
    """A planar fault defined by a point on the fault plane, its dip and
    strike, and a throw (vertical separation).

    `throw > 0` drops the hanging wall down (normal fault); `throw < 0`
    raises it (reverse fault). The fault plane is infinite; the
    hanging-wall/footwall split for a given horizon is computed by
    comparing that horizon's *own* elevation, at each map location, to
    the fault plane's elevation at that same location. This reproduces
    the usual "cutoff line" you'd draw where the fault plane intersects
    a horizon, without needing an explicit 3D solid model.

    This is a synthetic-forward-modelling approximation, not a
    kinematic restoration: horizons are displaced independently in the
    order you apply the fault, using their pre-fault geometry to decide
    which side of the plane they're on.
    """

    def __init__(self, dip, strike, throw, x0=0.0, y0=0.0, z0=0.0,
                 name="fault"):
        self.dip = dip
        self.strike = strike
        self.throw = throw
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.name = name

    def plane(self, X, Y):
        return plane_z(X, Y, self.dip, self.strike, self.x0, self.y0, self.z0)

    def hanging_wall_mask(self, X, Y, Z):
        """True where (X, Y, Z) is on the downthrown (hanging-wall) side."""
        return Z < self.plane(X, Y)

    def displace(self, X, Y, Z):
        """Return a new horizon array with the fault throw applied."""
        mask = self.hanging_wall_mask(X, Y, Z)
        Znew = Z.copy()
        Znew[mask] -= self.throw
        return Znew


def normal_fault(surface, X, Y, fault_x=0.0, fault_y=0.0, dip=90.0,
                  strike=0.0, throw=0.0):
    f = Fault(dip=dip, strike=strike, throw=throw, x0=fault_x, y0=fault_y)
    return f.displace(X, Y, surface)


# ----------------------------------------------------------------------
# Unconformities
# ----------------------------------------------------------------------

class Unconformity:
    """An erosional truncation surface.

    Call `truncate()` on every horizon currently in the stack: any part
    of an older horizon that pokes up above the unconformity surface is
    cut back down to it (beds "truncate" against the unconformity).
    Deposition can then resume on top of the returned surface. Give the
    unconformity a different dip/strike to the underlying beds to get an
    angular unconformity; leave them matching (and set noise_scale=0)
    for a purely erosional disconformity.
    """

    def __init__(self, dip, strike, x0=0.0, y0=0.0, z0=0.0,
                 noise_scale=0.0, corr_length=15, name="unconformity"):
        self.dip, self.strike = dip, strike
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.noise_scale = noise_scale
        self.corr_length = corr_length
        self.name = name

    def surface(self, X, Y):
        Z = plane_z(X, Y, self.dip, self.strike, self.x0, self.y0, self.z0)
        if self.noise_scale:
            noise = gaussian_filter(np.random.normal(size=Z.shape),
                                     sigma=self.corr_length)
            Z = Z + noise * self.noise_scale
        return Z

    def truncate(self, X, Y, Z):
        """Erode a single horizon down to this unconformity surface."""
        return np.minimum(Z, self.surface(X, Y))


# ----------------------------------------------------------------------
# Density fields
# ----------------------------------------------------------------------

class DensityModel:
    """Density (kg/m^3) for one layer: a constant mean plus optional
    spatially correlated noise, a vertical (within-layer) gradient, and
    a lateral gradient.

    Callable at *arbitrary* (x, y, z) points (not just the model's
    native grid), so it can be evaluated directly at TreeMesh cell
    centres. `z_top`/`z_base` (the local top/base horizon elevation at
    each query point) are required to compute the within-layer relative
    depth used for `vertical_gradient`.
    """

    def __init__(self, X, Y, mean=2350.0, std=0.0,
                 vertical_gradient=0.0, lateral_gradient=(0.0, 0.0),
                 corr_length=40, seed=None):
        self.mean = mean
        self.std = std
        self.vertical_gradient = vertical_gradient
        self.lateral_gradient = lateral_gradient

        if std:
            rng = np.random.default_rng(seed)
            noise = gaussian_filter(rng.normal(size=X.shape), sigma=corr_length)
            noise_std = noise.std()
            if noise_std > 0:
                noise = noise / noise_std
        else:
            noise = np.zeros_like(X)

        x = X[0, :]
        y = Y[:, 0]
        self._interp = RegularGridInterpolator(
            (y, x), noise, bounds_error=False, fill_value=None)

    def __call__(self, x, y, z, z_top, z_base):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        z_top = np.atleast_1d(np.asarray(z_top, dtype=float))
        z_base = np.atleast_1d(np.asarray(z_base, dtype=float))

        noise = self._interp(np.column_stack([y, x]))
        span = np.maximum(z_top - z_base, 1e-6)
        rel_depth = np.clip((z_top - z) / span, 0.0, 1.0)  # 0 at top, 1 at base

        rho = (self.mean
               + self.std * noise
               + self.vertical_gradient * (1.0 - rel_depth)
               + self.lateral_gradient[0] * x
               + self.lateral_gradient[1] * y)
        return rho


def sample_density_points(X, Y, base_layer, top_layer, density_model,
                           n_points=10000, seed=None):
    ny, nx = top_layer.shape
    rng = np.random.default_rng(seed)

    ix = rng.integers(0, nx, n_points)
    iy = rng.integers(0, ny, n_points)

    x = X[iy, ix]
    y = Y[iy, ix]
    z_top = top_layer[iy, ix]
    z_base = base_layer[iy, ix]

    u = rng.random(n_points)
    z = z_base + u * (z_top - z_base)

    rho = density_model(x, y, z, z_top, z_base)
    return np.column_stack((x, y, z, rho))


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------

@dataclass
class Layer:
    name: str
    top: str      # name of the horizon bounding the top of the layer
    base: str     # name of the horizon bounding the base of the layer
    density: Optional[float] = None            # uniform density, kg/m^3
    density_model: Optional[DensityModel] = None  # spatially varying, kg/m^3


# ----------------------------------------------------------------------
# StratModel
# ----------------------------------------------------------------------

class StratModel:
    """An ordered stack of horizons (elevation surfaces) and the layers
    between them, on a shared regular XY grid."""

    def __init__(self, nx, ny, dx):
        self.nx, self.ny, self.dx = nx, ny, dx

        x = np.arange(nx, dtype=float) * dx
        y = np.arange(ny, dtype=float) * dx
        self.X, self.Y = np.meshgrid(x, y)

        self.horizons: Dict[str, np.ndarray] = {}
        self.order: List[str] = []          # bottom -> top
        self.layers: Dict[str, Layer] = {}
        self.layer_order: List[str] = []    # bottom -> top
        self.events: List[str] = []         # human-readable event log

    # -- horizon construction --------------------------------------

    def add_basement(self, name="basement", noise_scale=100, dip=2, strike=0):
        Z = plane_z(self.X, self.Y, dip, strike)
        if noise_scale:
            noise = gaussian_filter(np.random.normal(size=Z.shape), sigma=10)
            Z = Z + noise * noise_scale

        self.horizons[name] = Z
        self.order.append(name)
        self.events.append(f"basement '{name}': dip={dip}, strike={strike}")
        return Z

    def deposit(self, name, mean=40, noise_scale=15, rel_dip=0, rel_strike=0,
                min_thickness=1.0, base=None):
        """Deposit a new horizon `name` on top of `base` (defaults to the
        current top of the stack), and register the `Layer` between
        them."""
        if not self.order:
            raise ValueError("call add_basement() before deposit()")

        base_name = base or self.order[-1]
        base_Z = self.horizons[base_name]

        if noise_scale:
            noise = gaussian_filter(np.random.normal(size=base_Z.shape), sigma=20)
        else:
            noise = 0
            noise_scale = 0
        thickness = mean + noise * noise_scale
        thickness = thickness + unit_dip(self.X, self.Y, dip=rel_dip, strike=rel_strike)
        thickness = np.maximum(thickness, min_thickness)

        Z = base_Z + thickness
        self.horizons[name] = Z
        self.order.append(name)

        layer_name = f"{base_name}__{name}"
        self.layers[layer_name] = Layer(name=layer_name, top=name, base=base_name)
        self.layer_order.append(layer_name)
        self.events.append(f"deposit '{name}' on '{base_name}': mean={mean}")
        return Z

    # -- structural events (apply to the whole stack so relationships
    #    between horizons - and therefore layer thicknesses - are
    #    preserved) ----------------------------------------------------

    def apply_fold(self, amplitude, wavelength, azimuth=0.0, phase=0.0,
                    horizons: Optional[Sequence[str]] = None):
        disp = fold_field(self.X, self.Y, amplitude, wavelength, azimuth, phase)
        for name in (horizons or self.order):
            self.horizons[name] = self.horizons[name] + disp
        self.events.append(
            f"fold: amplitude={amplitude}, wavelength={wavelength}, azimuth={azimuth}")

    def apply_fault(self, fault: Fault, horizons: Optional[Sequence[str]] = None):
        for name in (horizons or self.order):
            self.horizons[name] = fault.displace(self.X, self.Y, self.horizons[name])
        self.events.append(
            f"fault '{fault.name}': dip={fault.dip}, strike={fault.strike}, throw={fault.throw}")

    def apply_unconformity(self, unconf: Unconformity,
                            horizons: Optional[Sequence[str]] = None):
        """Truncate the existing stack against `unconf`, then add its
        surface as a new horizon on top so subsequent `deposit()` calls
        build onto it."""
        surf = unconf.surface(self.X, self.Y)
        for hname in (horizons or list(self.order)):
            self.horizons[hname] = np.minimum(self.horizons[hname], surf)

        self.horizons[unconf.name] = surf
        self.order.append(unconf.name)
        self.events.append(
            f"unconformity '{unconf.name}': dip={unconf.dip}, strike={unconf.strike}")
        return surf

    # -- physical properties -----------------------------------------

    def set_density(self, layer_name, mean=2350.0, std=0.0,
                     vertical_gradient=0.0, lateral_gradient=(0.0, 0.0),
                     corr_length=40, seed=None):
        """Assign a (possibly spatially varying) density, in kg/m^3, to a
        layer. `std=0` (default) gives a uniform layer density."""
        if std == 0 and vertical_gradient == 0 and lateral_gradient == (0.0, 0.0):
            self.layers[layer_name].density = mean
            self.layers[layer_name].density_model = None
        else:
            dm = DensityModel(self.X, self.Y, mean=mean, std=std,
                               vertical_gradient=vertical_gradient,
                               lateral_gradient=lateral_gradient,
                               corr_length=corr_length, seed=seed)
            self.layers[layer_name].density_model = dm
            self.layers[layer_name].density = None

    # -- helpers ---------------------------------------------------

    def horizon_interpolator(self, name) -> RegularGridInterpolator:
        Z = self.horizons[name]
        x = self.X[0, :]
        y = self.Y[:, 0]
        return RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=None)

    def write_tsurf(self, name, filename):
        write_tsurf(filename, self.X, self.Y, self.horizons[name])


# ----------------------------------------------------------------------
# GOCAD TSurf export
# ----------------------------------------------------------------------

def write_tsurf(filename, X, Y, Z):
    ny, nx = Z.shape

    with open(filename, "w") as f:
        f.write("GOCAD TSurf 1\n")
        f.write("HEADER {\n")
        f.write("name:surface\n")
        f.write("}\n")
        f.write("TFACE\n")

        vid = 1
        ids = np.zeros_like(Z, dtype=int)

        for i in range(ny):
            for j in range(nx):
                ids[i, j] = vid
                f.write(f"VRTX {vid} {X[i, j]} {Y[i, j]} {Z[i, j]}\n")
                vid += 1

        for i in range(ny - 1):
            for j in range(nx - 1):
                a = ids[i, j]
                b = ids[i, j + 1]
                c = ids[i + 1, j]
                d = ids[i + 1, j + 1]
                f.write(f"TRGL {a} {b} {c}\n")
                f.write(f"TRGL {b} {d} {c}\n")

        f.write("END\n")


# ----------------------------------------------------------------------
# TreeMesh discretisation
# ----------------------------------------------------------------------

def build_treemesh(model: StratModel, cell_size,
                    domain_pad=None, z_pad_below=None, z_pad_above=None,
                    octree_levels_topo=(1, 1), octree_levels_layers=(1, 1),
                    diagonal_balance=True):
    """Discretize a `StratModel` into a SimPEG `discretize` TreeMesh.

    The mesh is built with `discretize.utils.mesh_builder_xyz` and
    refined with `TreeMesh.refine_surface` at the ground surface (the
    youngest/top horizon) and at every internal horizon, so layer
    contacts get extra resolution without refining the whole volume.

    Requires `discretize` (and, for a full geophysical pipeline,
    `simpeg`) to be installed - not part of this module's own
    dependencies. Tested against the `discretize.TreeMesh.refine_surface`
    / `discretize.utils.mesh_builder_xyz` / `discretize.utils.active_from_xyz`
    API (discretize >= 0.9); if you're on an older discretize you may
    need `discretize.utils.refine_tree_xyz` instead - check your
    installed version if this raises an AttributeError/ImportError.

    Parameters
    ----------
    model : StratModel
    cell_size : float
        Base (finest) cell size, same units as the model's dx.
    domain_pad : float, optional
        Horizontal padding beyond the model footprint. Defaults to
        5 * cell_size.
    z_pad_below, z_pad_above : float, optional
        Extra vertical padding below the basement / above the ground
        surface. Defaults to 20% of the model's total relief / 5 cells.
    octree_levels_topo, octree_levels_layers : tuple of int
        `padding_cells_by_level` passed to `refine_surface` for the
        ground surface and for internal horizons respectively.

    Returns
    -------
    mesh : discretize.TreeMesh
    density : (mesh.n_cells,) ndarray
        Density in kg/m^3, NaN where no layer/density was assigned
        (including air cells above topography).
    active : (mesh.n_cells,) ndarray of bool
        True for cells below the ground surface (from active_from_xyz).
    """
    from discretize.utils import mesh_builder_xyz, active_from_xyz

    if not model.order:
        raise ValueError("model has no horizons - build the stack first")

    top_name = model.order[-1]
    base_name = model.order[0]

    top_Z = model.horizons[top_name]
    base_Z = model.horizons[base_name]

    xyz_topo = np.column_stack([model.X.ravel(), model.Y.ravel(), top_Z.ravel()])

    relief = float(top_Z.max() - base_Z.min())
    pad = domain_pad if domain_pad is not None else 5 * cell_size
    z_below = z_pad_below if z_pad_below is not None else 0.2 * relief
    z_above = z_pad_above if z_pad_above is not None else 5 * cell_size

    mesh = mesh_builder_xyz(
        xyz_topo,
        [cell_size, cell_size, cell_size],
        padding_distance=[[pad, pad], [pad, pad], [z_below, z_above]],
        depth_core=relief,
        mesh_type="tree",
    )

    mesh.refine_surface(xyz_topo, padding_cells_by_level=list(octree_levels_topo),
                         finalize=False)

    for name in model.order[1:-1]:
        Zh = model.horizons[name]
        xyz_h = np.column_stack([model.X.ravel(), model.Y.ravel(), Zh.ravel()])
        mesh.refine_surface(xyz_h, padding_cells_by_level=list(octree_levels_layers),
                             finalize=False)

    mesh.finalize()

    active = active_from_xyz(mesh, xyz_topo)

    cc = mesh.cell_centers
    density = np.full(mesh.n_cells, np.nan)

    interpolators = {name: model.horizon_interpolator(name) for name in model.order}
    pts_yx = np.column_stack([cc[:, 1], cc[:, 0]])

    for layer_name in model.layer_order:
        layer = model.layers[layer_name]
        z_top = interpolators[layer.top](pts_yx)
        z_base = interpolators[layer.base](pts_yx)

        in_layer = active & (cc[:, 2] <= z_top) & (cc[:, 2] > z_base)
        if not np.any(in_layer):
            continue

        if layer.density_model is not None:
            density[in_layer] = layer.density_model(
                cc[in_layer, 0], cc[in_layer, 1], cc[in_layer, 2],
                z_top[in_layer], z_base[in_layer])
        elif layer.density is not None:
            density[in_layer] = layer.density

    return mesh, density, active
