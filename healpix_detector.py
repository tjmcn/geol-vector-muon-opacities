"""
healpix_detector.py
===================

Simulated detector whose field of view is tiled by HEALPix cells, each
cell being a pyramid with its apex at the detector and pointing upwards.

* `cell_shape='triangle'` (default): every HEALPix pixel (a curvilinear
  quadrilateral) is split into two spherical triangles, so each cell is a
  triangular pyramid = tetrahedron (apex at the detector + 3 base vertices).
* `cell_shape='quad'`: one four-sided pyramid per HEALPix pixel (the true
  pixel footprint).

Conventions
-----------
* Local frame: +x east, +y north, +z up (the detector axis = HEALPix pole).
* Elevation is measured up from the horizon (zenith = +90 deg). Note this
  is `z = +sin(el)`; your `deg_to_vec` uses `z = -sin(el)`, so negate z or
  flip the elevation sign if you feed these into code that expects that.
* Azimuth is compass-style: clockwise from north (+y), matching the
  `x = cos(el) sin(az), y = cos(el) cos(az)` convention in `deg_to_vec`.
* Pixel indices use HEALPix NESTED ordering (`pixel_ids`). See `selftest()`
  for how to cross-check them against healpy if you have it installed.

Angular resolution
------------------
HEALPix pixels have equal area 4*pi/(12 nside^2) sr, so the (linear)
resolution is `sqrt(area)` = 58.63 deg / nside. `nside` must be a power of
two for NESTED ordering, so the requested resolution is rounded to the next
FINER power-of-two grid; the resolution actually obtained is reported by
`HealpixDetector.resolution_deg`. Triangle cells have half a pixel's area.

Quick start
-----------
    from healpix_detector import make_healpix_detector

    det = make_healpix_detector(origin=(1024, 1024, 0),
                                angular_resolution_deg=5,
                                min_elevation_deg=45)
    print(det)
    det.plot_sky()

    results = trace_rays_density(model, det.origin, det.central_dirs)

    # write the bins to HDF5 (needs h5py): opacity, opacity_unc, bin_vertices,
    # bin_solid_angle, bin_index  (dummy opacity/uncertainty unless passed arrays).
    # bin_vertices is (n_bins, n_corners, 2) = [phi, theta]; choose the convention with
    # vertices_format= ('az_el' default = [azimuth clockwise from north, elevation], deg).
    det.to_h5("detector.h5")                      # group "site_0"
    det.to_h5("detector.h5", group=None)          # datasets at the file root instead
    det_import = read_detector_h5("detector.h5")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# HEALPix base-face tables (healpix_base conventions): ring row / longitude
# offset of each of the 12 base pixels.
_JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
_JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)

# sqrt(4*pi/12) in degrees: linear resolution of an nside=1 pixel.
_RES_NSIDE1_DEG = float(np.degrees(np.sqrt(np.pi / 3.0)))


# ----------------------------------------------------------------------
# Resolution <-> nside
# ----------------------------------------------------------------------

def nside_for_resolution(resolution_deg: float) -> int:
    """Smallest power-of-two nside whose pixel resolution is <= `resolution_deg`."""
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    exact = _RES_NSIDE1_DEG / resolution_deg
    if exact <= 1.0:
        return 1
    return int(2 ** int(np.ceil(np.log2(exact) - 1e-9)))


def resolution_for_nside(nside: int) -> float:
    """Linear pixel resolution (deg) = sqrt(pixel area)."""
    return _RES_NSIDE1_DEG / nside


# ----------------------------------------------------------------------
# HEALPix geometry (NESTED ordering)
# ----------------------------------------------------------------------

def _check_nside(nside: int) -> int:
    nside = int(nside)
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a power of two, got {nside}")
    return nside


def _nest_to_face_xy(pix: np.ndarray, nside: int):
    """NESTED pixel index -> (base face, ix, iy) within that face."""
    order = int(np.log2(nside))
    npface = nside * nside
    face = pix // npface
    ipf = pix % npface
    ix = np.zeros_like(ipf)
    iy = np.zeros_like(ipf)
    for b in range(order):
        ix |= ((ipf >> (2 * b)) & 1) << b
        iy |= ((ipf >> (2 * b + 1)) & 1) << b
    return face, ix, iy


def _plane_to_vec(x: np.ndarray, y: np.ndarray, x_face: np.ndarray) -> np.ndarray:
    """Inverse HEALPix projection: plane (x, y) -> unit vectors on the sphere.

    Equatorial zone (|y| <= pi/4): z = 8y/(3 pi), phi = x.
    Polar caps: sigma = 2 - 4|y|/pi, z = sign(y)(1 - sigma^2/3),
    phi = x_face + (x - x_face)/sigma, where `x_face` is the longitude of the
    centre of the base face the point belongs to.
    """
    absy = np.abs(y)
    sigma = np.clip(2.0 - 4.0 * absy / np.pi, 0.0, None)

    z_eq = 8.0 * y / (3.0 * np.pi)
    z_pol = np.sign(y) * (1.0 - sigma ** 2 / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_pol = np.where(sigma > 1e-12, x_face + (x - x_face) / sigma, x_face)

    equatorial = absy <= np.pi / 4.0
    z = np.where(equatorial, z_eq, z_pol)
    phi = np.where(equatorial, x, phi_pol)

    s = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=-1)


def healpix_pixel_geometry(nside: int, pix: Optional[np.ndarray] = None):
    """Centres and corners of HEALPix pixels (NESTED ordering).

    Returns
    -------
    centers : (N, 3) unit vectors
    corners : (N, 4, 3) unit vectors ordered S, E, N, W. This is
        counter-clockwise when viewed from outside the sphere, i.e.
        (v_i x v_{i+1}) . centre > 0.
    """
    nside = _check_nside(nside)
    if pix is None:
        pix = np.arange(12 * nside * nside, dtype=np.int64)
    pix = np.asarray(pix, dtype=np.int64)

    face, ix, iy = _nest_to_face_xy(pix, nside)
    jr, jp = _JRLL[face], _JPLL[face]
    x_face = (np.pi / 4.0) * jp

    def vec(du, dv):
        u = (ix + du) / nside
        v = (iy + dv) / nside
        x = (np.pi / 4.0) * (jp + u - v)
        y = (np.pi / 4.0) * (2 - jr + u + v)
        return _plane_to_vec(x, y, x_face)

    centers = vec(0.5, 0.5)
    corners = np.stack([vec(0, 0), vec(1, 0), vec(1, 1), vec(0, 1)], axis=1)
    return centers, corners


# ----------------------------------------------------------------------
# Small spherical-geometry helpers
# ----------------------------------------------------------------------

def spherical_triangle_solid_angle(a, b, c):
    """Solid angle (sr) of the geodesic triangle with unit-vector corners a, b, c
    (Van Oosterom & Strakhov): tan(W/2) = |a.(b x c)| / (1 + a.b + b.c + c.a)."""
    num = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    den = 1.0 + np.sum(a * b, -1) + np.sum(b * c, -1) + np.sum(c * a, -1)
    return 2.0 * np.arctan2(num, den)


def vec_to_az_el(v):
    """Unit vectors (z up) -> (azimuth_deg clockwise from north, elevation_deg)."""
    v = np.asarray(v, dtype=float)
    el = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    az = (90.0 - np.degrees(np.arctan2(v[..., 1], v[..., 0]))) % 360.0
    return az, el


def az_el_to_vec(az_deg, el_deg):
    """Inverse of `vec_to_az_el` (z = +sin(el), i.e. elevation positive upwards)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], -1)


def _fan_triangles(vertex_dirs):
    """(M,K,3) cell vertices -> (M,T,3,3) triangles (T=1 for K=3, T=2 for K=4)."""
    if vertex_dirs.shape[1] == 3:
        return vertex_dirs[:, None, :, :]
    return np.stack([vertex_dirs[:, [0, 1, 2]], vertex_dirs[:, [0, 2, 3]]], axis=1)


# ----------------------------------------------------------------------
# The detector
# ----------------------------------------------------------------------

@dataclass(repr=False)
class HealpixDetector:
    """A set of pyramidal acceptance cells with a common apex at `origin`.

    Attributes (M = number of cells, K = 3 for triangles / 4 for quads)
    ----------
    origin           : (3,) apex position (the detector location)
    nside            : HEALPix nside of the parent grid
    cell_shape       : 'triangle' or 'quad'
    pixel_ids        : (M,) NESTED id of the parent HEALPix pixel
    sub_index        : (M,) 0/1 which half of the pixel (triangles); 0 for quads
    vertex_dirs      : (M, K, 3) unit vectors from the apex to each base vertex,
                       counter-clockwise seen from outside
    central_dirs     : (M, 3) one representative direction per cell: normalised
                       mean of the vertex directions (triangles) or the HEALPix
                       pixel centre (quads). Use this for one-ray-per-cell tracing.
    pixel_center_dirs: (M, 3) HEALPix centre of each cell's parent pixel
    solid_angles     : (M,) cell solid angle in steradians
    azimuth_deg, elevation_deg : (M,) of `central_dirs`
    """

    origin: np.ndarray
    nside: int
    cell_shape: str
    pixel_ids: np.ndarray
    sub_index: np.ndarray
    vertex_dirs: np.ndarray
    central_dirs: np.ndarray
    pixel_center_dirs: np.ndarray
    solid_angles: np.ndarray
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray

    # -- basic info ----------------------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.solid_angles)

    @property
    def resolution_deg(self) -> float:
        """Linear resolution of the parent HEALPix pixels (deg)."""
        return resolution_for_nside(self.nside)

    def __repr__(self):
        sa = self.solid_angles
        return (
            f"HealpixDetector(origin={np.asarray(self.origin).tolist()}, "
            f"nside={self.nside}, resolution={self.resolution_deg:.3f} deg, "
            f"shape='{self.cell_shape}', n_cells={self.n_cells}, "
            f"elevation=[{self.elevation_deg.min():.1f}, {self.elevation_deg.max():.1f}] deg, "
            f"solid angle per cell=[{sa.min():.3e}, {sa.max():.3e}] sr, "
            f"total={sa.sum():.4f} sr)"
        )

    # -- geometry at a given range ------------------------------------
    def vertices_at_range(self, length: float) -> np.ndarray:
        """(M, K, 3) positions of the pyramid base vertices at distance `length`
        from the apex (e.g. `length` = distance to topography)."""
        return np.asarray(self.origin, float) + float(length) * self.vertex_dirs

    def pyramid_volumes(self, length: float, base: str = "flat") -> np.ndarray:
        """Volume (m^3) of each pyramid out to range `length`.

        base='flat'     : base vertices joined by planar faces (a true tetrahedron
                          for triangle cells; two tetrahedra for quads).
        base='spherical': spherical sector, volume = solid_angle * length^3 / 3.
        """
        if base == "spherical":
            return self.solid_angles * length ** 3 / 3.0
        if base != "flat":
            raise ValueError("base must be 'flat' or 'spherical'")
        tris = _fan_triangles(self.vertex_dirs)  # (M, T, 3, 3)
        det = np.abs(np.einsum("...i,...i->...", tris[..., 0, :],
                               np.cross(tris[..., 1, :], tris[..., 2, :])))
        return det.sum(axis=1) * length ** 3 / 6.0

    # -- sampling ------------------------------------------------------
    def sample_directions(self, n_per_cell: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Directions drawn uniformly in SOLID ANGLE inside each cell.

        Returns (M, n_per_cell, 3) unit vectors. Uses rejection sampling
        (uniform points on each flat triangle, accepted with probability
        (d/r)^3 where d is the plane's distance from the apex), which is
        exact; averaging a quantity over these gives the cell's field-of-view
        average rather than a single central-ray value.
        """
        rng = np.random.default_rng(seed)
        M, n = self.n_cells, int(n_per_cell)
        tris = _fan_triangles(self.vertex_dirs)  # (M, T, 3, 3)
        T = tris.shape[1]

        omega = spherical_triangle_solid_angle(tris[..., 0, :], tris[..., 1, :], tris[..., 2, :])
        p = omega / omega.sum(axis=1, keepdims=True)
        choice = (rng.random((M, n, 1)) > np.cumsum(p, axis=1)[:, None, :]).sum(-1)
        choice = np.minimum(choice, T - 1)

        rows = np.arange(M)[:, None]
        A = tris[rows, choice, 0].reshape(-1, 3)
        B = tris[rows, choice, 1].reshape(-1, 3)
        C = tris[rows, choice, 2].reshape(-1, 3)

        normal = np.cross(B - A, C - A)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        d = np.abs(np.einsum("ij,ij->i", normal, A))

        out = np.zeros((M * n, 3))
        todo = np.arange(M * n)
        while todo.size:
            r1, r2 = rng.random(todo.size), rng.random(todo.size)
            flip = r1 + r2 > 1.0
            r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
            pts = A[todo] + r1[:, None] * (B[todo] - A[todo]) + r2[:, None] * (C[todo] - A[todo])
            r = np.linalg.norm(pts, axis=1)
            keep = rng.random(todo.size) < (d[todo] / r) ** 3
            out[todo[keep]] = pts[keep] / r[keep, None]
            todo = todo[~keep]
        return out.reshape(M, n, 3)

    # -- plotting ------------------------------------------------------
    def plot_sky(self, values=None, ax=None, cmap="viridis", edgecolor="k",
                 linewidth=0.3, label="", vmin=None, vmax=None):
        """Plan-view 'fisheye' map of the cells: radius = zenith angle, x = east,
        y = north. Colour by `values` (length M) if given, e.g. per-cell opacity."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))
        az, el = vec_to_az_el(self.vertex_dirs)
        r, a = 90.0 - el, np.radians(az)
        polys = np.stack([r * np.sin(a), r * np.cos(a)], axis=-1)  # (M, K, 2)

        pc = PolyCollection(polys, cmap=cmap, edgecolors=edgecolor, linewidths=linewidth)
        if values is not None:
            pc.set_array(np.asarray(values))
            pc.set_clim(vmin, vmax)
        else:
            pc.set_facecolor("none")
        ax.add_collection(pc)
        lim = 90.0 - self.elevation_deg.min() + self.resolution_deg
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("East (deg from zenith)")
        ax.set_ylabel("North (deg from zenith)")
        if values is not None:
            ax.figure.colorbar(pc, ax=ax, label=label)
        return ax

    def plot_3d(self, length: float = 1.0, values=None, ax=None, cmap="viridis",
                show_edges: bool = True, alpha: float = 0.6):
        """3D view of the pyramids out to range `length` (base faces + edges to the apex)."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ax is None:
            ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
        base = self.vertices_at_range(length)  # (M, K, 3)
        coll = Poly3DCollection(base, alpha=alpha, edgecolor="k", linewidths=0.3)
        if values is not None:
            vals = np.asarray(values, float)
            colours = plt.get_cmap(cmap)((vals - vals.min()) / max(np.ptp(vals), 1e-30))
            coll.set_facecolor(colours)
        else:
            coll.set_facecolor("tab:blue")
        ax.add_collection3d(coll)

        if show_edges:
            apex = np.asarray(self.origin, float)
            for cell in base:
                for v in cell:
                    ax.plot(*zip(apex, v), color="gray", lw=0.2)
        ax.scatter(*np.asarray(self.origin, float), color="red", marker="^", s=60)
        pts = np.vstack([base.reshape(-1, 3), np.asarray(self.origin, float)[None]])
        mid, half = pts.mean(0), 0.5 * np.ptp(pts, axis=0).max()
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_xlabel("x (east)")
        ax.set_ylabel("y (north)")
        ax.set_zlabel("z (up)")
        return ax

    # -- I/O ------------------------------------------------------------
    def encode_vertices(self, vertices_format: str = "az_el", angle_units: str = "deg",
                        unwrap_azimuth: bool = True) -> np.ndarray:
        """Bin corner directions as an (M, K, 2) array of angle pairs (or (M, K, 3)
        unit vectors for 'unit_vectors'). See `to_h5` for the formats."""
        if vertices_format == "unit_vectors":
            return np.asarray(self.vertex_dirs, dtype=float).copy()
        if vertices_format not in ("az_el", "az_zenith", "iso_phi_theta"):
            raise ValueError("vertices_format must be 'az_el', 'az_zenith', "
                             "'iso_phi_theta' or 'unit_vectors'")
        if angle_units not in ("deg", "rad"):
            raise ValueError("angle_units must be 'deg' or 'rad'")

        v = self.vertex_dirs
        az, el = vec_to_az_el(v)   
        el = 90-el                   # compass azimuth, elevation (deg)
        cen_az = self.azimuth_deg[:, None]
        az = np.where(np.hypot(v[..., 0], v[..., 1]) < 1e-9, cen_az, az)  # zenith corner
        if unwrap_azimuth:
            az = cen_az + (az - cen_az + 180.0) % 360.0 - 180.0
        else:
            az = az % 360.0

        if vertices_format == "az_el":
            first, second = az, el
        elif vertices_format == "az_zenith":
            first, second = az, 90.0 - el
        else:                                         # iso_phi_theta
            first, second = 90.0 - az, 90.0 - el
            if not unwrap_azimuth:
                first = first % 360.0
        # round off ~1e-14 float noise (and -0.0) so values like 0 / 360 come out clean
        out = np.round(np.stack([first, second], axis=-1), 10) + 0.0
        return np.radians(out) if angle_units == "rad" else out

    def reference_direction(self) -> Tuple[float, float]:
        """(azimuth_deg, elevation_deg) of the solid-angle-weighted mean cell
        direction. For a detector with full azimuth coverage this is straight
        up (elevation 90, azimuth 0 by convention since azimuth is undefined)."""
        m = (self.central_dirs * self.solid_angles[:, None]).sum(axis=0)
        norm = np.linalg.norm(m)
        if norm == 0.0:
            return 0.0, 90.0
        u = m / norm
        az, el = vec_to_az_el(u)
        if np.hypot(u[0], u[1]) < 1e-9:
            az = 0.0
        return float(az), float(el)

    def to_h5(self, path: str, group: Optional[str] = "site_0", mode: str = "w",
              overwrite_group: bool = False,
              opacity=None, opacity_unc=None,
              dummy_opacity: float = 0.0, dummy_unc: float = 1.0,
              data_col: str = "opacity", unc_col: str = "opacity_unc",
              vertices_col: str = "bin_vertices",
              solid_angle_col: str = "bin_solid_angle",
              index_col: str = "bin_index",
              reference_azimuth: Optional[float] = None,
              reference_elevation: Optional[float] = None,
              vertices_format: str = "az_el", angle_units: str = "deg",
              unwrap_azimuth: bool = True,
              extra_datasets: bool = False) -> str:
        """Write the detector bins to an HDF5 file (requires h5py).

        Datasets written (names are the `*_col` arguments), one entry per bin:

            data_col         "opacity"          (M,)      dummy values, or `opacity`
            unc_col          "opacity_unc"      (M,)      dummy values, or `opacity_unc`
            vertices_col     "bin_vertices"     (M,K,2)   [phi, theta] of each bin corner
                                                          (K=3 for triangles, 4 for quads),
                                                          counter-clockwise seen from
                                                          outside; see `vertices_format`
                                                          (or (M,K,3) unit vectors)
            solid_angle_col  "bin_solid_angle"  (M,)      steradians
            index_col        "bin_index"        (M,)      int64, 0..M-1

        Group attributes: `location` / `detector_x,y,z`, `reference_azimuth`,
        `reference_elevation` (degrees; default = the solid-angle-weighted mean bin
        direction, so a reader's "reference vs bin centroid" cross-check passes),
        `nside`, `resolution_deg`, `cell_shape`, `n_bins`, `n_vertices`.

        Parameters
        ----------
        group : name of the HDF5 group to write into (default "site_0"); pass None
            to write the datasets at the file root instead.
        mode : 'w' (new file, overwrites) or 'a' (add a group to an existing file,
            e.g. to hold several detectors in one file).
        overwrite_group : in mode 'a', replace datasets that already exist.
        opacity, opacity_unc : optional (M,) arrays to store instead of the dummy
            values (e.g. traced opacities and their uncertainties).
        dummy_opacity, dummy_unc : constant placeholder values. `dummy_unc` defaults
            to 1.0 rather than 0 so that anything computing 1/uncertainty weights
            doesn't divide by zero.
        vertices_format : how each corner direction is encoded in `bin_vertices`,
            as [first, second] columns:
              'az_el'     (default) [phi, theta] = [azimuth clockwise from north,
                          elevation above the horizon (90 = vertical)]. This is the
                          convention the muon forward-modelling package expects.
              'az_zenith' [azimuth clockwise from north, zenith angle]
              'iso_phi_theta' [phi, theta] in the ISO *physics* convention: phi
                          counter-clockwise from +x (east), theta = polar angle from
                          +z (up). NOT the same as the package's phi/theta.
              'unit_vectors' shape (M,K,3): unit vectors (x east, y north, z up)
        angle_units : 'deg' (default) or 'rad' for the angle formats.
        unwrap_azimuth : keep each bin's azimuths continuous (a bin straddling the
            0/360 seam gets e.g. 359 and 361 instead of 359 and 1) so that anything
            averaging or interpolating the angles works; values may then fall outside
            [0, 360). False wraps everything into [0, 360). A corner exactly at the
            zenith (azimuth undefined) takes its bin's centroid azimuth.
        extra_datasets : also write `bin_central_dir`, `bin_azimuth_deg`,
            `bin_elevation_deg` and `healpix_pixel_id` (off by default so the file
            holds only the five columns above).
        """
        import h5py

        if mode not in ("w", "a"):
            raise ValueError("mode must be 'w' or 'a'")
        n = self.n_cells
        vertices = self.encode_vertices(vertices_format, angle_units, unwrap_azimuth)

        def column(values, dummy, name):
            if values is None:
                return np.full(n, float(dummy))
            arr = np.asarray(values, dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            return arr

        datasets = {
            data_col: column(opacity, dummy_opacity, "opacity"),
            unc_col: column(opacity_unc, dummy_unc, "opacity_unc"),
            vertices_col: vertices,
            solid_angle_col: np.asarray(self.solid_angles, dtype=float),
            index_col: np.arange(n, dtype=np.int64),
        }
        if extra_datasets:
            datasets.update({
                "bin_central_dir": np.asarray(self.central_dirs, dtype=float),
                "bin_azimuth_deg": np.asarray(self.azimuth_deg, dtype=float),
                "bin_elevation_deg": np.asarray(self.elevation_deg, dtype=float),
                "healpix_pixel_id": np.asarray(self.pixel_ids, dtype=np.int64),
            })
        if len(datasets) != (9 if extra_datasets else 5):
            raise ValueError("column names must all be distinct "
                             "(and not clash with the extra_datasets names)")

        ref_az, ref_el = self.reference_direction()
        if reference_azimuth is not None:
            ref_az = float(reference_azimuth)
        if reference_elevation is not None:
            ref_el = float(reference_elevation)
        origin = np.asarray(self.origin, dtype=float)

        root_layout = group in (None, "", "/")
        with h5py.File(path, mode) as f:
            if root_layout:
                grp = f
            else:
                if group in f:
                    if not overwrite_group:
                        raise ValueError(f"group '{group}' already exists in {path}; "
                                         "pass overwrite_group=True to replace it")
                    del f[group]
                grp = f.create_group(group)

            for name, arr in datasets.items():
                if name in grp:
                    if not overwrite_group:
                        raise ValueError(f"dataset '{name}' already exists in {path}; "
                                         "pass overwrite_group=True to replace it")
                    del grp[name]
                grp.create_dataset(name, data=arr)

            grp.attrs["location"] = origin
            grp.attrs["detector_x"] = float(origin[0])
            grp.attrs["detector_y"] = float(origin[1])
            grp.attrs["detector_z"] = float(origin[2])
            grp.attrs["reference_azimuth"] = ref_az
            grp.attrs["reference_elevation"] = ref_el
            grp.attrs["nside"] = int(self.nside)
            grp.attrs["resolution_deg"] = float(self.resolution_deg)
            grp.attrs["cell_shape"] = str(self.cell_shape)
            grp.attrs["n_bins"] = int(n)
            grp.attrs["n_vertices"] = int(self.vertex_dirs.shape[1])
            grp.attrs["vertices_format"] = str(vertices_format)
            grp.attrs["vertices_angle_units"] = str(angle_units) if vertices_format != "unit_vectors" else "none"
            grp.attrs["vertices_columns"] = {
                "az_el": "phi=azimuth_clockwise_from_north, theta=elevation_above_horizon",
                "az_zenith": "azimuth_clockwise_from_north, zenith_angle",
                "iso_phi_theta": "phi_ccw_from_east, theta_polar_from_up",
                "unit_vectors": "x_east, y_north, z_up",
            }[vertices_format]
        return path

    def to_npz(self, path: str):
        np.savez(path, origin=self.origin, nside=self.nside, cell_shape=self.cell_shape,
                 pixel_ids=self.pixel_ids, sub_index=self.sub_index,
                 vertex_dirs=self.vertex_dirs, central_dirs=self.central_dirs,
                 pixel_center_dirs=self.pixel_center_dirs, solid_angles=self.solid_angles,
                 azimuth_deg=self.azimuth_deg, elevation_deg=self.elevation_deg)


# ----------------------------------------------------------------------

def make_healpix_detector(origin=(0.0, 0.0, 0.0),
                          angular_resolution_deg: Optional[float] = None,
                          nside: Optional[int] = None,
                          cell_shape: str = "triangle",
                          min_elevation_deg: float = 0.0,
                          max_elevation_deg: float = 90.0,
                          az_range_deg: Optional[Tuple[float, float]] = None,
                          select: str = "centroid",
                          split: str = "shortest") -> HealpixDetector:
    """Build an upward-pointing detector tiled by HEALPix pyramids.

    Parameters
    ----------
    origin : (3,) detector location (pyramid apex).
    angular_resolution_deg : desired linear cell size; rounded to the next finer
        valid HEALPix grid (see `HealpixDetector.resolution_deg` for what you got).
    nside : give this instead of a resolution to fix the grid directly (power of 2).
    cell_shape : 'triangle' (two tetrahedra per pixel) or 'quad' (one 4-sided
        pyramid per pixel).
    min_elevation_deg, max_elevation_deg : keep cells inside this elevation band
        (90 = straight up). E.g. min_elevation_deg=45 keeps the cone within 45 deg
        of the zenith.
    az_range_deg : optional (az_min, az_max), compass azimuth clockwise from north;
        az_min > az_max wraps through north, e.g. (350, 10).
    select : how a cell is tested against the elevation band -
        'centroid' (its central direction), 'all_vertices' (fully inside), or
        'any_vertex' (touches the band).
    split : for triangle cells, which pixel diagonal to cut along:
        'shortest' (per pixel, gives the most even triangles), 'ns', or 'ew'.
    """
    if (angular_resolution_deg is None) == (nside is None):
        raise ValueError("give exactly one of angular_resolution_deg or nside")
    if cell_shape not in ("triangle", "quad"):
        raise ValueError("cell_shape must be 'triangle' or 'quad'")
    if select not in ("centroid", "all_vertices", "any_vertex"):
        raise ValueError("select must be 'centroid', 'all_vertices' or 'any_vertex'")
    if split not in ("shortest", "ns", "ew"):
        raise ValueError("split must be 'shortest', 'ns' or 'ew'")

    nside = _check_nside(nside) if nside is not None else nside_for_resolution(angular_resolution_deg)
    centers, corners = healpix_pixel_geometry(nside)  # corners: S, E, N, W
    pix = np.arange(len(centers), dtype=np.int64)

    if cell_shape == "quad":
        verts = corners
        parent, sub = pix, np.zeros_like(pix)
        central = centers.copy()
        sa = (spherical_triangle_solid_angle(verts[:, 0], verts[:, 1], verts[:, 2])
              + spherical_triangle_solid_angle(verts[:, 0], verts[:, 2], verts[:, 3]))
    else:
        S, E, N, W = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
        if split == "ns":
            use_ns = np.ones(len(pix), bool)
        elif split == "ew":
            use_ns = np.zeros(len(pix), bool)
        else:
            use_ns = np.linalg.norm(S - N, axis=1) <= np.linalg.norm(E - W, axis=1)
        sel = use_ns[:, None, None]
        t0 = np.where(sel, np.stack([S, E, N], 1), np.stack([S, E, W], 1))
        t1 = np.where(sel, np.stack([S, N, W], 1), np.stack([E, N, W], 1))
        verts = np.stack([t0, t1], axis=1).reshape(-1, 3, 3)  # (2*npix, 3, 3)
        parent = np.repeat(pix, 2)
        sub = np.tile([0, 1], len(pix))
        central = verts.sum(axis=1)
        central /= np.linalg.norm(central, axis=1, keepdims=True)
        sa = spherical_triangle_solid_angle(verts[:, 0], verts[:, 1], verts[:, 2])

    pixel_centers = centers[parent]

    # ---- select cells -------------------------------------------------
    tol = 1e-9
    zlo = np.sin(np.radians(min_elevation_deg)) - tol
    zhi = np.sin(np.radians(max_elevation_deg)) + tol
    if select == "centroid":
        keep = (central[:, 2] >= zlo) & (central[:, 2] <= zhi)
    else:
        vz = verts[..., 2]
        inside = (vz >= zlo) & (vz <= zhi)
        keep = inside.all(1) if select == "all_vertices" else inside.any(1)

    if az_range_deg is not None:
        lo, hi = az_range_deg
        span = (hi - lo) % 360.0
        if span == 0.0 and hi != lo:
            span = 360.0
        az_c, _ = vec_to_az_el(central)
        keep &= ((az_c - lo) % 360.0) <= span + 1e-9

    if not keep.any():
        raise ValueError("no cells fall inside the requested elevation/azimuth limits")

    az, el = vec_to_az_el(central[keep])
    return HealpixDetector(
        origin=np.asarray(origin, dtype=float),
        nside=nside,
        cell_shape=cell_shape,
        pixel_ids=parent[keep],
        sub_index=sub[keep],
        vertex_dirs=verts[keep],
        central_dirs=central[keep],
        pixel_center_dirs=pixel_centers[keep],
        solid_angles=sa[keep],
        azimuth_deg=az,
        elevation_deg=el,
    )


# ----------------------------------------------------------------------
# HDF5 read-back
# ----------------------------------------------------------------------

def read_detector_h5(path: str, group: Optional[str] = "site_0",
                     data_col: str = "opacity", unc_col: str = "opacity_unc",
                     vertices_col: str = "bin_vertices",
                     solid_angle_col: str = "bin_solid_angle",
                     index_col: str = "bin_index") -> dict:
    """Read back a file written by `HealpixDetector.to_h5`.

    Returns a dict keyed by the column names you pass (the same names used
    when writing), plus an 'attrs' dict of the group attributes. Handy for
    checking that a file has the layout your downstream loader expects.
    """
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        grp = f if group in (None, "", "/") else f[group]
        for col in (data_col, unc_col, vertices_col, solid_angle_col, index_col):
            out[col] = grp[col][()]
        out["attrs"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in grp.attrs.items()}
    return out


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _ring_scheme_centers(nside: int) -> np.ndarray:
    """Independent pixel centres from the closed-form HEALPix RING formulas
    (Gorski et al. 2005), for checking the NESTED construction above."""
    pts = []
    for i in range(1, 4 * nside):
        if i < nside:                       # north polar cap
            z, nj, dphi, offset = 1.0 - i * i / (3.0 * nside ** 2), 4 * i, np.pi / (2 * i), 0.5
        elif i <= 3 * nside:                # equatorial belt
            z = 4.0 / 3.0 - 2.0 * i / (3.0 * nside)
            nj, dphi = 4 * nside, np.pi / (2 * nside)
            offset = ((i - nside + 1) % 2) / 2.0
        else:                               # south polar cap
            ii = 4 * nside - i
            z, nj, dphi, offset = -(1.0 - ii * ii / (3.0 * nside ** 2)), 4 * ii, np.pi / (2 * ii), 0.5
        phi = dphi * (np.arange(1, nj + 1) - offset)
        s = np.sqrt(1 - z * z)
        pts.append(np.stack([s * np.cos(phi), s * np.sin(phi), np.full_like(phi, z)], -1))
    return np.vstack(pts)


def selftest(nsides=(1, 2, 4, 8, 16), verbose=True):
    """Geometric invariants that must hold for a correct HEALPix tessellation.
    Raises AssertionError on failure."""
    from scipy.spatial import cKDTree

    for ns in nsides:
        centers, corners = healpix_pixel_geometry(ns)
        npix = 12 * ns * ns
        assert len(centers) == npix

        # 1. centres coincide (as a set) with the closed-form ring-scheme centres
        ring = _ring_scheme_centers(ns)
        assert len(ring) == npix
        dist, idx = cKDTree(ring).query(centers)
        assert dist.max() < 1e-9 and len(set(idx.tolist())) == npix, f"centres mismatch nside={ns}"

        # 2. corners are CCW from outside and enclose the centre
        for k in range(4):
            a, b = corners[:, k], corners[:, (k + 1) % 4]
            assert np.all(np.einsum("ij,ij->i", np.cross(a, b), centers) > 0), "corner order not CCW"

        # 3. neighbouring pixels share vertices: V = 12 nside^2 + 2 (Euler)
        uniq = np.unique(np.round(corners.reshape(-1, 3), 9), axis=0)
        assert len(uniq) == 12 * ns * ns + 2, f"vertex count {len(uniq)} nside={ns}"

        # 4. the geodesic tiling covers the sphere exactly, for both cell shapes
        for shape in ("triangle", "quad"):
            det = make_healpix_detector(nside=ns, cell_shape=shape, min_elevation_deg=-90)
            assert abs(det.solid_angles.sum() - 4 * np.pi) < 1e-9, f"area {shape} nside={ns}"
        if verbose:
            print(f"nside={ns:3d}: OK  ({npix} pixels, {len(uniq)} vertices)")

    # 5. optional cross-check of the NESTED indexing against healpy
    try:
        import healpy as hp
    except ImportError:
        if verbose:
            print("healpy not installed - NESTED index cross-check skipped")
        return
    for ns in nsides:
        c, _ = healpix_pixel_geometry(ns)
        ref = np.array(hp.pix2vec(ns, np.arange(12 * ns * ns), nest=True)).T
        assert np.abs(c - ref).max() < 1e-9, f"NESTED indexing differs from healpy at nside={ns}"
    if verbose:
        print("NESTED indexing matches healpy.pix2vec(nest=True)")


if __name__ == "__main__":
    selftest()
