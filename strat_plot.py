"""
strat_plot.py
=============

Side-view (vertical cross-section) plotting for `StratModel` (see
`stratmodel.py`), coloured by layer density. Built directly from the
horizon fields and `DensityModel`s.

Usage
-----
    from stratmodel import StratModel
    from strat_plot import plot_layer_stack_slice

    fig, ax, raster = plot_layer_stack_slice(model, normal='Y')
    plt.show()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_layer_stack_slice(model, normal: str = "Y", loc: Optional[float] = None,
                            n_along: Optional[int] = None, n_z: int = 400,
                            z_pad_frac: float = 0.05, ax=None, cmap: str = "turbo",
                            vmin: Optional[float] = None, vmax: Optional[float] = None,
                            norm=None, show_horizon_lines: bool = True,
                            horizon_color: str = "k",
                            horizon_lw: float = 0.8, label_horizons: bool = False,
                            equal_aspect: bool = False, colorbar: bool = True,
                            cbar_label: Optional[str] = None,
                            color_by: str = "density"):
    """Plot a vertical cross-section through a `StratModel`, coloured by
    density or by layer thickness, sampled directly from the horizon
    surfaces and layer `DensityModel`s (not a TreeMesh).

    Parameters
    ----------
    model : StratModel
    normal : {'Y', 'X'}
        Slice orientation. `'Y'` (default) fixes Y = `loc` and profiles
        along X (an "E-W slice", matching `mesh.plot_slice(..., normal='Y')`
        in the demo). `'X'` fixes X = `loc` and profiles along Y.
    loc : float, optional
        The fixed coordinate. Defaults to the middle of the model's
        domain along that axis.
    n_along : int, optional
        Number of samples along the slice. Defaults to the model's
        native grid resolution along that axis (`nx` or `ny`); increase
        for a smoother raster on a coarse model.
    n_z : int
        Vertical raster resolution.
    z_pad_frac : float
        Extra vertical padding above topography / below the basement,
        as a fraction of the slice's total relief.
    ax : matplotlib Axes, optional
        Plot into an existing axes; otherwise a new figure is created.
    color_by : {'density', 'thickness'}
        What the raster colour represents.
        `'density'` (default): each point's local density, as before -
        varies within a layer if it has a `DensityModel` (gradient,
        correlated noise, etc).
        `'thickness'`: each layer is coloured by its local isopach
        thickness - `top_horizon(x) - base_horizon(x)` at that position
        along the slice - so the whole vertical extent of the layer at
        a given `along` position gets one colour, and that colour
        varies only along the slice, not with depth within the layer.
        Useful for spotting thinning/thickening (onlap, growth faults,
        depocentres) independent of density.
    cmap, vmin, vmax : colormap and colour limits for the raster.
        `vmin`/`vmax` are ignored if `norm` is given (matplotlib
        disallows combining them).
    norm : matplotlib.colors.Normalize, optional
        Pass a `BoundaryNorm` (with a matching `ListedColormap` as
        `cmap`) to pin specific colours to specific value ranges, e.g.:

            from matplotlib.colors import ListedColormap, BoundaryNorm
            boundaries = [2000, 2300, 2500, 2700, 3000]  # kg/m^3
            colors = ['#3b4cc0', '#7ea8dd', '#f4a582', '#b2182b']
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(boundaries, cmap.N)
            plot_layer_stack_slice(model, cmap=cmap, norm=norm)
    show_horizon_lines : bool
        Overplot each horizon as a line - makes contacts crisp
        regardless of raster resolution, and shows unconformities /
        onlap even where the colour either side happens to match.
    label_horizons : bool
        Annotate each horizon line with its name at the right edge.
    equal_aspect : bool
        If True, use `ax.set_aspect('equal')` (true-scale cross-section,
        often heavily "squashed" looking for basins that are wide
        relative to their relief). Default False lets matplotlib scale
        the axes independently, which is the usual way these are shown.
    colorbar : bool
        Add a colorbar for the raster.
    cbar_label : str, optional
        Defaults to "Density (kg/m^3)" or "Thickness (m)" depending on
        `color_by`.

    Returns
    -------
    fig, ax, raster : Figure, Axes, dict
        `raster` holds the underlying arrays: `along` (1D coordinate
        along the slice), `z` (1D vertical coordinate), `values` (2D,
        shape (n_z, n_along), NaN outside the model - the plotted
        quantity, density or thickness per `color_by`), `color_by`
        (which one it is), and `horizons` (dict of horizon name -> 1D
        elevation profile along the slice), in case you want to reuse
        the sampled data elsewhere. For backwards compatibility
        `raster['density']` is also set when `color_by='density'`.
    """
    if color_by not in ("density", "thickness"):
        raise ValueError("color_by must be 'density' or 'thickness'")
    if cbar_label is None:
        cbar_label = "Density (kg/m$^3$)" if color_by == "density" else "Thickness (m)"

    normal = normal.upper()
    if normal not in ("X", "Y"):
        raise ValueError("normal must be 'X' or 'Y'")

    if normal == "Y":
        if loc is None:
            loc = float(model.Y.mean())
        n_along = n_along or model.nx
        along = np.linspace(float(model.X.min()), float(model.X.max()), n_along)
        x_line = along
        y_line = np.full_like(along, loc)
        along_label = "X (m)"
    else:
        if loc is None:
            loc = float(model.X.mean())
        n_along = n_along or model.ny
        along = np.linspace(float(model.Y.min()), float(model.Y.max()), n_along)
        y_line = along
        x_line = np.full_like(along, loc)
        along_label = "Y (m)"

    # Sample every horizon's elevation along the slice once.
    horizon_profiles = {}
    for name in model.order:
        interp = model.horizon_interpolator(name)
        horizon_profiles[name] = interp(np.column_stack([y_line, x_line]))

    z_bottom = min(prof.min() for prof in horizon_profiles.values())
    z_top_all = max(prof.max() for prof in horizon_profiles.values())
    relief = max(z_top_all - z_bottom, 1e-6)
    z_lo = z_bottom - z_pad_frac * relief
    z_hi = z_top_all + z_pad_frac * relief
    z = np.linspace(z_lo, z_hi, n_z)

    ALONG, Z = np.meshgrid(along, z)  # shape (n_z, n_along)
    values = np.full(ALONG.shape, np.nan)

    for layer_name in model.layer_order:
        layer = model.layers[layer_name]
        z_top = horizon_profiles[layer.top][None, :]   # (1, n_along)
        z_base = horizon_profiles[layer.base][None, :]  # (1, n_along)
        mask = (Z <= z_top) & (Z > z_base)
        if not np.any(mask):
            continue

        if color_by == "thickness":
            thickness_profile = horizon_profiles[layer.top] - horizon_profiles[layer.base]
            thickness_grid = np.broadcast_to(thickness_profile[None, :], ALONG.shape)
            values[mask] = thickness_grid[mask]
        elif layer.density_model is not None:
            zt_b = np.broadcast_to(z_top, ALONG.shape)[mask]
            zb_b = np.broadcast_to(z_base, ALONG.shape)[mask]
            if normal == "Y":
                x_sel, y_sel = ALONG[mask], np.full(mask.sum(), loc)
            else:
                x_sel, y_sel = np.full(mask.sum(), loc), ALONG[mask]
            values[mask] = layer.density_model(x_sel, y_sel, Z[mask], zt_b, zb_b)
        elif layer.density is not None:
            values[mask] = layer.density

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = ax.figure

    raster_masked = np.ma.masked_invalid(values)
    if norm is not None:
        pcm = ax.pcolormesh(along, z, raster_masked, cmap=cmap, norm=norm,
                             shading="auto")
    else:
        pcm = ax.pcolormesh(along, z, raster_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading="auto")

    if show_horizon_lines:
        for name in model.order:
            ax.plot(along, horizon_profiles[name], color=horizon_color,
                    lw=horizon_lw, zorder=3)
            if label_horizons:
                ax.text(along[-1], horizon_profiles[name][-1], f" {name}",
                        va="center", ha="left", fontsize=8, zorder=4)

    if colorbar:
        fig.colorbar(pcm, ax=ax, label=cbar_label)

    ax.set_xlabel(along_label)
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Layer stack {color_by} - {normal}={loc:.1f} slice")
    if equal_aspect:
        ax.set_aspect("equal")

    raster = {"along": along, "z": z, "values": values, "color_by": color_by,
              "horizons": horizon_profiles}
    if color_by == "density":
        raster["density"] = values
    return fig, ax, raster
