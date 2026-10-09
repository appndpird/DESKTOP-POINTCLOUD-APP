"""
Trial-wide surface models: DSM, DTM and CHM GeoTIFFs plus an annotated map.

Why trial-wide
--------------
The terrain is only observable where the laser reaches soil - the alleys
between plots - so a terrain model must be built from the whole trial, not
per plot. The surface model shares one grid so the canopy height model is a
single subtraction, and plot polygons only enter at the zonal step.

Products (all float32 GeoTIFFs on the same grid, NaN = no data)
    <prefix>_DSM.tif   per-cell maximum z            (digital surface model)
    <prefix>_DTM.tif   terrain from the chosen method (digital terrain model)
    <prefix>_CHM.tif   DSM - DTM, negatives clipped to 0 (canopy height model)
    <prefix>_CHM_annotated.png / _DTM_annotated.png   plots outlined and
                       labelled with a per-plot value (e.g. h_p99 in cm)

DTM methods (`dtm_mode`)
    "exterior"  smooth surface fitted to alley returns (PhenoApp ground_model);
                the default and the only same-flight method that survives a
                closed canopy
    "inplot"    exterior cell-minimum surface, overridden inside each plot by
                that plot's 1st-percentile z ("ground inside the zone" - for
                raised beds)
    "hag"       ground implied by an existing HeightAboveGround dimension
                (SMRF): cell median of z - hag
    "external"  a DTM GeoTIFF from another date/source, resampled onto the DSM
                grid; the vertical offset against this flight's alley minima
                is measured and (optionally) removed, and the fit refuses when
                the two disagree too much (poor co-registration)

Public API
----------
build_surface_models(x, y, z, plots_gdf, out_dir, prefix, crs, hag=None,
                     dsm_res=0.10, dtm_mode="exterior", dtm_cell=1.0,
                     margin_m=6.0, external_dtm=None, shift_external=True,
                     max_external_mad=0.15, progress_cb=None) -> dict
annotate_plot_map(raster_tif, plots_gdf, values, out_png, ...) -> path
zonal_chm_stats(chm_tif, plots_gdf) -> DataFrame (Plot_ID, chm_max, chm_p99,
                     chm_p95, chm_mean, chm_cover_frac)
"""
from __future__ import annotations
import os
import numpy as np


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _grid_from_bounds(bounds, res, margin):
    minx, miny, maxx, maxy = bounds
    x0 = np.floor((minx - margin) / res) * res
    y1 = np.ceil((maxy + margin) / res) * res
    nx = int(np.ceil((maxx + margin - x0) / res))
    ny = int(np.ceil((y1 - (miny - margin)) / res))
    return x0, y1, nx, ny          # origin is the top-left corner (north-up)


def _cell_reduce(x, y, z, x0, y1, res, nx, ny, how="max"):
    """Reduce points to a (ny, nx) grid; rows run north->south."""
    import pandas as pd
    ix = np.floor((x - x0) / res).astype(np.int64)
    iy = np.floor((y1 - y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key = iy[ok] * nx + ix[ok]
    s = pd.Series(z[ok]).groupby(key)
    agg = {"max": s.max, "min": s.min, "median": s.median, "count": s.size}[how]()
    out = np.full(nx * ny, np.nan, dtype=np.float64)
    out[agg.index.to_numpy()] = agg.to_numpy(dtype=float)
    return out.reshape(ny, nx)


def _fill_nearest(a):
    """Fill NaN cells from the nearest valid cell (Euclidean)."""
    from scipy.ndimage import distance_transform_edt
    m = np.isnan(a)
    if not m.any():
        return a
    if m.all():
        raise RuntimeError("Raster has no valid cells.")
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _cell_centres(x0, y1, res, nx, ny):
    cx = x0 + (np.arange(nx) + 0.5) * res
    cy = y1 - (np.arange(ny) + 0.5) * res
    return np.meshgrid(cx, cy)


def _write_tif(path, arr, x0, y1, res, crs, nodata=np.nan):
    import rasterio
    from rasterio.transform import from_origin
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs=crs, transform=from_origin(x0, y1, res, res),
                nodata=nodata, compress="deflate", tiled=False)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def _plots_union(plots_gdf, buffer_m):
    geoms = plots_gdf.geometry
    u = geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union
    return u.buffer(buffer_m)


def _contains(geom, x, y):
    from shapely import contains_xy
    return contains_xy(geom, x, y)


# ----------------------------------------------------------------------
# DTM builders
# ----------------------------------------------------------------------
def _exterior_min_cells(x, y, z, plots_gdf, x0, y1, cell, nx_c, ny_c, buffer_m=0.3):
    """Per-cell minimum z of returns outside the (buffered) plots."""
    u = _plots_union(plots_gdf, buffer_m)
    ext = ~_contains(u, x, y)
    zmin = _cell_reduce(x[ext], y[ext], z[ext], x0, y1, cell, nx_c, ny_c, "min")
    CX, CY = _cell_centres(x0, y1, cell, nx_c, ny_c)
    ok = np.isfinite(zmin)
    return CX[ok], CY[ok], zmin[ok]


def _dtm_exterior(x, y, z, plots_gdf, x0, y1, res, nx, ny, dtm_cell):
    from .ground_model import fit_exterior_ground
    # ground observations: cell minima at a fine cell, fitted on 2 m cells
    fine = max(0.25, dtm_cell / 4.0)
    nxc = int(np.ceil(nx * res / fine)); nyc = int(np.ceil(ny * res / fine))
    cx, cy, cz = _exterior_min_cells(x, y, z, plots_gdf, x0, y1, fine, nxc, nyc)
    if len(cz) < 200:
        raise RuntimeError(
            f"Only {len(cz)} exterior ground cells around the plots - the LAS "
            "must extend beyond the trial (alleys/tracks) for an exterior DTM.")
    gm = fit_exterior_ground(cx, cy, cz, plots_gdf, cell=max(2.0, dtm_cell * 2))
    CX, CY = _cell_centres(x0, y1, res, nx, ny)
    dtm = gm.predict(CX.ravel(), CY.ravel()).reshape(ny, nx)
    diag = {"method": "exterior", "terms": gm.terms,
            "rms_resid_m": gm.diagnostics.get("rms_resid_m"),
            "n_ground_cells": gm.diagnostics.get("n_ground_cells")}
    return dtm, diag


def _dtm_inplot(x, y, z, plots_gdf, x0, y1, res, nx, ny, dtm_cell):
    """Exterior cell-min surface (nearest-filled) overridden by each plot's p1."""
    nxc = int(np.ceil(nx * res / dtm_cell)); nyc = int(np.ceil(ny * res / dtm_cell))
    u = _plots_union(plots_gdf, 0.3)
    ext = ~_contains(u, x, y)
    coarse = _cell_reduce(x[ext], y[ext], z[ext], x0, y1, dtm_cell, nxc, nyc, "min")
    coarse = _fill_nearest(coarse)
    # upsample coarse -> fine grid by index mapping
    CX, CY = _cell_centres(x0, y1, res, nx, ny)
    ix = np.clip(np.floor((CX - x0) / dtm_cell).astype(int), 0, nxc - 1)
    iy = np.clip(np.floor((y1 - CY) / dtm_cell).astype(int), 0, nyc - 1)
    dtm = coarse[iy, ix]
    n_over = 0
    for geom in plots_gdf.geometry:
        bx0, by0, bx1, by1 = geom.bounds
        m = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        if not m.any():
            continue
        inside = _contains(geom, x[m], y[m])
        if inside.sum() < 20:
            continue
        p1 = float(np.quantile(z[m][inside], 0.01))
        cm = (CX >= bx0) & (CX <= bx1) & (CY >= by0) & (CY <= by1)
        cin = cm.copy(); cin[cm] = _contains(geom, CX[cm], CY[cm])
        dtm[cin] = p1; n_over += 1
    return dtm, {"method": "inplot", "plots_overridden": n_over, "coarse_cell_m": dtm_cell}


def _dtm_hag(x, y, z, hag, x0, y1, res, nx, ny, dtm_cell):
    if hag is None:
        raise RuntimeError("DTM mode 'hag' needs HeightAboveGround: enable SMRF on the Project tab.")
    nxc = int(np.ceil(nx * res / dtm_cell)); nyc = int(np.ceil(ny * res / dtm_cell))
    g = _cell_reduce(x, y, z - hag, x0, y1, dtm_cell, nxc, nyc, "median")
    g = _fill_nearest(g)
    CX, CY = _cell_centres(x0, y1, res, nx, ny)
    ix = np.clip(np.floor((CX - x0) / dtm_cell).astype(int), 0, nxc - 1)
    iy = np.clip(np.floor((y1 - CY) / dtm_cell).astype(int), 0, nyc - 1)
    return g[iy, ix], {"method": "hag", "coarse_cell_m": dtm_cell}


def _dtm_external(path, x, y, z, plots_gdf, x0, y1, res, nx, ny, crs,
                  shift, max_mad):
    """Resample an external DTM onto the DSM grid and check it against this
    flight's alley minima. Refuses when the disagreement is too large."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    dst = np.full((ny, nx), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=from_origin(x0, y1, res, res),
                  dst_crs=crs, dst_nodata=np.nan, src_nodata=src.nodata,
                  resampling=Resampling.bilinear)
    dtm = dst.astype(np.float64)
    # co-registration check on exterior cells
    nxc = int(np.ceil(nx * res / 1.0)); nyc = int(np.ceil(ny * res / 1.0))
    cx, cy, cz = _exterior_min_cells(x, y, z, plots_gdf, x0, y1, 1.0, nxc, nyc)
    ix = np.clip(np.floor((cx - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((y1 - cy) / res).astype(int), 0, ny - 1)
    d = dtm[iy, ix] - cz
    d = d[np.isfinite(d)]
    if len(d) < 50:
        raise RuntimeError("External DTM does not overlap the alleys around this trial.")
    med = float(np.median(d)); mad = float(np.median(np.abs(d - med)) * 1.4826)
    diag = {"method": "external", "source": path, "offset_vs_alleys_m": med,
            "robust_sd_m": mad, "n_check_cells": int(len(d)), "shift_applied_m": 0.0}
    if mad > max_mad:
        raise RuntimeError(
            f"External DTM disagrees with this flight's alley ground: robust SD "
            f"{mad:.2f} m (limit {max_mad:.2f} m), offset {med:+.2f} m. The two "
            "datasets are not co-registered well enough (RTK on both dates or "
            "shared GCPs are required). DTM not used.")
    if shift:
        dtm = dtm - med; diag["shift_applied_m"] = -med
    return dtm, diag


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------
def build_surface_models(x, y, z, plots_gdf, out_dir, prefix, crs, hag=None,
                         dsm_res=0.10, dtm_mode="exterior", dtm_cell=1.0,
                         margin_m=6.0, external_dtm=None, shift_external=True,
                         max_external_mad=0.15, progress_cb=None) -> dict:
    """Build DSM, DTM and CHM GeoTIFFs for the whole trial.

    x, y, z    : point arrays in the working CRS (all points, no subsampling
                 needed - reduction is done with a groupby)
    plots_gdf  : plot polygons in the same CRS (Plot_ID column optional)
    crs        : e.g. "EPSG:7850"
    Returns dict with paths, grid definition and DTM diagnostics.
    """
    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)
    os.makedirs(out_dir, exist_ok=True)
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    x0, y1, nx, ny = _grid_from_bounds(plots_gdf.total_bounds, dsm_res, margin_m)
    # keep only points inside the raster extent (speeds everything up)
    m = (x >= x0) & (x < x0 + nx * dsm_res) & (y <= y1) & (y > y1 - ny * dsm_res)
    x, y, z = x[m], y[m], z[m]
    hag = np.asarray(hag, float)[m] if hag is not None else None

    _p(5, f"DSM: max z per {dsm_res} m cell ({nx} x {ny})")
    dsm = _cell_reduce(x, y, z, x0, y1, dsm_res, nx, ny, "max")
    dsm_valid = np.isfinite(dsm)
    dsm = _fill_nearest(dsm)

    _p(40, f"DTM: {dtm_mode}")
    if dtm_mode == "exterior":
        dtm, diag = _dtm_exterior(x, y, z, plots_gdf, x0, y1, dsm_res, nx, ny, dtm_cell)
    elif dtm_mode == "inplot":
        dtm, diag = _dtm_inplot(x, y, z, plots_gdf, x0, y1, dsm_res, nx, ny, dtm_cell)
    elif dtm_mode == "hag":
        dtm, diag = _dtm_hag(x, y, z, hag, x0, y1, dsm_res, nx, ny, dtm_cell)
    elif dtm_mode == "external":
        if not external_dtm:
            raise RuntimeError("DTM mode 'external' needs a DTM GeoTIFF path.")
        dtm, diag = _dtm_external(external_dtm, x, y, z, plots_gdf, x0, y1, dsm_res,
                                  nx, ny, crs, shift_external, max_external_mad)
    else:
        raise ValueError(f"unknown dtm_mode '{dtm_mode}'")

    _p(75, "CHM = DSM - DTM")
    chm = dsm - dtm
    neg_frac = float((chm[dsm_valid] < -0.05).mean()) if dsm_valid.any() else 0.0
    chm = np.clip(chm, 0.0, None)
    # no-data where the DSM had no returns at all
    dsm_out = np.where(dsm_valid, dsm, np.nan)
    chm_out = np.where(dsm_valid, chm, np.nan)

    _p(85, "writing GeoTIFFs")
    paths = {
        "dsm": _write_tif(os.path.join(out_dir, f"{prefix}_DSM.tif"), dsm_out, x0, y1, dsm_res, crs),
        "dtm": _write_tif(os.path.join(out_dir, f"{prefix}_DTM.tif"), dtm, x0, y1, dsm_res, crs),
        "chm": _write_tif(os.path.join(out_dir, f"{prefix}_CHM.tif"), chm_out, x0, y1, dsm_res, crs),
    }
    _p(100, "surface models written")
    return {"paths": paths, "x0": x0, "y1": y1, "res": dsm_res, "nx": nx, "ny": ny,
            "dtm": diag, "dsm_cells_with_returns": float(dsm_valid.mean()),
            "chm_negative_frac": neg_frac}


# ----------------------------------------------------------------------
# zonal statistics from the CHM (for cross-checking against point traits)
# ----------------------------------------------------------------------
def zonal_chm_stats(chm_tif, plots_gdf, height_cut=0.15):
    import pandas as pd, rasterio
    from rasterio.features import geometry_mask
    rows = []
    with rasterio.open(chm_tif) as src:
        a = src.read(1); tr = src.transform
        for i, row in plots_gdf.reset_index(drop=True).iterrows():
            geom = row.geometry
            mask = geometry_mask([geom], out_shape=a.shape, transform=tr, invert=True)
            v = a[mask]; v = v[np.isfinite(v)]
            pid = row.get("Plot_ID", i + 1)
            if len(v) == 0:
                rows.append({"Plot_ID": pid}); continue
            rows.append({"Plot_ID": pid, "chm_cells": int(len(v)), "chm_max": float(v.max()),
                         "chm_p99": float(np.quantile(v, 0.99)), "chm_p95": float(np.quantile(v, 0.95)),
                         "chm_mean": float(v.mean()), "chm_cover_frac": float((v > height_cut).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# annotated map
# ----------------------------------------------------------------------
def annotate_plot_map(raster_tif, plots_gdf, values: dict, out_png, title="",
                      value_label="", fmt="{:.0f}", cmap="viridis", vmin=None, vmax=None,
                      label_col="Plot_ID", show_ids=False, dpi=200, font_pt=5.5):
    """Render a raster with plot outlines and one annotation per plot.

    values : {Plot_ID: number} written at each plot centroid (NaN -> '-').
    """
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    with rasterio.open(raster_tif) as src:
        a = src.read(1); b = src.bounds
    finite = a[np.isfinite(a)]
    if vmin is None: vmin = float(np.percentile(finite, 2)) if finite.size else 0.0
    if vmax is None: vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    w_m, h_m = b.right - b.left, b.top - b.bottom
    fig_w = 11.0; fig_h = max(4.0, fig_w * h_m / w_m + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(a, extent=(b.left, b.right, b.bottom, b.top), cmap=cmap,
                   norm=Normalize(vmin, vmax), interpolation="nearest", origin="upper")
    plots_gdf.boundary.plot(ax=ax, color="white", linewidth=0.5, alpha=0.9)
    for _, row in plots_gdf.iterrows():
        pid = row.get(label_col, None)
        v = values.get(pid, np.nan) if values is not None else np.nan
        if isinstance(v, (tuple, list)):
            ok = all(u is not None and np.isfinite(u) for u in v)
        else:
            ok = v is not None and np.isfinite(v)
        txt = fmt.format(v) if ok else "-"
        if show_ids: txt = f"{pid}\n{txt}"
        c = row.geometry.centroid
        ax.text(c.x, c.y, txt, ha="center", va="center", fontsize=font_pt, color="white",
                fontweight="bold", path_effects=None,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45))
    ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)
    ax.set_aspect("equal"); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=7)
    if value_label: cb.set_label(value_label, fontsize=8)
    if title: ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
