"""
Plot-level canopy-top height (transferable plant-height metric).

Why: percentiles of *all* points (h_p95 / h_p99) depend on how many returns
penetrate the canopy, which changes with flying height, scan rate, canopy
density, growth stage and wind; and a trial-wide ground surface can drift by
several centimetres from one part of the field to another. The pipeline here
ties every plot to its own ground and describes the *upper canopy surface*
instead of the point distribution:

  1. noise filter        statistical outlier removal on the upper canopy
                         slice, plus isolated points far above the canopy
  2. plot-local ground   robust plane through bare-soil cell minima in a
                         ring around the plot (alleys), excluding neighbours
  3. density normalisation  2 cm voxel thinning (one point per voxel)
  4. top-surface raster  cell max of height above local ground on a 5 cm
                         grid, plot edge trimmed by an inward buffer
  5. metrics             P90 of the cell maxima (recommended), plus P95,
                         P99, max, mean and the legacy point percentiles on
                         the same local ground for continuity
  6. quality flags       n points, cell coverage, ground cells used,
                         ground fit rms, local-minus-trial ground offset,
                         noise points removed

Public API
----------
canopy_top_metrics(px, py, pz, rx, ry, rz, polygon, ...) -> dict
canopy_top_all(x, y, z, plots, ...) -> DataFrame (one row per plot)
CANOPY_TOP_KEYS                                  trait keys produced
"""

from __future__ import annotations
import numpy as np

CANOPY_TOP_KEYS = ["cth_p90", "cth_p95", "cth_p99", "cth_max", "cth_mean", "cth_cover",
                   "cth_pt_p99", "cth_pt_max", "cth_ground_cells", "cth_ground_rms",
                   "cth_ground_offset", "cth_n_noise", "cth_ok"]


# ---------------------------------------------------------------- helpers
def _robust_plane(px, py, pz, order=1, iters=4):
    def A(u, v):
        cols = [np.ones_like(u), u, v]
        if order == 2:
            cols += [u * u, v * v, u * v]
        return np.column_stack(cols)
    x0, y0 = px.mean(), py.mean(); u, v = px - x0, py - y0
    keep = np.ones(len(u), bool); c = None
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A(u[keep], v[keep]), pz[keep], rcond=None)
        r = A(u, v) @ c - pz
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826 + 1e-4
        new = np.abs(r) < 2.5 * mad
        if new.sum() < max(6, 0.5 * len(u)) or np.array_equal(new, keep):
            break
        keep = new
    r = (A(u, v) @ c - pz)[keep]
    return (x0, y0, order, c), float(np.sqrt(np.mean(r ** 2))), int(keep.sum())


def _plane_z(model, qx, qy):
    x0, y0, order, c = model; u, v = qx - x0, qy - y0
    cols = [np.ones_like(u), u, v]
    if order == 2:
        cols += [u * u, v * v, u * v]
    return np.column_stack(cols) @ c


def _cell_reduce(px, py, pv, cell, fn):
    ix = np.floor(px / cell).astype(np.int64); iy = np.floor(py / cell).astype(np.int64)
    key = ix * 1_000_003 + iy
    order = np.argsort(key, kind="stable"); ks = key[order]; vs = pv[order]
    first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    red = fn.reduceat(vs, first)
    cx = (ix[order][first] + 0.5) * cell; cy = (iy[order][first] + 0.5) * cell
    return cx, cy, red


def _cell_percentile(px, py, pz, cell, pctl):
    """Per-cell percentile of z (like the exterior ground model observation)."""
    ix = np.floor(px / cell).astype(np.int64); iy = np.floor(py / cell).astype(np.int64)
    key = ix * 1_000_003 + iy
    order = np.argsort(key, kind="stable"); ks = key[order]; zs = pz[order]
    first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]; last = np.r_[first[1:], len(ks)]
    cz = np.array([np.percentile(zs[a:b], pctl) for a, b in zip(first, last)])
    cx = (ix[order][first] + 0.5) * cell; cy = (iy[order][first] + 0.5) * cell
    return cx, cy, cz


def _sor_upper(px, py, pz, k=8, nsig=2.0, top_gap=0.25):
    """Statistical outlier removal restricted to the top 10% of points, plus
    removal of isolated points more than `top_gap` above the 99.5th pct."""
    keep = np.ones(len(pz), bool)
    if len(pz) < 50:
        return keep
    from scipy.spatial import cKDTree
    hi = np.flatnonzero(pz >= np.quantile(pz, 0.90))
    P = np.column_stack([px[hi], py[hi], pz[hi]])
    d, _ = cKDTree(P).query(P, k=min(k + 1, len(P)))
    md = d[:, 1:].mean(1)
    keep[hi[md > md.mean() + nsig * md.std()]] = False
    keep &= ~(pz > np.quantile(pz, 0.995) + top_gap)
    return keep


def _voxel_thin(px, py, ph, vox):
    ix = np.floor(px / vox).astype(np.int64); iy = np.floor(py / vox).astype(np.int64)
    iz = np.floor(ph / vox).astype(np.int64)
    key = (ix * 1_000_003 + iy) * 1_000_003 + iz
    _, idx = np.unique(key, return_index=True)
    return idx


# ---------------------------------------------------------------- per plot
def canopy_top_metrics(px, py, pz, rx, ry, rz, polygon, ext_ground=None,
                       cell=0.05, trim=0.15, vox=0.02, sor=True,
                       ground_cell=0.25, ground_pctl=5.0, min_ground_cells=15):
    """Metrics for one plot.

    px, py, pz : points inside the plot polygon (absolute z)
    rx, ry, rz : points in the ring around the plot (alleys; neighbours excluded)
    polygon    : shapely polygon of the plot
    ext_ground : optional array of trial-wide ground z at (px, py) used as a
                 fallback when the ring has too few bare-soil cells, and to
                 report the local-minus-trial ground offset
    Returns a dict with CANOPY_TOP_KEYS.
    """
    from shapely import contains_xy
    out = {k: np.nan for k in CANOPY_TOP_KEYS}
    out["cth_ok"] = 0
    if len(pz) < 100:
        return out
    px = np.asarray(px, float); py = np.asarray(py, float); pz = np.asarray(pz, float)
    if sor:
        keep = _sor_upper(px, py, pz)
        out["cth_n_noise"] = int((~keep).sum())
        px, py, pz = px[keep], py[keep], pz[keep]
    else:
        out["cth_n_noise"] = 0
    # --- local ground
    g = None
    if rx is not None and len(rx) > 200:
        cx, cy, cz = _cell_percentile(np.asarray(rx, float), np.asarray(ry, float), np.asarray(rz, float),
                                      ground_cell, ground_pctl)
        soil = np.ones(len(cz), bool)
        if soil.sum() >= min_ground_cells:
            model, rms, nused = _robust_plane(cx[soil], cy[soil], cz[soil], order=1)
            if nused >= min_ground_cells:
                g = _plane_z(model, px, py)
                out["cth_ground_cells"] = nused; out["cth_ground_rms"] = rms
    if g is None:
        if ext_ground is None:
            g = np.full(len(pz), np.quantile(pz, 0.01))   # last resort: in-plot p1
        else:
            g = np.asarray(ext_ground, float)
            if sor:
                g = g[keep]
        out["cth_ground_cells"] = 0
    elif ext_ground is not None:
        e = np.asarray(ext_ground, float)
        if sor:
            e = e[keep]
        out["cth_ground_offset"] = float(np.mean(g - e))
    h = pz - g
    out["cth_pt_p99"] = float(np.quantile(h, 0.99)); out["cth_pt_max"] = float(h.max())
    # --- thinning + top surface inside the trimmed polygon
    ti = _voxel_thin(px, py, h, vox) if vox and vox > 0 else np.arange(len(h))
    inner = polygon.buffer(-trim) if trim > 0 else polygon
    if inner.is_empty or inner.area <= 0:
        inner = polygon
    sel = ti[contains_xy(inner, px[ti], py[ti])]
    if len(sel) < 20:
        return out
    _, _, cm = _cell_reduce(px[sel], py[sel], h[sel], cell, np.maximum)
    out["cth_cover"] = float(len(cm) / max(inner.area / cell ** 2, 1.0))
    out["cth_p90"] = float(np.quantile(cm, 0.90)); out["cth_p95"] = float(np.quantile(cm, 0.95))
    out["cth_p99"] = float(np.quantile(cm, 0.99)); out["cth_max"] = float(cm.max())
    out["cth_mean"] = float(cm.mean())
    out["cth_ok"] = int(out["cth_ground_cells"] >= min_ground_cells and out["cth_cover"] >= 0.5)
    return out


def canopy_top_all(x, y, z, plots, ext_ground_model=None, ring_in=0.25, ring_out=1.0,
                   progress_cb=None, use_ring=True, **kw):
    """Run canopy_top_metrics for every plot of a GeoDataFrame.

    x, y, z          : the whole point cloud (absolute z)
    ext_ground_model : optional GroundModel (core.ground_model) for fallback
                       and the offset diagnostic
    ring_in/out      : ring = polygon.buffer(ring_out) minus every plot
                       polygon buffered by ring_in (neighbours excluded)
    Returns a DataFrame with Plot_ID + CANOPY_TOP_KEYS.
    """
    import pandas as pd
    from shapely import contains_xy
    from shapely.ops import unary_union
    all_buf = unary_union(list(plots.geometry.buffer(ring_in)))
    rows = []
    n = len(plots)
    for i, (_, r) in enumerate(plots.iterrows()):
        poly = r.geometry
        ring = poly.buffer(ring_out).difference(all_buf)
        bx0, by0, bx1, by1 = ring.bounds
        m = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        xs, ys, zs = x[m], y[m], z[m]
        ip = contains_xy(poly, xs, ys); ir = contains_xy(ring, xs, ys)
        ext = ext_ground_model.predict(xs[ip], ys[ip]) if ext_ground_model is not None else None
        if use_ring:
            d = canopy_top_metrics(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, ext_ground=ext, **kw)
        else:
            d = canopy_top_metrics(xs[ip], ys[ip], zs[ip], None, None, None, poly, ext_ground=ext, **kw)
        d["Plot_ID"] = r.get("Plot_ID", i + 1)
        rows.append(d)
        if progress_cb and (i % 8 == 0 or i == n - 1):
            progress_cb(int(100 * (i + 1) / n), f"canopy top: plot {i + 1}/{n}")
    return pd.DataFrame(rows)[["Plot_ID"] + CANOPY_TOP_KEYS]


# ---------------------------------------------------------------- reference targets
def measure_targets(x, y, z, targets, ext_ground_model=None, radius=0.5, **kw):
    """Measure rigid reference targets of known height with the same pipeline.

    targets : DataFrame with columns E, N (target centre, working CRS),
              height_m (known height above the soil) and optional name,
              radius_m (footprint radius, default `radius`).
    Returns the DataFrame with measured cth_p90 / cth_max / cth_pt_p99 (m).
    """
    import pandas as pd
    from shapely import contains_xy
    from shapely.geometry import Point
    rows = []
    for _, t in targets.iterrows():
        rad = float(t.get("radius_m", radius) or radius)
        poly = Point(float(t["E"]), float(t["N"])).buffer(rad)
        ring = poly.buffer(1.0).difference(poly.buffer(0.25))
        bx0, by0, bx1, by1 = ring.bounds
        m = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        xs, ys, zs = x[m], y[m], z[m]
        ip = contains_xy(poly, xs, ys); ir = contains_xy(ring, xs, ys)
        ext = ext_ground_model.predict(xs[ip], ys[ip]) if ext_ground_model is not None else None
        d = canopy_top_metrics(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, ext_ground=ext,
                               trim=0.0, sor=False, **kw)
        d.update(name=t.get("name", ""), E=t["E"], N=t["N"], height_m=float(t["height_m"]), n_points=int(ip.sum()))
        rows.append(d)
    return pd.DataFrame(rows)


def fit_target_calibration(tg, metric="cth_p90"):
    """known = offset + scale x measured, fitted on the targets. With <3 targets
    only the offset is fitted (scale = 1). Returns a dict with the fit and the
    per-target residuals."""
    ok = tg[metric].notna() & tg["height_m"].notna()
    m = tg.loc[ok, metric].to_numpy(float); k = tg.loc[ok, "height_m"].to_numpy(float)
    if len(m) == 0:
        raise ValueError("no measurable targets")
    if len(m) >= 3 and np.ptp(m) > 0.1:
        scale, offset = np.polyfit(m, k, 1)
    else:
        scale, offset = 1.0, float(np.mean(k - m))
    pred = offset + scale * m
    return dict(metric=metric, n_targets=int(len(m)), offset=float(offset), scale=float(scale),
                rmse_m=float(np.sqrt(np.mean((pred - k) ** 2))),
                residuals_m=[float(v) for v in (pred - k)], measured_m=[float(v) for v in m],
                known_m=[float(v) for v in k])
