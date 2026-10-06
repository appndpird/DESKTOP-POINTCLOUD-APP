"""
LiDAR360-style per-plot preprocessing, point classification and feature extraction (PhenoApp v3).

Pipeline per plot (bounding window of the polygon + a 0.25-1.0 m alley ring, neighbours excluded):
  1. Remove outliers      statistical outlier removal (k neighbours, mean distance > mean + n sd) and
                          isolated points > 30 cm above the 99.9th percentile      -> class 7 (noise)
  2. Classify ground      robust plane through the alley ring (25 cm cells, 5th percentile of z) refined
                          with in-plot points within +-8 cm; ground = |h| <= 8 cm   -> class 2 (ground)
  3. Normalise            h = z - plane(x, y)
  4. Classify canopy      canopy = h > 10 cm                                        -> class 5 (high vegetation)
                          0.08 < h <= 0.10 m                                        -> class 3 (low vegetation)
  5. Features             canopy-only elevation metrics (H_mean, H50, H75, H90, H95, H99, H999, max, sd, cv,
                          skew, kurt, canopy relief ratio, 10 density layers, top-N mean), cover on a 5 cm grid,
                          voxel canopy volume (5 cm), gap fraction Pgap (ground returns / all returns) and
                          multi-layer gap fraction (20 cm layers), LAI proxy -ln(Pgap)/0.5, 3D profile area,
                          canopy-top surface P90/P95/P99 (5 cm cell maxima, 15 cm edge trim), rumple index,
                          roughness, return metrics, intensity normalised by the alley ground, densities, QC.

Validated on Muresk NUE 2025 (ruler height): canopy-only percentiles + canopy-top metrics + normalised
intensity gave R2 0.50 / RMSE 3.6 cm with PLS or ridge (repeated 10-fold), and normalised canopy intensity
was the strongest single biomass correlate at anthesis (r 0.47), as in Bates et al. (2026).

Public API
----------
preprocess_plot(px, py, pz, rx, ry, rz, poly, ...) -> dict(keep, h, cls, ground_model, ground_rms, ground_cells, inorm)
lidar_plot_features(...)                            -> dict of features for one plot
lidar_features_all(x, y, z, intensity, return_number, number_of_returns, plots, progress_cb, write_classified_dir)
                                                    -> DataFrame (Plot_ID + features); optionally writes classified
                                                       per-plot LAS files (classification = 2/3/5/7)
LIDAR_V3_KEYS                                       feature column names
"""

from __future__ import annotations
import os
import numpy as np

CANOPY_H, GROUND_H, CELL, VOX = 0.10, 0.08, 0.05, 0.05
LAYERS = np.arange(0.2, 1.61, 0.2)
LIDAR_V3_CORE = ["H95", "H99", "H999", "H_topN_mean", "H_mean", "H50", "H_sd", "cth_p95", "cth_p99", "tip_thin", "cover_5cm", "Pgap",
                 "LAI_proxy", "vox_volume_m3_per_m2", "profile_area_m", "canopy_pts_per_m2", "roughness", "rumple", "H_crr",
                 "I_canopy_mean", "I_canopy_p90"]


def _cell_reduce(px, py, pv, cell, fn):
    ix = np.floor(px / cell).astype(np.int64); iy = np.floor(py / cell).astype(np.int64); key = ix * 1_000_003 + iy
    o = np.argsort(key, kind="stable"); ks = key[o]; first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    return fn.reduceat(pv[o], first), ix[o][first], iy[o][first]


def sor_mask(px, py, pz, k=10, nsig=3.0, top_gap=0.30):
    """True = keep. Statistical outlier removal + isolated points far above the canopy."""
    if len(pz) < k + 2:
        return np.ones(len(pz), bool)
    from scipy.spatial import cKDTree
    P = np.column_stack([px, py, pz]); d, _ = cKDTree(P).query(P, k=k + 1); md = d[:, 1:].mean(1)
    keep = md <= md.mean() + nsig * md.std()
    keep &= ~(pz > np.quantile(pz, 0.999) + top_gap)
    return keep


def preprocess_plot(px, py, pz, rx, ry, rz, poly, ri=None, k=10, nsig=3.0):
    """Outlier removal, ground plane, normalisation and classification for one plot.
    Returns dict(keep, h, cls, ground_source, ground_rms, ground_cells, inorm, model)."""
    from .canopy_top import _cell_percentile, _robust_plane, _plane_z
    keep = sor_mask(px, py, pz, k, nsig)
    qx, qy, qz = px[keep], py[keep], pz[keep]
    model = None; rms = np.nan; nused = 0; src = "inplot_p1"
    if rx is not None and len(rx) > 200:
        cx, cy, cz = _cell_percentile(np.asarray(rx, float), np.asarray(ry, float), np.asarray(rz, float), 0.25, 5.0)
        model, rms, nused = _robust_plane(cx, cy, cz); src = "ring"
        h0 = qz - _plane_z(model, qx, qy); near = np.abs(h0) < 0.08
        if near.sum() > 50:
            m2, rms2, n2 = _robust_plane(np.r_[cx, qx[near]], np.r_[cy, qy[near]], np.r_[cz, qz[near]])
            if rms2 < 0.06:
                model, rms, nused, src = m2, rms2, n2, "ring+inplot"
    if model is not None:
        g = _plane_z(model, qx, qy)
    else:
        g = np.full(len(qz), np.quantile(qz, 0.01) if len(qz) else 0.0)
    h = qz - g
    cls = np.full(len(pz), 7, np.uint8)            # removed points = noise
    c = np.where(h > CANOPY_H, 5, np.where(np.abs(h) <= GROUND_H, 2, np.where(h > GROUND_H, 3, 1))).astype(np.uint8)
    cls[np.flatnonzero(keep)] = c
    inorm = float(np.median(ri)) if (ri is not None and len(ri) > 50 and np.median(ri) > 0) else 1.0
    return dict(keep=keep, h=h, cls=cls, ground_source=src, ground_rms=float(rms) if np.isfinite(rms) else np.nan,
                ground_cells=int(nused), inorm=inorm, model=model)


def lidar_plot_features(px, py, pz, pi, prn, pnr, rx, ry, rz, ri, poly, pre=None):
    """Feature dict for one plot (see module docstring). `pre` = result of preprocess_plot (computed if None)."""
    from scipy.stats import skew, kurtosis
    from shapely import contains_xy
    out = dict(n_raw=int(len(pz)), qc_ok=0)
    if len(pz) < 100:
        return out
    pre = pre or preprocess_plot(px, py, pz, rx, ry, rz, poly, ri)
    keep = pre["keep"]; h = pre["h"]; px, py = px[keep], py[keep]
    pi = pi[keep] if pi is not None else None; prn = prn[keep] if prn is not None else None; pnr = pnr[keep] if pnr is not None else None
    out.update(n_noise_removed=int((~keep).sum()), ground_source=pre["ground_source"], ground_cells=pre["ground_cells"], ground_rms_cm=pre["ground_rms"] * 100)
    ground = np.abs(h) <= GROUND_H; canopy = h > CANOPY_H
    n = len(h); nc = int(canopy.sum()); ng = int(ground.sum())
    out.update(n_points=n, n_canopy=nc, n_ground=ng, pts_per_m2=n / poly.area, canopy_pts_per_m2=nc / poly.area)
    if nc < 30:
        return out
    hc = h[canopy]; q = np.quantile(hc, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
    out.update(H_mean=hc.mean(), H25=q[0], H50=q[1], H75=q[2], H90=q[3], H95=q[4], H99=q[5], H999=q[6], H_max=hc.max(), H_sd=hc.std(),
               H_cv=hc.std() / hc.mean(), H_skew=float(skew(hc)), H_kurt=float(kurtosis(hc)), H_iqr=q[2] - q[0],
               H_crr=(hc.mean() - hc.min()) / max(hc.max() - hc.min(), 1e-6), H_topN_mean=np.sort(hc)[-max(10, int(0.01 * nc)):].mean(),
               Hall_mean=h.mean(), Hall_p95=np.quantile(h, 0.95), Hall_p99=np.quantile(h, 0.99))
    edges = np.linspace(0, q[6], 11); dens = np.histogram(np.clip(hc, 0, q[6]), bins=edges)[0] / nc
    for i, d in enumerate(dens):
        out[f"dens_{i+1}"] = float(d)
    allcells = set(zip(*_cell_reduce(px, py, h, CELL, np.maximum)[1:]))
    cancells = set(zip(*_cell_reduce(px[canopy], py[canopy], hc, CELL, np.maximum)[1:]))
    n_cells_poly = max(poly.area / CELL ** 2, 1.0)
    out.update(cover_5cm=min(len(cancells) / n_cells_poly, 1.0), cover_5cm_of_sampled=len(cancells) / max(len(allcells), 1), cells_sampled_frac=len(allcells) / n_cells_poly)
    pgap = ng / n; out.update(Pgap=pgap, LAI_proxy=float(-np.log(max(pgap, 1e-3)) / 0.5))
    for L in LAYERS:
        out[f"Pgap_below_{int(L*100):03d}cm"] = float((h < L).mean())
    hs = np.linspace(0, max(q[6], 0.1), 50); out["profile_area_m"] = float(np.trapezoid([(h > v).mean() for v in hs], hs))
    vx = np.floor(px[canopy] / VOX).astype(np.int64); vy = np.floor(py[canopy] / VOX).astype(np.int64); vz = np.floor(hc / VOX).astype(np.int64)
    nvox = len(np.unique((vx * 1_000_003 + vy) * 1_000_003 + vz))
    out.update(vox_volume_m3=nvox * VOX ** 3, vox_volume_m3_per_m2=nvox * VOX ** 3 / poly.area, PVI_m=out["cover_5cm"] * q[4])
    inner = poly.buffer(-0.15); sel = contains_xy(inner, px, py) if not inner.is_empty else np.ones(n, bool)
    if sel.sum() < 20:
        sel = np.ones(n, bool); inner = poly
    cm, cxi, cyi = _cell_reduce(px[sel], py[sel], h[sel], CELL, np.maximum)
    out.update(cth_p90=np.quantile(cm, 0.90), cth_p95=np.quantile(cm, 0.95), cth_p99=np.quantile(cm, 0.99), cth_max=cm.max(), cth_mean=cm.mean(),
               cth_cover=len(cm) / max(inner.area / CELL ** 2, 1), tip_thin=cm.max() - np.quantile(cm, 0.95), roughness=cm.std())
    W = cxi.max() - cxi.min() + 1; Hh = cyi.max() - cyi.min() + 1
    grid = np.full((Hh, W), np.nan); grid[cyi - cyi.min(), cxi - cxi.min()] = cm
    gy, gx = np.gradient(np.nan_to_num(grid, nan=float(np.nanmean(grid))), CELL); out["rumple"] = float(np.nanmean(np.sqrt(1 + gx ** 2 + gy ** 2)))
    if prn is not None and pnr is not None:
        out.update(frac_first_return=float((prn == 1).mean()), frac_single_return=float((pnr == 1).mean()), mean_returns_per_pulse=float(pnr.mean()),
                   frac_canopy_first=float((prn[canopy] == 1).mean()))
    if pi is not None:
        inorm = pre["inorm"]; ic = pi[canopy] / inorm
        out.update(I_canopy_mean=float(ic.mean()), I_canopy_p50=float(np.median(ic)), I_canopy_p90=float(np.quantile(ic, 0.90)), I_canopy_sd=float(ic.std()),
                   I_ground_norm=float(np.median(pi[ground]) / inorm) if ng > 10 else np.nan, I_ring_median_raw=float(inorm), I_canopy_x_cover=float(ic.mean() * out["cover_5cm"]))
    out["qc_ok"] = int(out["cover_5cm"] > 0.3 and (pre["ground_cells"] >= 15 or pre["ground_source"] == "inplot_p1") and out["n_noise_removed"] < 0.05 * n)
    return out


def lidar_features_all(x, y, z, intensity, return_number, number_of_returns, plots, progress_cb=None,
                       write_classified_dir=None, las_header=None, points=None, ring_in=0.25, ring_out=1.0):
    """Run the v3 pipeline for every plot of a GeoDataFrame over the whole cloud arrays.

    write_classified_dir : if given (and las_header/points from the LASManager), one LAS per plot is
                           written with classification 2 ground / 3 low veg / 5 canopy / 7 noise.
    Returns a DataFrame with Plot_ID, area_m2 and all features.
    """
    import pandas as pd
    from shapely import contains_xy
    from shapely.ops import unary_union
    all_buf = unary_union(list(plots.geometry.buffer(ring_in)))
    rows = []; n = len(plots)
    if write_classified_dir:
        os.makedirs(write_classified_dir, exist_ok=True)
    for i, (_, r) in enumerate(plots.iterrows()):
        poly = r.geometry; ring = poly.buffer(ring_out).difference(all_buf); bx = ring.bounds
        m = (x >= bx[0]) & (x <= bx[2]) & (y >= bx[1]) & (y <= bx[3]); idx_m = np.flatnonzero(m)
        xs, ys, zs = x[m], y[m], z[m]
        ip = contains_xy(poly, xs, ys); ir = contains_xy(ring, xs, ys)
        pi = intensity[m][ip] if intensity is not None else None; ri = intensity[m][ir] if intensity is not None else None
        prn = return_number[m][ip] if return_number is not None else None; pnr = number_of_returns[m][ip] if number_of_returns is not None else None
        pre = preprocess_plot(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, ri) if ip.sum() >= 100 else None
        d = lidar_plot_features(xs[ip], ys[ip], zs[ip], pi, prn, pnr, xs[ir], ys[ir], zs[ir], ri, poly, pre)
        d["Plot_ID"] = r.get("Plot_ID", i + 1); d["area_m2"] = poly.area; rows.append(d)
        if write_classified_dir and pre is not None and las_header is not None and points is not None:
            try:
                import laspy, re
                sub = laspy.LasData(header=las_header); sub.points = points[idx_m[ip]].copy()
                sub.classification = pre["cls"]
                nm = re.sub(r"[^A-Za-z0-9_-]", "_", str(r.get("B/R", f"P{d['Plot_ID']}")))
                sub.write(os.path.join(write_classified_dir, f"plot_{d['Plot_ID']}_{nm}_classified.las"))
            except Exception:
                pass
        if progress_cb and (i % 8 == 0 or i == n - 1):
            progress_cb(int(100 * (i + 1) / n), f"LiDAR v3 features: plot {i + 1}/{n}")
    df = pd.DataFrame(rows)
    cols = ["Plot_ID", "area_m2", "qc_ok"] + [c for c in df.columns if c not in ("Plot_ID", "area_m2", "qc_ok")]
    return df[cols]
