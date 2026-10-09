"""
LiDAR360-style per-plot preprocessing, point classification and feature extraction (PhenoApp v3; v3.2 update 2026-10-09).

Pipeline per plot (bounding window of the polygon + a 0.25-1.0 m alley ring, neighbours excluded):
  1. Noise rule (selectable, NOISE_RULES; default "gap")                                -> class 7 (noise)
       gap     : points more than 30 cm above the 99.9th percentile (true spikes only)   [DEFAULT since v3.2]
       none    : nothing flagged
       legacy  : canopy_top._sor_upper - SOR on the top 10 % of points (k=8, 2 sd) + points > 25 cm above p99.5
       sor     : statistical outlier removal on all points (k neighbours, mean distance > mean + nsig sd) + the gap rule
                 (the v3.0 default with k=10, nsig=3)
     Validation, Muresk NUE 2025-09-30 (128 plots, ruler heights): the gap rule flagged 1 point in the trial; every SOR
     variant removed sparse canopy tips (heads/awns, 0.3-1.5 % of points, half of them in the top 5 % of the canopy) and
     lowered the ruler agreement of cth_p95 (r 0.70 / LOOCV 3.70 cm unfiltered; 0.68 / 3.79 at 5 sd; 0.66 / 3.88 at 3 sd).
     Flagged points are KEPT and marked (class 7, withheld bit in written LAS) unless drop_noise=True; every metric
     ignores them either way.
  2. Classify ground      robust plane through the alley ring (25 cm cells, 5th percentile of z) refined
                          with in-plot points within +-8 cm; ground = |h| <= 8 cm   -> class 2 (ground)
  3. Normalise            h = z - plane(x, y)   (h_all is returned for every point, noise included)
  4. Classify canopy      canopy = h > 10 cm                                        -> class 5 (high vegetation)
                          0.08 < h <= 0.10 m                                        -> class 3 (low vegetation)
  5. Features             canopy-only elevation metrics (H_mean, H50, H75, H90, H95, H99, H999, max, sd, cv,
                          skew, kurt, canopy relief ratio, 10 density layers from the 10 cm canopy threshold to H999,
                          top-N mean), cover on a 5 cm grid, voxel canopy volume (5 cm), gap fraction Pgap
                          (ground returns / all returns) and multi-layer gap fraction (20 cm layers), LAI proxy
                          -ln(Pgap)/0.5, 3D profile area, canopy-top surface P90/P95/P99 (5 cm cell maxima, 15 cm
                          edge trim), rumple index, roughness, return metrics, intensity normalised by the alley
                          ground, densities, QC.
     v3.2: the same gap metrics on the 20 cm-trimmed INTERIOR (Pgap_interior, LAI_proxy_interior, cover_2cm_interior,
     ground_visible_2cm_interior) plus cover_2cm / ground_visible_2cm / ground_edge_frac on the whole polygon. Refit plot
     polygons include alley strips along the long sides, so whole-polygon Pgap is inflated by soil that is not inside the
     canopy (Muresk 2025-09-30: 68 % of ground returns lie within 20 cm of the edge; Pgap 8.3 % whole vs 3.5 % interior;
     the interior / 2 cm versions correlate better with biomass). At 30,000 pts/m2 cover_5cm saturates at 1.0 in half of
     the plots; cover_2cm_interior and ground_visible_2cm_interior replace it in LIDAR_V3_CORE.

Validated on Muresk NUE 2025 (ruler height): canopy-only percentiles + canopy-top metrics + normalised
intensity gave R2 0.50 / RMSE 3.6 cm with PLS or ridge (repeated 10-fold), and normalised canopy intensity
was the strongest single biomass correlate at anthesis (r 0.47), as in Bates et al. (2026).

Public API
----------
noise_mask(px, py, pz, rule="gap", k=10, nsig=3.0, top_gap=0.30) -> keep mask (True = keep)
preprocess_plot(px, py, pz, rx, ry, rz, poly, ri=None, k=10, nsig=3.0, rule="gap", top_gap=0.30)
                                                    -> dict(keep, h, h_all, cls, ground_source, ground_rms, ground_cells, inorm, model, rule)
lidar_plot_features(...)                            -> dict of features for one plot
lidar_features_all(x, y, z, intensity, return_number, number_of_returns, plots, progress_cb, write_classified_dir,
                   las_header, points, ring_in, ring_out, noise_rule="gap", k=10, nsig=3.0, top_gap=0.30, drop_noise=False)
                                                    -> DataFrame (Plot_ID + features); optionally writes classified
                                                       per-plot LAS files (classification 2/3/5/7, withheld bit on noise,
                                                       HeightAboveGround extra dimension) + a sidecar JSON per plot
LIDAR_V3_CORE                                       feature column names used by the Height Models / Biomass ML tabs
NOISE_RULES                                         ("gap", "none", "legacy", "sor")
"""

from __future__ import annotations
import os
import json
import numpy as np

CANOPY_H, GROUND_H, CELL, VOX = 0.10, 0.08, 0.05, 0.05
TRIM, FINE = 0.20, 0.02                      # interior trim (m) and fine cell size (m) for the v3.2 gap metrics
LAYERS = np.arange(0.2, 1.61, 0.2)
NOISE_RULES = ("gap", "none", "legacy", "sor")
LIDAR_V3_CORE = ["H95", "H99", "H999", "H_topN_mean", "H_mean", "H50", "H_sd", "cth_p95", "cth_p99", "tip_thin",
                 "cover_2cm_interior", "ground_visible_2cm_interior", "Pgap_interior", "LAI_proxy_interior",
                 "vox_volume_m3_per_m2", "profile_area_m", "canopy_pts_per_m2", "roughness", "rumple", "H_crr",
                 "I_canopy_mean", "I_canopy_p90"]


def _cell_reduce(px, py, pv, cell, fn):
    ix = np.floor(px / cell).astype(np.int64); iy = np.floor(py / cell).astype(np.int64); key = ix * 1_000_003 + iy
    o = np.argsort(key, kind="stable"); ks = key[o]; first = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    return fn.reduceat(pv[o], first), ix[o][first], iy[o][first]


def _n_cells(px, py, cell):
    """Number of distinct cells of size `cell` occupied by the points."""
    if len(px) == 0:
        return 0
    return int(np.unique(np.floor(px / cell).astype(np.int64) * 1_000_003 + np.floor(py / cell).astype(np.int64)).size)


def sor_mask(px, py, pz, k=10, nsig=3.0, top_gap=0.30):
    """True = keep. Statistical outlier removal + isolated points far above the canopy."""
    if len(pz) < k + 2:
        return np.ones(len(pz), bool)
    from scipy.spatial import cKDTree
    P = np.column_stack([px, py, pz]); d, _ = cKDTree(P).query(P, k=k + 1); md = d[:, 1:].mean(1)
    keep = md <= md.mean() + nsig * md.std()
    keep &= ~(pz > np.quantile(pz, 0.999) + top_gap)
    return keep


def noise_mask(px, py, pz, rule="gap", k=10, nsig=3.0, top_gap=0.30):
    """True = keep. `rule` is one of NOISE_RULES (see the module docstring)."""
    rule = (rule or "gap").lower()
    if rule == "none" or len(pz) < 10:
        return np.ones(len(pz), bool)
    if rule == "gap":
        return ~(pz > np.quantile(pz, 0.999) + top_gap)
    if rule == "legacy":
        from .canopy_top import _sor_upper
        return _sor_upper(px, py, pz)
    if rule == "sor":
        return sor_mask(px, py, pz, k, nsig, top_gap)
    raise ValueError(f"unknown noise rule '{rule}' - use one of {NOISE_RULES}")


def preprocess_plot(px, py, pz, rx, ry, rz, poly, ri=None, k=10, nsig=3.0, rule="gap", top_gap=0.30):
    """Noise flagging, ground plane, normalisation and classification for one plot.
    Returns dict(keep, h, h_all, cls, ground_source, ground_rms, ground_cells, inorm, model, rule)."""
    from .canopy_top import _cell_percentile, _robust_plane, _plane_z
    px = np.asarray(px, float); py = np.asarray(py, float); pz = np.asarray(pz, float)
    keep = noise_mask(px, py, pz, rule, k, nsig, top_gap)
    qx, qy, qz = px[keep], py[keep], pz[keep]
    model = None; rms = np.nan; nused = 0; src = "inplot_p1"
    if rx is not None and len(rx) > 200:
        cx, cy, cz = _cell_percentile(np.asarray(rx, float), np.asarray(ry, float), np.asarray(rz, float), 0.25, 5.0)
        model, rms, nused = _robust_plane(cx, cy, cz); src = "ring"
        h0 = qz - _plane_z(model, qx, qy); near = np.abs(h0) < GROUND_H
        if near.sum() > 50:
            m2, rms2, n2 = _robust_plane(np.r_[cx, qx[near]], np.r_[cy, qy[near]], np.r_[cz, qz[near]])
            if rms2 < 0.06:
                model, rms, nused, src = m2, rms2, n2, "ring+inplot"
    if model is not None:
        g_all = _plane_z(model, px, py)
    else:
        g_all = np.full(len(pz), np.quantile(pz, 0.01) if len(pz) else 0.0)
    h_all = pz - g_all; h = h_all[keep]
    cls = np.full(len(pz), 7, np.uint8)            # flagged points = noise
    c = np.where(h > CANOPY_H, 5, np.where(np.abs(h) <= GROUND_H, 2, np.where(h > GROUND_H, 3, 1))).astype(np.uint8)
    cls[np.flatnonzero(keep)] = c
    inorm = float(np.median(ri)) if (ri is not None and len(ri) > 50 and np.median(ri) > 0) else 1.0
    return dict(keep=keep, h=h, h_all=h_all, cls=cls, ground_source=src, ground_rms=float(rms) if np.isfinite(rms) else np.nan,
                ground_cells=int(nused), inorm=inorm, model=model, rule=rule)


def lidar_plot_features(px, py, pz, pi, prn, pnr, rx, ry, rz, ri, poly, pre=None):
    """Feature dict for one plot (see module docstring). `pre` = result of preprocess_plot (computed if None)."""
    from scipy.stats import skew, kurtosis
    from shapely import contains_xy
    out = dict(n_raw=int(len(pz)), qc_ok=0)
    if len(pz) < 100:
        return out
    pre = pre or preprocess_plot(px, py, pz, rx, ry, rz, poly, ri)
    keep = pre["keep"]; h = pre["h"]; px, py = np.asarray(px, float)[keep], np.asarray(py, float)[keep]
    pi = np.asarray(pi, float)[keep] if pi is not None else None
    prn = np.asarray(prn)[keep] if prn is not None else None; pnr = np.asarray(pnr)[keep] if pnr is not None else None
    out.update(n_noise_removed=int((~keep).sum()), noise_rule=pre.get("rule", "sor"), ground_source=pre["ground_source"],
               ground_cells=pre["ground_cells"], ground_rms_cm=pre["ground_rms"] * 100)
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
    # density layers from the canopy threshold up to H999 (v3.0 started at 0, which left dens_1 always empty)
    top = max(q[6], CANOPY_H + 0.01)
    edges = np.linspace(CANOPY_H, top, 11); dens = np.histogram(np.clip(hc, CANOPY_H, top), bins=edges)[0] / nc
    for i, d in enumerate(dens):
        out[f"dens_{i+1}"] = float(d)
    allcells = set(zip(*_cell_reduce(px, py, h, CELL, np.maximum)[1:]))
    cancells = set(zip(*_cell_reduce(px[canopy], py[canopy], hc, CELL, np.maximum)[1:]))
    n_cells_poly = max(poly.area / CELL ** 2, 1.0)
    out.update(cover_5cm=min(len(cancells) / n_cells_poly, 1.0), cover_5cm_of_sampled=len(cancells) / max(len(allcells), 1), cells_sampled_frac=len(allcells) / n_cells_poly)
    pgap = ng / n; out.update(Pgap=pgap, LAI_proxy=float(-np.log(max(pgap, 1e-3)) / 0.5))
    for L in LAYERS:
        out[f"Pgap_below_{int(L*100):03d}cm"] = float((h < L).mean())
    # ---- v3.2: fine-cell cover / ground visibility on the whole polygon and on the 20 cm interior
    n_all2 = _n_cells(px, py, FINE)
    out.update(cover_2cm=_n_cells(px[canopy], py[canopy], FINE) / max(n_all2, 1), ground_visible_2cm=_n_cells(px[ground], py[ground], FINE) / max(n_all2, 1))
    inner20 = poly.buffer(-TRIM)
    sel20 = contains_xy(inner20, px, py) if (not inner20.is_empty and inner20.area > 0) else np.ones(n, bool)
    n20 = int(sel20.sum())
    out["ground_edge_frac"] = float(1.0 - (ground & sel20).sum() / ng) if ng > 0 else np.nan
    if n20 >= 50:
        pg20 = float((ground & sel20).sum() / n20); n_in2 = _n_cells(px[sel20], py[sel20], FINE)
        out.update(n_interior=n20, interior_area_frac=float(inner20.area / poly.area) if not inner20.is_empty else 1.0,
                   Pgap_interior=pg20, LAI_proxy_interior=float(-np.log(max(pg20, 1e-3)) / 0.5),
                   cover_2cm_interior=_n_cells(px[sel20 & canopy], py[sel20 & canopy], FINE) / max(n_in2, 1),
                   ground_visible_2cm_interior=_n_cells(px[sel20 & ground], py[sel20 & ground], FINE) / max(n_in2, 1))
        for L in LAYERS:
            out[f"Pgap_interior_below_{int(L*100):03d}cm"] = float((h[sel20] < L).mean())
    else:
        out.update(n_interior=n20, interior_area_frac=np.nan, Pgap_interior=np.nan, LAI_proxy_interior=np.nan,
                   cover_2cm_interior=np.nan, ground_visible_2cm_interior=np.nan)
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


def _write_classified_las(path, las_header, points, sel, cls, h_all, drop_noise, meta):
    """One classified per-plot LAS (+ sidecar JSON): classification 2/3/5/7, withheld bit on noise, HeightAboveGround."""
    import laspy
    if drop_noise:
        ok = cls != 7; sel, cls, h_all = sel[ok], cls[ok], h_all[ok]
    # a fresh point format per file: never mutate the LASManager's shared header / record formats
    src_pf = las_header.point_format
    pf = laspy.PointFormat(src_pf.id)
    for ed in src_pf.extra_dimensions:
        pf.add_extra_dimension(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype, description=getattr(ed, "description", "") or ""))
    if "HeightAboveGround" not in pf.dimension_names:
        pf.add_extra_dimension(laspy.ExtraBytesParams(name="HeightAboveGround", type=np.float32, description="z - plot-local ground plane"))
    hdr = laspy.LasHeader(point_format=pf, version=las_header.version)
    hdr.scales = las_header.scales; hdr.offsets = las_header.offsets
    for v in las_header.vlrs:
        hdr.vlrs.append(v)
    src = points[sel]
    rec = laspy.ScaleAwarePointRecord.zeros(len(sel), header=hdr)
    for name in src_pf.dimension_names:
        rec[name] = src[name]
    rec["classification"] = cls; rec["withheld"] = (cls == 7).astype(np.uint8); rec["HeightAboveGround"] = np.asarray(h_all, np.float32)
    sub = laspy.LasData(hdr); sub.points = rec
    sub.header.generating_software = "PhenoApp v3 lidar_features"
    sub.write(path)
    meta = dict(meta); u, c = np.unique(cls, return_counts=True); meta["class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
    meta["n_points_written"] = int(len(cls)); meta["noise_points"] = "dropped from file" if drop_noise else "kept: classification 7 + withheld bit"
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=float)


def lidar_features_all(x, y, z, intensity, return_number, number_of_returns, plots, progress_cb=None,
                       write_classified_dir=None, las_header=None, points=None, ring_in=0.25, ring_out=1.0,
                       noise_rule="gap", k=10, nsig=3.0, top_gap=0.30, drop_noise=False):
    """Run the v3 pipeline for every plot of a GeoDataFrame over the whole cloud arrays.

    noise_rule / k / nsig / top_gap : see NOISE_RULES and noise_mask (default "gap", validated on Muresk 2025).
    drop_noise           : False (default) keeps flagged points in the classified LAS (class 7 + withheld bit);
                           True removes them from the file. Features never use them.
    write_classified_dir : if given (and las_header/points from the LASManager), one LAS per plot is written with
                           classification 2 ground / 3 low veg / 5 canopy / 7 noise, the withheld bit on noise points
                           and a HeightAboveGround extra dimension, plus a sidecar JSON (ground plane, counts, rule).
    Returns a DataFrame with Plot_ID, area_m2 and all features.
    """
    import pandas as pd
    from shapely import contains_xy
    from shapely.ops import unary_union
    from .canopy_top import _plane_z
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
        pre = (preprocess_plot(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, ri, k=k, nsig=nsig, rule=noise_rule, top_gap=top_gap)
               if ip.sum() >= 100 else None)
        d = lidar_plot_features(xs[ip], ys[ip], zs[ip], pi, prn, pnr, xs[ir], ys[ir], zs[ir], ri, poly, pre)
        d["Plot_ID"] = r.get("Plot_ID", i + 1); d["area_m2"] = poly.area; d["noise_rule"] = noise_rule; rows.append(d)
        if write_classified_dir and pre is not None and las_header is not None and points is not None:
            try:
                import re
                nm = re.sub(r"[^A-Za-z0-9_-]", "_", str(r.get("B/R", f"P{d['Plot_ID']}")))
                model = pre["model"]; x0, y0, order, coef = model if model is not None else (None, None, None, None)
                meta = dict(Plot_ID=d["Plot_ID"], BR=str(r.get("B/R", "")), noise_rule=noise_rule, sor_k=k, sor_nsig=nsig, top_gap_m=top_gap,
                            classes={"1": "below ground (h < -8 cm)", "2": "ground (|h| <= 8 cm)", "3": "low vegetation (8-10 cm)", "5": "canopy (> 10 cm)", "7": "noise"},
                            ground=dict(source=pre["ground_source"], cells_used=pre["ground_cells"], rms_m=pre["ground_rms"], ring_in_m=ring_in, ring_out_m=ring_out,
                                        plane=dict(x0=x0, y0=y0, order=order, coef=[float(v) for v in coef] if coef is not None else None,
                                                   formula="g = coef[0] + coef[1]*(x-x0) + coef[2]*(y-y0)"),
                                        z_at_centroid=float(_plane_z(model, np.array([poly.centroid.x]), np.array([poly.centroid.y]))[0]) if model is not None else None),
                            intensity_ring_median=pre["inorm"], n_noise=int((~pre["keep"]).sum()), polygon_area_m2=float(poly.area))
                _write_classified_las(os.path.join(write_classified_dir, f"plot_{d['Plot_ID']}_{nm}_classified.las"),
                                      las_header, points, idx_m[ip], pre["cls"], pre["h_all"], drop_noise, meta)
            except Exception:
                pass
        if progress_cb and (i % 8 == 0 or i == n - 1):
            progress_cb(int(100 * (i + 1) / n), f"LiDAR v3 features: plot {i + 1}/{n}")
    df = pd.DataFrame(rows)
    cols = ["Plot_ID", "area_m2", "qc_ok"] + [c for c in df.columns if c not in ("Plot_ID", "area_m2", "qc_ok")]
    return df[cols]
