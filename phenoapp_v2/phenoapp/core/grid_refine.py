"""
Refine plot polygons to the crop actually visible in a canopy height model.

Auto-align (auto_align.py) fixes a rigid shift of the whole grid. This module
fixes what a rigid shift cannot: plots that are individually off along the
sowing direction (seeder start/stop varies by range and drilling direction)
and plot lengths that do not match the sown length.

Per plot, in the polygon's own frame (u along the long axis, v across):
  * along  - average the CHM across the inner 60 % of the width, find the two
             positions where it falls to half its plateau (the bare alleys at
             each end), take their midpoint as the crop centre and their
             distance as the crop length
  * across - neighbours usually touch at maturity, so half-height edges do
             not exist; instead find the furrow minima expected at +-pitch/2
             and use them only as a per-range median (individual furrows are
             noisy), and only when a furrow is actually visible

Then rebuild each polygon: same width and orientation, re-centred along u,
length = crop length minus two margins, clipped to [original length -
shrink_max, original length + extend_max]. Plots whose detection fails a
plausibility test fall back to the range median offset and original length.

Public API
----------
refine_grid_to_canopy(chm_tif, plots_gdf, margin_m=0.20, extend_max_m=0.5,
                      shrink_max_m=0.5, along_search_m=1.2, apply_across=True,
                      group_col=None, progress_cb=None)
    -> (refined GeoDataFrame, per-plot report DataFrame, diagnostics dict)
render_refine_qa(chm_tif, old_gdf, new_gdf, report, out_png) -> out_png
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
def _frame(geom):
    xs, ys = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)[:-1].T
    c = np.array([xs.mean(), ys.mean()])
    e1 = np.array([xs[1] - xs[0], ys[1] - ys[0]]); e2 = np.array([xs[2] - xs[1], ys[2] - ys[1]])
    L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
    a_u, hl, hw = (e1 / L1, L1 / 2, L2 / 2) if L1 >= L2 else (e2 / L2, L2 / 2, L1 / 2)
    a_v = np.array([-a_u[1], a_u[0]])
    return c, a_u, a_v, hl, hw


def _half_height_edges(profile, coords, plateau_q=0.9, frac=0.5):
    ok = np.isfinite(profile)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan
    p = np.where(ok, profile, 0.0)
    core = np.abs(coords) < 0.6 * np.abs(coords).max()
    plateau = float(np.quantile(p[core], plateau_q)) if core.any() else float(np.nanmax(p))
    if plateau <= 0:
        return np.nan, np.nan, plateau
    thr = frac * plateau
    i0 = int(np.argmin(np.abs(coords)))
    lo = i0
    while lo > 0 and p[lo] >= thr:
        lo -= 1
    hi = i0
    while hi < len(p) - 1 and p[hi] >= thr:
        hi += 1
    def cross(i, j):
        return coords[i] if p[i] == p[j] else coords[i] + (thr - p[i]) * (coords[j] - coords[i]) / (p[j] - p[i])
    e_lo = cross(lo, lo + 1) if lo < i0 else np.nan
    e_hi = cross(hi - 1, hi) if hi > i0 else np.nan
    return e_lo, e_hi, plateau


def _estimate_pitch(plots_gdf, hw):
    """Median centre-to-centre distance to the nearest neighbour (across rows)."""
    c = np.array([[g.centroid.x, g.centroid.y] for g in plots_gdf.geometry])
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    pitch = float(np.median(nn))
    return pitch if 2 * hw < pitch < 6 * hw else 2 * hw + 0.4


def refine_grid_to_canopy(chm_tif, plots_gdf, margin_m=0.20, extend_max_m=0.5,
                          shrink_max_m=0.5, along_search_m=1.2, apply_across=True,
                          group_col=None, progress_cb=None):
    import pandas as pd, rasterio
    from shapely.geometry import Polygon

    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    with rasterio.open(chm_tif) as src:
        chm = src.read(1).astype(float); b = src.bounds; res = src.transform.a
    ny, nx = chm.shape
    CX = b.left + (np.arange(nx) + 0.5) * res; CY = b.top - (np.arange(ny) + 0.5) * res
    GX, GY = np.meshgrid(CX, CY)

    gdf = plots_gdf.reset_index(drop=True).copy()
    if group_col is None:
        group_col = next((c for c in ("Range", "Bank", "range", "bank") if c in gdf.columns), None)
    groups = gdf[group_col].astype(str).values if group_col else np.array(["all"] * len(gdf))
    _, _, _, hl0, hw0 = _frame(gdf.geometry.iloc[0])
    pitch = _estimate_pitch(gdf, hw0)

    rows, frames, prof_v_by_group = [], [], {}
    for i, geom in enumerate(gdf.geometry):
        if i % 16 == 0: _p(int(70 * i / len(gdf)), f"measuring plot {i + 1}/{len(gdf)}")
        c, a_u, a_v, hl, hw = _frame(geom); frames.append((c, a_u, a_v, hl, hw))
        pad = along_search_m + 0.5
        minx, miny, maxx, maxy = geom.buffer(pad).bounds
        sel = (GX >= minx) & (GX <= maxx) & (GY >= miny) & (GY <= maxy)
        px, py, pz = GX[sel], GY[sel], chm[sel]
        u = (px - c[0]) * a_u[0] + (py - c[1]) * a_u[1]; v = (px - c[0]) * a_v[0] + (py - c[1]) * a_v[1]
        # along profile
        inner_v = np.abs(v) < 0.6 * hw
        ub = np.arange(-(hl + along_search_m), hl + along_search_m + 0.1, 0.1)
        prof_u = np.array([np.nanmean(pz[inner_v & (np.abs(u - uu) < 0.1)]) if np.any(inner_v & (np.abs(u - uu) < 0.1)) else np.nan for uu in ub])
        u_lo, u_hi, plateau = _half_height_edges(prof_u, ub)
        # across profile (for furrows)
        inner_u = np.abs(u) < 0.8 * hl
        vb = np.arange(-(pitch / 2 + 0.6), pitch / 2 + 0.61, 0.05)
        prof_v = np.array([np.nanmean(pz[inner_u & (np.abs(v - vv) < 0.05)]) if np.any(inner_u & (np.abs(v - vv) < 0.05)) else np.nan for vv in vb])
        prof_v_by_group.setdefault(groups[i], []).append(prof_v)
        crop_len = u_hi - u_lo; off_u = (u_hi + u_lo) / 2
        rows.append(dict(idx=i, group=groups[i], poly_len=2 * hl, poly_w=2 * hw, crop_len=crop_len, along_off=off_u,
                         plateau_m=plateau, mean_inside_before=float(np.nanmean(pz[(np.abs(u) < hl) & (np.abs(v) < hw)]))))
    rep = pd.DataFrame(rows)

    # plausibility: crop length within [poly-shrink_max-0.3, poly+extend_max+1.0], offset within search
    ok = (rep.crop_len.between(rep.poly_len - shrink_max_m - 0.3, rep.poly_len + extend_max_m + 1.0)
          & (rep.along_off.abs() < along_search_m * 0.6))
    rep["along_ok"] = ok
    grp_med = rep[ok].groupby("group").along_off.median()
    rep["along_off_applied"] = np.where(ok, rep.along_off, rep.group.map(grp_med).fillna(0.0))
    rep["new_len"] = np.where(ok, np.clip(rep.crop_len - 2 * margin_m, rep.poly_len - shrink_max_m, rep.poly_len + extend_max_m), rep.poly_len)

    # across: per-group furrow check on the folded median profile
    across_applied, furrow_depth = {}, {}
    vb = np.arange(-(pitch / 2 + 0.6), pitch / 2 + 0.61, 0.05)
    for g, profs in prof_v_by_group.items():
        med = np.nanmedian(np.array(profs), axis=0)
        def furrow(lo, hi):
            m = (vb >= lo) & (vb <= hi) & np.isfinite(med)
            return (vb[m][np.argmin(med[m])], med[m].min()) if m.sum() >= 3 else (np.nan, np.nan)
        (f_lo, z_lo), (f_hi, z_hi) = furrow(-pitch / 2 - 0.45, -pitch / 2 + 0.45), furrow(pitch / 2 - 0.45, pitch / 2 + 0.45)
        centre = float(np.nanmedian(med[np.abs(vb) < 0.5 * hw0]))
        depth = centre - np.nanmean([z_lo, z_hi]) if np.isfinite(z_lo) and np.isfinite(z_hi) else np.nan
        furrow_depth[g] = depth
        visible = np.isfinite(depth) and depth > 0.04 * max(centre, 1e-6) and depth > 0.02
        # the two furrows must be a plausible pair: spacing within 15 % of the pitch and a
        # modest shift (< 0.25 m) - otherwise the minima are canopy texture, not furrows
        shift = float((f_lo + f_hi) / 2) if visible else 0.0
        plausible = visible and abs((f_hi - f_lo) - pitch) < 0.15 * pitch and abs(shift) < 0.25
        across_applied[g] = shift if (apply_across and plausible) else 0.0
    rep["across_off_applied"] = rep.group.map(across_applied)
    rep["furrow_visible"] = rep.group.map(lambda g: bool(np.isfinite(furrow_depth[g]) and across_applied[g] != 0.0 or (np.isfinite(furrow_depth[g]) and furrow_depth[g] > 0.02)))

    # rebuild polygons
    _p(80, "rebuilding polygons")
    new_geoms = []
    for i, (c, a_u, a_v, hl, hw) in enumerate(frames):
        r = rep.iloc[i]
        c2 = c + a_u * r.along_off_applied + a_v * r.across_off_applied
        L = r.new_len
        new_geoms.append(Polygon([c2 + a_u * s * L / 2 + a_v * t * hw for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
    out = gdf.copy(); out["geometry"] = new_geoms
    out["plot_len"] = np.round(rep.new_len.values, 2); out["along_off"] = np.round(rep.along_off_applied.values, 3)
    out["across_off"] = np.round(rep.across_off_applied.values, 3); out["crop_len"] = np.round(rep.crop_len.values, 2)
    out["refine_ok"] = rep.along_ok.values

    # after-metrics: mean CHM inside the new polygon and whether an end still sticks out
    _p(90, "checking result")
    ends_out_before = ends_out_after = 0
    after = []
    for i, geom in enumerate(new_geoms):
        c, a_u, a_v, hl, hw = _frame(geom); r = rep.iloc[i]
        minx, miny, maxx, maxy = geom.buffer(0.3).bounds
        sel = (GX >= minx) & (GX <= maxx) & (GY >= miny) & (GY <= maxy)
        u = (GX[sel] - c[0]) * a_u[0] + (GY[sel] - c[1]) * a_u[1]; v = (GX[sel] - c[0]) * a_v[0] + (GY[sel] - c[1]) * a_v[1]
        after.append(float(np.nanmean(chm[sel][(np.abs(u) < hl) & (np.abs(v) < hw)])))
        if r.along_ok:
            c0, au0, _, hl0_, _ = frames[i]
            lo, hi = r.along_off - r.crop_len / 2, r.along_off + r.crop_len / 2          # crop edges in the old frame
            ends_out_before += int((-hl0_ < lo) or (hl0_ > hi))
            nlo, nhi = r.along_off_applied - r.new_len / 2, r.along_off_applied + r.new_len / 2
            ends_out_after += int((nlo < lo - 1e-6) or (nhi > hi + 1e-6))
    rep["mean_inside_after"] = after
    if "Plot_ID" in gdf.columns: rep.insert(0, "Plot_ID", gdf["Plot_ID"].values)
    diag = dict(n=len(gdf), n_along_ok=int(ok.sum()), pitch_m=pitch,
                along_off_median_by_group={k: float(v) for k, v in grp_med.items()},
                across_off_by_group=across_applied, furrow_depth_by_group={k: (float(v) if np.isfinite(v) else None) for k, v in furrow_depth.items()},
                crop_len_median=float(np.nanmedian(rep.crop_len[ok])) if ok.any() else None,
                poly_len=float(rep.poly_len.median()), new_len_median=float(rep.new_len.median()),
                ends_outside_crop_before=ends_out_before, ends_outside_crop_after=ends_out_after,
                mean_chm_inside_before=float(rep.mean_inside_before.mean()), mean_chm_inside_after=float(np.nanmean(after)))
    _p(100, "grid refined")
    return out, rep, diag


def render_refine_qa(chm_tif, old_gdf, new_gdf, report, out_png, n_zoom=6):
    import rasterio, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    with rasterio.open(chm_tif) as src:
        chm = src.read(1); b = src.bounds; res = src.transform.a
    idx = report.sort_values("along_off", key=np.abs, ascending=False).idx.values[:n_zoom]
    fig, axes = plt.subplots(1, n_zoom, figsize=(3.2 * n_zoom, 3.6))
    for ax, i in zip(np.atleast_1d(axes), idx):
        g = new_gdf.geometry.iloc[i]; minx, miny, maxx, maxy = g.buffer(1.2).bounds
        r0 = max(0, int((b.top - maxy) / res)); r1 = min(chm.shape[0], int((b.top - miny) / res))
        c0 = max(0, int((minx - b.left) / res)); c1 = min(chm.shape[1], int((maxx - b.left) / res))
        ax.imshow(chm[r0:r1, c0:c1], extent=(b.left + c0 * res, b.left + c1 * res, b.top - r1 * res, b.top - r0 * res), cmap="viridis", origin="upper")
        old_gdf.geometry.iloc[[i]].boundary.plot(ax=ax, color="red", linewidth=1.2, linestyle="--")
        new_gdf.geometry.iloc[[i]].boundary.plot(ax=ax, color="white", linewidth=1.6)
        r = report.iloc[i]; pid = r.get("Plot_ID", i + 1)
        ax.set_title(f"Plot {pid}: along {r.along_off_applied:+.2f} m, len {r.poly_len:.1f}->{r.new_len:.1f} m", fontsize=8)
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Grid refinement - red dashed = before, white = after (largest along-plot corrections)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png
