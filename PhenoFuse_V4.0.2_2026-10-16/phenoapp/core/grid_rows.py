"""
Correct a plot grid from the crop rows and wheel tracks seen in the LiDAR point cloud, and detect the rows (PhenoFuse 4.0.3).

Why another refinement: grid_refine.py moves each plot on a coarse canopy raster and treats the across-plot
position as one shift per range. On a closed or leaning canopy that leaves plots 10-30 cm off their seeded
rows. This module works at 2 cm in every plot's own frame, straight from the point cloud, and uses the
regular structure of a drilled trial:

  * wheel tracks between plots repeat with one pitch along a range (Muresk: 2.00 m) -> a periodic "comb" is
    fitted to the whole range, far more robust than a minimum found plot by plot;
  * each plot is re-centred between its two tracks (across) and on the alleys at its ends (along), with
    equal end margins;
  * the crop rows (seeder pitch, Muresk 23 cm) are fitted as a second comb inside the seeded span and each
    tooth is tested for a real ridge, which gives the row count, pitch, bundle offset and per-row contrast;
    rows are written out as lines for row-wise analyses (border rows, seeding gaps, lodging).

Evidence per plot, from a 2 cm raster of (count, min z, max z) in the plot frame (u along, v across) with a
margin around the polygon: ground = 5th percentile of the cell minima in 30 cm blocks; canopy top = cell
max - ground; cover = canopy top > 10 cm; ground visible = cell min within 5 cm of the ground (where the
laser reached the soil - the plant BASES, which do not move when the canopy leans at maturity).
Tracks and alleys are read from whichever of cover / ground-visible / canopy-top shows them with the most
contrast on this flight (open canopy: cover; closed, leaning canopy: ground visibility).

Public API
----------
bin_plot_windows(las_path, grid_gdf, step=0.02, pad_along=1.3, pad_across=1.0, progress_cb=None) -> windows
bin_raster_windows(raster_tif, grid_gdf, ...) -> windows (canopy raster instead of a LAS; no ground visibility)
correct_grid_from_rows(windows, grid_gdf, margin_m=0.20, row_pitch_m=None, keep_width=True, ...)
    -> (corrected GeoDataFrame, rows GeoDataFrame (LineStrings), report DataFrame, diagnostics dict)
rows_from_profile(v, profile, lo, hi, pitch_prior=None) -> dict(rows, pitch, phase, present, contrast, ...)
rows_from_points(u, v, h, hw, hl, pitch_prior=None) -> the same for one plot's normalised points (per-plot LAS,
    no margin): used by the LiDAR Features tab for n_rows / row_pitch_m / row_offset_m / row_contrast
save_row_outputs(stem, new_gdf, rows_gdf, report, diag) -> dict of written paths
render_rows_qa(windows, old_gdf, new_gdf, rows_gdf, report, diag, out_png) -> out_png
"""
from __future__ import annotations
import os
import json
import numpy as np

COVER_H = 0.10          # m above local ground: a cell with canopy top above this counts as crop cover
GROUND_VIS_H = 0.05     # m: a cell whose lowest return is within this of the ground model "sees the ground"
GROUND_BLOCK = 0.30     # m: block size of the ground model (5th percentile of the cell minima)
ROW_PITCH_RANGE = (0.17, 0.32)   # m: admissible seeder row spacing when no prior is given (cereal plot seeders: 18-30 cm)


# ----------------------------------------------------------------------------- frames
def _frame(geom, ref_u=None):
    xs, ys = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)[:-1].T
    c = np.array([xs.mean(), ys.mean()])
    e1 = np.array([xs[1] - xs[0], ys[1] - ys[0]]); e2 = np.array([xs[2] - xs[1], ys[2] - ys[1]])
    L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
    a_u, hl, hw = (e1 / L1, L1 / 2, L2 / 2) if L1 >= L2 else (e2 / L2, L2 / 2, L1 / 2)
    if ref_u is not None and float(a_u @ ref_u) < 0:
        a_u = -a_u
    a_v = np.array([-a_u[1], a_u[0]])
    return c, a_u, a_v, hl, hw


def _reference_axis(gdf):
    _, a_u, _, _, _ = _frame(gdf.geometry.iloc[0])
    if a_u[1] < 0 or (a_u[1] == 0 and a_u[0] < 0):
        a_u = -a_u
    return a_u


def _frames(gdf, pad_along, pad_across, step):
    ref_u = _reference_axis(gdf); out = []
    for pid, geom in zip(gdf["Plot_ID"].astype(int).values, gdf.geometry):
        c, a_u, a_v, hl, hw = _frame(geom, ref_u)
        HU, HV = hl + pad_along, hw + pad_across
        out.append(dict(pid=pid, c=c, a_u=a_u, a_v=a_v, hl=hl, hw=hw, HU=HU, HV=HV,
                        nu=int(np.ceil(2 * HU / step)), nv=int(np.ceil(2 * HV / step)), R=float(np.hypot(HU, HV))))
    return ref_u, out


# ----------------------------------------------------------------------------- binning
def bin_plot_windows(las_path, grid_gdf, step=0.02, pad_along=1.3, pad_across=1.0, progress_cb=None, chunk=4_000_000):
    """One pass over the LAS: per plot, (count, min z, max z) on a `step` grid in the plot frame."""
    import laspy
    ref_u, fr = _frames(grid_gdf, pad_along, pad_across, step)
    acc = {f["pid"]: dict(n=np.zeros((f["nu"], f["nv"]), np.int32), zmin=np.full((f["nu"], f["nv"]), np.inf, np.float32),
                          zmax=np.full((f["nu"], f["nv"]), -np.inf, np.float32)) for f in fr}
    bb = {f["pid"]: (f["c"][0] - f["R"], f["c"][0] + f["R"], f["c"][1] - f["R"], f["c"][1] + f["R"]) for f in fr}
    done = 0
    with laspy.open(las_path) as src:
        total = max(src.header.point_count, 1)
        for ch in src.chunk_iterator(chunk):
            x = np.asarray(ch.x, float); y = np.asarray(ch.y, float); z = np.asarray(ch.z, np.float32)
            ok0 = np.ones(len(x), bool)
            if "classification" in ch.point_format.dimension_names:
                ok0 &= np.asarray(ch.classification) != 7
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            for f in fr:
                bx0, bx1, by0, by1 = bb[f["pid"]]
                if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
                    continue
                m = ok0 & (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
                if not m.any():
                    continue
                dx = x[m] - f["c"][0]; dy = y[m] - f["c"][1]
                u = dx * f["a_u"][0] + dy * f["a_u"][1]; v = dx * f["a_v"][0] + dy * f["a_v"][1]
                iu = np.floor((u + f["HU"]) / step).astype(np.int64); iv = np.floor((v + f["HV"]) / step).astype(np.int64)
                k = (iu >= 0) & (iu < f["nu"]) & (iv >= 0) & (iv < f["nv"])
                if not k.any():
                    continue
                a = acc[f["pid"]]; iu, iv, zz = iu[k], iv[k], z[m][k]
                np.add.at(a["n"], (iu, iv), 1); np.minimum.at(a["zmin"], (iu, iv), zz); np.maximum.at(a["zmax"], (iu, iv), zz)
            done += len(x)
            if progress_cb:
                progress_cb(int(60 * done / total), f"reading point cloud {done / 1e6:.0f} / {total / 1e6:.0f} M points")
    return dict(step=step, ref_u=ref_u, frames=fr, plots=acc, source=os.path.basename(las_path), kind="las")


def bin_raster_windows(raster_tif, grid_gdf, step=0.02, pad_along=1.3, pad_across=1.0, progress_cb=None):
    """Same structure from a canopy raster (CHM or vegetation index): zmax = raster value, zmin = zmax (no ground visibility)."""
    import rasterio
    from rasterio.windows import from_bounds
    ref_u, fr = _frames(grid_gdf, pad_along, pad_across, step); acc = {}
    with rasterio.open(raster_tif) as src:
        nd = src.nodata
        for i, f in enumerate(fr):
            R = f["R"]; w = from_bounds(f["c"][0] - R, f["c"][1] - R, f["c"][0] + R, f["c"][1] + R, src.transform)
            a = src.read(1, window=w).astype(np.float32); tr = src.window_transform(w)
            if nd is not None: a[a == nd] = np.nan
            ny, nx = a.shape
            xs = tr.c + (np.arange(nx) + 0.5) * tr.a; ys = tr.f + (np.arange(ny) + 0.5) * tr.e
            X, Y = np.meshgrid(xs, ys); dx = X - f["c"][0]; dy = Y - f["c"][1]
            u = dx * f["a_u"][0] + dy * f["a_u"][1]; v = dx * f["a_v"][0] + dy * f["a_v"][1]
            iu = np.floor((u.ravel() + f["HU"]) / step).astype(np.int64); iv = np.floor((v.ravel() + f["HV"]) / step).astype(np.int64)
            vals = a.ravel(); k = (iu >= 0) & (iu < f["nu"]) & (iv >= 0) & (iv < f["nv"]) & np.isfinite(vals)
            n = np.zeros((f["nu"], f["nv"]), np.int32); zmax = np.full((f["nu"], f["nv"]), -np.inf, np.float32)
            np.add.at(n, (iu[k], iv[k]), 1); np.maximum.at(zmax, (iu[k], iv[k]), vals[k])
            if (n == 0).any() and (n > 0).any():
                from scipy.ndimage import distance_transform_edt
                idx = distance_transform_edt(n == 0, return_distances=False, return_indices=True)
                zmax = zmax[tuple(idx)]; n = np.maximum(n, 1)
            acc[f["pid"]] = dict(n=n, zmin=zmax.copy(), zmax=zmax)
            if progress_cb and i % 16 == 0: progress_cb(int(60 * i / len(fr)), f"sampling raster, plot {i + 1}/{len(fr)}")
    return dict(step=step, ref_u=ref_u, frames=fr, plots=acc, source=os.path.basename(raster_tif), kind="raster")


# ----------------------------------------------------------------------------- evidence
def _smooth(p, k):
    from scipy.ndimage import uniform_filter1d
    p = np.asarray(p, float)
    fill = np.nanmean(p) if np.isfinite(p).any() else 0.0
    return uniform_filter1d(np.nan_to_num(p, nan=fill), max(int(round(k)), 1))


def _ground_model(zmin, have, step):
    from scipy.ndimage import uniform_filter
    nu, nv = zmin.shape; bs = max(int(round(GROUND_BLOCK / step)), 1)
    gb = np.full((nu // bs + 1, nv // bs + 1), np.nan)
    for i in range(gb.shape[0]):
        for j in range(gb.shape[1]):
            blk = zmin[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs]; blk = blk[np.isfinite(blk)]
            if blk.size > 10: gb[i, j] = np.percentile(blk, 5)
    if not np.isfinite(gb).any():
        gb[:] = 0.0
    gb = np.where(np.isfinite(gb), gb, np.nanmedian(gb)); gb = uniform_filter(gb, 3, mode="nearest")
    return np.repeat(np.repeat(gb, bs, axis=0), bs, axis=1)[:nu, :nv]


def _plot_evidence(win, f, step, raster=False, shift=(0.0, 0.0), hl=None, hw=None):
    """canopy-top raster h (m) in the plot frame + across (v) and along (u) profiles of canopy top, cover and ground visibility.
    shift = (du, dv): position of the INPUT polygon centre in the window frame (the window was binned around another grid);
    hl, hw = the input polygon's half length / width (default: the window's)."""
    n = win["n"]; zmin = win["zmin"].astype(float); zmax = win["zmax"].astype(float)
    have = n > 0; zmin[~have] = np.nan; zmax[~have] = np.nan
    hl = f["hl"] if hl is None else hl; hw = f["hw"] if hw is None else hw
    nu, nv = n.shape; u = -f["HU"] + (np.arange(nu) + 0.5) * step - shift[0]; v = -f["HV"] + (np.arange(nv) + 0.5) * step - shift[1]
    if raster:
        ground = np.zeros_like(zmax); h = zmax; gvis = np.full_like(zmax, np.nan)
    else:
        ground = _ground_model(zmin, have, step); h = zmax - ground
        gvis = np.where(have, ((zmin - ground) < GROUND_VIS_H).astype(float), np.nan)
    cover = np.where(have, (h > COVER_H).astype(float), np.nan)
    inner_u = np.abs(u) < hl - 0.4; inner_v = np.abs(v) < hw - 0.2
    with np.errstate(all="ignore"):
        prof = dict(top_v=np.nanmean(h[inner_u], axis=0), cov_v=np.nanmean(cover[inner_u], axis=0), gvis_v=np.nanmean(gvis[inner_u], axis=0),
                    top_u=np.nanmean(h[:, inner_v], axis=1), cov_u=np.nanmean(cover[:, inner_v], axis=1), gvis_u=np.nanmean(gvis[:, inner_v], axis=1))
        canopy = float(np.nanpercentile(h[np.ix_(inner_u, inner_v)], 90)) if (inner_u.any() and inner_v.any()) else np.nan
    return dict(u=u, v=v, h=h, canopy_m=canopy, **prof)


def _gap_evidence(e, kind, axis):
    """profile in [0, 1]-like units whose MINIMA are the bare gaps (tracks across, alleys along)."""
    p = e[f"{kind}_{axis}"]
    if kind == "cov": return p
    if kind == "gvis": return 1.0 - p
    c = e["canopy_m"] if np.isfinite(e["canopy_m"]) and e["canopy_m"] > 0.05 else np.nanmax(p)
    return np.clip(p / max(c, 1e-6), 0, 1.5)


def _contrast(prof, x, zones):
    """(plateau - minimum inside the zones) / plateau, median over zones; plateau = 75th percentile of the profile."""
    p = np.asarray(prof, float); ok = np.isfinite(p)
    if ok.sum() < 10: return np.nan
    plateau = np.nanpercentile(p, 75); vals = []
    for lo, hi in zones:
        m = (x >= lo) & (x <= hi) & ok
        if m.sum() > 3: vals.append((plateau - np.nanmin(p[m])) / max(plateau, 1e-6))
    return float(np.median(vals)) if vals else np.nan


# ----------------------------------------------------------------------------- combs
def _fit_comb(positions, weights, pitch0, tol=0.10, pitch_range=0.12, n_pitch=49, n_phase=60):
    """Fit t_k = phase + k * pitch to candidate positions (robust: maximise the weight within `tol` of a tooth).
    Returns (pitch, phase, inlier weight)."""
    positions = np.asarray(positions, float); weights = np.asarray(weights, float)
    ok = np.isfinite(positions) & np.isfinite(weights) & (weights > 0); positions, weights = positions[ok], weights[ok]
    best = (float(pitch0), 0.0, 0.0)
    if len(positions) < 2:
        return best
    for P in np.linspace(pitch0 * (1 - pitch_range), pitch0 * (1 + pitch_range), n_pitch):
        ph = np.mod(positions, P)
        for ph0 in np.linspace(0, P, n_phase, endpoint=False):
            d = np.abs((ph - ph0 + P / 2) % P - P / 2); w = float(weights[d < tol].sum())
            if w > best[2]:
                best = (float(P), float(ph0), w)
    P, ph0, _ = best
    k = np.round((positions - ph0) / P); d = positions - (ph0 + k * P); inl = np.abs(d) < tol
    if inl.sum() >= 3 and np.ptp(k[inl]) > 0:
        try:
            A = np.c_[k[inl], np.ones(inl.sum())]; sol, *_ = np.linalg.lstsq(A, positions[inl], rcond=None)
            if np.isfinite(sol).all() and pitch0 * (1 - pitch_range) * 0.9 < sol[0] < pitch0 * (1 + pitch_range) * 1.1:
                P, ph0 = float(sol[0]), float(sol[1])
        except Exception:
            pass
    return float(P), float(ph0), float(best[2])


def _row_signal(profile, step):
    """detrended, unit-variance ridge signal of an across-plot profile (rows = peaks)."""
    p = _smooth(profile, 0.04 / step); det = p - _smooth(p, 0.5 / step)
    sd = np.nanstd(det)
    return det / sd if sd > 0 else det


def row_pitch_from_profile(signal, step, lo=None, hi=None):
    """(pitch m, power share) of the strongest periodicity of a ridge signal within ROW_PITCH_RANGE (or lo..hi)."""
    lo, hi = (lo or ROW_PITCH_RANGE[0]), (hi or ROW_PITCH_RANGE[1])
    s = np.nan_to_num(np.asarray(signal, float)); s = s - s.mean()
    if len(s) < 20: return np.nan, 0.0
    n = 4096; F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n)) ** 2; fq = np.fft.rfftfreq(n, d=step)
    band = (fq >= 1 / hi) & (fq <= 1 / lo)
    if not band.any() or F[band].sum() <= 0: return np.nan, 0.0
    k = int(np.argmax(np.where(band, F, 0))); return float(1 / fq[k]), float(F[k] / F[band].sum())


def rows_from_profile(v, profile, lo, hi, pitch_prior=None, step=None, edge_gap=0.06, signal2=None):
    """Crop rows from an across-plot profile whose PEAKS are the rows (canopy top), between the seeded-span limits lo..hi
    (the track edges). `signal2` = optional second ridge profile (e.g. 1 - ground visibility) added to the canopy top.

    The row PITCH is the flight's seeder spacing (pitch_prior, estimated once per flight from the periodogram of all
    plots); only the PHASE is fitted per plot, by maximising the ridge signal at the teeth minus the signal at the
    mid-gaps. Teeth lie between lo + edge_gap and hi - edge_gap (outer rows sit ~6 cm inside the track edge).
    Each tooth gets a contrast (ridge - mid-gaps); n_rows_geom = number of teeth that fit the span, n_rows = teeth with
    positive contrast (a non-positive contrast flags a missing / lodged row).
    Returns dict(rows (all teeth), teeth, present, contrast, pitch, phase, n_rows, n_rows_geom, offset, span_lo, span_hi, pitch_share)."""
    v = np.asarray(v, float); step = step or float(np.median(np.diff(v)))
    empty = dict(rows=[], teeth=[], present=[], contrast=[], pitch=np.nan, phase=np.nan, n_rows=0, n_rows_geom=0, offset=np.nan, span_lo=lo, span_hi=hi, pitch_share=0.0)
    inside = (v > lo) & (v < hi)
    if inside.sum() < 10 or not np.isfinite(profile).any():
        return empty
    sig = _row_signal(profile, step)
    if signal2 is not None and np.isfinite(signal2).any():
        sig = sig + _row_signal(signal2, step)
    P, share = row_pitch_from_profile(sig[inside], step)
    if pitch_prior is not None and np.isfinite(pitch_prior):
        P = float(pitch_prior)
    if not np.isfinite(P) or P <= 0:
        return empty
    # number of rows that fit the seeded span: the outer rows sit about a quarter pitch inside the track edges
    n_fit = int(round((hi - lo - P / 2) / P)) + 1
    if n_fit < 1:
        return empty
    def val(pos):
        m = np.abs(v - pos) < 0.03
        return float(np.nanmean(sig[m])) if m.any() else np.nan
    centre = (lo + hi) / 2; half = (n_fit - 1) * P / 2; best = (None, -np.inf)
    for off in np.arange(-P / 2, P / 2, step / 2):                      # comb of n_fit teeth, phase searched around the centred position
        teeth = centre + off + np.arange(n_fit) * P - half
        if teeth[0] < lo - 0.02 or teeth[-1] > hi + 0.02: continue
        s = np.nansum([val(t) for t in teeth]) - np.nansum([val(t + P / 2) for t in teeth[:-1]])
        if s > best[1]: best = (teeth, s)
    teeth = best[0]
    if teeth is None:
        return empty
    contrast = np.array([val(t) - np.nanmean([val(t - P / 2), val(t + P / 2)]) for t in teeth])
    present = np.isfinite(contrast) & (contrast > 0)
    return dict(rows=teeth.tolist(), teeth=teeth.tolist(), present=present.tolist(), contrast=contrast.tolist(), pitch=float(P), phase=float(teeth[0] % P),
                n_rows=int(present.sum()), n_rows_geom=int(len(teeth)), offset=float(teeth.mean() - (lo + hi) / 2), span_lo=float(lo), span_hi=float(hi), pitch_share=float(share))


def rows_from_points(u, v, h, hw, hl, pitch_prior=None, step=0.02):
    """Rows of ONE plot from its normalised points (per-plot LAS of the LiDAR Features tab: no margin, so the
    seeded span is the polygon width). Returns rows_from_profile(...) + 'profile' (2 cm canopy-top across profile)."""
    u = np.asarray(u, float); v = np.asarray(v, float); h = np.asarray(h, float)
    m = (np.abs(u) < max(hl - 0.3, 0.5)) & (np.abs(v) <= hw) & np.isfinite(h)
    if m.sum() < 200:
        return dict(rows=[], teeth=[], present=[], contrast=[], pitch=np.nan, phase=np.nan, n_rows=0, offset=np.nan, span_lo=-hw, span_hi=hw, profile=None)
    nu = int(np.ceil(2 * max(hl, 0.5) / step)); nv = int(np.ceil(2 * hw / step))
    iu = np.clip(np.floor((u[m] + hl) / step).astype(int), 0, nu - 1); iv = np.clip(np.floor((v[m] + hw) / step).astype(int), 0, nv - 1)
    top = np.full((nu, nv), -np.inf); np.maximum.at(top, (iu, iv), h[m]); top[~np.isfinite(top)] = np.nan
    with np.errstate(all="ignore"): prof = np.nanmean(top, axis=0)
    vv = -hw + (np.arange(nv) + 0.5) * step
    out = rows_from_profile(vv, prof, -hw + 0.02, hw - 0.02, pitch_prior=pitch_prior, step=step); out["profile"] = prof
    return out


# ----------------------------------------------------------------------------- main
def correct_grid_from_rows(windows, grid_gdf, margin_m=0.15, row_pitch_m=None, keep_width=True, max_across_shift_m=0.6, max_along_shift_m=1.0,
                           extend_max_m=0.6, shrink_max_m=0.6, group_col=None, progress_cb=None, mode="per_plot", end_frac=0.4):
    """See the module docstring. `windows` = bin_plot_windows(...) or bin_raster_windows(...).
    mode 'per_plot': every plot is re-centred on its own tracks / alleys (fit a trial's reference grid on the flight with
                     the clearest rows, usually before canopy closure);
         'rigid'   : the SAME shift (median across, median along) is applied to every plot and lengths are kept - use it to
                     register a reference grid to another flight of the same trial (flight-to-flight georeferencing offsets
                     of 0.1-0.5 m are common), where the plots must not be re-fitted individually on a leaning canopy.
    row_pitch_m: seeder row spacing; None = estimated from the flight (median per-plot periodogram pitch).
    end_frac: fraction of the inside-to-alley drop at which the plot end is read (0.4 reaches the sparse end plants)."""
    import pandas as pd
    from scipy.signal import find_peaks
    from shapely.geometry import Polygon, LineString

    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    step = windows["step"]; raster = windows.get("kind") == "raster"
    fr = {f["pid"]: f for f in windows["frames"]}; ref_u = windows["ref_u"]; ref_v = np.array([-ref_u[1], ref_u[0]])
    gdf = grid_gdf.reset_index(drop=True).copy(); gdf["Plot_ID"] = gdf["Plot_ID"].astype(int)
    cxy = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry]); V = cxy @ ref_v; U = cxy @ ref_u
    hw0 = float(np.median([fr[p]["hw"] for p in gdf.Plot_ID])); hl0 = float(np.median([fr[p]["hl"] for p in gdf.Plot_ID]))
    if group_col and group_col in gdf.columns:
        groups = gdf[group_col].astype(str).values
    else:
        # across-lines: plots whose along positions are within one plot length of each other stand side by side and share
        # one track comb (the Range / Row attributes mean different things in different trials, so they are not used)
        order = np.argsort(U); gid = np.zeros(len(U), int); g = 0
        for a, b in zip(order[:-1], order[1:]):
            if U[b] - U[a] > hl0: g += 1
            gid[b] = g
        groups = np.array([f"L{k + 1:02d}" for k in gid])
    d = np.abs(V[None, :] - V[:, None]); d[(np.abs(U[None, :] - U[:, None]) > hl0) | (d < 1e-6)] = np.inf
    nn = d.min(axis=1); pitch0 = float(np.median(nn[np.isfinite(nn)])) if np.isfinite(nn).any() else 2 * hw0 + 0.4

    # frames of the INPUT polygons (the windows may have been binned around another grid of the same trial)
    fin = {}
    for pid, geom in zip(gdf.Plot_ID, gdf.geometry):
        c, a_u, a_v, hl, hw = _frame(geom, ref_u); f = fr[pid]; dvec = c - f["c"]
        fin[pid] = dict(c=c, a_u=a_u, a_v=a_v, hl=hl, hw=hw, shift=(float(dvec @ f["a_u"]), float(dvec @ f["a_v"])))
    # 1. per-plot evidence + choice of the gap evidence (tracks / alleys) for this flight
    ev = {}
    for i, pid in enumerate(gdf.Plot_ID):
        fi = fin[pid]
        ev[pid] = _plot_evidence(windows["plots"][pid], fr[pid], step, raster=raster, shift=fi["shift"], hl=fi["hl"], hw=fi["hw"])
        if i % 16 == 0: _p(60 + int(15 * i / len(gdf)), f"canopy profiles, plot {i + 1}/{len(gdf)}")
    canopy = float(np.nanmedian([e["canopy_m"] for e in ev.values()]))
    kinds = ["top"] if raster else ["cov", "gvis", "top"]
    def median_contrast(kind, axis):
        vals = []
        for pid in gdf.Plot_ID:
            e = ev[pid]; f = fin[pid]
            if axis == "v": zones = [(-f["hw"] - 0.45, -f["hw"] + 0.25), (f["hw"] - 0.25, f["hw"] + 0.45)]; x = e["v"]
            else: zones = [(-f["hl"] - 1.0, -f["hl"] + 0.3), (f["hl"] - 0.3, f["hl"] + 1.0)]; x = e["u"]
            vals.append(_contrast(_smooth(_gap_evidence(e, kind, axis), 0.08 / step), x, zones))
        return float(np.nanmedian(vals)) if np.isfinite(vals).any() else -1.0
    c_v = {k: median_contrast(k, "v") for k in kinds}; c_u = {k: median_contrast(k, "u") for k in kinds}
    kind_v = max(c_v, key=c_v.get); kind_u = max(c_u, key=c_u.get)

    # 2. range-wide track comb (global across coordinate)
    tracks_by_group, comb_by_group, prof_by_group, trackw_by_group = {}, {}, {}, {}
    for g in np.unique(groups):
        sel = np.where(groups == g)[0]
        HVg = fr[gdf.Plot_ID[sel[0]]]["HV"]; vmin = V[sel].min() - HVg; vmax = V[sel].max() + HVg
        gv = np.arange(vmin, vmax + step, step); S = np.zeros(len(gv)); N = np.zeros(len(gv))
        for i in sel:
            e = ev[gdf.Plot_ID[i]]; prof = _gap_evidence(e, kind_v, "v"); ok = np.isfinite(prof)
            j = np.clip(np.round((V[i] + e["v"][ok] - vmin) / step).astype(int), 0, len(gv) - 1)
            np.add.at(S, j, prof[ok]); np.add.at(N, j, 1)
        with np.errstate(invalid="ignore"): P = np.where(N > 0, S / np.maximum(N, 1), np.nan)
        Ps = _smooth(P, 0.08 / step)
        plateau = np.nanpercentile(Ps, 75); floor = np.nanpercentile(Ps, 5); prom = 0.25 * (plateau - floor) if plateau > floor else 1e-3
        pk, pr = find_peaks(-Ps, distance=int(0.5 * pitch0 / step), prominence=prom)
        pos = gv[pk]; wts = pr["prominences"]
        Pfit, ph, winl = _fit_comb(pos, wts, pitch0)
        k0 = int(np.floor((vmin - ph) / Pfit)) - 1; k1 = int(np.ceil((vmax - ph) / Pfit)) + 1
        teeth = ph + np.arange(k0, k1 + 1) * Pfit
        # track width at half depth around the teeth (median)
        widths = []
        for t in teeth:
            m = np.abs(gv - t) < 0.6
            if m.sum() < 10: continue
            seg = Ps[m]; xs = gv[m]; k = int(np.argmin(np.abs(xs - t))); thr = seg[k] + 0.5 * (plateau - seg[k])
            a = k
            while a > 0 and seg[a - 1] < thr: a -= 1
            b = k
            while b < len(seg) - 1 and seg[b + 1] < thr: b += 1
            if 0 < a and b < len(seg) - 1: widths.append(xs[b] - xs[a])
        trackw_by_group[g] = float(np.median(widths)) if widths else 0.4
        tracks_by_group[g] = teeth; prof_by_group[g] = (gv, Ps, pos)
        comb_by_group[g] = dict(pitch=float(Pfit), phase=float(ph), n_minima=int(len(pos)), inlier_weight=float(winl), track_width_m=trackw_by_group[g],
                                n_teeth_in_range=int(((teeth > V[sel].min() - pitch0) & (teeth < V[sel].max() + pitch0)).sum()))
    _p(80, "fitting plots between their tracks")
    # seeded span of every plot (between the edges of its two tracks) and the flight row pitch: median of the per-plot
    # periodogram pitches of the ridge signal (canopy top + inter-row ground visibility) inside the span
    spans = {}; pitches = []
    for i, pid in enumerate(gdf.Plot_ID):
        f = fin[pid]; e = ev[pid]; g = groups[i]; teeth = tracks_by_group[g]; tw = trackw_by_group[g]
        mids = (teeth[:-1] + teeth[1:]) / 2; j = int(np.argmin(np.abs(mids - V[i])))
        t_lo, t_hi = teeth[j] - V[i], teeth[j + 1] - V[i]; spans[pid] = (t_lo, t_hi, tw)
        v = e["v"]; m = (v > t_lo + tw / 2 + 0.03) & (v < t_hi - tw / 2 - 0.03)
        sig = _row_signal(e["top_v"], step)
        if not raster and np.isfinite(e["gvis_v"]).any(): sig = sig + _row_signal(1 - e["gvis_v"], step)
        p_, share = row_pitch_from_profile(sig[m], step) if m.sum() > 20 else (np.nan, 0.0); pitches.append((p_, share))
    pitches = np.array(pitches); okp = np.isfinite(pitches[:, 0])
    row_pitch_flight = float(np.median(pitches[okp, 0])) if okp.any() else np.nan
    row_pitch_spread = float(np.percentile(pitches[okp, 0], 75) - np.percentile(pitches[okp, 0], 25)) if okp.any() else np.nan
    row_pitch_est = float(row_pitch_m) if row_pitch_m else row_pitch_flight
    rows_reliable = bool(np.isfinite(row_pitch_flight) and row_pitch_spread <= 0.03) or bool(row_pitch_m)

    # 3. per plot
    recs, row_lines, new_geoms = [], [], []
    for i, pid in enumerate(gdf.Plot_ID):
        f = fin[pid]; e = ev[pid]; g = groups[i]; t_lo, t_hi, tw = spans[pid]
        across = float((t_lo + t_hi) / 2); across_ok = abs(across) <= max_across_shift_m; across_applied = across if across_ok else 0.0
        prof = _smooth(_gap_evidence(e, kind_v, "v"), 0.08 / step); v = e["v"]
        def local_min(t):
            m = np.abs(v - t) < 0.25
            return float(v[m][np.argmin(prof[m])]) if m.sum() > 3 else np.nan
        loc_lo, loc_hi = local_min(t_lo), local_min(t_hi)
        span_lo, span_hi = t_lo + tw / 2, t_hi - tw / 2                      # seeded span between the track edges
        # along: alley at each end (half-drop of the chosen evidence between its inside level and the alley floor)
        pu = _smooth(_gap_evidence(e, kind_u, "u"), 0.10 / step); u = e["u"]; hl = f["hl"]
        inside_u = np.nanmedian(pu[np.abs(u) < hl - 0.5])
        def end(side):
            outer = (u < -hl - 0.3) if side < 0 else (u > hl + 0.3)
            if not outer.any() or not np.isfinite(inside_u): return np.nan
            floor_u = np.nanmin(pu[outer]); thr = floor_u + end_frac * (inside_u - floor_u)
            if inside_u - floor_u < 0.15 * max(inside_u, 0.05): return np.nan
            if side < 0:
                idx = np.where((u < -hl + 0.8) & (pu < thr))[0]; return float(u[idx[-1]]) if len(idx) else np.nan
            idx = np.where((u > hl - 0.8) & (pu < thr))[0]; return float(u[idx[0]]) if len(idx) else np.nan
        u0, u1 = end(-1), end(+1)
        crop_len = u1 - u0 if np.isfinite(u0) and np.isfinite(u1) else np.nan
        along = (u0 + u1) / 2 if np.isfinite(crop_len) else np.nan
        along_ok = bool(np.isfinite(along) and abs(along) <= max_along_shift_m and (2 * hl - shrink_max_m - 0.3) <= crop_len <= (2 * hl + extend_max_m + 1.0))
        # rows: row comb (flight pitch, per-plot phase) on the canopy-top ridges + inter-row ground visibility inside the seeded span
        rw = rows_from_profile(v, e["top_v"], span_lo, span_hi, pitch_prior=row_pitch_est if np.isfinite(row_pitch_est) else None, step=step,
                               signal2=None if (raster or not np.isfinite(e["gvis_v"]).any()) else 1 - e["gvis_v"])
        recs.append(dict(idx=i, Plot_ID=pid, group=g, poly_len=2 * hl, poly_w=2 * f["hw"], track_lo=t_lo, track_hi=t_hi, track_pitch=t_hi - t_lo, track_w=tw,
                         seeded_w=span_hi - span_lo, local_min_lo=loc_lo, local_min_hi=loc_hi, across_off=across, across_ok=across_ok, across_applied=across_applied,
                         crop_start=u0, crop_end=u1, crop_len=crop_len, along_off=along, along_ok=along_ok,
                         n_rows=rw["n_rows_geom"], n_rows_ok=rw["n_rows"], row_pitch=rw["pitch"], row_offset=rw["offset"], rows_v=";".join(f"{x:.3f}" for x in rw["rows"]),
                         row_contrast=float(np.nanmean(rw["contrast"])) if len(rw["contrast"]) else np.nan, row_pitch_share=rw["pitch_share"],
                         rows_weak=";".join(f"{t:.3f}" for t, pres in zip(rw["teeth"], rw["present"]) if not pres), canopy_m=e["canopy_m"]))
    rep = pd.DataFrame(recs)
    grp_med = rep[rep.along_ok].groupby("group").along_off.median()
    rep["along_applied"] = np.where(rep.along_ok, rep.along_off, rep.group.map(grp_med).fillna(0.0))
    rep["new_len"] = np.where(rep.along_ok, np.clip(rep.crop_len - 2 * margin_m, rep.poly_len - shrink_max_m, rep.poly_len + extend_max_m), rep.poly_len)
    if mode == "rigid":
        du = float(np.nanmedian(rep.along_off[rep.along_ok])) if rep.along_ok.any() else 0.0
        dv = float(np.nanmedian(rep.across_off[rep.across_ok])) if rep.across_ok.any() else 0.0
        rep["along_applied"] = du; rep["across_applied"] = dv; rep["new_len"] = rep.poly_len
    # 4. polygons + row lines
    for i, pid in enumerate(gdf.Plot_ID):
        f = fin[pid]; r = rep.iloc[i]
        c2 = f["c"] + f["a_u"] * r.along_applied + f["a_v"] * r.across_applied
        L = r.new_len; W = 2 * f["hw"] if keep_width else float(np.clip(r.seeded_w + 2 * margin_m, 0.5, r.track_pitch - 0.1))
        new_geoms.append(Polygon([c2 + f["a_u"] * s * L / 2 + f["a_v"] * t * W / 2 for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]))
        if r.rows_v:
            for k, vv in enumerate(float(x) for x in r.rows_v.split(";")):
                voff = vv - r.across_applied                     # relative to the corrected centre
                p0 = c2 + f["a_v"] * voff - f["a_u"] * L / 2; p1 = c2 + f["a_v"] * voff + f["a_u"] * L / 2
                row_lines.append(dict(Plot_ID=pid, row_idx=k + 1, v_off_m=round(voff, 3), length_m=round(L, 2), geometry=LineString([p0, p1])))
    out = gdf.copy(); out["geometry"] = new_geoms
    out["plot_len"] = np.round(rep.new_len.values, 2); out["along_off"] = np.round(rep.along_applied.values, 3); out["across_off"] = np.round(rep.across_applied.values, 3)
    out["crop_len"] = np.round(rep.crop_len.values, 2); out["seeded_w"] = np.round(rep.seeded_w.values, 3); out["n_rows"] = rep.n_rows.values; out["n_rows_ok"] = rep.n_rows_ok.values
    out["row_pitch"] = np.round(rep.row_pitch.values, 3); out["row_off"] = np.round(rep.row_offset.values, 3)
    out["refine_ok"] = (rep.along_ok & rep.across_ok).values; out["fit_src"] = f"{mode}:{kind_v}/{kind_u}"
    import geopandas as gpd
    cols = ["Plot_ID", "row_idx", "v_off_m", "length_m", "geometry"]
    rows_gdf = gpd.GeoDataFrame(row_lines, geometry="geometry", crs=gdf.crs) if row_lines else gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=gdf.crs)
    diag = dict(n=len(gdf), source=windows.get("source"), evidence_across=kind_v, evidence_along=kind_u, contrast_across=c_v, contrast_along=c_u, canopy_m=canopy,
                pitch_grid_m=pitch0, comb_by_group=comb_by_group, track_width_m=float(np.median(list(trackw_by_group.values()))),
                n_across_ok=int(rep.across_ok.sum()), n_along_ok=int(rep.along_ok.sum()),
                across_abs_median_m=float(rep.across_off.abs().median()), across_abs_p90_m=float(rep.across_off.abs().quantile(0.9)),
                across_mean_m=float(rep.across_applied.mean()), along_mean_m=float(rep.along_applied.mean()),
                along_abs_median_m=float(rep.along_off.abs().median()) if rep.along_ok.any() else None,
                crop_len_median_m=float(np.nanmedian(rep.crop_len)) if rep.along_ok.any() else None, poly_len_m=float(rep.poly_len.median()), new_len_median_m=float(rep.new_len.median()),
                seeded_w_median_m=float(np.nanmedian(rep.seeded_w)), rows_per_plot={int(k): int(v) for k, v in rep.n_rows.value_counts().sort_index().items()},
                row_pitch_median_m=float(np.nanmedian(rep.row_pitch)) if rep.row_pitch.notna().any() else None, row_pitch_prior_m=row_pitch_m,
                row_pitch_flight_m=row_pitch_flight if np.isfinite(row_pitch_flight) else None, row_pitch_used_m=row_pitch_est if np.isfinite(row_pitch_est) else None,
                row_pitch_iqr_m=row_pitch_spread if np.isfinite(row_pitch_spread) else None,
                rows_reliable=rows_reliable, rows_ok_per_plot={int(k): int(v) for k, v in rep.n_rows_ok.value_counts().sort_index().items()},
                mode=mode, end_frac=end_frac, margin_m=margin_m, keep_width=keep_width, _profiles={g: (gv.tolist(), Ps.tolist(), pos.tolist()) for g, (gv, Ps, pos) in prof_by_group.items()})
    _p(100, "grid corrected")
    return out, rows_gdf, rep, diag


def render_rows_gallery(windows, new_gdf, report, out_dir, per_page=16, prefix="rows_gallery"):
    """One panel per plot (canopy top in the corrected plot frame, fitted rows, tracks, seeded span), 16 plots per PNG page.
    Returns the list of PNG paths."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    step = windows["step"]; fr = {f["pid"]: f for f in windows["frames"]}; raster = windows.get("kind") == "raster"
    rep = report.set_index("Plot_ID"); new = new_gdf.set_index(new_gdf["Plot_ID"].astype(int)); pids = list(rep.index); paths = []
    vmax = max(float(np.nanmedian(rep.canopy_m)) * 1.3, 0.2) if "canopy_m" in rep else 1.0
    for page in range(0, len(pids), per_page):
        chunk = pids[page:page + per_page]; ncol = 4; nrow = int(np.ceil(len(chunk) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.3 * nrow)); axs = np.atleast_1d(axs).ravel()
        for ax in axs[len(chunk):]: ax.axis("off")
        for ax, pid in zip(axs, chunk):
            f = fr[pid]; e = _plot_evidence(windows["plots"][pid], f, step, raster=raster); r = rep.loc[pid]
            ax.imshow(np.clip(e["h"], 0, vmax).T, origin="lower", extent=[e["u"][0], e["u"][-1], e["v"][0], e["v"][-1]], cmap="viridis", aspect="equal")
            c, a_u, a_v, hl, hw = _frame(new.loc[pid].geometry, windows["ref_u"]); d = c - f["c"]; ou = d @ f["a_u"]; ov = d @ f["a_v"]
            ax.add_patch(plt.Rectangle((ou - hl, ov - hw), 2 * hl, 2 * hw, fill=False, ec="white", lw=1.2))
            for t in (r.track_lo, r.track_hi): ax.axhline(t, color="orange", lw=0.7, ls=":")
            if isinstance(r.rows_v, str) and r.rows_v:
                for vv in (float(x) for x in r.rows_v.split(";")): ax.axhline(vv, color="cyan", lw=0.5, ls="--", xmin=0.18, xmax=0.82)
            if isinstance(r.rows_weak, str) and r.rows_weak:
                for vv in (float(x) for x in r.rows_weak.split(";")): ax.axhline(vv, color="magenta", lw=0.8, ls=":", xmin=0.18, xmax=0.82)
            br = new.loc[pid].get("B/R", "") if "B/R" in new.columns else ""
            ax.set_title(f"plot {pid} {br}: {int(r.n_rows)} rows ({int(r.n_rows_ok)} clear), pitch {r.row_pitch * 100:.1f} cm, bundle {r.row_offset * 100:+.0f} cm", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Rows per plot ({windows.get('source')}): white = corrected plot, orange = wheel tracks, cyan = rows, magenta = weak row", fontsize=9)
        fig.tight_layout(); p = os.path.join(out_dir, f"{prefix}_{page // per_page + 1:02d}.png"); fig.savefig(p, dpi=100); plt.close(fig); paths.append(p)
    return paths


def save_row_outputs(stem, new_gdf, rows_gdf, report, diag):
    """<stem>.shp (corrected grid), <stem>_rows.shp (row lines), <stem>_rows.csv (rows per plot), <stem>_report.csv, <stem>_diag.json."""
    from .grid_io import save_grid
    paths = {"grid": save_grid(new_gdf, stem + ".shp")}
    if len(rows_gdf):
        rows_gdf.to_file(stem + "_rows.shp", driver="ESRI Shapefile"); paths["rows"] = stem + "_rows.shp"
    report.round(3).to_csv(stem + "_report.csv", index=False); paths["report"] = stem + "_report.csv"
    report[["Plot_ID", "n_rows", "n_rows_ok", "row_pitch", "row_pitch_share", "row_offset", "row_contrast", "rows_v", "rows_weak", "track_lo", "track_hi", "seeded_w", "across_applied", "along_applied", "crop_len"]].round(3) \
        .to_csv(stem + "_rows.csv", index=False); paths["rows_csv"] = stem + "_rows.csv"
    json.dump({k: v for k, v in diag.items() if not k.startswith("_")}, open(stem + "_diag.json", "w"), indent=1, default=float); paths["diag"] = stem + "_diag.json"
    return paths


def render_rows_qa(windows, old_gdf, new_gdf, rows_gdf, report, diag, out_png, n_zoom=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    step = windows["step"]; fr = {f["pid"]: f for f in windows["frames"]}; raster = windows.get("kind") == "raster"
    rep = report.set_index("Plot_ID"); old = old_gdf.set_index(old_gdf["Plot_ID"].astype(int)); new = new_gdf.set_index(new_gdf["Plot_ID"].astype(int))
    pick = list(rep.across_off.abs().sort_values(ascending=False).index[:n_zoom // 2]) + list(rep.along_off.abs().sort_values(ascending=False).index[:n_zoom - n_zoom // 2])
    pick = list(dict.fromkeys(pick))[:n_zoom]
    ng = min(len(diag["_profiles"]), 4); fig = plt.figure(figsize=(3.3 * n_zoom, 4.4 + 1.6 * ng))
    gs = fig.add_gridspec(1 + ng, n_zoom, height_ratios=[4] + [1.4] * ng)
    for k, pid in enumerate(pick):
        ax = fig.add_subplot(gs[0, k]); f = fr[pid]; e = _plot_evidence(windows["plots"][pid], f, step, raster=raster)
        ax.imshow(np.clip(e["h"], 0, max(diag["canopy_m"] * 1.3, 0.2)).T, origin="lower", extent=[e["u"][0], e["u"][-1], e["v"][0], e["v"][-1]], cmap="viridis", aspect="equal")
        for gdf_, col, lw in ((old, "red", 1.0), (new, "white", 1.6)):
            c, a_u, a_v, hl, hw = _frame(gdf_.loc[pid].geometry, windows["ref_u"]); d = c - f["c"]; ou = d @ f["a_u"]; ov = d @ f["a_v"]
            ax.add_patch(plt.Rectangle((ou - hl, ov - hw), 2 * hl, 2 * hw, fill=False, ec=col, lw=lw, ls="--" if col == "red" else "-"))
        r = rep.loc[pid]
        for t in (r.track_lo, r.track_hi): ax.axhline(t, color="orange", lw=0.8, ls=":")
        if isinstance(r.rows_v, str) and r.rows_v:
            for vv in (float(x) for x in r.rows_v.split(";")): ax.axhline(vv, color="cyan", lw=0.5, ls="--", xmin=0.2, xmax=0.8)
        if isinstance(r.rows_weak, str) and r.rows_weak:
            for vv in (float(x) for x in r.rows_weak.split(";")): ax.axhline(vv, color="magenta", lw=0.8, ls=":", xmin=0.2, xmax=0.8)
        ax.set_title(f"plot {pid}: across {r.across_applied * 100:+.0f} cm, along {r.along_applied * 100:+.0f} cm\n{int(r.n_rows)} rows ({int(r.n_rows_ok)} clear), pitch {r.row_pitch * 100:.1f} cm", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j, (g, (gv, Ps, pos)) in enumerate(list(diag["_profiles"].items())[:ng]):
        ax = fig.add_subplot(gs[1 + j, :]); gv = np.array(gv); Ps = np.array(Ps)
        ax.plot(gv, Ps, "k-", lw=0.8); cb = diag["comb_by_group"][g]
        for t in cb["phase"] + np.arange(-2, 400) * cb["pitch"]:
            if gv[0] <= t <= gv[-1]: ax.axvline(t, color="orange", lw=0.6)
        ax.plot(pos, np.interp(pos, gv, Ps), "r.", ms=3); ax.set_ylabel(f"range {g}", fontsize=8); ax.tick_params(labelsize=7)
    fig.suptitle(f"Grid corrected from rows - tracks from {diag['evidence_across']}, alleys from {diag['evidence_along']} ({windows.get('source')}); "
                 f"red dashed = before, white = after, orange = tracks, cyan = rows, magenta = weak row", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png
