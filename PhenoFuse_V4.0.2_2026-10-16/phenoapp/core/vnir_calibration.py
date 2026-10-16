"""
Reflectance check and empirical-line calibration of a VNIR cube (PhenoFuse 4.0.4).

Why: a GRYFN VNIR orthomosaic can be delivered as at-sensor RADIANCE (float32, no reflectance scale factor, the oxygen-A
760 nm and water-vapour 940 nm absorption dips of sunlight still in every spectrum, NIR about equal to red). Every
vegetation index built on it is meaningless. If the processing cannot be redone, an empirical line recovers reflectance
band by band: R(l) = gain(l) * L(l) [+ offset(l)], with the gain measured on a surface whose reflectance is known.

Reference options (best first)
  * targets   : polygons of reference panels / tarps in THIS flight with a known reflectance spectrum (or a constant
                reflectance per target). Same date, same illumination: the proper empirical line. Two or more targets of
                different brightness also give an offset per band.
  * cube      : a calibrated reflectance cube of the same trial (another flight of the same day, or another date) - the
                bare soil of the trial (wheel tracks, alleys) is used as a pseudo-invariant target. Across DATES this
                assumes the soil's reflectance did not change: soil moisture, roughness and residue do change it, but
                mostly as a wavelength-flat factor, so ratio indices survive and absolute reflectance is uncertain by
                roughly 10-20 %. Across flights of the same day it is nearly exact.
Both are reported with their internal consistency (gain repeatability between two halves of the samples) and a
before / after figure, and the output cube carries the recipe in its header.

Public API
----------
calibration_check(cube_path) -> dict            is this cube reflectance or radiance? (absorption features, NIR/red, dtype, scale)
gain_from_reference_cube(cube_path, ref_cube_path, grid=None, soil_ndvi_max=0.20, ...) -> dict(wavelength, gain, offset, ...)
gain_from_targets(cube_path, targets, spectra_csv=None) -> dict(...)
apply_calibration(cube_path, gain, offset, out_path=None, scale=10000) -> out_path   (ENVI uint16 x scale, header with wavelengths)
render_calibration_qa(cube_path, result, out_png) -> out_png
"""
from __future__ import annotations
import os
import re
import json
import numpy as np

SOIL_NDVI_MAX = 0.20


def _wavelengths(stem_or_bin):
    from .vnir import parse_envi_wavelengths, _find_header
    return parse_envi_wavelengths(_find_header(stem_or_bin))


def _scale_factor(bin_path):
    from .vnir import _find_header
    try:
        h = open(_find_header(bin_path), "r", errors="ignore").read()
        m = re.search(r"reflectance scale factor\s*=\s*([\d.]+)", h, re.I)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _feature(spec, wl, centre, half=6, gap=8):
    """value at `centre` relative to the median of two flanking windows (absorption dip < 1, peak > 1)."""
    i = int(np.argmin(np.abs(wl - centre))); lo = max(i - gap - half, 0); hi = min(i + gap + half, len(spec) - 1)
    flank = np.r_[spec[lo:max(i - gap, lo + 1)], spec[min(i + gap, hi - 1):hi]]
    return float(spec[i] / np.median(flank)) if len(flank) and np.median(flank) > 0 else np.nan


def calibration_check(cube_path, window=400):
    """Sample three windows across the cube and judge the calibration from the spectral shape."""
    import rasterio
    wl = _wavelengths(cube_path); sc = _scale_factor(cube_path)
    with rasterio.open(cube_path) as S:
        cx, cy = S.width // 2, S.height // 2; parts = []
        for dx in (-S.width // 4, 0, S.width // 4):
            w = rasterio.windows.Window(max(cx + dx - window // 2, 0), max(cy - window // 2, 0), window, window)
            parts.append(S.read(window=w).reshape(S.count, -1))
        a = np.concatenate(parts, axis=1).astype(float); dtype = str(S.dtypes[0])
    valid = (a > 0).any(0); a = a[:, valid]
    spec = np.median(a, axis=1); b = lambda v: int(np.argmin(np.abs(wl - v)))
    r, n = a[b(670)], a[b(800)]; ok = (r > 0) & (n > 0); ndvi = (n[ok] - r[ok]) / (n[ok] + r[ok]) if ok.any() else np.array([np.nan])
    o2a, h2o = _feature(spec, wl, 762), _feature(spec, wl, 940, half=8, gap=14)
    nir_red = float(spec[b(800)] / spec[b(670)]) if spec[b(670)] > 0 else np.nan
    red_edge = float((spec[b(750)] - spec[b(700)]) / spec[b(700)]) if spec[b(700)] > 0 else np.nan
    looks = "reflectance"
    reasons = []
    if np.isfinite(o2a) and o2a < 0.85: reasons.append(f"oxygen-A absorption dip at 760 nm ({o2a:.2f} of the flanks)")
    if np.isfinite(h2o) and h2o < 0.80: reasons.append(f"water-vapour dip at 940 nm ({h2o:.2f})")
    if np.isfinite(nir_red) and nir_red < 1.12 and np.nanpercentile(ndvi, 95) < 0.15: reasons.append(f"NIR/red {nir_red:.2f} and NDVI 95th percentile {np.nanpercentile(ndvi, 95):.2f}: no vegetation or soil signature")
    if len(reasons) >= 2 or (reasons and "absorption" in reasons[0] and nir_red < 1.12): looks = "radiance"
    elif reasons: looks = "doubtful"
    if sc is None and "float" in dtype and looks == "reflectance" and np.nanmax(spec) > 2.0: looks = "doubtful"; reasons.append("float values above 2 without a reflectance scale factor")
    return dict(cube=cube_path, dtype=dtype, scale_factor=sc, o2a_ratio=o2a, h2o_ratio=h2o, nir_red=nir_red, red_edge_rise=red_edge,
                ndvi_p50=float(np.nanmedian(ndvi)), ndvi_p95=float(np.nanpercentile(ndvi, 95)), looks_like=looks, reasons=reasons,
                median_spectrum=spec.tolist(), wavelength=wl.tolist())


def _sample_at(cube_path, X, Y):
    """(bands x n) values of a cube at map coordinates X, Y (nearest pixel) and a mask of the points inside the raster."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(cube_path) as S:
        b = S.bounds; inside = (X >= b.left) & (X <= b.right) & (Y >= b.bottom) & (Y <= b.top)
        if not inside.any():
            return np.zeros((S.count, len(X))), inside
        w = from_bounds(max(X[inside].min() - 1, b.left), max(Y[inside].min() - 1, b.bottom), min(X[inside].max() + 1, b.right), min(Y[inside].max() + 1, b.top), S.transform).round_offsets().round_lengths()
        a = S.read(window=w); tr = S.window_transform(w)
        r, c = rasterio.transform.rowcol(tr, X, Y); r = np.clip(np.asarray(r), 0, a.shape[1] - 1); c = np.clip(np.asarray(c), 0, a.shape[2] - 1)
        return a[:, r, c].astype(float), inside


def gain_from_reference_cube(cube_path, ref_cube_path, grid=None, soil_ndvi_max=SOIL_NDVI_MAX, lattice_m=0.10, n_samples=80000, progress_cb=None):
    """Per-band gain = median reference reflectance over the bare soil / median value of this cube at the same places.
    grid: GeoDataFrame of the plots (its extent, plus 1 m, limits the sampling); None = overlap of the two cubes."""
    import rasterio
    wl = _wavelengths(cube_path); wr = _wavelengths(ref_cube_path)
    if len(wl) != len(wr) or np.max(np.abs(wl - wr)) > 1.0:
        raise RuntimeError("the two cubes do not share the same band list; the reference must come from the same sensor configuration")
    sc_ref = _scale_factor(ref_cube_path) or 10000.0
    with rasterio.open(cube_path) as S, rasterio.open(ref_cube_path) as R:
        b1, b2 = S.bounds, R.bounds
        minx, miny, maxx, maxy = max(b1.left, b2.left), max(b1.bottom, b2.bottom), min(b1.right, b2.right), min(b1.top, b2.top)
    if grid is not None and len(grid):
        gb = grid.total_bounds; minx, miny, maxx, maxy = max(minx, gb[0] - 1), max(miny, gb[1] - 1), min(maxx, gb[2] + 1), min(maxy, gb[3] + 1)
    if maxx - minx < 2 or maxy - miny < 2:
        raise RuntimeError("the cubes (and the grid) do not overlap")
    xs = np.arange(minx + 0.5, maxx - 0.5, lattice_m); ys = np.arange(miny + 0.5, maxy - 0.5, lattice_m); X, Y = np.meshgrid(xs, ys); X = X.ravel(); Y = Y.ravel()
    if len(X) > n_samples:
        k = np.random.default_rng(0).choice(len(X), n_samples, replace=False); X, Y = X[k], Y[k]
    if progress_cb: progress_cb(10, f"sampling {len(X)} positions in both cubes")
    A, oka = _sample_at(ref_cube_path, X, Y); B, okb = _sample_at(cube_path, X, Y); A = A / sc_ref
    ok = oka & okb & (A > 0).all(0) & (B > 0).all(0); A, B = A[:, ok], B[:, ok]
    if A.shape[1] < 200:
        raise RuntimeError(f"only {A.shape[1]} positions have valid values in every band of both cubes")
    bi = lambda v: int(np.argmin(np.abs(wl - v))); ndvi = (A[bi(800)] - A[bi(670)]) / (A[bi(800)] + A[bi(670)]); soil = ndvi < soil_ndvi_max
    if soil.sum() < 100:
        raise RuntimeError(f"only {int(soil.sum())} bare-soil positions (reference NDVI < {soil_ndvi_max}); raise the soil NDVI limit")
    if progress_cb: progress_cb(60, f"{int(soil.sum())} soil positions - computing the gain")
    med = lambda M: np.median(M, axis=1)
    gain = med(A[:, soil]) / med(B[:, soil]); idx = np.flatnonzero(soil)
    g1 = med(A[:, idx[::2]]) / med(B[:, idx[::2]]); g2 = med(A[:, idx[1::2]]) / med(B[:, idx[1::2]])
    gain_all = med(A) / med(B)
    # check on the vegetated part of the reference: the calibrated cube's crop spectrum
    crop = ndvi > max(0.5, soil_ndvi_max + 0.2)
    return dict(method="reference cube (bare-soil pseudo-invariant target)", reference=ref_cube_path, wavelength=wl.tolist(), gain=gain.tolist(), offset=[0.0] * len(wl),
                gain_half1=g1.tolist(), gain_half2=g2.tolist(), repeatability_pct=float(np.median(np.abs(g1 / g2 - 1)) * 100), n_positions=int(A.shape[1]), n_soil=int(soil.sum()),
                soil_vs_all_gain_ratio=float(np.median(gain / gain_all)), soil_ndvi_max=soil_ndvi_max,
                spectrum_soil_ref=med(A[:, soil]).tolist(), spectrum_soil_cal=(med(B[:, soil]) * gain).tolist(),
                spectrum_crop_ref=(med(A[:, crop]).tolist() if crop.sum() > 20 else None), spectrum_crop_cal=((med(B[:, crop]) * gain).tolist() if crop.sum() > 20 else None),
                n_crop=int(crop.sum()))


def gain_from_targets(cube_path, targets, spectra_csv=None, refl_col="refl", name_col="name", progress_cb=None):
    """targets: GeoDataFrame of reference polygons (panels / tarps) in this flight, with either a constant reflectance in
    column `refl_col` (0-1) or a name in `name_col` matching a column of spectra_csv (wavelength_nm + one column per target,
    reflectance 0-1). One target: gain only. Two or more targets of different brightness: gain and offset per band
    (least squares)."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds
    import pandas as pd
    wl = _wavelengths(cube_path); spec = pd.read_csv(spectra_csv) if spectra_csv else None
    L, Rr, names = [], [], []
    with rasterio.open(cube_path) as S:
        for _, t in targets.iterrows():
            g = t.geometry; minx, miny, maxx, maxy = g.bounds
            w = from_bounds(minx, miny, maxx, maxy, S.transform).round_offsets().round_lengths(); a = S.read(window=w).astype(float); tr = S.window_transform(w)
            m = ~geometry_mask([g], out_shape=a.shape[1:], transform=tr, invert=False) & (a > 0).all(0)
            if m.sum() < 10: continue
            L.append(np.median(a[:, m], axis=1))
            if spec is not None and name_col in t and str(t[name_col]) in spec.columns:
                Rr.append(np.interp(wl, spec["wavelength_nm"].to_numpy(float), spec[str(t[name_col])].to_numpy(float))); names.append(str(t[name_col]))
            elif refl_col in t and np.isfinite(float(t[refl_col])):
                Rr.append(np.full(len(wl), float(t[refl_col]))); names.append(f"{t.get(name_col, 'target')} ({float(t[refl_col]):.2f})")
            else:
                L.pop()
    if not L:
        raise RuntimeError("no usable reference target (needs pixels inside the cube and a reflectance spectrum or a constant reflectance)")
    L = np.array(L); Rr = np.array(Rr)
    if len(L) >= 2:
        gain = np.zeros(len(wl)); off = np.zeros(len(wl))
        for b in range(len(wl)):
            A = np.c_[L[:, b], np.ones(len(L))]; sol, *_ = np.linalg.lstsq(A, Rr[:, b], rcond=None); gain[b], off[b] = sol
        if (gain <= 0).any(): gain = np.median(Rr / L, axis=0); off = np.zeros(len(wl))
    else:
        gain = (Rr / L)[0]; off = np.zeros(len(wl))
    return dict(method=f"reference targets ({', '.join(names)})", reference=", ".join(names), wavelength=wl.tolist(), gain=gain.tolist(), offset=off.tolist(),
                gain_half1=gain.tolist(), gain_half2=gain.tolist(), repeatability_pct=0.0, n_positions=int(len(L)), n_soil=0, soil_vs_all_gain_ratio=1.0,
                spectrum_soil_ref=None, spectrum_soil_cal=None, spectrum_crop_ref=None, spectrum_crop_cal=None, n_crop=0, targets=names)


def apply_calibration(cube_path, gain, offset=None, out_path=None, scale=10000, progress_cb=None, note=""):
    """Write <cube>_reflectance_empirical.bin/.hdr: uint16 reflectance x scale on the same grid; 0 stays nodata."""
    import rasterio
    from .vnir import _find_header
    gain = np.asarray(gain, float); offset = np.zeros(len(gain)) if offset is None else np.asarray(offset, float)
    out = out_path or os.path.splitext(cube_path)[0] + "_reflectance_empirical.bin"
    with rasterio.open(cube_path) as S:
        prof = dict(driver="ENVI", width=S.width, height=S.height, count=S.count, dtype="uint16", crs=S.crs, transform=S.transform, interleave="band")
        with rasterio.open(out, "w", **prof) as D:
            for b in range(1, S.count + 1):
                a = S.read(b); v = np.where(a > 0, np.clip((a * gain[b - 1] + offset[b - 1]) * scale, 1, 1.5 * scale), 0).astype(np.uint16); D.write(v, b)
                if progress_cb and (b % 10 == 0 or b == S.count): progress_cb(int(100 * b / S.count), f"writing calibrated band {b}/{S.count}")
            D.update_tags(reflectance_scale_factor=str(scale), calibration="empirical line - PhenoFuse " + note)
    h = open(_find_header(cube_path), "r", errors="ignore").read(); hdr = os.path.splitext(out)[0] + ".hdr"; h2 = open(hdr, "r", errors="ignore").read()
    wlblock = re.search(r"wavelength\s*=\s*\{[^}]*\}", h, re.S | re.I); units = re.search(r"wavelength units\s*=\s*\w+", h, re.I)
    extra = f"\nreflectance scale factor = {scale}\n" + (wlblock.group(0) if wlblock else "") + "\n" + (units.group(0) if units else "wavelength units = nm") + "\n"
    if "wavelength" not in h2.lower():
        with open(hdr, "a") as f: f.write(extra)
    return out


def save_calibration(stem, result, check_before=None, check_after=None):
    """<stem>_gain.csv, <stem>_README.txt, <stem>.json"""
    import pandas as pd
    pd.DataFrame(dict(wavelength_nm=result["wavelength"], gain=result["gain"], offset=result["offset"], gain_half1=result["gain_half1"], gain_half2=result["gain_half2"])).to_csv(stem + "_gain.csv", index=False)
    d = {k: v for k, v in result.items() if not k.startswith("spectrum_")}
    d["check_before"] = {k: v for k, v in (check_before or {}).items() if k not in ("median_spectrum", "wavelength")}
    d["check_after"] = {k: v for k, v in (check_after or {}).items() if k not in ("median_spectrum", "wavelength")}
    json.dump(d, open(stem + ".json", "w"), indent=1, default=float)
    cross_date = "reference cube" in result["method"]
    txt = (f"{os.path.basename(stem)}.bin = the cube converted to reflectance by an empirical line, band by band: R = gain * value + offset.\n"
           f"Method: {result['method']}; reference: {result['reference']}\n"
           f"Samples: {result['n_positions']} positions, {result['n_soil']} bare-soil positions; gain repeatability between two halves of the samples: {result['repeatability_pct']:.2f} %\n")
    if check_before: txt += f"Before: looks like {check_before['looks_like']} (O2-A dip {check_before['o2a_ratio']:.2f}, H2O dip {check_before['h2o_ratio']:.2f}, NIR/red {check_before['nir_red']:.2f}, NDVI p50 {check_before['ndvi_p50']:.2f})\n"
    if check_after: txt += f"After : looks like {check_after['looks_like']} (O2-A {check_after['o2a_ratio']:.2f}, H2O {check_after['h2o_ratio']:.2f}, NIR/red {check_after['nir_red']:.2f}, NDVI p50 {check_after['ndvi_p50']:.2f})\n"
    txt += ("Caveats: " + ("the bare soil is assumed to have the same reflectance in both cubes. Across dates soil moisture, roughness and residue change it, mostly as a "
                          "wavelength-flat factor: ratio indices (NDVI, red-edge, CIRE...) are reliable, absolute reflectance is uncertain by roughly 10-20 %. Across flights of the "
                          "same day the gain is nearly exact. " if cross_date else "the targets' reflectance spectra are taken as given; a single target gives gain only. ")
            + "No atmospheric path correction beyond what the reference carries. A reprocessing to reflectance by the sensor software supersedes this file.\n")
    open(stem + "_README.txt", "w", encoding="utf-8").write(txt)
    return stem + "_README.txt"


def render_calibration_qa(result, check_before, check_after, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    wl = np.array(result["wavelength"]); fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(wl, check_before["median_spectrum"], "k-"); ax[0].set_title(f"before: median spectrum (looks like {check_before['looks_like']})", fontsize=9)
    for v, lab in ((762, "O2-A"), (940, "H2O")): ax[0].axvline(v, color="grey", ls=":", lw=0.8); ax[0].text(v, ax[0].get_ylim()[1] * 0.95, lab, fontsize=7, ha="center")
    ax[1].plot(wl, result["gain"], label="gain"); ax[1].plot(wl, result["gain_half1"], lw=0.6, alpha=0.6, label="half 1"); ax[1].plot(wl, result["gain_half2"], lw=0.6, alpha=0.6, label="half 2"); ax[1].set_title(f"gain per band - {result['method'].split(' (')[0]}; repeatability {result['repeatability_pct']:.2f} %", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].plot(wl, np.asarray(check_after["median_spectrum"], float) / (check_after.get("scale_factor") or 1.0), "k-", label="after: median spectrum (reflectance)")
    if result.get("spectrum_soil_ref"): ax[2].plot(wl, result["spectrum_soil_ref"], label="soil, reference"); ax[2].plot(wl, result["spectrum_soil_cal"], "--", label="soil, calibrated")
    if result.get("spectrum_crop_ref"): ax[2].plot(wl, result["spectrum_crop_ref"], label="crop pixels, reference"); ax[2].plot(wl, result["spectrum_crop_cal"], "--", label="same pixels, calibrated")
    ax[2].set_title(f"after: looks like {check_after['looks_like']} (NIR/red {check_after['nir_red']:.2f}, NDVI p50 {check_after['ndvi_p50']:.2f})", fontsize=9); ax[2].legend(fontsize=7)
    for a_ in ax: a_.set_xlabel("wavelength (nm)", fontsize=8); a_.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig); return out_png


def calibrate_cube(cube_path, mode, reference=None, grid=None, targets=None, spectra_csv=None, soil_ndvi_max=SOIL_NDVI_MAX, progress_cb=None):
    """One call: check, derive the gain (mode 'cube' or 'targets'), write the calibrated cube, QA figure, gain table, README.
    Returns dict(out_cube, check_before, check_after, result, qa_png, readme)."""
    import geopandas as gpd
    if progress_cb: progress_cb(2, "checking the cube")
    before = calibration_check(cube_path)
    if mode == "cube":
        result = gain_from_reference_cube(cube_path, reference, grid=grid, soil_ndvi_max=soil_ndvi_max, progress_cb=progress_cb)
    else:
        t = targets if not isinstance(targets, str) else gpd.read_file(targets)
        import rasterio
        with rasterio.open(cube_path) as S:
            if S.crs is not None and t.crs is not None and t.crs != S.crs: t = t.to_crs(S.crs)
        result = gain_from_targets(cube_path, t, spectra_csv=spectra_csv, progress_cb=progress_cb)
    out = apply_calibration(cube_path, result["gain"], result["offset"], progress_cb=lambda p, m: progress_cb(60 + int(p * 0.38), m) if progress_cb else None, note=result["method"])
    after = calibration_check(out); stem = os.path.splitext(out)[0]
    readme = save_calibration(stem, result, before, after); qa = render_calibration_qa(result, before, after, stem + "_qa.png")
    if progress_cb: progress_cb(100, "done")
    return dict(out_cube=out, check_before=before, check_after=after, result=result, qa_png=qa, readme=readme)
