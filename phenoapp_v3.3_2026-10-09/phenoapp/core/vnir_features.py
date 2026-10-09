"""
Cleaned VNIR per-plot features (PhenoApp v3).

Cleaning: a band value of 0 is a clipped (negative) reflectance and is treated as missing; bands in the
excluded ranges (sensor edge 0-415 nm, O2-A 755-770 nm, water vapour 928-962 nm) are never used.
Soil mask: an adaptive NDVI(800/670) threshold per flight (Otsu split, bounded 0.15-0.50); when the
histogram has no soil/crop contrast (senesced canopy) the mask is disabled and every valid pixel counts.
Features per plot (sampling region): per-pixel indices (recommended set) averaged over all valid pixels
and over vegetation pixels (mean, sd), vegetation fraction, valid-pixel QC, broad-band means of the
vegetation mean spectrum, red-edge derivative features, Guyot red-edge position, PCA scores (5) of the
log/SNV vegetation spectra, and the vegetation mean spectrum itself.

Public API
----------
vnir_features_all(cube, plots, regions, indices=None, progress_cb=None)
    -> (features DataFrame, vegetation mean spectra DataFrame, per-band valid-fraction DataFrame)
Every per-plot band value and index is NaN where it cannot be trusted: band valid on fewer than MIN_VALID_FRAC
(0.5) of the region pixels, band inside an excluded wavelength range, or a cube that fails the reflectance check.
DEFAULT_V3_INDICES
"""

from __future__ import annotations
import numpy as np

DEFAULT_V3_INDICES = ["NDVI_nb", "NDRE740", "NDRE720", "NDREI", "REP", "S2REP", "LCI", "MTCI", "CIRE", "CIre705", "VOG1", "OSAVI", "SAVI", "GNDVI",
                      "kNDVI", "NIRv", "WDRVI", "MCARI", "TCARI", "TCARIOSAVI", "mND705", "PRI", "WBI", "NDWI970", "SR705", "ND705", "GCC", "CIG",
                      "RENDVI", "PSRI"]
VNIR_V3_CORE = ["NDRE740_veg", "NDRE720_veg", "REP_veg", "LCI_veg", "MTCI_veg", "CIRE_veg", "NDVI_nb_veg", "OSAVI_veg", "GNDVI_veg", "kNDVI_veg",
                "NIRv_veg", "PRI_veg", "WBI_veg", "fcover_vnir", "refl_N", "refl_RE1", "refl_R", "red_edge_slope_max", "red_edge_pos_deriv_nm",
                "spec_pc1", "spec_pc2", "spec_pc3"]


def _otsu(v):
    v = v[np.isfinite(v)]
    if len(v) < 100:
        return 0.3
    hist, edges = np.histogram(v, bins=200, range=(-0.2, 1.0)); c = edges[:-1] + np.diff(edges) / 2
    w1 = np.cumsum(hist); w2 = w1[-1] - w1; m1 = np.cumsum(hist * c) / np.maximum(w1, 1); m2 = (np.cumsum((hist * c)[::-1])[::-1] / np.maximum(w2, 1))
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[1:]) ** 2
    return float(c[np.argmax(var)])


def vnir_features_all(cube, plots, regions, indices=None, progress_cb=None, excluded=None):
    import pandas as pd
    from .spectral_indices import IndexCatalogue, DEFAULT_EXCLUDED_NM, MIN_VALID_FRAC, _needed_bands, evaluate_formula, constants_env, guess_scale
    excluded = DEFAULT_EXCLUDED_NM if excluded is None else excluded
    src = cube._src; wl = np.asarray(cube.wavelengths, float); scale = guess_scale(cube)
    cat = IndexCatalogue(); acr = [a for a in (indices or DEFAULT_V3_INDICES) if cat.computable(a, wl, excluded)]
    need = _needed_bands(cat, acr, wl, excluded)
    lab = cube._labels(regions).ravel(); sel = np.flatnonzero(lab); lab_sel = lab[sel]; n_plots = len(regions)
    region_px = np.bincount(lab, minlength=n_plots + 1)[1:]
    # keep the cube's own dtype: a float32 cube (e.g. GRYFN output without the uint16 x10000 scaling) must not be
    # truncated to integers
    S = np.zeros((len(sel), src.count), np.dtype(src.dtypes[0]))
    for b in range(1, src.count + 1):
        S[:, b - 1] = src.read(b).ravel()[sel]
        if progress_cb and b % 20 == 0:
            progress_cb(int(50 * b / src.count), f"VNIR v3: reading band {b}/{src.count}")
    excl = np.zeros(len(wl), bool)
    for lo, hi in excluded:
        excl |= (wl >= lo) & (wl <= hi)
    def band(nm): return int(np.argmin(np.abs(wl - nm)))
    r670 = S[:, band(670)].astype(np.float32); r800 = S[:, band(800)].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where((r670 > 0) & (r800 > 0), (r800 - r670) / (r800 + r670), np.nan)
    t_raw = _otsu(ndvi); veg_t = float(np.clip(t_raw, 0.15, 0.50))
    if t_raw < 0.12 or float(np.nanmean(ndvi >= veg_t)) < 0.10:
        veg_t = float("nan"); veg = np.isfinite(ndvi)
    else:
        veg = ndvi >= veg_t
    vals = {}
    for bname, idx in need.items():
        sub = S[:, idx].astype(np.float32); ok = (sub > 0).all(1); vals[bname] = np.where(ok, sub.mean(1) / scale, np.nan)
    def agg(arr, mask):
        ok = np.isfinite(arr) & mask; gl = np.where(ok, lab_sel, 0)
        n = np.bincount(gl, minlength=n_plots + 1)[1:]; s = np.bincount(gl, weights=np.where(ok, arr, 0), minlength=n_plots + 1)[1:]
        s2 = np.bincount(gl, weights=np.where(ok, arr * arr, 0), minlength=n_plots + 1)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / np.maximum(n, 1), np.nan); sd = np.sqrt(np.maximum(np.where(n > 1, s2 / np.maximum(n, 1) - mean ** 2, np.nan), 0))
        return mean, sd, n
    n_all = np.bincount(lab_sel, minlength=n_plots + 1)[1:]
    feats = {"Plot_ID": list(plots["Plot_ID"]) if "Plot_ID" in plots else list(range(1, n_plots + 1)), "region_px": region_px, "frac_in_cube": n_all / np.maximum(region_px, 1)}
    feats["fcover_vnir"] = np.bincount(np.where(veg, lab_sel, 0), minlength=n_plots + 1)[1:] / np.maximum(np.bincount(np.where(np.isfinite(ndvi), lab_sel, 0), minlength=n_plots + 1)[1:], 1)
    for nm in (470, 550, 670):
        feats[f"valid_{nm}_frac"] = np.bincount(np.where(S[:, band(nm)] > 0, lab_sel, 0), minlength=n_plots + 1)[1:] / np.maximum(n_all, 1)
    allm = np.ones(len(sel), bool)
    for j, a in enumerate(acr):
        e = cat.entries[a]; env = {b_: vals[b_] for b_ in e.bands}; env.update(constants_env(e))
        r = evaluate_formula(e.formula, env).astype(np.float32)
        m_all, sd_all, n1 = agg(r, allm); m_veg, sd_veg, _ = agg(r, veg)
        vfrac = n1 / np.maximum(n_all, 1); bad = vfrac < MIN_VALID_FRAC        # index valid on too few pixels (clipped band) -> NaN
        feats[a] = np.where(bad, np.nan, m_all); feats[f"{a}_sd"] = np.where(bad, np.nan, sd_all)
        feats[f"{a}_veg"] = np.where(bad, np.nan, m_veg); feats[f"{a}_veg_sd"] = np.where(bad, np.nan, sd_veg); feats[f"{a}_valid"] = vfrac
        if progress_cb:
            progress_cb(50 + int(35 * (j + 1) / len(acr)), f"VNIR v3: index {a}")
    spec = np.full((n_plots, len(wl)), np.nan)
    band_vfrac = np.zeros((n_plots, len(wl)))        # per plot and band: fraction of region pixels with a valid (non-zero) value
    for b in range(len(wl)):
        col = S[:, b].astype(np.float32); ok = (col > 0) & veg; gl = np.where(ok, lab_sel, 0)
        n = np.bincount(gl, minlength=n_plots + 1)[1:]; s = np.bincount(gl, weights=np.where(ok, col, 0), minlength=n_plots + 1)[1:]
        band_vfrac[:, b] = np.bincount(np.where(col > 0, lab_sel, 0), minlength=n_plots + 1)[1:] / np.maximum(n_all, 1)
        # NaN when the band is clipped on too many pixels or lies in an excluded range: the spectra table then shows
        # the gap instead of a mean of the surviving brighter pixels
        usable = (n > 0) & (band_vfrac[:, b] >= MIN_VALID_FRAC) & (not excl[b])
        spec[:, b] = np.where(usable, s / np.maximum(n, 1) / scale, np.nan)
    specc = spec.copy()
    for k_, (lo, hi) in {"G": (510, 600), "R": (620, 690), "RE1": (695, 715), "RE2": (730, 750), "RE3": (765, 795), "N": (780, 900), "N2": (850, 880)}.items():
        feats[f"refl_{k_}"] = np.nanmean(specc[:, (wl >= lo) & (wl <= hi)], 1)
    seg = (wl >= 680) & (wl <= 760) & ~excl; w_seg = wl[seg]; d = np.gradient(spec[:, seg], w_seg, axis=1)   # excluded (O2-A) bands never enter the derivative
    feats["red_edge_slope_max"] = np.nanmax(d, 1); feats["red_edge_pos_deriv_nm"] = w_seg[np.nanargmax(np.nan_to_num(d, nan=-1e9), 1)]
    for nm in (705, 725, 740):
        feats[f"deriv_{nm}"] = d[:, np.argmin(np.abs(w_seg - nm))]
    def R(nm): return spec[:, band(nm)]
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["REP_guyot"] = 700 + 40 * ((R(670) + R(780)) / 2 - R(700)) / (R(740) - R(700))
        feats["red_depth"] = 1 - R(670) / ((R(550) + R(780)) / 2); feats["nir_plateau_slope"] = (R(900) - R(800)) / 100
    df = pd.DataFrame(feats)
    Xs = specc[:, ~excl]; good_b = np.isfinite(Xs).mean(0) >= 0.95; Xs = Xs[:, good_b]; okr = np.isfinite(Xs).all(1) if Xs.shape[1] else np.zeros(n_plots, bool)   # PCA on bands usable in >= 95 % of plots
    if okr.sum() > 10:
        from sklearn.decomposition import PCA
        Xl = np.log10(np.clip(Xs[okr], 1e-4, None)); Xl = (Xl - Xl.mean(1, keepdims=True)) / (Xl.std(1, keepdims=True) + 1e-9)
        sc = np.full((n_plots, 5), np.nan); sc[okr] = PCA(5).fit_transform(Xl)
        for i in range(5):
            df[f"spec_pc{i+1}"] = sc[:, i]
    df["veg_ndvi_threshold"] = veg_t
    df["red_ok"] = (df.valid_670_frac > 0.5).astype(int)
    # red-clipped plots (670 nm zero on half the pixels or more): every index that uses a band in the red absorption
    # well (620-690 nm) is computed from the bright half of the pixels only and biased high -> censor it to NaN so the
    # model code drops the feature (>5 % missing) or median-fills it, instead of training on it silently
    red_dep = [a for a in acr if any(((wl[need[b_]] >= 620) & (wl[need[b_]] <= 690)).any() for b_ in cat.entries[a].bands)]
    red_bad = df.red_ok == 0
    for a in red_dep:
        for suf in ("", "_sd", "_veg", "_veg_sd"):
            df.loc[red_bad, f"{a}{suf}"] = np.nan
    df.loc[red_bad, ["refl_R", "red_depth", "REP_guyot"]] = np.nan
    # reflectance sanity check for the whole flight: a cube that is still radiance / DN (float, no reflectance scale
    # factor, values far above 1, flat spectrum) gives NDVI ~ 0 everywhere; its spectral features are meaningless
    from .spectral_indices import parse_scale
    hdr_scale = parse_scale(getattr(cube, "hdr", "") or "", default=0.0)
    is_int = np.issubdtype(np.dtype(src.dtypes[0]), np.integer)
    med_ndvi = float(np.nanmedian(df["NDVI_nb"])) if "NDVI_nb" in df.columns and np.isfinite(df["NDVI_nb"]).any() else np.nan
    med_nir = float(np.nanmedian(df["refl_N"])) if np.isfinite(df["refl_N"]).any() else np.nan
    refl_ok = True
    if np.isfinite(med_ndvi) and med_ndvi < 0.05:
        refl_ok = False                                   # no vegetation signal anywhere in the flight
    if (not is_int) and hdr_scale <= 0 and np.isfinite(med_nir) and med_nir > 2.0:
        refl_ok = False                                   # float cube with values far above reflectance range
    df["vnir_reflectance_ok"] = int(refl_ok)
    df["vnir_qc_ok"] = ((df.frac_in_cube > 0.9) & (df.fcover_vnir > 0.2) & refl_ok).astype(int)
    if not refl_ok:
        # radiance / DN cube: every spectral value is wrong, so every feature except the bookkeeping and QC columns is NaN
        keep = {"Plot_ID", "region_px", "frac_in_cube", "valid_470_frac", "valid_550_frac", "valid_670_frac", "veg_ndvi_threshold",
                "red_ok", "vnir_reflectance_ok", "vnir_qc_ok"}
        for c in df.columns:
            if c not in keep:
                df[c] = np.nan
        spec[:] = np.nan
    wl_cols = [f"{w:.1f}" for w in wl]
    sp = pd.DataFrame(spec, columns=wl_cols); sp.insert(0, "Plot_ID", df["Plot_ID"].values)
    vf = pd.DataFrame(band_vfrac, columns=wl_cols); vf.insert(0, "Plot_ID", df["Plot_ID"].values)
    if progress_cb:
        progress_cb(100, "VNIR v3 features done")
    return df, sp, vf
