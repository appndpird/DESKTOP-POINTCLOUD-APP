"""
RGB modality (PhenoFuse 4.0): per-plot crops of the RGB orthomosaic and per-plot RGB features.

Per-plot crops
  One GeoTIFF per plot, cut from the orthomosaic on ITS OWN pixel grid: the read window is the integer pixel
  window of the plot polygon (no resampling, no reprojection), the dtype, CRS and transform are the source's,
  the compression is lossless DEFLATE. Pixel values are therefore exact copies of the orthomosaic as any
  reader decodes it (the source itself is usually JPEG-in-TIFF, so "exact" means identical to what QGIS or
  rasterio read from the big file). Pixels outside the polygon are written as 0 and flagged by an internal
  GDAL mask (transparent in QGIS, no nodata tag). plots_rgb_index.csv lists every file.

Per-plot features (rgb_features.csv; every column starts with rgb_)
  over the sampling region of the LiDAR / VNIR tabs (whole plot inset or central band):
  mean R, G, B (0-255), chromatic coordinates r, g, b, and the vegetation indices ExG, ExR, ExGR, GLI, VARI,
  NGRDI, MGRVI, RGBVI, CIVE, TGI, VEG, COM, GCC - each as the mean over every valid pixel, the mean over
  VEGETATION pixels (_veg) and the sd (_sd); vegetation cover (rgb_cover: share of pixels above an ExG threshold
  found by Otsu on the whole flight, bounded 0.02-0.25), HSV means, grey-level sd and Laplacian variance
  (texture / sharpness), pixel counts and QC flags (rgb_qc_ok: >= 90 % of the region inside the orthomosaic
  and not nodata).

Public API
----------
extract_rgb_plots(ortho_path, plots, out_dir, write_tifs=True, region_mode="whole", band_width=0.5,
                  progress_cb=None) -> (features DataFrame, index DataFrame)
RGB_CORE                                  feature names used by the modelling tabs
"""
from __future__ import annotations
import os
import re
import numpy as np

RGB_INDEX_NAMES = ["ExG", "ExR", "ExGR", "GLI", "VARI", "NGRDI", "MGRVI", "RGBVI", "CIVE", "TGI", "VEG", "COM", "GCC"]
RGB_CORE = ["rgb_ExG_veg", "rgb_ExGR_veg", "rgb_GLI_veg", "rgb_VARI_veg", "rgb_NGRDI_veg", "rgb_MGRVI_veg", "rgb_RGBVI_veg",
            "rgb_TGI_veg", "rgb_GCC_veg", "rgb_cover", "rgb_ExG", "rgb_NGRDI", "rgb_g", "rgb_H_mean", "rgb_S_mean", "rgb_V_mean",
            "rgb_gray_sd", "rgb_lap_var"]


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))


def _otsu(v, lo=0.02, hi=0.25, default=0.10):
    v = v[np.isfinite(v)]
    if len(v) < 500:
        return default
    hist, edges = np.histogram(v, bins=256, range=(-0.3, 0.6)); c = edges[:-1] + np.diff(edges) / 2
    w1 = np.cumsum(hist); w2 = w1[-1] - w1
    m1 = np.cumsum(hist * c) / np.maximum(w1, 1); m2 = np.cumsum((hist * c)[::-1])[::-1] / np.maximum(w2, 1)
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[1:]) ** 2
    t = float(c[np.argmax(var)])
    return float(np.clip(t, lo, hi))


def rgb_indices(R, G, B):
    """Dict of per-pixel index arrays from float R, G, B in 0-1."""
    s = R + G + B
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(s > 0, R / s, np.nan); g = np.where(s > 0, G / s, np.nan); b = np.where(s > 0, B / s, np.nan)
        d = {"r": r, "g": g, "b": b,
             "ExG": 2 * g - r - b, "ExR": 1.4 * r - g, "GLI": (2 * G - R - B) / (2 * G + R + B),
             "VARI": (G - R) / (G + R - B), "NGRDI": (G - R) / (G + R), "MGRVI": (G ** 2 - R ** 2) / (G ** 2 + R ** 2),
             "RGBVI": (G ** 2 - R * B) / (G ** 2 + R * B), "CIVE": 0.441 * r - 0.811 * g + 0.385 * b + 18.78745,
             "TGI": -0.5 * (190 * (R - G) - 120 * (R - B)), "GCC": g}
        d["ExGR"] = d["ExG"] - d["ExR"]
        d["VEG"] = np.where((r > 0) & (b > 0), g / (np.power(r, 0.667) * np.power(b, 0.333) + 1e-9), np.nan)
        d["COM"] = 0.25 * d["ExG"] + 0.30 * d["ExGR"] + 0.33 * d["CIVE"] + 0.12 * d["VEG"]
    for k in ("VARI", "GLI", "NGRDI", "MGRVI", "RGBVI"):
        d[k] = np.where(np.isfinite(d[k]), np.clip(d[k], -1.5, 1.5), np.nan)
    return d


def _hsv(R, G, B):
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B); df = mx - mn
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(df == 0, 0.0, np.where(mx == R, (60 * ((G - B) / df) + 360) % 360,
                                             np.where(mx == G, 60 * ((B - R) / df) + 120, 60 * ((R - G) / df) + 240)))
        s = np.where(mx == 0, 0.0, df / mx)
    return h, s, mx


def extract_rgb_plots(ortho_path, plots, out_dir, write_tifs=True, region_mode="whole", band_width=0.5,
                      progress_cb=None, name_col="B/R", region_inset=0.10):
    """plots: GeoDataFrame in the working CRS (reprojected to the orthomosaic CRS when they differ)."""
    import pandas as pd
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.features import geometry_mask
    from .regions import plot_region
    os.makedirs(out_dir, exist_ok=True)
    src = rasterio.open(ortho_path)
    try:
        if src.crs is not None and plots.crs is not None and src.crs != plots.crs:
            plots = plots.to_crs(src.crs)
        tr = src.transform; nb = min(src.count, 3)
        if src.count < 3:
            raise RuntimeError(f"the orthomosaic has {src.count} band(s); an RGB orthomosaic with 3 (or 4) bands is needed")
        alpha = 4 if src.count >= 4 else None
        nodata = src.nodata
        geoms = list(plots.geometry); pids = list(plots["Plot_ID"]) if "Plot_ID" in plots else list(range(1, len(geoms) + 1))
        names = list(plots[name_col]) if name_col in plots else [f"P{p}" for p in pids]
        regions = [plot_region(g, mode=region_mode, band_width=band_width, inset=region_inset) for g in geoms]
        n = len(geoms); rows = []; index = []; exg_all = []; per_plot = []
        for i, (pid, nm, g, reg) in enumerate(zip(pids, names, geoms, regions)):
            w = from_bounds(*g.bounds, tr).round_offsets().round_lengths()
            try:
                w = w.intersection(Window(0, 0, src.width, src.height))
            except Exception:
                w = None
            if w is None or w.width <= 0 or w.height <= 0:
                rows.append(dict(Plot_ID=pid, rgb_n_px=0, rgb_frac_in_ortho=0.0, rgb_qc_ok=0)); per_plot.append(None); continue
            wt = rasterio.windows.transform(w, tr); shape = (int(w.height), int(w.width))
            arr = src.read(list(range(1, nb + 1)), window=w)                      # (3, h, w) exact pixel copy
            inside = ~geometry_mask([g], out_shape=shape, transform=wt, invert=False)
            in_region = ~geometry_mask([reg], out_shape=shape, transform=wt, invert=False)
            valid = inside.copy()
            if alpha:
                valid &= src.read(alpha, window=w) > 0
            if nodata is not None:
                valid &= ~(arr == nodata).all(0)
            valid &= ~(arr == 0).all(0)                                            # black fill outside the mosaic
            if write_tifs:
                prof = dict(driver="GTiff", width=shape[1], height=shape[0], count=nb, dtype=src.dtypes[0], crs=src.crs, transform=wt,
                            compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256, photometric="RGB", BIGTIFF="IF_SAFER")
                path = os.path.join(out_dir, f"plot_{pid}_{_safe(nm)}.tif")
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                    with rasterio.open(path, "w", **prof) as dst:
                        dst.write(np.where(inside[None], arr, 0).astype(arr.dtype))
                        dst.write_mask((inside * 255).astype("uint8"))
                        dst.update_tags(Plot_ID=str(pid), plot_name=str(nm), source=os.path.basename(ortho_path),
                                        pixel_values="exact copy of the orthomosaic pixels (same grid, dtype, CRS; lossless DEFLATE)",
                                        clip="plot polygon via internal GDAL mask; outside = 0 + masked")
                        for b_, ci in zip(range(1, nb + 1), ("red", "green", "blue")):
                            dst.set_band_description(b_, ci)
                index.append(dict(Plot_ID=pid, name=nm, path=path, width=shape[1], height=shape[0], px_inside=int(inside.sum())))
            sel = valid & in_region
            scale = 255.0 if np.issubdtype(arr.dtype, np.integer) and arr.max() > 1 else (65535.0 if arr.dtype == np.uint16 else 1.0)
            if arr.dtype == np.uint16:
                scale = 65535.0
            R = arr[0][sel].astype(np.float32) / scale; G = arr[1][sel].astype(np.float32) / scale; B = arr[2][sel].astype(np.float32) / scale
            d = rgb_indices(R, G, B) if sel.any() else None
            if d is not None:
                exg_all.append(d["ExG"][np.isfinite(d["ExG"])][::max(1, int(sel.sum() // 20000))])
            per_plot.append(dict(pid=pid, R=R, G=G, B=B, d=d, n_region=int(in_region.sum()), n_sel=int(sel.sum()),
                                 gray=(0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]).astype(np.float32) / scale, sel=sel, in_region=in_region))
            if progress_cb and (i % 4 == 0 or i == n - 1):
                progress_cb(int(80 * (i + 1) / n), f"RGB plots: {i + 1}/{n}")
        thr = _otsu(np.concatenate(exg_all)) if exg_all else 0.10
        for i, pp in enumerate(per_plot):
            if pp is None:
                continue
            row = dict(Plot_ID=pp["pid"], rgb_n_px=pp["n_sel"], rgb_frac_in_ortho=pp["n_sel"] / max(pp["n_region"], 1), rgb_veg_threshold=thr)
            d = pp["d"]
            if d is None or pp["n_sel"] < 50:
                row["rgb_qc_ok"] = 0; rows.append(row); continue
            R, G, B = pp["R"], pp["G"], pp["B"]
            veg = d["ExG"] > thr
            row.update(rgb_R_mean=float(R.mean() * 255), rgb_G_mean=float(G.mean() * 255), rgb_B_mean=float(B.mean() * 255),
                       rgb_r=float(np.nanmean(d["r"])), rgb_g=float(np.nanmean(d["g"])), rgb_b=float(np.nanmean(d["b"])),
                       rgb_cover=float(veg.mean()))
            for k in RGB_INDEX_NAMES:
                v = d[k]; ok = np.isfinite(v)
                row[f"rgb_{k}"] = float(v[ok].mean()) if ok.any() else np.nan
                row[f"rgb_{k}_sd"] = float(v[ok].std()) if ok.any() else np.nan
                vv = v[ok & veg]
                row[f"rgb_{k}_veg"] = float(vv.mean()) if len(vv) > 10 else np.nan
            h, s, v_ = _hsv(R, G, B)
            row.update(rgb_H_mean=float(np.nanmean(h)), rgb_S_mean=float(np.nanmean(s)), rgb_V_mean=float(np.nanmean(v_)),
                       rgb_H_veg=float(np.nanmean(h[veg])) if veg.sum() > 10 else np.nan, rgb_S_veg=float(np.nanmean(s[veg])) if veg.sum() > 10 else np.nan)
            gray = pp["gray"]; m = pp["sel"]
            row["rgb_gray_sd"] = float(gray[m].std())
            lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
            mm = m[1:-1, 1:-1]
            row["rgb_lap_var"] = float(lap[mm].var()) if mm.any() else np.nan
            row["rgb_qc_ok"] = int(row["rgb_frac_in_ortho"] >= 0.9 and pp["n_sel"] >= 200)
            rows.append(row)
            if progress_cb and (i % 8 == 0 or i == n - 1):
                progress_cb(80 + int(20 * (i + 1) / n), f"RGB features: {i + 1}/{n}")
    finally:
        src.close()
    df = pd.DataFrame(rows)
    for c in ("Plot", "B/R", "Bank", "Row", "Range"):
        if c in plots.columns and c not in df.columns:
            df.insert(1, c, list(plots[c]))
    idx = pd.DataFrame(index)
    if len(idx):
        idx.to_csv(os.path.join(out_dir, "plots_rgb_index.csv"), index=False)
    return df, idx
