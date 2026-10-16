"""
Per-plot LiDAR extraction (PhenoFuse 4.0).

For every plot polygon:
  1. Choose the sampling region (whole plot or central band - see regions.py)
  2. Find points inside the region (bbox prefilter + vectorized contains)
  3. Optionally write a per-plot LAS file (always the FULL plot polygon). With LiDAR v3 on, the per-plot file IS the
     classified one (2 ground / 3 low veg / 5 canopy / 7 noise, HeightAboveGround) - one file per plot, not two.
  4. Compute the selected traits over the region; with LiDAR v3, the canopy-only v3 features as well.
  5. Write ONE table, lidar_features.csv, into the per-plot folder: legacy traits + canopy-top heights + v3 features
     (a v3 column whose name clashes with a legacy trait gets the _v3 suffix). No separate _lidar_v3.csv any more.

Height reference options (ground_mode):
  "hag"      : use the LAS manager's HAG (SMRF cache) or plot-local fallback
  "exterior" : fit an exterior ground surface from points outside the plots (recommended for dense pasture)

Public API
----------
extract_all_plots(las_manager, plots_gdf, out_dir, out_csv, selected_traits, write_las=True, ...) -> pd.DataFrame
"""

from __future__ import annotations
import os
import re
import json
import numpy as np
import pandas as pd
import laspy
from shapely.vectorized import contains as sv_contains

from .traits import compute_plot_traits
from .regions import plot_region


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))


def extract_all_plots(
    mgr,                # LASManager (already loaded)
    plots,              # GeoDataFrame in working CRS
    out_dir: str,
    out_csv: str,
    selected_traits: list[str],
    write_las: bool = True,
    height_cut: float = 0.15,
    voxel: float = 0.05,
    biomass_k: float = 1.0,
    ground_mode: str = "hag",
    region_mode: str = "whole",
    band_width: float = 0.5,
    region_inset: float = 0.10,
    progress_cb=None,
    cancel_flag=None,
    cth_ground: str = "local",
    targets_csv: str | None = None,
    lidar_v3: bool = False,
    classified_dir: str | None = None,
    noise_rule: str = "gap",
    noise_nsig: float = 3.0,
    drop_noise: bool = False,
    canopy_h: float | None = None,
    ground_h: float | None = None,
) -> pd.DataFrame:
    if write_las or classified_dir:
        os.makedirs(out_dir, exist_ok=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # with v3 classification on, the classified per-plot LAS replaces the plain copy (same folder, one file per plot)
    write_plain = write_las and not (lidar_v3 and classified_dir)

    x = mgr.x; y = mgr.y; z = mgr.z
    hag = mgr.hag
    ground_diag = None
    gm = None

    if ground_mode == "exterior":
        from .ground_model import fit_exterior_ground
        if progress_cb:
            progress_cb(1, "Fitting exterior ground surface...")
        gm = fit_exterior_ground(x, y, z, plots)
        hag = gm.hag(x, y, z)
        ground_diag = gm.diagnostics
        if progress_cb:
            progress_cb(3, (f"Ground surface: {len(gm.terms)} terms, "
                            f"rms {gm.diagnostics['rms_resid_m']*100:.1f} cm "
                            f"over {gm.diagnostics['n_ground_cells']} cells"))

    n_plots = len(plots)
    rows = []

    for idx, plot in plots.reset_index(drop=True).iterrows():
        if cancel_flag and cancel_flag():
            break

        pid = plot.get("Plot_ID", idx + 1)
        br  = _safe_name(plot.get("B/R", f"P{pid}"))
        poly = plot.geometry
        region = plot_region(poly, mode=region_mode,
                             band_width=band_width, inset=region_inset)

        minx, miny, maxx, maxy = region.bounds
        m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        if m.any():
            inside = sv_contains(region, x[m], y[m])
            keep = np.where(m)[0][inside]
        else:
            keep = np.array([], dtype=int)

        if write_plain:
            fminx, fminy, fmaxx, fmaxy = poly.bounds
            fm = (x >= fminx) & (x <= fmaxx) & (y >= fminy) & (y <= fmaxy)
            if fm.any():
                finside = sv_contains(poly, x[fm], y[fm])
                fkeep = np.where(fm)[0][finside]
            else:
                fkeep = np.array([], dtype=int)
            if len(fkeep) > 0:
                sub = laspy.LasData(header=mgr.header)
                sub.points = mgr.points[fkeep].copy()
                try:
                    sub.write(os.path.join(out_dir, f"plot_{pid}_{br}.las"))
                except Exception:
                    pass

        traits = compute_plot_traits(
            x[keep], y[keep], z[keep],
            (hag[keep] if hag is not None else None),
            region, selected_traits,
            voxel=voxel, height_cut=height_cut, biomass_k=biomass_k,
        )

        row = {
            "Plot_ID": pid,
            "B/R":     plot.get("B/R", ""),
            "Bank":    int(plot["Bank"]) if "Bank" in plot else None,
            "Row":     int(plot["Row"])  if "Row"  in plot else None,
            "region_mode": region_mode,
            "region_area_m2": float(region.area),
        }
        for k, v in traits.items():
            row[k] = json.dumps(v) if isinstance(v, list) else v
        rows.append(row)

        if progress_cb:
            progress_cb(int(100 * (idx + 1) / n_plots), f"Plot {idx+1}/{n_plots}")

    df = pd.DataFrame(rows)
    if ground_diag is not None:
        df.attrs["ground_model"] = ground_diag

    # ---- canopy-top surface heights (top-surface raster on the chosen ground) ----
    cth_keys = [k for k in selected_traits if k.startswith("cth_")]
    if cth_keys and len(df):
        from .canopy_top import canopy_top_all, CANOPY_TOP_KEYS, measure_targets, fit_target_calibration
        if progress_cb:
            progress_cb(99, "Canopy-top heights (top-surface raster)...")
        if gm is None:
            from .ground_model import fit_exterior_ground
            try:
                gm = fit_exterior_ground(x, y, z, plots)
            except Exception:
                gm = None
        ct = canopy_top_all(x, y, z, plots.reset_index(drop=True), ext_ground_model=gm,
                            use_ring=(cth_ground == "local"),
                            progress_cb=(lambda p, m: progress_cb(99, m)) if progress_cb else None)
        keep = ["Plot_ID"] + [k for k in CANOPY_TOP_KEYS if k in cth_keys or k == "cth_ok"]
        attrs = dict(df.attrs)
        df = df.drop(columns=[c for c in df.columns if c.startswith("cth_")])
        df = df.merge(ct[keep], on="Plot_ID", how="left")
        df.attrs.update(attrs)
        df.attrs["cth_ground"] = cth_ground
        if targets_csv and os.path.exists(targets_csv):
            try:
                tg = measure_targets(x, y, z, pd.read_csv(targets_csv), gm)
                cal = fit_target_calibration(tg)
                df.attrs["target_calibration"] = cal
                for k in ("cth_p90", "cth_p95", "cth_p99", "cth_max"):
                    if k in df.columns:
                        df[k + "_cal"] = cal["offset"] + cal["scale"] * df[k]
                if out_csv:
                    tg.to_csv(os.path.splitext(out_csv)[0] + "_targets.csv", index=False)
                    with open(os.path.splitext(out_csv)[0] + "_target_calibration.json", "w") as f:
                        json.dump(cal, f, indent=2)
            except Exception as e:
                df.attrs["target_calibration_error"] = str(e)
    # ---- LiDAR v3: per-plot noise removal, ground/canopy classification, canopy-only features ----
    if lidar_v3 and len(df):
        from . import lidar_features as lf
        lf.set_thresholds(canopy_h, ground_h)
        if progress_cb:
            progress_cb(99, "LiDAR v3: preprocessing, classification and features...")
        inten = getattr(mgr, "intensity", None); rn = getattr(mgr, "return_number", None); nr = getattr(mgr, "number_of_returns", None)
        if inten is None or rn is None:
            try:
                pts = mgr.points; dims = set(mgr.header.point_format.dimension_names)
                inten = np.asarray(pts.intensity, float) if "intensity" in dims else None
                rn = np.asarray(pts.return_number) if "return_number" in dims else None
                nr = np.asarray(pts.number_of_returns) if "number_of_returns" in dims else None
            except Exception:
                inten = rn = nr = None
        v3 = lf.lidar_features_all(x, y, z, inten, rn, nr, plots.reset_index(drop=True),
                                   progress_cb=(lambda p, m: progress_cb(99, m)) if progress_cb else None,
                                   write_classified_dir=classified_dir, las_header=getattr(mgr, "header", None),
                                   points=getattr(mgr, "points", None) if classified_dir else None,
                                   noise_rule=noise_rule, nsig=noise_nsig, drop_noise=drop_noise)
        attrs = dict(df.attrs); attrs.update(noise_rule=noise_rule, drop_noise=drop_noise, canopy_h=lf.CANOPY_H, ground_h=lf.GROUND_H)
        dup = [c for c in v3.columns if c in df.columns and c != "Plot_ID"]
        v3 = v3.rename(columns={c: f"{c}_v3" for c in dup})     # keep legacy columns, suffix the v3 duplicates (cth_*, qc_ok ...)
        df = df.merge(v3, on="Plot_ID", how="left"); df.attrs.update(attrs); df.attrs["lidar_v3"] = True
    if out_csv:
        df.to_csv(out_csv, index=False)
        meta = {k: v for k, v in df.attrs.items()}
        meta.update(las=getattr(mgr, "las_path", ""), n_plots=int(len(df)), region_mode=region_mode, band_width=band_width,
                    ground_mode=ground_mode, lidar_v3=bool(lidar_v3), classified_las=bool(lidar_v3 and classified_dir),
                    per_plot_dir=out_dir, columns=list(df.columns))
        try:
            with open(os.path.splitext(out_csv)[0] + "_provenance.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass
        if ground_diag is not None:
            try:
                with open(os.path.splitext(out_csv)[0] + "_ground_model.json", "w") as f:
                    json.dump(ground_diag, f, indent=2)
            except Exception:
                pass
    return df
