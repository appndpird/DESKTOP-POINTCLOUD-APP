"""
Per-plot extraction.

For every plot polygon:
  1. Choose the sampling region (whole plot or central band — see regions.py)
  2. Find points inside the region (bbox prefilter + vectorized contains)
  3. Optionally write a per-plot LAS file (always the FULL plot polygon)
  4. Compute the selected traits over the region
  5. Append a row to the metrics CSV

Height reference options (ground_mode):
  "hag"      : use the LAS manager's HAG (SMRF cache) or plot-local fallback
  "exterior" : fit an exterior ground surface from points outside the plots
               (recommended for dense pasture — see ground_model.py)

Public API
----------
extract_all_plots(las_manager, plots_gdf, out_dir, out_csv,
                  selected_traits, write_las=True,
                  height_cut=0.15, voxel=0.05,
                  ground_mode="hag", region_mode="whole",
                  band_width=0.5, region_inset=0.10,
                  progress_cb=None, cancel_flag=None) -> pd.DataFrame
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
) -> pd.DataFrame:
    if write_las:
        os.makedirs(out_dir, exist_ok=True)

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

        # -- points in the sampling region (for traits) --
        minx, miny, maxx, maxy = region.bounds
        m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        if m.any():
            inside = sv_contains(region, x[m], y[m])
            keep = np.where(m)[0][inside]
        else:
            keep = np.array([], dtype=int)

        # -- per-plot LAS uses the FULL plot polygon --
        if write_las:
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
                    pass  # don't crash the whole batch over one plot

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
            if isinstance(v, list):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        rows.append(row)

        if progress_cb:
            pct = int(100 * (idx + 1) / n_plots)
            progress_cb(pct, f"Plot {idx+1}/{n_plots}")

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
        # 'exterior' = trial-wide smooth surface (validated best on Muresk 2025);
        # 'local' = per-plot alley ring plane (fallback trial-wide)
        ct = canopy_top_all(x, y, z, plots.reset_index(drop=True), ext_ground_model=gm,
                            use_ring=(cth_ground == "local"),
                            progress_cb=(lambda p, m: progress_cb(99, m)) if progress_cb else None)
        keep = ["Plot_ID"] + [k for k in CANOPY_TOP_KEYS if k in cth_keys or k == "cth_ok"]
        attrs = dict(df.attrs)                      # pandas merge drops attrs
        # plots with no points get placeholder None columns from compute_plot_traits;
        # drop them so the merge does not produce _x/_y suffixes
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
    if out_csv:
        df.to_csv(out_csv, index=False)
        if ground_diag is not None:
            try:
                with open(os.path.splitext(out_csv)[0] + "_ground_model.json",
                          "w") as f:
                    json.dump(ground_diag, f, indent=2)
            except Exception:
                pass
    return df
