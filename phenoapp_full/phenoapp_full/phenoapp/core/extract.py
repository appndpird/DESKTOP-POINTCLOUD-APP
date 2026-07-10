"""
Per-plot extraction.

For every plot polygon:
  1. Find points inside the polygon (bbox prefilter + vectorized contains)
  2. Optionally write a per-plot LAS file
  3. Compute the selected traits
  4. Append a row to the metrics CSV

Public API
----------
extract_all_plots(las_manager, plots_gdf, out_dir, out_csv,
                  selected_traits, write_las=True,
                  height_cut=0.15, voxel=0.05, progress_cb=None,
                  cancel_flag=None) -> pd.DataFrame
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
    progress_cb=None,
    cancel_flag=None,
) -> pd.DataFrame:
    if write_las:
        os.makedirs(out_dir, exist_ok=True)

    x = mgr.x; y = mgr.y; z = mgr.z
    hag = mgr.hag

    n_plots = len(plots)
    rows = []

    for idx, plot in plots.reset_index(drop=True).iterrows():
        if cancel_flag and cancel_flag():
            break

        pid = plot.get("Plot_ID", idx + 1)
        br  = _safe_name(plot.get("B/R", f"P{pid}"))
        poly = plot.geometry

        minx, miny, maxx, maxy = poly.bounds
        m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        if m.any():
            inside = sv_contains(poly, x[m], y[m])
            keep = np.where(m)[0][inside]
        else:
            keep = np.array([], dtype=int)

        if len(keep) > 0 and write_las:
            sub = laspy.LasData(header=mgr.header)
            sub.points = mgr.points[keep].copy()
            try:
                sub.write(os.path.join(out_dir, f"plot_{pid}_{br}.las"))
            except Exception:
                pass  # don't crash the whole batch over one plot

        # Compute traits using HAG when available
        traits = compute_plot_traits(
            x[keep], y[keep], z[keep],
            (hag[keep] if hag is not None else None),
            poly, selected_traits,
            voxel=voxel, height_cut=height_cut, biomass_k=biomass_k,
        )

        row = {
            "Plot_ID": pid,
            "B/R":     plot.get("B/R", ""),
            "Bank":    int(plot["Bank"]) if "Bank" in plot else None,
            "Row":     int(plot["Row"])  if "Row"  in plot else None,
        }
        # Serialize list-typed traits (e.g. vert_profile) as JSON
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
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
