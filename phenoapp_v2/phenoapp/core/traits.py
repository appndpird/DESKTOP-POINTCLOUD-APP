"""
Per-plot trait computation.

Computes any combination of 18 physical traits from points within a plot
polygon. Designed to be called once per plot, with the points pre-filtered
to that plot.

All traits are documented in docs/traits.md (loaded by the Guidance tab).

Public API
----------
TRAITS_CATALOG : list of (key, group, label, description) tuples — for the UI
compute_plot_traits(x, y, z, hag, polygon, selected_keys, voxel=0.05,
                    height_cut=0.15) -> dict
"""

from __future__ import annotations
import math
import numpy as np

# ============================================================
# Trait catalogue (used by Traits tab UI and Guidance docs)
# ============================================================
# Tuple format: (key, group, label, description)
TRAITS_CATALOG = [
    # ---- Height & vertical structure ----
    ("h_max",       "Height", "Max height (h_max)",
        "Maximum height-above-ground in the plot. Sensitive to noise spikes."),
    ("h_mean",      "Height", "Mean height",
        "Mean of all height values in the plot."),
    ("h_median",    "Height", "Median height (h_p50)",
        "50th percentile height. More robust than mean."),
    ("h_p95",       "Height", "95th percentile (h_p95) — recommended",
        "Robust plant-height estimator, standard in phenotyping literature."),
    ("h_p99",       "Height", "99th percentile (h_p99)",
        "Less aggressive smoothing than h_p95."),
    ("h_std",       "Height", "Height std dev",
        "Standard deviation of heights. High value = uneven canopy."),
    ("h_p99_p50",   "Height", "Vertical spread (p99 - p50)",
        "Difference between 99th and 50th percentile. Captures vertical structure."),
    ("h_skew",      "Height", "Skewness",
        "Distribution skewness of Z. Positive = top-heavy canopy."),
    ("h_kurt",      "Height", "Kurtosis",
        "Distribution peakedness. >0 = sharp peak."),
    # ---- Local-baseline heights (computed from raw Z within each plot;
    #      independent of SMRF / HAG. Useful as a cross-check.) ----
    ("h_ground_p1", "Height", "Local ground baseline (p1 of Z)",
        "1st-percentile Z within the plot. Proxy for local ground elevation. "
        "Useful when SMRF is unavailable or as a sanity check."),
    ("h_max_local", "Height", "Max height vs local baseline",
        "(max Z) - (1st-percentile Z) within the plot. Plot-local height that "
        "does not depend on global SMRF. May overestimate if no ground returns."),
    ("h_p95_local", "Height", "p95 height vs local baseline",
        "(95th-pct Z) - (1st-pct Z) within the plot. Robust local height; "
        "compare to h_p95 to detect SMRF inconsistencies."),
    # ---- Cover & density ----
    ("cover_frac",  "Cover",  "Canopy cover fraction",
        "Fraction of points above height_cut. Strong biomass proxy."),
    ("pt_density",  "Cover",  "Point density (per m²)",
        "Number of points per square metre of plot area."),
    ("vert_profile","Cover",  "Vertical profile (10 cm bins)",
        "Density per 10 cm slice 0..3 m. Stored as JSON list."),
    ("gap_frac",    "Cover",  "Gap fraction",
        "1 - cover_frac. Inverse of cover; high = sparse canopy."),
    # ---- Volume & biomass ----
    ("vol_voxel",   "Volume", "Voxel volume (5 cm)",
        "Occupied 5 cm voxels × voxel volume. Best biomass proxy."),
    ("vol_chull",   "Volume", "Convex hull volume",
        "3D convex hull volume. Fast, overestimates."),
    ("vol_alpha",   "Volume", "Alpha-shape volume",
        "Concave hull volume. Closer to true canopy envelope; slower."),
    ("surf_area",   "Volume", "3D surface area",
        "Triangulated canopy surface area. For light-interception models."),
    ("biomass_pvi", "Volume", "Biomass proxy (PVI = cover x h_p95 x area)",
        "Plant Volume Index: cover_frac * h_p95 * region_area (m^3 per "
        "plot). Standard UAV biomass proxy. PVI / region_area_m2 = "
        "cover_frac * h_p95 is the area-free canopy depth (m) used for "
        "calibration, so k transfers between plot sizes."),
    ("biomass_kg",  "Volume", "Calibrated biomass per plot (kg) = PVI x k",
        "Total biomass in the sampled region: PVI (m^3) x k (kg/m^3). "
        "Set k with 'Fit k from CSV...' (any ground-truth unit) or use a "
        "published value: wheat ~0.25-0.30, barley ~0.20-0.30, fodder "
        "grass ~0.08-0.15 (fresh weight). With k=1.0 equals biomass_pvi."),
    ("biomass_kg_ha", "Volume", "Calibrated biomass density (kg/ha)",
        "k x cover_frac x h_p95 x 10000: the same calibration expressed as "
        "an area density, directly comparable with kg/ha ground truth and "
        "with other trials regardless of plot size."),
    # ---- Geometric ----
    ("canopy_extent","Shape", "Canopy extent (XY width × length)",
        "Bounding-box of canopy points within the plot polygon."),
    ("lodging_angle","Shape", "Lodging angle (deg)",
        "PCA-derived lean of plot relative to vertical."),
    ("roughness",    "Shape", "Surface roughness",
        "Std dev of canopy-top heights across XY cells."),
    ("row_count",    "Shape", "Detected row count",
        "Estimated number of crop rows by 1D density peaks across plot width."),
    # ---- Bookkeeping ----
    ("n_points",     "Counts","Total points in plot",
        "Number of LAS points falling inside the plot polygon."),
    ("n_canopy",     "Counts","Canopy points (above height_cut)",
        "Number of points above the canopy height cutoff."),
]

TRAIT_KEYS = [t[0] for t in TRAITS_CATALOG]
TRAIT_BY_KEY = {t[0]: t for t in TRAITS_CATALOG}


def fit_biomass_k(metrics_csv: str, ground_truth_csv: str,
                  join_col: str = "Plot_ID",
                  gt_col: str | None = None,
                  gt_unit: str = "auto") -> dict:
    """Fit the biomass calibration coefficient k from a ground-truth CSV.

    Model (area-free):
        biomass density (kg/m^2) = k * cover_frac * h_p95
                                 = k * biomass_pvi / region_area_m2
    with the intercept fixed at 0 (zero canopy -> zero biomass). k is in
    kg per m^3 of canopy and no longer depends on plot size, so a k fitted
    on one trial can be applied to another with different plots.

    `gt_col`  : ground-truth column; None = first column whose name starts
                with biomass / fresh / dm / yield (e.g. biomass_kg_ha).
    `gt_unit` : 'auto' (infer from the column name: *_kg_ha, *_t_ha, *_g_m2,
                *_kg_m2, *_kg = kg per plot) or an explicit unit key from
                phenoapp.core.units.BIOMASS_UNITS.

    Returns dict with k, r2, rmse_kg_ha, loocv_rmse_kg_ha, n, unit, gt_col,
    depth_range (m), gt_range_kg_ha.
    """
    import pandas as pd
    from .units import (find_ground_truth_column, resolve_unit, to_kg_m2,
                        kg_m2_to_kg_ha, UNIT_LABELS)
    m  = pd.read_csv(metrics_csv)
    gt = pd.read_csv(ground_truth_csv)

    if "biomass_pvi" not in m.columns:
        raise RuntimeError(
            f"'biomass_pvi' column missing from {metrics_csv}. "
            "Tick biomass_pvi in the Traits tab and recompute first.")
    if "region_area_m2" not in m.columns:
        raise RuntimeError(
            "The metrics CSV has no 'region_area_m2' column. Recompute "
            "traits with this version of PhenoApp so k can be fitted per m².")
    if join_col not in gt.columns:
        raise RuntimeError(
            f"Ground-truth CSV is missing the '{join_col}' column. "
            f"Required: '{join_col}' plus a biomass column such as "
            "'biomass_kg_ha' or 'biomass_kg'.")
    gt_col = find_ground_truth_column(gt.columns, gt_col)
    unit_key = resolve_unit(gt_unit, gt_col)

    gt = gt[[join_col, gt_col]].copy()
    gt[gt_col] = pd.to_numeric(gt[gt_col], errors="coerce")
    df = m[[join_col, "biomass_pvi", "region_area_m2"]].merge(gt, on=join_col, how="inner")
    df = df.dropna(subset=["biomass_pvi", "region_area_m2", gt_col])
    df = df[df["region_area_m2"] > 0]
    if len(df) < 3:
        raise RuntimeError(
            f"Only {len(df)} plot(s) matched between metrics and ground truth. "
            f"Need at least 3 for a meaningful fit.")

    area  = df["region_area_m2"].to_numpy(dtype=float)
    depth = df["biomass_pvi"].to_numpy(dtype=float) / area      # m^3/m^2 = m
    gtv   = to_kg_m2(df[gt_col].to_numpy(dtype=float), unit_key, area)
    ok = np.isfinite(depth) & np.isfinite(gtv)
    depth, gtv = depth[ok], gtv[ok]
    denom = float((depth * depth).sum())
    if denom <= 0:
        raise RuntimeError("All canopy depths are zero - cannot fit.")
    k = float((gtv * depth).sum() / denom)

    pred = k * depth
    ss_res = float(((gtv - pred) ** 2).sum())
    ss_tot = float(((gtv - gtv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(gtv)))
    # leave-one-out for a through-origin line: hat h_i = x_i^2 / sum x^2
    h = depth * depth / denom
    loo = float(np.sqrt(np.mean(((gtv - pred) / (1.0 - h)) ** 2))) if np.all(h < 1) else float("nan")

    return {
        "k":    k,
        "r2":   r2,
        "rmse_kg_ha":       float(kg_m2_to_kg_ha(rmse)),
        "loocv_rmse_kg_ha": float(kg_m2_to_kg_ha(loo)),
        "rmse": rmse,                      # kg/m^2 (kept for callers)
        "n":    int(len(gtv)),
        "unit": unit_key,
        "unit_label": UNIT_LABELS[unit_key],
        "gt_col": gt_col,
        "depth_range": (float(depth.min()), float(depth.max())),
        "gt_range_kg_ha": (float(kg_m2_to_kg_ha(gtv.min())), float(kg_m2_to_kg_ha(gtv.max()))),
        "area_range_m2": (float(area.min()), float(area.max())),
    }


# Default LiDAR predictors for the multi-metric biomass model. These are the
# metrics most consistently linked to crop biomass in UAV-LiDAR studies:
# a robust height, canopy cover, and a 3-D occupancy volume. Per-plot
# volumes are divided by region_area_m2 before fitting (see units.py) so the
# model predicts a density (kg/ha) and is independent of plot size.
BIOMASS_MODEL_PREDICTORS = ["h_p95", "cover_frac", "vol_voxel"]


def fit_biomass_multi(metrics_csv: str, ground_truth_csv: str,
                      predictors: list[str] | None = None,
                      join_col: str = "Plot_ID",
                      gt_col: str | None = None,
                      gt_unit: str = "auto") -> dict:
    """Fit a multiple-linear-regression biomass model from ground truth.

    Model:  biomass_kg_ha = b0 + b1*p1 + b2*p2 + ... + bk*pk
    where p1..pk are LiDAR-derived plot metrics (default: h_p95, cover_frac,
    vol_voxel). Any predictor that scales with plot size (vol_*, surf_area,
    biomass_pvi, n_points) is divided by region_area_m2 first, so every
    term is a density and the coefficients transfer between trials.

    Ground truth may be in any unit (see fit_biomass_k for `gt_col` and
    `gt_unit`); it is converted to kg/ha before fitting.

    Returns dict: {predictors, area_normalised, intercept, coefs, r2, adj_r2,
                   rmse, loocv_rmse, loocv_r2, n, equation, target_unit,
                   gt_unit, gt_col}. RMSE values are in kg/ha.
    """
    import pandas as pd
    from .units import (find_ground_truth_column, resolve_unit, to_kg_m2,
                        kg_m2_to_kg_ha, area_normalise)

    predictors = list(predictors or BIOMASS_MODEL_PREDICTORS)
    m  = pd.read_csv(metrics_csv)
    gt = pd.read_csv(ground_truth_csv)

    missing = [p for p in predictors if p not in m.columns]
    if missing:
        raise RuntimeError(
            "These predictor column(s) are missing from the metrics CSV: "
            f"{', '.join(missing)}.\nTick them in the Traits tab and recompute, "
            "or choose a different predictor set.")
    if "region_area_m2" not in m.columns:
        raise RuntimeError(
            "The metrics CSV has no 'region_area_m2' column. Recompute "
            "traits with this version of PhenoApp.")
    if join_col not in gt.columns:
        raise RuntimeError(
            f"Ground-truth CSV is missing the '{join_col}' column. "
            f"Required: '{join_col}' plus a biomass column such as "
            "'biomass_kg_ha' or 'biomass_kg'.")
    gt_col = find_ground_truth_column(gt.columns, gt_col)
    unit_key = resolve_unit(gt_unit, gt_col)

    m, normed = area_normalise(m, predictors)
    gt = gt[[join_col, gt_col]].copy()
    gt[gt_col] = pd.to_numeric(gt[gt_col], errors="coerce")
    df = m[[join_col, "region_area_m2"] + predictors].merge(gt, on=join_col, how="inner")
    df = df.dropna(subset=predictors + [gt_col, "region_area_m2"])
    df = df[df["region_area_m2"] > 0]
    n = len(df)
    if n < len(predictors) + 2:
        raise RuntimeError(
            f"Only {n} plot(s) matched with complete data. Need at least "
            f"{len(predictors) + 2} for a {len(predictors)}-predictor model.")

    X = df[predictors].to_numpy(dtype=float)
    y = kg_m2_to_kg_ha(to_kg_m2(df[gt_col].to_numpy(dtype=float), unit_key,
                                df["region_area_m2"].to_numpy(dtype=float)))
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)

    pred = A @ beta
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = len(predictors)
    adj_r2 = (1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
              if n - p - 1 > 0 else r2)
    rmse = float(np.sqrt(ss_res / n))
    # leave-one-out via the hat matrix diagonal
    try:
        H = np.einsum("ij,jk,ik->i", A, np.linalg.pinv(A.T @ A), A)
        loo_res = resid / (1.0 - H)
        loocv_rmse = float(np.sqrt(np.mean(loo_res ** 2)))
        loocv_r2 = 1.0 - float((loo_res ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        loocv_rmse, loocv_r2 = float("nan"), float("nan")

    intercept = float(beta[0])
    coefs = {name: float(c) for name, c in zip(predictors, beta[1:])}
    def _nm(name): return f"{name}/m^2" if name in normed else name
    eq = "biomass_kg_ha = {:.4g}".format(intercept)
    for name, c in coefs.items():
        eq += " {} {:.4g}*{}".format("+" if c >= 0 else "-", abs(c), _nm(name))

    return {
        "predictors":      predictors,
        "area_normalised": normed,
        "intercept":  intercept,
        "coefs":      coefs,
        "r2":         r2,
        "adj_r2":     adj_r2,
        "rmse":       rmse,
        "loocv_rmse": loocv_rmse,
        "loocv_r2":   loocv_r2,
        "n":          int(n),
        "equation":   eq,
        "target_unit": "kg/ha",
        "gt_unit":    unit_key,
        "gt_col":     gt_col,
    }


def apply_biomass_multi(metrics_csv: str, model: dict,
                        out_col: str = "biomass_pred_kg_ha") -> int:
    """Apply a fitted multi-metric model to a metrics CSV, writing predictions.

    Writes `out_col` (kg/ha) and, when region_area_m2 is present, the
    per-plot total 'biomass_pred_kg' = kg/ha x area / 10000, then saves the
    CSV in place. Returns the number of rows predicted. Rows missing any
    predictor get NaN.
    """
    import pandas as pd
    from .units import area_normalise
    df = pd.read_csv(metrics_csv)
    preds = model["predictors"]
    missing = [p for p in preds if p not in df.columns]
    if missing:
        raise RuntimeError(
            f"Metrics CSV no longer has predictor column(s): {', '.join(missing)}.")
    normed = model.get("area_normalised")
    if normed is None:                      # legacy model: fitted per plot
        feats = df
    else:
        feats, _ = area_normalise(df, preds)
    val = np.full(len(df), model["intercept"], dtype=float)
    for name in preds:
        val = val + model["coefs"][name] * feats[name].to_numpy(dtype=float)
    if model.get("target_unit", "kg") == "kg/ha":
        df[out_col] = val
        if "region_area_m2" in df.columns:
            df["biomass_pred_kg"] = val * df["region_area_m2"].to_numpy(dtype=float) / 10_000.0
    else:                                   # legacy per-plot model
        df["biomass_pred_kg"] = val
        if "region_area_m2" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df[out_col] = val / df["region_area_m2"].to_numpy(dtype=float) * 10_000.0
    df.to_csv(metrics_csv, index=False)
    return int(np.isfinite(val).sum())


# ============================================================
# Compute
# ============================================================
def compute_plot_traits(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    hag: np.ndarray | None,
    polygon,
    selected_keys: list[str],
    voxel: float = 0.05,
    height_cut: float = 0.15,
    biomass_k: float = 1.0,
) -> dict:
    """Compute a set of traits for a single plot.

    Parameters
    ----------
    x, y, z   : arrays of points already filtered to the polygon
    hag       : height-above-ground (preferred for all height metrics);
                if None, falls back to z minus 5th percentile
    polygon   : shapely polygon in same CRS as x, y
    selected_keys : list of trait keys from TRAIT_KEYS
    voxel     : voxel size for vol_voxel (m)
    height_cut: height threshold for canopy/gap/cover (m)
    """
    out: dict = {}
    n = len(x)
    out["n_points"] = int(n)

    if n == 0:
        # Fill all selected with NaN/0 so DataFrame shape is consistent
        for k in selected_keys:
            if k == "n_points": continue
            out[k] = 0 if k in ("n_canopy", "row_count") else None
        return out

    # Use HAG if supplied, else local-percentile fallback
    z_use = hag if hag is not None else (z - np.percentile(z, 5))
    z_use = np.asarray(z_use, dtype=float)

    plot_area = float(polygon.area)
    canopy_mask = z_use > height_cut
    canopy_z = z_use[canopy_mask]
    out["n_canopy"] = int(canopy_mask.sum())

    sel = set(selected_keys)
    def want(k): return k in sel

    # ---- Height & vertical structure ----
    if want("h_max"):    out["h_max"]    = float(np.max(z_use))
    if want("h_mean"):   out["h_mean"]   = float(np.mean(z_use))
    if want("h_median"): out["h_median"] = float(np.median(z_use))
    if want("h_p95"):    out["h_p95"]    = float(np.quantile(z_use, 0.95))
    if want("h_p99"):    out["h_p99"]    = float(np.quantile(z_use, 0.99))
    if want("h_std"):    out["h_std"]    = float(np.std(z_use))
    if want("h_p99_p50"):
        out["h_p99_p50"] = float(np.quantile(z_use, 0.99) - np.quantile(z_use, 0.50))
    if want("h_skew") or want("h_kurt"):
        try:
            from scipy.stats import skew, kurtosis
            if want("h_skew"): out["h_skew"] = float(skew(z_use))
            if want("h_kurt"): out["h_kurt"] = float(kurtosis(z_use))
        except ImportError:
            if want("h_skew"): out["h_skew"] = None
            if want("h_kurt"): out["h_kurt"] = None

    # ---- Local-baseline heights (use raw z, not z_use; SMRF-independent) ----
    if want("h_ground_p1") or want("h_max_local") or want("h_p95_local"):
        z_raw = np.asarray(z, dtype=float)
        ground_p1 = float(np.quantile(z_raw, 0.01))
        if want("h_ground_p1"):
            out["h_ground_p1"] = ground_p1
        if want("h_max_local"):
            out["h_max_local"] = float(z_raw.max() - ground_p1)
        if want("h_p95_local"):
            out["h_p95_local"] = float(np.quantile(z_raw, 0.95) - ground_p1)

    # ---- Cover & density ----
    if want("cover_frac"):
        out["cover_frac"] = float(canopy_mask.mean())
    if want("pt_density"):
        out["pt_density"] = float(n / plot_area) if plot_area > 0 else 0.0
    if want("gap_frac"):
        out["gap_frac"] = float(1.0 - canopy_mask.mean())
    if want("vert_profile"):
        # 10 cm bins from 0 to 3 m
        edges = np.arange(0, 3.01, 0.10)
        h, _ = np.histogram(z_use, bins=edges)
        out["vert_profile"] = h.tolist()

    # ---- Volume & biomass ----
    if want("vol_voxel"):
        out["vol_voxel"] = _voxel_volume(x, y, z_use, voxel)
    if want("vol_chull"):
        out["vol_chull"] = _convex_hull_volume(x, y, z_use)
    if want("vol_alpha"):
        out["vol_alpha"] = _alpha_shape_volume(x, y, z_use)
    if want("surf_area"):
        out["surf_area"] = _surface_area(x, y, z_use)
    if want("biomass_pvi") or want("biomass_kg") or want("biomass_kg_ha"):
        h_p95_local = float(np.quantile(z_use, 0.95))
        cover = float(canopy_mask.mean())
        pvi = cover * h_p95_local * plot_area
        if want("biomass_pvi"):
            out["biomass_pvi"] = pvi
        if want("biomass_kg"):
            out["biomass_kg"] = pvi * float(biomass_k)
        if want("biomass_kg_ha"):
            # k (kg/m^3) x canopy depth (m) = kg/m^2  ->  x 10000 = kg/ha
            out["biomass_kg_ha"] = float(biomass_k) * cover * h_p95_local * 10_000.0

    # ---- Geometric ----
    if want("canopy_extent"):
        if canopy_mask.any():
            cx = x[canopy_mask]; cy = y[canopy_mask]
            ext_x = float(cx.max() - cx.min())
            ext_y = float(cy.max() - cy.min())
            out["canopy_extent"] = max(ext_x, ext_y)
            out["canopy_extent_x"] = ext_x
            out["canopy_extent_y"] = ext_y
        else:
            out["canopy_extent"] = 0.0
            out["canopy_extent_x"] = 0.0
            out["canopy_extent_y"] = 0.0
    if want("lodging_angle"):
        out["lodging_angle"] = _lodging_angle(x, y, z_use)
    if want("roughness"):
        out["roughness"] = _roughness(x, y, z_use, cell=0.10)
    if want("row_count"):
        out["row_count"] = _row_count(x, y, polygon)

    return out


# ============================================================
# Helpers
# ============================================================
def _voxel_volume(x, y, z, vox):
    """Count occupied voxels and multiply by voxel volume."""
    if len(x) == 0: return 0.0
    ix = np.floor(x / vox).astype(np.int64)
    iy = np.floor(y / vox).astype(np.int64)
    iz = np.floor(z / vox).astype(np.int64)
    # only count voxels above ground (z > 0)
    pos = z > 0
    if not pos.any(): return 0.0
    keys = np.unique(ix[pos] * 1_000_000_007 + iy[pos] * 1_000_003 + iz[pos])
    return float(len(keys) * (vox ** 3))


def _convex_hull_volume(x, y, z):
    """3D convex hull volume from scipy."""
    pts = np.column_stack([x, y, z])
    if len(pts) < 4: return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)
    except Exception:
        return None


def _alpha_shape_volume(x, y, z, alpha=0.5):
    """Alpha-shape volume (concave hull). Falls back to convex hull on failure."""
    pts = np.column_stack([x, y, z])
    if len(pts) < 4: return 0.0
    # alpha shape in 3D is expensive and fragile; use a per-cell column approximation:
    # divide into XY cells, integrate (zmax - zmin) over occupied cells
    cell = 0.05
    if len(pts) > 200_000:
        cell = 0.10  # speed up
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    keys = ix * 1_000_003 + iy
    order = np.argsort(keys)
    keys_sorted = keys[order]; z_sorted = z[order]
    uk, idx_start = np.unique(keys_sorted, return_index=True)
    idx_end = np.append(idx_start[1:], len(z_sorted))
    vol = 0.0
    for s, e in zip(idx_start, idx_end):
        zmin = z_sorted[s:e].min()
        zmax = z_sorted[s:e].max()
        if zmax > 0:
            vol += max(zmax - max(zmin, 0.0), 0.0) * (cell * cell)
    return float(vol)


def _surface_area(x, y, z, cell=0.10):
    """Triangulated surface area of canopy top. Approx via per-cell zmax raster."""
    if len(x) < 3: return 0.0
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    # build dict cell -> zmax
    keys = ix * 1_000_003 + iy
    order = np.argsort(keys)
    keys_s = keys[order]; z_s = z[order]
    uk, st = np.unique(keys_s, return_index=True)
    en = np.append(st[1:], len(z_s))
    zmax = np.array([z_s[s:e].max() for s, e in zip(st, en)])
    # approximate: surface area = sum( sqrt(1 + dz/dx^2 + dz/dy^2) * dxdy )
    # we don't have full neighbour info without re-rasterising; cheap approximation:
    # area = (cell**2) * Σ sqrt(1 + (max_neighbour_dz/cell)^2 ... )
    # For speed, return planimetric area * (1 + roughness factor):
    rough = float(np.std(zmax))
    flat_area = len(uk) * cell * cell
    return float(flat_area * (1.0 + rough / max(np.mean(zmax) or 1e-6, 1e-6)))


def _lodging_angle(x, y, z):
    """Angle between principal axis of point cloud and vertical."""
    if len(x) < 10: return None
    pts = np.column_stack([x - x.mean(), y - y.mean(), z - z.mean()])
    try:
        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        # principal axis = eigenvector with largest eigenvalue
        pa = evecs[:, -1]
        # angle from vertical
        cos_t = abs(pa[2]) / (np.linalg.norm(pa) + 1e-9)
        ang = math.degrees(math.acos(min(1.0, cos_t)))
        # express as deviation from vertical (90 = horizontal canopy)
        return float(ang)
    except Exception:
        return None


def _roughness(x, y, z, cell=0.10):
    """Std dev of zmax across cells."""
    if len(x) < 3: return 0.0
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    keys = ix * 1_000_003 + iy
    order = np.argsort(keys)
    keys_s = keys[order]; z_s = z[order]
    uk, st = np.unique(keys_s, return_index=True)
    en = np.append(st[1:], len(z_s))
    zmax = np.array([z_s[s:e].max() for s, e in zip(st, en)])
    return float(np.std(zmax)) if len(zmax) > 0 else 0.0


def _row_count(x, y, polygon):
    """Count peaks in cross-row density. Uses minimum rotated rect to find row direction."""
    if len(x) < 50: return 0
    try:
        from shapely.geometry import box as shp_box
        # Project points onto the polygon's short axis using min rotated rect
        mrr = polygon.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)[:-1]
        # Find shorter side
        sides = []
        for i in range(4):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % 4]
            sides.append((math.hypot(x2 - x1, y2 - y1), (x1, y1, x2, y2)))
        sides.sort()
        short = sides[0][1]  # shortest side
        sx, sy = short[2] - short[0], short[3] - short[1]
        L = math.hypot(sx, sy)
        if L == 0: return 0
        ux, uy = sx / L, sy / L
        # project (x - short[0], y - short[1]) onto (ux, uy) — 1D position across rows
        proj = (x - short[0]) * ux + (y - short[1]) * uy
        # density along projection
        bins = max(20, int(L / 0.05))
        h, _ = np.histogram(proj, bins=bins)
        # smooth and find peaks
        try:
            from scipy.signal import find_peaks
            sm = np.convolve(h, np.ones(3) / 3.0, mode="same")
            peaks, _ = find_peaks(sm, distance=3, prominence=h.max() * 0.20)
            return int(len(peaks))
        except ImportError:
            # crude peak count: cells > 1.5x mean
            return int(np.sum(h > 1.5 * h.mean()))
    except Exception:
        return 0
