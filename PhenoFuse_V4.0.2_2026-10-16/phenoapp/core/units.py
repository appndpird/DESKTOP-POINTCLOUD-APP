"""
Biomass unit handling — one place that turns any ground-truth unit into an
area density so calibrations transfer between trials and plot sizes.

Why density
-----------
Ground truth almost always arrives as a density (kg/ha, t/ha, g/m²) because
it is scaled up from a quadrat cut. LiDAR features that scale with plot size
(PVI, voxel volume, hull volume, surface area) are per-plot totals. Fitting a
per-plot total against a density silently assumes every plot has the same
area — which breaks the moment a calibration is reused on another trial.

The rule used throughout PhenoApp:

    * ground truth      -> kg/m²  (internal)  and kg/ha (reported)
    * area-type feature -> divided by region_area_m2, so PVI/area = cover x h
    * k (kg/m³) and regression coefficients are therefore area-free

Unit keys
---------
    kg_ha   kilograms per hectare        (agronomic standard)
    t_ha    tonnes per hectare
    g_m2    grams per square metre
    kg_m2   kilograms per square metre
    kg_plot kilograms per plot           (needs region_area_m2)
    g_plot  grams per plot               (needs region_area_m2)
    auto    infer from the column name   (see infer_unit_from_column)
"""
from __future__ import annotations
import re
import numpy as np

# key -> (label shown in the UI, factor to kg/m², needs plot area)
BIOMASS_UNITS: dict[str, tuple[str, float, bool]] = {
    "kg_ha":   ("kg/ha  (kilograms per hectare)",    1.0 / 10_000.0, False),
    "t_ha":    ("t/ha  (tonnes per hectare)",        1.0 / 10.0,     False),
    "g_m2":    ("g/m²  (grams per square metre)",    1.0 / 1_000.0,  False),
    "kg_m2":   ("kg/m²  (kilograms per square metre)", 1.0,          False),
    "kg_plot": ("kg per plot  (total in the sampled region)", 1.0,   True),
    "g_plot":  ("g per plot  (total in the sampled region)", 1.0 / 1_000.0, True),
}
UNIT_KEYS = list(BIOMASS_UNITS.keys())
UNIT_LABELS = {k: v[0] for k, v in BIOMASS_UNITS.items()}
AUTO_LABEL = "Auto-detect from column name"

# columns that hold per-plot totals (scale with area) and must be divided by
# region_area_m2 before they are used as predictors of a density target
AREA_SCALED_FEATURES = {"biomass_pvi", "biomass_kg", "vol_voxel", "vol_chull",
                        "vol_alpha", "surf_area", "n_points", "n_canopy"}

_SUFFIX_UNIT = [
    # order matters: 'kg_m2' must win over 'g_m2', 'kg_plot' over 'g_plot'
    (r"(?<![a-z])(kg[_/ ]?ha|kgha)$", "kg_ha"),
    (r"(?<![a-z])(t[_/ ]?ha|tha|tonnes?[_ ]?ha)$", "t_ha"),
    (r"(?<![a-z])(kg[_/ ]?m2|kg[_/ ]?m\^?2|kgm2)$", "kg_m2"),
    (r"(?<![a-z])(g[_/ ]?m2|g[_/ ]?m\^?2|gm2)$", "g_m2"),
    (r"(?<![a-z])(kg([_ ]?plot)?|kilograms?)$", "kg_plot"),
    (r"(?<![a-z])(g([_ ]?plot)?|grams?)$", "g_plot"),
]


def infer_unit_from_column(col: str) -> str | None:
    """Guess the unit from a column name such as 'biomass_kg_ha' or 'fresh_kg'.

    Returns a unit key or None when the name carries no unit.
    """
    c = str(col).strip().lower().replace("(", "_").replace(")", "")
    c = re.sub(r"[\s\-]+", "_", c)
    for pat, key in _SUFFIX_UNIT:
        if re.search(pat, c):
            return key
    return None


def resolve_unit(unit: str | None, col: str, default: str = "kg_ha") -> str:
    """Turn a user choice ('auto' / key / label / None) into a unit key."""
    if unit in (None, "", "auto", AUTO_LABEL):
        return infer_unit_from_column(col) or default
    if unit in BIOMASS_UNITS:
        return unit
    for k, (label, _, _) in BIOMASS_UNITS.items():
        if unit == label:
            return k
    raise ValueError(f"Unknown biomass unit '{unit}'. Use one of {UNIT_KEYS}.")


def to_kg_m2(values, unit: str, area_m2=None) -> np.ndarray:
    """Convert ground-truth values in `unit` to kg/m²."""
    v = np.asarray(values, dtype=float)
    _, factor, needs_area = BIOMASS_UNITS[unit]
    v = v * factor
    if needs_area:
        if area_m2 is None:
            raise ValueError(
                f"Unit '{unit}' is a per-plot total; region_area_m2 is needed "
                "to convert it to a density. Recompute traits so the metrics "
                "CSV carries region_area_m2, or supply the area.")
        a = np.asarray(area_m2, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(a > 0, v / a, np.nan)
    return v


def from_kg_m2(density, unit: str, area_m2=None) -> np.ndarray:
    """Convert a kg/m² density back to `unit` (inverse of to_kg_m2)."""
    d = np.asarray(density, dtype=float)
    _, factor, needs_area = BIOMASS_UNITS[unit]
    if needs_area:
        if area_m2 is None:
            raise ValueError(f"Unit '{unit}' needs region_area_m2.")
        d = d * np.asarray(area_m2, dtype=float)
    return d / factor


def kg_m2_to_kg_ha(density) -> np.ndarray:
    return np.asarray(density, dtype=float) * 10_000.0


def kg_ha_to_kg_m2(kg_ha) -> np.ndarray:
    return np.asarray(kg_ha, dtype=float) / 10_000.0


def find_ground_truth_column(columns, preferred: str | None = None,
                             stems=("biomass", "fresh", "dm", "yield")) -> str:
    """Pick the ground-truth column in a user CSV.

    Order: exact `preferred` name; then any column starting with one of
    `stems` (e.g. biomass_kg, biomass_kg_ha, fresh_t_ha); fails otherwise.
    """
    cols = list(columns)
    if preferred and preferred in cols:
        return preferred
    low = {c.lower().strip(): c for c in cols}
    for stem in stems:
        for lc, c in low.items():
            if lc.startswith(stem) and lc not in ("plot_id", "plot"):
                return c
    raise RuntimeError(
        "Ground-truth CSV has no biomass column. Expected a column such as "
        "'biomass_kg_ha', 'biomass_kg', 'fresh_kg_ha' or 'fresh_kg' "
        f"(found: {', '.join(cols)}).")


def ground_truth_density(gt_df, metrics_df, gt_col: str, unit: str = "auto",
                         join_col: str = "Plot_ID"):
    """Merge a ground-truth table with plot metrics and add density columns.

    Returns (merged DataFrame, unit_key). The merged frame carries
    `<gt_col>` (as supplied), `gt_kg_m2`, `gt_kg_ha` and `region_area_m2`.
    Only plots present in both tables are kept.
    """
    import pandas as pd
    unit_key = resolve_unit(unit, gt_col)
    gt = gt_df[[join_col, gt_col]].copy()
    gt[gt_col] = pd.to_numeric(gt[gt_col], errors="coerce")
    keep = [join_col] + (["region_area_m2"] if "region_area_m2" in metrics_df.columns else [])
    merged = metrics_df[keep].merge(gt, on=join_col, how="inner")
    area = merged["region_area_m2"].to_numpy(float) if "region_area_m2" in merged else None
    merged["gt_kg_m2"] = to_kg_m2(merged[gt_col].to_numpy(float), unit_key, area)
    merged["gt_kg_ha"] = kg_m2_to_kg_ha(merged["gt_kg_m2"])
    return merged, unit_key


def area_normalise(df, features: list[str], area_col: str = "region_area_m2"):
    """Return a copy of `df` where area-scaled features are divided by area.

    Features not in AREA_SCALED_FEATURES are left untouched. Names are kept,
    so a model fitted on the normalised frame must be applied to a frame
    normalised the same way; the returned list says which columns changed.
    """
    out = df.copy()
    to_norm = [f for f in features if f in AREA_SCALED_FEATURES and f in out.columns]
    if to_norm:
        if area_col not in out.columns:
            raise RuntimeError(
                f"'{area_col}' is missing from the metrics CSV, so per-plot "
                f"features ({', '.join(to_norm)}) cannot be expressed per m². "
                "Recompute traits with this version of PhenoApp.")
        a = out[area_col].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in to_norm:
                out[f] = np.where(a > 0, out[f].to_numpy(float) / a, np.nan)
    return out, to_norm
