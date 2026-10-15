"""
Biomass (and generic trait) machine learning on LiDAR, VNIR and RGB plot features (PhenoFuse 4.0).

Modalities: LiDAR, VNIR, RGB - any one alone or any combination (fused). Every feature family belongs to a set of
modalities (FAMILY_MODALITIES); the tab offers the families whose modalities are ticked.

Feature families
  Simple_linear : one-predictor linear calibrations (the classic models of the v2 Biomass tab: mean / p95 height,
                  PVI, canopy volume, NDRE, WBI, RGB ExG, cover ...) - the reference every other model must beat
  LiDAR_core    : canopy-only percentiles, canopy-top metrics, cover, Pgap, LAI proxy, voxel volume, PVI,
                  profile area, density, roughness, rumple, normalised intensity, 20 cm gap layers
  VNIR_core     : vegetation-masked red-edge / NDVI-family / water indices, reflectance bands, derivative
                  features, spectral PCA scores, vegetation fraction  (VNIR mode "bands": the reflectance bands instead)
  RGB_core      : vegetation-masked RGB indices (ExG, ExGR, GLI, VARI, NGRDI, MGRVI, RGBVI, TGI, GCC), cover, HSV, texture
  Fused_core    : LiDAR + VNIR cores + CHM-weighted indices (index x cover, index x H95, NDRE x PVI)
  LiDAR+RGB_core, VNIR+RGB_core, Fused3_core (all three)
  Fused_all     : every available column of the selected modalities
  PCA families (pca_families.py): one PCA per block (95 % of the variance), fitted inside every training fold
Learners      : Ridge, PLS, SVR (RBF), GPR, RandomForest, ExtraTrees, XGBoost*, LightGBM*  (*if installed)
Validation    : repeated 10-fold (feature selection by random-forest importance inside each training fold),
                5-fold, leave-one-out, or fit-only; group-aware when several flights are pooled (cv_utils).
Metrics       : R2, RMSE, rRMSE, MAE, bias, r, accuracy (100 - MAPE).

Ground truth  : any per-plot column; biomass columns in any unit (kg/ha, t/ha, g/m2, kg/m2, kg or g per plot) are
                converted to kg/ha with ground_truth_kg_ha(); "fresh" or "dry" is just which column is chosen.

Public API
----------
FAMILIES, LEARNERS, FAMILY_MODALITIES, families_for(modalities), feature_family(df, family, ...)
ground_truth_kg_ha(gt, target, unit, area_by_plot) -> (gt, target_col, unit_label)
fit_trait_models(df, gt, target_col, families, learners, cv="rkf10", n_select=12, ...)
    -> (results list, predictions DataFrame, fitted {(family, learner): (model, feats)})
results_table(results)
save_model / load_model / apply_model
"""

from __future__ import annotations
import os
import numpy as np

from .lidar_features import LIDAR_V3_CORE
from .vnir_features import VNIR_V3_CORE
from .rgb_features import RGB_CORE
from .pca_families import (PCA_FAMILIES, FAMILY_BLOCKS, family_inputs, pca_preprocessor, make_pca_learner, pca_summary, spectra_matrix,
                           numeric_feature_columns, split_blocks, vnir_block)
from .cv_utils import make_splits

def _isnum(series):
    from pandas.api.types import is_numeric_dtype
    return bool(is_numeric_dtype(series))


LIDAR_CORE_BIOMASS = ["H95", "H99", "H_mean", "H50", "H_sd", "H_crr", "cth_p95", "cover_5cm", "Pgap", "LAI_proxy", "vox_volume_m3_per_m2", "PVI_m",
                      "profile_area_m", "canopy_pts_per_m2", "roughness", "rumple", "I_canopy_mean", "I_canopy_p90", "I_canopy_x_cover",
                      "Pgap_below_020cm", "Pgap_below_040cm", "Pgap_below_060cm", "dens_1", "dens_5", "dens_9",
                      "cover_2cm_interior", "ground_visible_2cm_interior", "Pgap_interior", "LAI_proxy_interior"]
VNIR_CORE_BIOMASS = VNIR_V3_CORE + ["WDRVI_veg", "NDVI_nb", "NDRE740", "OSAVI", "refl_RE2", "refl_G", "red_depth", "NDWI970_veg", "NDWI_veg", "WI_NDVI_veg"]
RGB_CORE_BIOMASS = list(RGB_CORE)
WEIGHTED = {"NDRE740_veg_x_cover": ("NDRE740_veg", "cover_5cm"), "OSAVI_veg_x_cover": ("OSAVI_veg", "cover_5cm"), "GNDVI_veg_x_cover": ("GNDVI_veg", "cover_5cm"),
            "NDVI_nb_veg_x_cover": ("NDVI_nb_veg", "cover_5cm"), "NDRE740_veg_x_H95": ("NDRE740_veg", "H95"), "OSAVI_veg_x_H95": ("OSAVI_veg", "H95"),
            "REP_veg_x_H95": ("REP_veg", "H95"), "LCI_veg_x_H95": ("LCI_veg", "H95"), "NDRE_x_PVI": ("NDRE740_veg", "PVI_m"),
            "rgb_ExG_veg_x_H95": ("rgb_ExG_veg", "H95"), "rgb_cover_x_H95": ("rgb_cover", "H95")}
FAMILIES = ["Simple_linear", "LiDAR_core", "VNIR_core", "RGB_core", "Fused_core", "LiDAR+RGB_core", "VNIR+RGB_core", "Fused3_core", "Fused_all"] + PCA_FAMILIES
FAMILY_MODALITIES = {"Simple_linear": set(), "LiDAR_core": {"lidar"}, "VNIR_core": {"vnir"}, "RGB_core": {"rgb"}, "Fused_core": {"lidar", "vnir"},
                     "LiDAR+RGB_core": {"lidar", "rgb"}, "VNIR+RGB_core": {"vnir", "rgb"}, "Fused3_core": {"lidar", "vnir", "rgb"}, "Fused_all": set(),
                     "LiDAR_PCA": {"lidar"}, "VNIR_PCA": {"vnir"}, "RGB_PCA": {"rgb"}, "Fused_PCA": {"lidar", "vnir"}, "Fused_PCA_joint": {"lidar", "vnir"},
                     "LiDAR+RGB_PCA": {"lidar", "rgb"}, "VNIR+RGB_PCA": {"vnir", "rgb"}, "Fused3_PCA": {"lidar", "vnir", "rgb"},
                     "VNIR_bands_PCA": {"vnir"}, "Fused_bands_PCA": {"lidar", "vnir"}}
FAMILY_TIPS = {
    "Simple_linear": "One predictor, straight line: the classic calibrations (H95 / mean height / PVI / canopy volume, NDRE, WBI, RGB ExG, cover). Each predictor is reported as its own model.",
    "LiDAR_core": "Canopy-only LiDAR features (percentiles, cover, gap fraction, volume, intensity); random-forest top-k selection inside each fold.",
    "VNIR_core": "Vegetation-masked spectral indices, reflectance bands, red-edge shape, spectral PCA scores; selection inside each fold. VNIR mode 'bands' uses the valid reflectance bands instead of the indices.",
    "RGB_core": "Vegetation-masked RGB indices (ExG, ExGR, GLI, VARI, NGRDI, MGRVI, RGBVI, TGI, GCC), RGB cover, HSV and texture; selection inside each fold.",
    "Fused_core": "LiDAR core + VNIR core + CHM-weighted indices (index x cover, index x H95, NDRE x PVI); selection inside each fold.",
    "LiDAR+RGB_core": "LiDAR core + RGB core (+ ExG x H95, RGB cover x H95); selection inside each fold.",
    "VNIR+RGB_core": "VNIR core + RGB core; selection inside each fold.",
    "Fused3_core": "LiDAR + VNIR + RGB cores + weighted indices; selection inside each fold.",
    "Fused_all": "Every available column of the ticked modalities; selection inside each fold.",
    "LiDAR_PCA": "PCA family: every finite LiDAR feature standardised and reduced to the components holding 95 % of the variance (fitted inside each fold); no selection.",
    "VNIR_PCA": "PCA family: every finite VNIR feature (indices, bands or both per the VNIR mode) -> PCA (95 % variance) inside each fold.",
    "RGB_PCA": "PCA family: every finite rgb_* feature -> PCA (95 % variance) inside each fold.",
    "Fused_PCA": "PCA family, late fusion: one PCA per modality (LiDAR, VNIR), the components are concatenated for the learner.",
    "Fused_PCA_joint": "PCA family, early fusion: one PCA on the concatenated LiDAR + VNIR features.",
    "LiDAR+RGB_PCA": "PCA family, late fusion: LiDAR PCA + RGB PCA.", "VNIR+RGB_PCA": "PCA family, late fusion: VNIR PCA + RGB PCA.",
    "Fused3_PCA": "PCA family, late fusion: LiDAR PCA + VNIR PCA + RGB PCA.",
    "VNIR_bands_PCA": "PCA on the mean vegetation spectrum (log10 reflectance, SNV per plot) over the bands usable on >= 95 % of the plots and outside the excluded ranges; needs the spectra table.",
    "Fused_bands_PCA": "LiDAR PCA + spectral-band PCA (late fusion); needs the spectra table."}
LEARNERS = ["Ridge", "PLS", "SVR_rbf", "GPR", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM"]
VNIR_MODES = {"indices": "spectral indices and shape features (default)", "bands": "valid reflectance bands only (no indices)",
              "both": "indices + reflectance bands"}
# (predictor candidates, modality) of the one-predictor linear calibrations; the first available column is used
SIMPLE_PREDICTORS = [(("H95", "h_p95"), "lidar"), (("H_mean", "h_mean"), "lidar"), (("cth_p95",), "lidar"), (("PVI_m", "biomass_pvi"), "lidar"),
                     (("vox_volume_m3_per_m2", "vol_voxel"), "lidar"), (("profile_area_m",), "lidar"), (("I_canopy_x_cover",), "lidar"),
                     (("cover_2cm_interior", "cover_5cm", "cover_frac"), "lidar"),
                     (("NDRE740_veg", "NDRE740", "NDRE"), "vnir"), (("NDRE720_veg",), "vnir"), (("REP_veg",), "vnir"), (("OSAVI_veg", "OSAVI"), "vnir"),
                     (("WBI_veg", "WBI"), "vnir"), (("NDWI970_veg", "NDWI970"), "vnir"), (("NDVI_nb_veg", "NDVI_nb", "NDVI"), "vnir"), (("kNDVI_veg",), "vnir"),
                     (("rgb_ExG_veg",), "rgb"), (("rgb_GLI_veg",), "rgb"), (("rgb_NGRDI_veg",), "rgb"), (("rgb_cover",), "rgb"), (("rgb_MGRVI_veg",), "rgb")]


def families_for(modalities, vnir_mode="indices"):
    """Families offered for a set of ticked modalities (single-modality families and every fusion of a subset)."""
    mods = {m.lower() for m in modalities}
    out = []
    for f in FAMILIES:
        req = FAMILY_MODALITIES[f]
        if f in ("Simple_linear", "Fused_all"):
            out.append(f); continue
        if req and req <= mods:
            if f in ("VNIR_bands_PCA", "Fused_bands_PCA") and vnir_mode == "bands":
                continue          # identical to VNIR_PCA / Fused_PCA in bands mode
            out.append(f)
    return out


def add_weighted(df):
    d = df.copy()
    for k, (a, b) in WEIGHTED.items():
        if a in d.columns and b in d.columns:
            d[k] = d[a] * d[b]
    return d


def _available(cols, d):
    out = []
    for c in cols:
        if c in d.columns and _isnum(d[c]):
            v = d[c].astype(float)
            if np.isfinite(v).mean() > 0.95 and v.std() > 0:
                out.append(c)
    return out


def _blocks(d, vnir_cols, rgb_cols, exclude=()):
    feats = numeric_feature_columns(d, exclude)
    return split_blocks(d, feats, vnir_cols, rgb_cols)


def feature_family(df, family, exclude=(), vnir_cols=None, rgb_cols=None, vnir_mode="indices", modalities=None):
    """Input columns of a family on a table."""
    d = add_weighted(df)
    d = d.drop(columns=[c for c in exclude if c in d.columns])
    if family in PCA_FAMILIES:
        return family_inputs(family, d, exclude, vnir_cols, rgb_cols, vnir_mode)[0]
    b = _blocks(d, vnir_cols, rgb_cols, exclude)
    def core(cols, block):
        # the curated core list when enough of it is present (>= 3 columns), else every usable column of the block
        c = _available(cols, d)
        return c if len(c) >= 3 else b[block]
    lidar_core = core(LIDAR_CORE_BIOMASS, "lidar")
    if vnir_mode == "bands":
        vnir_core = b["bands"]
    elif vnir_mode == "both":
        vnir_core = core(VNIR_CORE_BIOMASS, "vnir") + b["bands"]
    else:
        vnir_core = core(VNIR_CORE_BIOMASS, "vnir")
    rgb_core = core(RGB_CORE_BIOMASS, "rgb")
    w = [k for k in WEIGHTED if k in d.columns]
    if family == "LiDAR_core":
        return lidar_core
    if family == "VNIR_core":
        return vnir_core
    if family == "RGB_core":
        return rgb_core
    if family == "Fused_core":
        return _available(lidar_core + vnir_core + [k for k in w if not k.startswith("rgb_")], d)
    if family == "LiDAR+RGB_core":
        return _available(lidar_core + rgb_core + [k for k in w if k.startswith("rgb_")], d)
    if family == "VNIR+RGB_core":
        return _available(vnir_core + rgb_core, d)
    if family == "Fused3_core":
        return _available(lidar_core + vnir_core + rgb_core + w, d)
    if family == "Fused_all":
        mods = {m.lower() for m in (modalities or ("lidar", "vnir", "rgb"))}
        cols = []
        if "lidar" in mods: cols += b["lidar"]
        if "vnir" in mods: cols += vnir_block(b, vnir_mode)
        if "rgb" in mods: cols += b["rgb"]
        cols += [k for k in w if k not in cols]
        return _available(cols, d)
    if family == "LiDAR_height_core":
        return _available(LIDAR_V3_CORE, d)
    if family == "LiDAR+VNIR_height":
        return _available(LIDAR_V3_CORE + VNIR_V3_CORE, d)
    raise ValueError(family)


def make_learner(name, nfeat, scale_y=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV
    if name == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if name == "PLS":
        return GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(8, nfeat) + 1))}, cv=5, scoring="neg_root_mean_squared_error")
    if name == "SVR_rbf":
        eps = [0.05 * scale_y, 0.15 * scale_y, 0.3 * scale_y]; C = [1 * scale_y, 3 * scale_y, 10 * scale_y, 30 * scale_y]
        return GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": C, "svr__epsilon": eps, "svr__gamma": ["scale", 0.03]}, cv=5, scoring="neg_root_mean_squared_error")
    if name == "GPR":
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(nfeat) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, verbose=-1)
    if name == "Linear":
        from sklearn.linear_model import LinearRegression
        return make_pipeline(StandardScaler(), LinearRegression())
    raise ValueError(name)


def learner_available(name):
    try:
        if name == "XGBoost": import xgboost  # noqa
        if name == "LightGBM": import lightgbm  # noqa
        return True
    except Exception:
        return False


def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-9)) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), rRMSE=float(np.sqrt(np.mean(e ** 2)) / abs(y.mean()) * 100),
                MAE=float(np.abs(e).mean()), bias=float(e.mean()), r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, MAPE=mape, accuracy=100 - mape)


def _select(Xtr, ytr, k):
    from sklearn.ensemble import RandomForestRegressor
    if k is None or k >= Xtr.shape[1]:
        return list(range(Xtr.shape[1]))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return list(np.argsort(rf.feature_importances_)[::-1][:k])


PLOT_RULES = {"all": "all plots with usable LiDAR (VNIR families: VNIR QC passed)",
              "A": "valid bands only, rule A: every band valid on >= 50 % of the plot's pixels",
              "B": "valid bands only, rule B: every band valid on >= 95 % of the plot's pixels, plot inside the cube"}


def valid_band_plots(vnir_df, valid_fraction_df, rule="A", band_list=None):
    """Plot_IDs whose spectra have no band problem.
    vnir_df           : VNIR features (Plot_ID, vnir_qc_ok, vnir_reflectance_ok, frac_in_cube)
    valid_fraction_df : band valid-fraction table (Plot_ID + one column per wavelength: share of valid pixels)
    rule              : "A" (>= 50 % of pixels valid in every band) or "B" (>= 95 %, and >= 95 % of the plot inside the cube)
    band_list         : wavelengths (nm) to check; default = the bundled common band list (assets/dl_models/common_bands.csv),
                        i.e. the bands usable across flights; falls back to every band outside the excluded ranges."""
    import pandas as pd
    from .spectral_indices import DEFAULT_EXCLUDED_NM
    wcols = [c for c in valid_fraction_df.columns if c != "Plot_ID"]; wl = np.array([float(c) for c in wcols])
    if band_list is None:
        cb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dl_models", "common_bands.csv")
        if os.path.exists(cb):
            t = pd.read_csv(cb); band_list = t.wavelength_nm[t.common_95 == 1].to_numpy(float)
    if band_list is not None:
        use = np.array([np.min(np.abs(np.asarray(band_list, float) - w)) < 0.5 for w in wl])
    else:
        use = np.ones(len(wl), bool)
        for lo, hi in DEFAULT_EXCLUDED_NM: use &= ~((wl >= lo) & (wl <= hi))
    thr = 0.5 if str(rule).upper() == "A" else 0.95
    F = valid_fraction_df.set_index("Plot_ID")[np.array(wcols)[use]].astype(float)
    ok = (F >= thr).all(axis=1)
    V = vnir_df.set_index("Plot_ID")
    for col, cond in (("vnir_qc_ok", lambda s: s == 1), ("vnir_reflectance_ok", lambda s: s == 1)):
        if col in V.columns: ok &= cond(V[col].reindex(F.index).fillna(0))
    if str(rule).upper() == "B" and "frac_in_cube" in V.columns:
        ok &= V.frac_in_cube.reindex(F.index).fillna(0) >= 0.95
    return set(F.index[ok.to_numpy()])


def ground_truth_kg_ha(gt, target, unit="auto", area_by_plot=None):
    """Convert a biomass ground-truth column to kg/ha (column '<target>_kg_ha' unless already kg/ha).
    Non-biomass targets (height, dm_frac, ...) are returned unchanged. Returns (gt, target_col, unit_label)."""
    import pandas as pd
    from .units import resolve_unit, to_kg_m2, kg_m2_to_kg_ha, UNIT_LABELS, infer_unit_from_column
    g = gt.copy(); low = target.lower()
    if target not in g.columns:
        raise ValueError(f"ground truth has no column '{target}' (columns: {', '.join(str(c) for c in g.columns if c != 'Plot_ID')})")
    biomass_like = any(k in low for k in ("biomass", "fresh", "dm_", "dry", "yield", "weight", "_kg", "_g_", "t_ha")) and "frac" not in low and "height" not in low
    if not biomass_like:
        return g, target, ""
    u = resolve_unit(unit, target, default="kg_ha") if unit != "auto" else (infer_unit_from_column(target) or "kg_ha")
    if u == "kg_ha":
        return g, target, UNIT_LABELS["kg_ha"]
    area = None
    if area_by_plot is not None:
        area = pd.Series(area_by_plot).reindex(g["Plot_ID"]).to_numpy(float)
    dens = to_kg_m2(pd.to_numeric(g[target], errors="coerce").to_numpy(float), u, area_m2=area)
    newc = (target if target.endswith("_kg_ha") else f"{target}_kg_ha")
    g[newc] = kg_m2_to_kg_ha(dens)
    return g, newc, UNIT_LABELS[u]


def fit_trait_models(df, gt, target_col, families, learners, cv="rkf10", n_select=12, progress_cb=None, scale_y=None, vnir_cols=None, spectra=None,
                     plot_ids=None, groups=None, rgb_cols=None, vnir_mode="indices", modalities=None, excluded=None):
    """Nested cross-validated fit of `target_col` (from gt, joined on Plot_ID) on the chosen families x learners.
    vnir_cols / rgb_cols : columns of the VNIR / RGB feature tables (tells the families which block a column belongs to)
    spectra   : mean vegetation spectra table (Plot_ID + one column per wavelength) for the band families / VNIR mode 'bands'
    plot_ids  : optional set of Plot_IDs to use (e.g. valid_band_plots(...)); None = every plot with LiDAR QC ok
    groups    : optional group id per row of df (same plot on several flights -> one fold); None = plain splits
    excluded  : excluded wavelength ranges for the spectra matrix (default: the standard ranges)"""
    import pandas as pd
    from sklearn.base import clone
    d = add_weighted(df).merge(gt[["Plot_ID", target_col]].dropna(), on="Plot_ID", how="inner")
    if groups is not None:
        gmap = dict(zip(df["Plot_ID"], groups)); d["_group"] = d["Plot_ID"].map(gmap)
    if "qc_ok" in d.columns:
        d = d[d.qc_ok.fillna(1) == 1]           # NaN = no v3 QC for that flight -> keep
    if plot_ids is not None:
        d = d[d.Plot_ID.isin(plot_ids)]
    need_bands = vnir_mode in ("bands", "both") or any(f.endswith("bands_PCA") for f in families)
    if spectra is not None and need_bands:
        sb = spectra_matrix(spectra, excluded=excluded)
        if len(sb.columns):
            d = d.merge(sb, left_on="Plot_ID", right_index=True, how="left")
    if vnir_mode == "bands" and vnir_cols:
        keep_qc = ("vnir_qc_ok", "vnir_reflectance_ok", "frac_in_cube", "fcover_vnir", "red_ok")
        d = d.drop(columns=[c for c in vnir_cols if c in d.columns and c not in keep_qc])
    d = d.reset_index(drop=True); y = d[target_col].to_numpy(float)
    if len(y) < 10:
        raise ValueError(f"only {len(y)} plots with features and {target_col}")
    sy = scale_y or float(y.std())
    pred = pd.DataFrame({"Plot_ID": d.Plot_ID, f"{target_col}_measured": y})
    for c in ("flight", "Plot_ID_orig"):
        if c in d.columns:
            pred.insert(1, c, d[c].values)
    results, fitted = [], {}
    fam_list = list(families)
    total = sum((len(SIMPLE_PREDICTORS) if f == "Simple_linear" else len(learners)) for f in fam_list); done = 0
    mods = {m.lower() for m in (modalities or ("lidar", "vnir", "rgb"))}
    for fam in fam_list:
        is_pca = fam in PCA_FAMILIES
        uses_vnir = "vnir" in FAMILY_MODALITIES.get(fam, set()) or (fam == "Fused_all" and "vnir" in mods)
        dm = d[d.vnir_qc_ok.fillna(0) == 1] if (uses_vnir and "vnir_qc_ok" in d.columns) else d
        if "rgb" in FAMILY_MODALITIES.get(fam, set()) and "rgb_qc_ok" in dm.columns:
            dm = dm[dm.rgb_qc_ok.fillna(0) == 1]
        ym = dm[target_col].to_numpy(float); grp = dm["_group"].to_numpy() if "_group" in dm.columns else None
        splits = make_splits(len(ym), cv, grp)
        if fam == "Simple_linear":
            for cands, mod in SIMPLE_PREDICTORS:
                done += 1
                if mod not in mods:
                    continue
                col = next((c for c in cands if c in dm.columns and _isnum(dm[c])), None)
                if col is None:
                    continue
                okm = np.isfinite(dm[col].astype(float).to_numpy())
                if okm.sum() < 10 or dm.loc[okm, col].astype(float).std() == 0:
                    continue
                dsub = dm[okm]; X = dsub[[col]].astype(float).to_numpy(); yy = dsub[target_col].to_numpy(float)
                gsub = dsub["_group"].to_numpy() if grp is not None else None
                sp = make_splits(len(yy), cv, gsub)
                model = make_learner("Linear", 1); full = clone(model).fit(X, yy); pf = full.predict(X).ravel()
                b = np.polyfit(X[:, 0], yy, 1); lrn = f"linear[{col}]"
                res = dict(family="Simple_linear", learner=lrn, status="ok", n=int(len(yy)), n_features=1, features=[col], fit=metrics(pf, yy), cv_mode=cv,
                           calibration={"intercept": float(b[1]), "slope": float(b[0]), "equation": f"{target_col} = {b[1]:.4g} + {b[0]:.4g} x {col}"})
                pred.loc[pred.Plot_ID.isin(dsub.Plot_ID), f"pred_Simple_{col}"] = pf
                if sp:
                    ps = np.zeros(len(yy)); cnt = np.zeros(len(yy))
                    for tr, te in sp:
                        mm = clone(model).fit(X[tr], yy[tr]); ps[te] += mm.predict(X[te]).ravel(); cnt[te] += 1
                    pcv = ps / np.maximum(cnt, 1); res["cv"] = metrics(pcv, yy)
                    pred.loc[pred.Plot_ID.isin(dsub.Plot_ID), f"predcv_Simple_{col}"] = pcv
                fitted[("Simple_linear", lrn)] = (full, [col]); results.append(res)
                if progress_cb:
                    progress_cb(int(100 * done / max(total, 1)), f"Simple linear: {col}" + (f": cv RMSE {res['cv']['RMSE']:.3g}" if "cv" in res else ""))
            continue
        if is_pca:
            feats, blocks = family_inputs(fam, add_weighted(dm), exclude=(target_col,), vnir_cols=vnir_cols, rgb_cols=rgb_cols, vnir_mode=vnir_mode)
        else:
            feats, blocks = feature_family(dm, fam, exclude=(target_col,), vnir_cols=vnir_cols, rgb_cols=rgb_cols, vnir_mode=vnir_mode, modalities=modalities), None
        X = dm[feats].astype(float).fillna(dm[feats].astype(float).median()).to_numpy() if feats else None
        k = None if is_pca else (n_select if len(feats) > n_select + 3 else None)      # PCA families: no selection, the PCA is the reduction
        for ln in learners:
            done += 1
            if not learner_available(ln) or not feats:
                results.append(dict(family=fam, learner=ln, status="skipped (package missing or no features" + (" - needs a VNIR features CSV / spectra table" if is_pca else "") + ")")); continue
            model = make_pca_learner(ln, pca_preprocessor(blocks), sy) if is_pca else make_learner(ln, min(len(feats), k or len(feats)), sy)
            idx_full = _select(X, ym, k) if k else list(range(len(feats)))
            full = clone(model).fit(X[:, idx_full], ym); pf = full.predict(X[:, idx_full]).ravel()
            res = dict(family=fam, learner=ln, status="ok", n=int(len(ym)), n_features=len(feats), features=[feats[i] for i in idx_full], fit=metrics(pf, ym), cv_mode=cv)
            col = f"pred_{fam}_{ln}"; pred.loc[pred.Plot_ID.isin(dm.Plot_ID), col] = pf
            if splits:
                ps = np.zeros(len(ym)); cnt = np.zeros(len(ym)); selc = np.zeros(len(feats))
                for tr, te in splits:
                    idx = _select(X[tr], ym[tr], k); selc[idx] += 1
                    mm = clone(model).fit(X[tr][:, idx], ym[tr]); ps[te] += mm.predict(X[te][:, idx]).ravel(); cnt[te] += 1
                pcv = ps / np.maximum(cnt, 1); res["cv"] = metrics(pcv, ym); res["selection_frequency"] = dict(zip(feats, (selc / len(splits)).round(2).tolist()))
                pred.loc[pred.Plot_ID.isin(dm.Plot_ID), f"predcv_{fam}_{ln}"] = pcv
            fitted[(fam, ln)] = (full, [feats[i] for i in idx_full])
            if is_pca:
                res["pca"] = pca_summary(full); res["blocks"] = {name: [feats[i] for i in idx] for name, idx in blocks}
            results.append(res)
            if progress_cb:
                progress_cb(int(100 * done / max(total, 1)), f"{fam} / {ln}" + (f": cv RMSE {res['cv']['RMSE']:.3g}" if "cv" in res else ""))
    return results, pred, fitted


def results_table(results, unit=""):
    lines = [f"{'family':16s} {'learner':26s} {'n':>4s} {'nfeat':>5s} {'PCs':>4s} {'cvR2':>6s} {'cvRMSE':>9s} {'rRMSE%':>7s} {'cvMAE':>9s} {'cv r':>5s} {'acc%':>6s} {'fitR2':>6s}"]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"{r['family']:16s} {r['learner'][:26]:26s} {r['status']}"); continue
        c = r.get("cv"); f = r["fit"]
        pcs = f"{sum(v['n_components'] for v in r['pca'].values()):4d}" if r.get("pca") else "   -"
        cvs = f"{c['R2']:6.3f} {c['RMSE']:9.3g} {c['rRMSE']:7.1f} {c['MAE']:9.3g} {c['r']:5.2f} {c['accuracy']:6.1f}" if c else " " * 49
        lines.append(f"{r['family']:16s} {r['learner'][:26]:26s} {r['n']:4d} {len(r['features']):5d} {pcs} {cvs} {f['R2']:6.3f}")
    return "\n".join(lines)


def save_model(path, model, feats, meta=None, table=None):
    """table: the training feature table -> its medians are stored so a feature unusable on a new flight can be filled."""
    import joblib
    from .ensembles import feature_medians, feature_ranges
    joblib.dump({"model": model, "features": feats, "meta": meta or {},
                 "feature_medians": feature_medians(table, feats) if table is not None else {},
                 "feature_ranges": feature_ranges(table, feats) if table is not None else {}}, path); return path


def load_model(path):
    import joblib
    return joblib.load(path)


def apply_model(df, saved, offset=0.0, spectra=None, folds="all", excluded=None):
    """spectra: mean vegetation spectra table of the flight, needed by the band models (columns sb_<nm>).
    folds: for fold-ensemble files, "all" | "best3" | list of fold numbers (see core.ensembles)."""
    import pandas as pd
    from .ensembles import is_ensemble, is_fold_ensembles, predict_saved, required_features, fill_missing, range_check
    d = add_weighted(df); feats = required_features(saved, folds); filled = []
    sb = [f for f in feats if f.startswith("sb_")]
    if spectra is not None and sb:
        # exactly the bands the model was trained on (the flight's usable-band set may differ, e.g. clipped blue bands)
        sm = spectra_matrix(spectra, excluded=excluded, bands=sb); filled = list(sm.attrs.get("filled_bands", []))
        d = d.merge(sm, left_on="Plot_ID", right_index=True, how="left")
    d, filled_feats, replaced = fill_missing(saved, d, feats)
    if is_ensemble(saved) or is_fold_ensembles(saved):
        p, ok = predict_saved(saved, d, folds)
        out = pd.DataFrame({"Plot_ID": d.Plot_ID, "predicted": p + offset, "model_ok": ok.astype(int)})
    else:
        missing = [f for f in feats if f not in d.columns]
        if missing:
            raise ValueError("feature table lacks " + ", ".join(missing))
        X = d[feats].astype(float).to_numpy(); ok = np.isfinite(X).all(1); p = np.full(len(d), np.nan)
        if ok.any():
            p[ok] = saved["model"].predict(X[ok]).ravel() + offset
        out = pd.DataFrame({"Plot_ID": d.Plot_ID, "predicted": p, "model_ok": ok.astype(int)})
    out.attrs["filled_bands"] = filled; out.attrs["filled_features"] = filled_feats; out.attrs["replaced_features"] = replaced
    out.attrs["unusable_features"] = [f for f in feats if f in d.columns and not f.startswith("sb_") and d[f].isna().all()]
    out.attrs["out_of_range"] = range_check(saved, d, [f for f in feats if f not in replaced])
    out.attrs["n_range_checked"] = len([f for f in dict.fromkeys(feats) if f in saved.get("feature_ranges", {})]) if isinstance(saved, dict) else 0
    return out
