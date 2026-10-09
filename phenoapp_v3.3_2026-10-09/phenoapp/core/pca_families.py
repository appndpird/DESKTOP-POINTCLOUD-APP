"""
PCA feature families (PhenoApp v3.3) shared by the Biomass ML tab and the Height Models tab.

Instead of selecting a few features inside each training fold, a PCA family standardises every available feature of a
block and keeps the principal components that explain 95 % of its variance; the components are the inputs of the learner.
The scaler and the PCA are part of the model pipeline, so they are fitted inside every training fold (no leakage) and
saved with the model.

  LiDAR_PCA        : one PCA on every finite LiDAR feature
  VNIR_PCA         : one PCA on every finite VNIR index feature
  Fused_PCA        : separate PCA per modality, scores concatenated -> learner (late fusion)
  Fused_PCA_joint  : one PCA on the concatenated LiDAR + VNIR features (early fusion)
  VNIR_bands_PCA   : PCA on the mean vegetation spectrum (log10 reflectance, SNV per plot) over the bands that are
                     usable on >= 95 % of the plots and outside the excluded wavelength ranges
  Fused_bands_PCA  : LiDAR PCA + spectral-band PCA

Public API
----------
PCA_FAMILIES, PCA_VAR
numeric_feature_columns(df, exclude)            finite numeric feature columns of a table
split_blocks(df, feats, vnir_cols)              -> (lidar columns, vnir columns, band columns)
spectra_matrix(spectra_df, min_frac, excluded)  -> DataFrame Plot_ID-indexed, columns sb_<nm>
pca_preprocessor(blocks)                        ColumnTransformer: StandardScaler -> PCA(0.95) per block
make_pca_learner(name, pre, scale_y)            pipeline preprocessor -> learner
pca_summary(fitted_model)                       components kept and variance explained per block
"""

from __future__ import annotations
import re
import numpy as np

PCA_VAR = 0.95
PCA_FAMILIES = ["LiDAR_PCA", "VNIR_PCA", "Fused_PCA", "Fused_PCA_joint", "VNIR_bands_PCA", "Fused_bands_PCA"]
SB_PREFIX = "sb_"
_NON_FEATURES = {"Plot_ID", "Plot", "B/R", "Bank", "Row", "Range", "Block", "Rep", "region_mode", "region_area_m2", "area_m2", "qc_ok",
                 "vnir_qc_ok", "vnir_reflectance_ok", "red_ok", "ground_source", "n_raw", "n_noise_removed", "ground_cells", "ground_rms_cm",
                 "I_ring_median_raw", "region_px", "frac_in_cube", "veg_ndvi_threshold", "cth_ok", "cth_ground_cells", "cth_n_noise", "n_usable_bands",
                 "cube_reflectance_ok", "noise_rule", "gt_column", "has_ground_truth", "valid_lidar", "valid_vnir", "valid_vnir_lenient", "valid_fused"}
_VNIR_NAME = re.compile(r"(_veg$|_veg_sd$|^refl_|^red_depth|^red_edge|^rededge|^REP|^S2REP|^LCI|^MTCI|^CIRE|^CIre|^NDRE|^NDREI|^NDVI|^GNDVI|^kNDVI|"
                        r"^OSAVI|^SAVI|^NIRv|^WDRVI|^VOG|^MCARI|^TCARI|^WBI|^PRI|^EVI|^MSR|^spec_pc|^pc\d|^fcover_vnir|^veg_frac|^deriv_|^dr_|^log_|^snv_|^sb_)")


def numeric_feature_columns(df, exclude=(), min_finite=0.95):
    """Numeric columns that look like features: not bookkeeping, not targets / predictions / QC flags, finite on
    >= min_finite of the rows and not constant."""
    cols = []
    for c in df.columns:
        if c in _NON_FEATURES or c in exclude or df[c].dtype == object:
            continue
        if c.endswith(("_valid", "_measured", "_sd_valid")) or c.startswith(("pred", "biomass", "height_cm", "ruler", "dm_", "fresh_", "valid_")):
            continue
        v = df[c].astype(float).to_numpy()
        if np.isfinite(v).mean() >= min_finite and np.nanstd(v) > 0:
            cols.append(c)
    return cols


def split_blocks(df, feats, vnir_cols=None):
    """Split feature columns into (lidar, vnir, bands). VNIR membership comes from `vnir_cols` (the columns of the VNIR
    features CSV) when given, otherwise from the index naming conventions. Band columns start with 'sb_'."""
    vset = set(vnir_cols or [])
    bands = [c for c in feats if c.startswith(SB_PREFIX)]
    vnir = [c for c in feats if c not in bands and (c in vset or (not vset and _VNIR_NAME.search(c)))]
    lidar = [c for c in feats if c not in bands and c not in vnir]
    return lidar, vnir, bands


def spectra_matrix(spectra_df, min_frac=0.95, excluded=None):
    """Mean vegetation spectra table (Plot_ID + one column per wavelength in nm, NaN = unusable band) -> DataFrame
    indexed by Plot_ID of log10 reflectance after per-plot SNV, over the bands usable on >= min_frac of the plots and
    outside the excluded ranges; remaining NaN cells are filled with the band median over plots."""
    import pandas as pd
    if excluded is None:
        from .spectral_indices import DEFAULT_EXCLUDED_NM
        excluded = DEFAULT_EXCLUDED_NM
    S = spectra_df.set_index("Plot_ID")
    wcols = [c for c in S.columns if re.fullmatch(r"[0-9.]+", str(c))]
    S = S[wcols]; wl = np.array([float(c) for c in wcols])
    ex = np.zeros(len(wl), bool)
    for lo, hi in excluded:
        ex |= (wl >= lo) & (wl <= hi)
    X = S.to_numpy(float)
    X = np.where(X > 0, X, np.nan)                                       # clipped / NaN bands are not reflectance
    ok = (np.isfinite(X).mean(0) >= min_frac) & ~ex
    if ok.sum() < 3:
        return pd.DataFrame(index=S.index)
    X = X[:, ok]
    if np.nanmax(X) > 1.5:                                                # reflectance x 10000 -> reflectance
        X = X / 10000.0
    X = np.log10(np.clip(X, 1e-4, None))
    mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True) + 1e-9
    X = (X - mu) / sd                                                    # SNV per plot on its valid bands
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0, keepdims=True))
    return pd.DataFrame(X, index=S.index, columns=[f"{SB_PREFIX}{c}" for c in np.array(wcols)[ok]])


def pca_blocks(family, lidar, vnir, bands):
    """(block name, column list) pairs of a family."""
    if family == "LiDAR_PCA":
        return [("lidar", lidar)]
    if family == "VNIR_PCA":
        return [("vnir", vnir)]
    if family == "Fused_PCA":
        return [("lidar", lidar), ("vnir", vnir)]
    if family == "Fused_PCA_joint":
        return [("joint", lidar + vnir)]
    if family == "VNIR_bands_PCA":
        return [("bands", bands)]
    if family == "Fused_bands_PCA":
        return [("lidar", lidar), ("bands", bands)]
    raise ValueError(family)


def family_inputs(family, df, exclude=(), vnir_cols=None):
    """Ordered input columns and blocks (name -> column indices) of a PCA family on a table. Empty when a required
    block has no usable columns (e.g. no spectra table, or VNIR all NaN)."""
    feats = numeric_feature_columns(df, exclude)
    lidar, vnir, bands = split_blocks(df, feats, vnir_cols)
    blocks = pca_blocks(family, lidar, vnir, bands)
    if any(len(cols) < 2 for _, cols in blocks):
        return [], []
    cols = [c for _, cs in blocks for c in cs]
    pos = {c: i for i, c in enumerate(cols)}
    return cols, [(name, [pos[c] for c in cs]) for name, cs in blocks]


def pca_preprocessor(blocks):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    trs = []
    for name, idx in blocks:
        if len(idx) >= 2:
            trs.append((name, make_pipeline(StandardScaler(), PCA(n_components=PCA_VAR, svd_solver="full")), list(idx)))
        elif len(idx) == 1:
            trs.append((name, StandardScaler(), list(idx)))
    return ColumnTransformer(trs, remainder="drop")


def make_pca_learner(name, pre, scale_y=1.0):
    """Pipeline: PCA preprocessor -> learner (the tool's learner settings; GPR with an isotropic kernel because the
    number of components is only known after the PCA is fitted)."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV

    def pipe(est):
        return Pipeline([("pre", clone(pre)), ("m", est)])
    if name == "Ridge":
        return pipe(RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if name == "PLS":
        return GridSearchCV(pipe(PLSRegression(scale=True)), {"m__n_components": [1, 2, 3, 4, 5, 6]}, cv=5, scoring="neg_root_mean_squared_error")
    if name == "SVR_rbf":
        eps = [0.05 * scale_y, 0.15 * scale_y, 0.3 * scale_y]; C = [1 * scale_y, 3 * scale_y, 10 * scale_y, 30 * scale_y]
        return GridSearchCV(pipe(SVR()), {"m__C": C, "m__epsilon": eps, "m__gamma": ["scale", 0.03, 0.1]}, cv=5, scoring="neg_root_mean_squared_error")
    if name == "GPR":
        return pipe(GaussianProcessRegressor(ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0))
    if name == "RandomForest":
        return pipe(RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.7, random_state=0, n_jobs=-1))
    if name == "ExtraTrees":
        return pipe(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=0, n_jobs=-1))
    if name == "XGBoost":
        import xgboost as xgb
        return pipe(xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0))
    if name == "LightGBM":
        import lightgbm as lgb
        return pipe(lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, random_state=0, verbose=-1))
    raise ValueError(name)


def pca_summary(model):
    """{block: {n_components, explained_variance}} of a fitted PCA pipeline (GridSearchCV wrappers are unwrapped)."""
    from sklearn.decomposition import PCA
    est = getattr(model, "best_estimator_", model)
    pre = getattr(est, "named_steps", {}).get("pre")
    out = {}
    if pre is None or not hasattr(pre, "transformers_"):
        return out
    for name, tr, _ in pre.transformers_:
        if hasattr(tr, "steps") and isinstance(tr.steps[-1][1], PCA):
            p = tr.steps[-1][1]
            out[name] = dict(n_components=int(p.n_components_), explained_variance=float(p.explained_variance_ratio_.sum()))
    return out
