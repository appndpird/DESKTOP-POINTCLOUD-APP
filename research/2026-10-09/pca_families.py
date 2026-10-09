"""
PCA-feature families shared by height_models.py and biomass_models.py.

Each family is a scikit-learn pipeline fitted INSIDE every cross-validation training fold (no leakage):
    block(s) -> StandardScaler -> PCA(keep 95 % of the variance) -> learner
  LiDAR_PCA   : one PCA on all finite LiDAR features
  VNIR_PCA    : one PCA on all finite VNIR features (NaN cells median-filled beforehand by the caller)
  Fused_PCA   : separate PCA per modality (block-wise), scores concatenated -> learner   (late fusion of the two modalities)
  Fused_PCA_joint : one PCA on the concatenated LiDAR + VNIR features                     (early fusion)
Extra pass-through columns (e.g. trial / stage flags) are appended untouched.
Learners: Ridge, SVR (RBF, grid), GPR (isotropic RBF), RandomForest, ExtraTrees, MLP, XGBoost / LightGBM when installed.
"""
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
try: import xgboost as xgb
except Exception: xgb = None
try: import lightgbm as lgb
except Exception: lgb = None
PCA_VAR = 0.95

def pca_preprocessor(blocks, n_passthrough=0):
    """blocks: list of (name, column index list) of the input matrix; a block with one column is standardised only."""
    trs = []
    for name, idx in blocks:
        if len(idx) >= 2: trs.append((name, make_pipeline(StandardScaler(), PCA(n_components=PCA_VAR, svd_solver="full")), list(idx)))
        elif len(idx) == 1: trs.append((name, StandardScaler(), list(idx)))
    return ColumnTransformer(trs, remainder="passthrough")

def pca_learners(pre, svr_C, svr_eps):
    """dict name -> estimator; every estimator starts with a clone of the PCA preprocessor"""
    def pipe(est): return Pipeline([("pre", clone(pre)), ("m", est)])
    L = {"Ridge": pipe(RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "SVR_rbf": GridSearchCV(pipe(SVR()), {"m__C": svr_C, "m__epsilon": svr_eps, "m__gamma": ["scale", 0.03, 0.1]}, cv=5, scoring="neg_root_mean_squared_error"),
         "GPR": pipe(GaussianProcessRegressor(ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0)),
         "RandomForest": pipe(RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.7, random_state=0, n_jobs=-1)),
         "ExtraTrees": pipe(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=0, n_jobs=-1)),
         "MLP": pipe(MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=3000, early_stopping=True, random_state=0))}
    if xgb is not None: L["XGBoost"] = pipe(xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0))
    if lgb is not None: L["LightGBM"] = pipe(lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, random_state=0, verbose=-1))
    return L

def pca_blocks(family, n_lidar, n_vnir, n_bands=0):
    """column-index blocks for a family given a matrix laid out as [lidar cols | vnir index cols | spectral band cols | passthrough]"""
    li = list(range(n_lidar)); vi = list(range(n_lidar, n_lidar + n_vnir)); bi = list(range(n_lidar + n_vnir, n_lidar + n_vnir + n_bands))
    if family == "LiDAR_PCA": return [("lidar", li)]
    if family == "VNIR_PCA": return [("vnir", vi)]
    if family == "Fused_PCA": return [("lidar", li), ("vnir", vi)]
    if family == "Fused_PCA_joint": return [("joint", li + vi)]
    if family == "VNIR_bands_PCA": return [("bands", bi)]                     # PCA on the valid spectral bands (log, SNV per plot)
    if family == "Fused_bands_PCA": return [("lidar", li), ("bands", bi)]
    raise ValueError(family)

SB_PREFIX = "sb_"
def spectra_matrix(SP, min_frac=0.95, excluded=((0, 415), (755, 770), (928, 962))):
    """Vegetation mean spectra (Plot_ID + one column per wavelength, NaN = band unusable) -> DataFrame indexed by Plot_ID
    of log10 reflectance after per-plot SNV, over the bands that are valid on >= min_frac of the plots and outside the
    excluded ranges; remaining NaN cells are filled with the band median (over plots). Columns 'sb_<nm>'."""
    import pandas as pd
    S = SP.set_index("Plot_ID"); wl = np.array([float(c) for c in S.columns])
    ex = np.zeros(len(wl), bool)
    for lo, hi in excluded: ex |= (wl >= lo) & (wl <= hi)
    ok = (S.notna().mean(0).to_numpy() >= min_frac) & ~ex
    X = S.loc[:, ok].to_numpy(float)
    X = np.log10(np.clip(X, 1e-4, None))
    mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True) + 1e-9; X = (X - mu) / sd      # SNV per plot on its valid bands
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0, keepdims=True))
    return pd.DataFrame(X, index=S.index, columns=[f"{SB_PREFIX}{c}" for c in S.columns[ok]])

def pca_summary(pre_fitted):
    """components kept and variance explained per block of a fitted preprocessor"""
    out = {}
    for name, tr, _ in pre_fitted.transformers_:
        if hasattr(tr, "steps") and isinstance(tr.steps[-1][1], PCA):
            p = tr.steps[-1][1]; out[name] = dict(n_components=int(p.n_components_), explained_variance=float(p.explained_variance_ratio_.sum()))
    return out
