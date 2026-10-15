"""
Plant-height models: per-plot LiDAR (+ VNIR, + RGB) features -> measured plant height (PhenoFuse 4.0).

Validated on a wheat trial whose ruler height was measured at maturity; error metrics refer to the maturity flight
only (a flight without a ruler reading of its own gets a predicted height, not an error score):

  cal_linear  : ruler = a + b x cth_p95.  The reference model (LOOCV R2 0.23, RMSE 4.5 cm on the validation flight).
  struct_*    : 7 canopy-structure features (cth_p95, tip thinness, cth_cover, cover_frac, roughness, vertical
                spread, point density): ridge / random forest / XGBoost
  lidar_v3_*  : the 22 canopy-only features of the per-plot classification, nested top-10 selection inside each fold
  fused_v3_*  : LiDAR v3 + VNIR features
  *_pca_*     : PCA sets (one PCA per modality, 95 % of the variance, fitted inside each fold) - LiDAR, VNIR, RGB,
                LiDAR + VNIR (late / joint), LiDAR + RGB, LiDAR + VNIR + RGB, reflectance bands

Every model belongs to a set of modalities (HEIGHT_MODEL_MODALITIES); the tab offers the models whose modalities are
ticked. Heights are modelled in centimetres. Pooled multi-flight training is group-aware (cv_utils.make_splits).

Public API
----------
HEIGHT_MODELS                       ordered dict key -> (label, feature list | "PCA:<set>", learner)
HEIGHT_MODEL_MODALITIES, models_for(modalities)
prepare_features(df, vnir_df, rgb_df) adds tip_thin / vspread when possible, converts m -> cm
fit_height_models(df, gt, keys, cv, ...) -> (results, predictions DataFrame, fitted dict)
save_height_model / load_height_model / apply_height_model
results_table(results)              plain-text table
"""

from __future__ import annotations
import os
import json
import numpy as np

STRUCT_FEATURES = ["cth_p95", "tip_thin", "cth_cover", "cover_frac", "roughness", "vspread", "pt_density"]
from .lidar_features import LIDAR_V3_CORE
from .vnir_features import VNIR_V3_CORE
from .rgb_features import RGB_CORE
from .pca_families import numeric_feature_columns, split_blocks, pca_preprocessor, make_pca_learner, pca_summary, spectra_matrix, vnir_block
from .cv_utils import make_splits

def _isnum(series):
    from pandas.api.types import is_numeric_dtype
    return bool(is_numeric_dtype(series))


HEIGHT_MODELS = {
    "cal_linear":   ("Linear calibration on cth_p95 (reference; ruler = a + b x cth_p95)", ["cth_p95"], "linear"),
    "cal_linear_H95": ("Linear calibration on H95 (canopy-only 95th percentile)", ["H95"], "linear"),
    "struct_ridge": ("Ridge on 7 canopy-structure features", STRUCT_FEATURES, "ridge"),
    "struct_rf":    ("Random forest on 7 canopy-structure features (transfers between dates)", STRUCT_FEATURES, "rf"),
    "struct_xgb":   ("XGBoost on 7 canopy-structure features, monotone in cth_p95", STRUCT_FEATURES, "xgb"),
    "lidar_v3_pls":   ("v3 LiDAR core (22 canopy features) - PLS, nested selection", LIDAR_V3_CORE, "pls_sel"),
    "lidar_v3_ridge": ("v3 LiDAR core - Ridge, nested selection", LIDAR_V3_CORE, "ridge_sel"),
    "lidar_v3_gpr":   ("v3 LiDAR core - Gaussian process (best date-to-date transfer with a target offset)", LIDAR_V3_CORE, "gpr_sel"),
    "vnir_v3_pls":    ("VNIR features only - PLS, nested selection", VNIR_V3_CORE, "pls_sel"),
    "rgb_pls":        ("RGB features only - PLS, nested selection", RGB_CORE, "pls_sel"),
    "fused_v3_pls":   ("v3 LiDAR + VNIR - PLS, nested selection", LIDAR_V3_CORE + VNIR_V3_CORE, "pls_sel"),
    "fused_v3_ridge": ("v3 LiDAR + VNIR - Ridge, nested selection", LIDAR_V3_CORE + VNIR_V3_CORE, "ridge_sel"),
    "fused_v3_rf":    ("v3 LiDAR + VNIR - Random forest, nested selection", LIDAR_V3_CORE + VNIR_V3_CORE, "rf_sel"),
    "lidar_rgb_gpr":  ("v3 LiDAR + RGB - Gaussian process, nested selection", LIDAR_V3_CORE + RGB_CORE, "gpr_sel"),
    "fused3_ridge":   ("v3 LiDAR + VNIR + RGB - Ridge, nested selection", LIDAR_V3_CORE + VNIR_V3_CORE + RGB_CORE, "ridge_sel"),
    "lidar_v3_pca_ridge":      ("LiDAR PCA (all LiDAR features -> PCA 95 %) - Ridge", "PCA:lidar", "pca_ridge"),
    "lidar_v3_pca_gpr":        ("LiDAR PCA - Gaussian process", "PCA:lidar", "pca_gpr"),
    "vnir_pca_ridge":          ("VNIR PCA (indices / bands per the VNIR mode) - Ridge", "PCA:vnir", "pca_ridge"),
    "rgb_pca_ridge":           ("RGB PCA - Ridge", "PCA:rgb", "pca_ridge"),
    "bands_pca_ridge":         ("Reflectance-band PCA (mean vegetation spectrum) - Ridge", "PCA:bands", "pca_ridge"),
    "fused_v3_pca_ridge":      ("LiDAR + VNIR PCA (one PCA per modality, late fusion) - Ridge", "PCA:fused", "pca_ridge"),
    "fused_v3_pca_gpr":        ("LiDAR + VNIR PCA (one PCA per modality) - Gaussian process", "PCA:fused", "pca_gpr"),
    "fused_v3_pca_rf":         ("LiDAR + VNIR PCA (one PCA per modality) - Random forest", "PCA:fused", "pca_rf"),
    "fused_v3_pcajoint_ridge": ("LiDAR + VNIR joint PCA (one PCA on both, early fusion) - Ridge", "PCA:joint", "pca_ridge"),
    "fused_bands_pca_gpr":     ("LiDAR PCA + reflectance-band PCA - Gaussian process", "PCA:fused_bands", "pca_gpr"),
    "lidar_rgb_pca_ridge":     ("LiDAR + RGB PCA (late fusion) - Ridge", "PCA:lidar_rgb", "pca_ridge"),
    "vnir_rgb_pca_ridge":      ("VNIR + RGB PCA (late fusion) - Ridge", "PCA:vnir_rgb", "pca_ridge"),
    "fused3_pca_ridge":        ("LiDAR + VNIR + RGB PCA (late fusion) - Ridge", "PCA:fused3", "pca_ridge"),
    "fused3_pca_gpr":          ("LiDAR + VNIR + RGB PCA (late fusion) - Gaussian process", "PCA:fused3", "pca_gpr"),
}
HEIGHT_MODEL_MODALITIES = {
    "cal_linear": {"lidar"}, "cal_linear_H95": {"lidar"}, "struct_ridge": {"lidar"}, "struct_rf": {"lidar"}, "struct_xgb": {"lidar"},
    "lidar_v3_pls": {"lidar"}, "lidar_v3_ridge": {"lidar"}, "lidar_v3_gpr": {"lidar"}, "vnir_v3_pls": {"vnir"}, "rgb_pls": {"rgb"},
    "fused_v3_pls": {"lidar", "vnir"}, "fused_v3_ridge": {"lidar", "vnir"}, "fused_v3_rf": {"lidar", "vnir"}, "lidar_rgb_gpr": {"lidar", "rgb"},
    "fused3_ridge": {"lidar", "vnir", "rgb"}, "lidar_v3_pca_ridge": {"lidar"}, "lidar_v3_pca_gpr": {"lidar"}, "vnir_pca_ridge": {"vnir"},
    "rgb_pca_ridge": {"rgb"}, "bands_pca_ridge": {"vnir"}, "fused_v3_pca_ridge": {"lidar", "vnir"}, "fused_v3_pca_gpr": {"lidar", "vnir"},
    "fused_v3_pca_rf": {"lidar", "vnir"}, "fused_v3_pcajoint_ridge": {"lidar", "vnir"}, "fused_bands_pca_gpr": {"lidar", "vnir"},
    "lidar_rgb_pca_ridge": {"lidar", "rgb"}, "vnir_rgb_pca_ridge": {"vnir", "rgb"}, "fused3_pca_ridge": {"lidar", "vnir", "rgb"}, "fused3_pca_gpr": {"lidar", "vnir", "rgb"}}
DEFAULT_HEIGHT_KEYS = ("cal_linear", "struct_ridge", "struct_rf", "lidar_v3_pls", "lidar_v3_gpr", "fused_v3_pls", "lidar_v3_pca_ridge", "fused_v3_pca_ridge", "fused3_pca_ridge")
_PCA_LEARNER = {"pca_ridge": "Ridge", "pca_gpr": "GPR", "pca_rf": "RandomForest"}
_PCA_SETS = {"lidar": ["lidar"], "vnir": ["vnir"], "rgb": ["rgb"], "bands": ["bands"], "fused": ["lidar", "vnir"], "joint": ["joint:lidar+vnir"],
             "fused_bands": ["lidar", "bands"], "lidar_rgb": ["lidar", "rgb"], "vnir_rgb": ["vnir", "rgb"], "fused3": ["lidar", "vnir", "rgb"]}


def models_for(modalities):
    mods = {m.lower() for m in modalities}
    return [k for k in HEIGHT_MODELS if HEIGHT_MODEL_MODALITIES[k] <= mods]


def _pca_feature_set(kind, m, vnir_df=None, rgb_df=None, vnir_cols=None, rgb_cols=None, vnir_mode="indices"):
    """Input columns and blocks of a PCA set on the merged metrics table."""
    allf = numeric_feature_columns(m, exclude=("height_cm", "height_measured_cm"))
    vcols = vnir_cols if vnir_cols is not None else ([c for c in vnir_df.columns if c != "Plot_ID"] if vnir_df is not None else None)
    rcols = rgb_cols if rgb_cols is not None else ([c for c in rgb_df.columns if c != "Plot_ID"] if rgb_df is not None else None)
    b = split_blocks(m, allf, vcols, rcols)
    blocks = []
    for spec in _PCA_SETS[kind]:
        if spec.startswith("joint:"):
            cols = []
            for part in spec[6:].split("+"):
                cols += vnir_block(b, vnir_mode) if part == "vnir" else list(b[part])
            blocks.append(("joint", cols))
        elif spec == "vnir":
            blocks.append(("vnir", vnir_block(b, vnir_mode)))
        else:
            blocks.append((spec, list(b[spec])))
    if any(len(c) < 2 for _, c in blocks):
        return [], None
    cols = [c for _, cs in blocks for c in cs]; pos = {c: i for i, c in enumerate(cols)}
    return cols, [(n, [pos[c] for c in cs]) for n, cs in blocks]


_CM_COLS = ("cth_p95", "cth_max", "cth_p90", "cth_p99", "cth_mean", "cth_pt_p99", "h_p99", "h_median", "h_p95", "h_max", "h_mean")
_V3_CM = ("H95", "H99", "H999", "H_topN_mean", "H_mean", "H50", "H_sd", "cth_p95_v3", "cth_p99_v3", "tip_thin_v3", "roughness_v3", "H25", "H75", "H90", "H_max", "H_iqr", "Hall_mean", "Hall_p95", "Hall_p99", "cth_p90_v3", "cth_max_v3", "cth_mean_v3")


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False


def prepare_features(df, vnir_df=None, rgb_df=None):
    """Return a copy with heights in cm and the derived structure features where their inputs exist (+ VNIR / RGB
    columns merged on Plot_ID). Nothing is required here: a model whose inputs are missing is skipped with a message."""
    d = df.copy()
    for extra in (vnir_df, rgb_df):
        if extra is not None:
            v = extra[[c for c in extra.columns if c == "Plot_ID" or c not in d.columns]]
            d = d.merge(v, on="Plot_ID", how="left")
    for c in list(_CM_COLS) + list(_V3_CM):
        if c in d.columns and _isnum(d[c]) and np.nanmax(np.abs(d[c].astype(float))) < 20:        # metres -> cm
            d[c] = d[c].astype(float) * 100.0
    if "cth_p95" not in d.columns and "cth_p95_v3" in d.columns:
        d["cth_p95"] = d["cth_p95_v3"]
    if "cth_max" in d.columns and "cth_p95" in d.columns:
        d["tip_thin"] = d["cth_max"] - d["cth_p95"]
    elif "tip_thin" not in d.columns and "tip_thin_v3" in d.columns:
        d["tip_thin"] = d["tip_thin_v3"]
    if "h_p99" in d.columns and "h_median" in d.columns:
        d["vspread"] = d["h_p99"] - d["h_median"]
    elif "H99" in d.columns and "H50" in d.columns:
        d["vspread"] = d["H99"] - d["H50"]
    if "pt_density" not in d.columns and "pts_per_m2" in d.columns:
        d["pt_density"] = d["pts_per_m2"]
    if "cover_frac" not in d.columns and "cover_5cm" in d.columns:
        d["cover_frac"] = d["cover_5cm"]
    if "cth_cover" not in d.columns and "cth_cover_v3" in d.columns:
        d["cth_cover"] = d["cth_cover_v3"]
    if "roughness" not in d.columns and "roughness_v3" in d.columns:
        d["roughness"] = d["roughness_v3"]
    return d


def _gt_height_cm(gt):
    """Pick the height column of a ground-truth table and return cm."""
    cols = [c for c in gt.columns if c != "Plot_ID" and ("height" in c.lower() or c.lower().startswith("h_"))]
    if not cols:
        cols = [c for c in gt.columns if c not in ("Plot_ID", "flight", "Plot", "Variety") and _isnum(gt[c])]
    if not cols:
        raise ValueError("ground-truth CSV needs Plot_ID and a height column (height_cm or height_m)")
    c = cols[0]
    v = gt[c].astype(float)
    if c.lower().endswith("_m") or (v.dropna().abs().max() < 5):
        v = v * 100.0
    return v, c


class _SelWrap:
    """learner with random-forest top-k feature selection fitted on the training data only"""
    def __init__(self, base, k=10): self.base = base; self.k = k; self.idx = None
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone
        self.idx = list(range(X.shape[1])) if X.shape[1] <= self.k else list(np.argsort(RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y).feature_importances_)[::-1][:self.k])
        self.model = clone(self.base).fit(X[:, self.idx], y); return self
    def predict(self, X): return np.asarray(self.model.predict(X[:, self.idx])).ravel()
    def get_params(self, deep=True): return {"base": self.base, "k": self.k}
    def set_params(self, **p):
        for k_, v in p.items(): setattr(self, k_, v)
        return self


def _make(learner, feats):
    from sklearn.pipeline import make_pipeline
    if learner.endswith("_sel"):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.linear_model import RidgeCV as _R
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        from sklearn.ensemble import RandomForestRegressor
        k = min(10, len(feats))
        base = {"pls_sel": GridSearchCV(make_pipeline(_SS(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(9, k + 1)))}, cv=5, scoring="neg_root_mean_squared_error"),
                "ridge_sel": make_pipeline(_SS(), _R(alphas=np.logspace(-2, 3, 30))),
                "gpr_sel": make_pipeline(_SS(), GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(k) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0)),
                "rf_sel": RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1)}[learner]
        return _SelWrap(base, k)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, RidgeCV
    if learner == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if learner == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if learner == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=-1)
    if learner == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=400, learning_rate=0.04, max_depth=3, min_child_weight=5, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=2.0, random_state=0, n_jobs=4, verbosity=0,
                                monotone_constraints="(" + ",".join("1" if f == "cth_p95" else "0" for f in feats) + ")")
    raise ValueError(learner)


def _metrics(pred, y):
    e = pred - y
    r = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float("nan")
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.abs(e).mean()), bias=float(e.mean()), r=r,
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), slope=float(np.polyfit(pred, y, 1)[0]) if np.std(pred) > 0 else float("nan"),
                RMSE=float(np.sqrt(np.mean(e ** 2))))


def fit_height_models(df, gt, keys, cv: str = "rkf10", progress_cb=None, vnir_df=None, rgb_df=None, groups=None, spectra=None,
                      vnir_cols=None, rgb_cols=None, vnir_mode="indices", excluded=None, band_transform="snv"):
    """Fit + validate the chosen models. band_transform: "snv" | "raw" for the reflectance band columns.

    df      : feature table (Plot_ID + LiDAR features; VNIR / RGB may already be merged or given as vnir_df / rgb_df)
    gt      : DataFrame Plot_ID + height column (cm, or m if the name ends in _m / values < 5)
    keys    : list of HEIGHT_MODELS keys
    cv      : 'rkf10' | 'kfold5' | 'loo' | 'none'; groups: optional group id per row of df (pooled flights)
    spectra : mean vegetation spectra table for the band PCA sets / VNIR mode 'bands'
    Returns (results list, predictions DataFrame, fitted {key: (model, feats)}).
    """
    import pandas as pd
    from sklearn.base import clone
    d = prepare_features(df, vnir_df, rgb_df)
    if groups is not None:
        d["_group"] = d["Plot_ID"].map(dict(zip(df["Plot_ID"], groups)))
    if spectra is not None and (vnir_mode in ("bands", "both") or any(HEIGHT_MODELS[k][1] in ("PCA:bands", "PCA:fused_bands") for k in keys)):
        sb = spectra_matrix(spectra, excluded=excluded, transform=band_transform)
        if len(sb.columns):
            d = d.merge(sb, left_on="Plot_ID", right_index=True, how="left")
    if vnir_mode == "bands" and vnir_cols:
        keep_qc = ("vnir_qc_ok", "vnir_reflectance_ok", "frac_in_cube", "fcover_vnir", "red_ok")
        d = d.drop(columns=[c for c in vnir_cols if c in d.columns and c not in keep_qc])
    y_all, gt_col = _gt_height_cm(gt)
    g = pd.DataFrame({"Plot_ID": gt["Plot_ID"], "height_cm": y_all}).dropna()
    m = d.merge(g, on="Plot_ID", how="inner").reset_index(drop=True)
    if len(m) < 10:
        raise ValueError(f"only {len(m)} plots have both features and a height - need at least 10")
    y = m["height_cm"].to_numpy(float)
    pred_df = pd.DataFrame({"Plot_ID": m["Plot_ID"], "height_measured_cm": y})
    for c in ("flight", "Plot_ID_orig", "cth_p95"):
        if c in m.columns:
            pred_df[c if c != "cth_p95" else "cth_p95_cm"] = m[c].values
    results, fitted = [], {}
    grp = m["_group"].to_numpy() if "_group" in m.columns else None
    splits = make_splits(len(y), cv, grp)
    for i, k in enumerate(keys):
        label, feats, learner = HEIGHT_MODELS[k]
        if learner == "xgb" and not xgboost_available():
            results.append(dict(key=k, label=label, status="skipped - xgboost not installed")); continue
        blocks = None
        if isinstance(feats, str) and feats.startswith("PCA:"):
            feats, blocks = _pca_feature_set(feats[4:], m, vnir_df, rgb_df, vnir_cols, rgb_cols, vnir_mode)
            if not feats:
                results.append(dict(key=k, label=label, status="skipped - no usable features for this PCA set (needs the VNIR / RGB features or the spectra table)")); continue
        else:
            feats = list(feats)
        miss = [f for f in feats if f not in m.columns]
        if miss and blocks is None:
            avail = [f for f in feats if f in m.columns]
            if learner.endswith("_sel") and len(avail) >= 3:
                feats = avail                                   # nested-selection models work on the features that exist
            else:
                results.append(dict(key=k, label=label, status="skipped - missing features: " + ", ".join(miss[:6]) + " (tick 'LiDAR v3' on the LiDAR Features tab / run the VNIR and RGB tabs)")); continue
        # a VNIR / RGB model only uses plots whose VNIR / RGB passed QC
        sel = np.ones(len(m), bool)
        if "vnir" in HEIGHT_MODEL_MODALITIES[k] and "vnir_qc_ok" in m.columns:
            sel &= m["vnir_qc_ok"].fillna(0).to_numpy() == 1
        if "rgb" in HEIGHT_MODEL_MODALITIES[k] and "rgb_qc_ok" in m.columns:
            sel &= m["rgb_qc_ok"].fillna(0).to_numpy() == 1
        X = m.loc[sel, feats].to_numpy(float); ym = y[sel]
        if len(ym) < 10:
            results.append(dict(key=k, label=label, status="skipped - fewer than 10 plots pass the modality QC")); continue
        if not np.isfinite(X).all():
            X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        sp = splits if sel.all() else make_splits(len(ym), cv, grp[sel] if grp is not None else None)
        model = make_pca_learner(_PCA_LEARNER[learner], pca_preprocessor(blocks)) if blocks is not None else _make(learner, feats)
        fit_model = clone(model).fit(X, ym)
        pfit = np.asarray(fit_model.predict(X)).ravel()
        res = dict(key=k, label=label, n=int(len(ym)), features=feats, status="ok", fit=_metrics(pfit, ym), family=k, learner=learner)
        if blocks is not None:
            res["pca"] = pca_summary(fit_model); res["blocks"] = {name: [feats[i] for i in idx] for name, idx in blocks}
        pred_df.loc[sel, f"pred_{k}"] = pfit
        if sp:
            ps = np.zeros(len(ym)); cnt = np.zeros(len(ym))
            for tr, te in sp:
                mm = clone(model).fit(X[tr], ym[tr]); ps[te] += np.asarray(mm.predict(X[te])).ravel(); cnt[te] += 1
            pcv = ps / np.maximum(cnt, 1)
            res["cv"] = _metrics(pcv, ym); res["cv_mode"] = cv
            pred_df.loc[sel, f"predcv_{k}"] = pcv
        if learner == "linear":
            b = np.polyfit(X[:, 0], ym, 1)
            res["calibration"] = {"intercept_cm": float(b[1]), "slope": float(b[0]), "predictor": feats[0]}
        fitted[k] = (fit_model, feats)
        results.append(res)
        if progress_cb:
            progress_cb(int(100 * (i + 1) / len(keys)), f"{label} done")
    pred_df["gt_column"] = gt_col
    return results, pred_df, fitted


def results_table(results) -> str:
    lines = [f"{'model':62s} {'n':>4s} {'cvRMSE':>7s} {'cvMAE':>6s} {'cv r':>5s} {'cvR2':>5s} {'slope':>5s} {'fitRMSE':>7s} {'fitR2':>5s}"]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"{r['label'][:62]:62s} {r.get('status')}"); continue
        c = r.get("cv"); f = r["fit"]
        cvs = f"{c['rmse']:7.2f} {c['mae']:6.2f} {c['r']:5.2f} {c['R2']:5.2f} {c['slope']:5.2f}" if c else f"{'-':>7s} {'-':>6s} {'-':>5s} {'-':>5s} {'-':>5s}"
        lines.append(f"{r['label'][:62]:62s} {r['n']:4d} {cvs} {f['rmse']:7.2f} {f['R2']:5.2f}")
    return "\n".join(lines)


def save_height_model(path, model, feats, meta=None, table=None):
    """table: the prepared training feature table -> its medians are stored so a feature unusable on a new flight can be filled."""
    import joblib
    from .ensembles import feature_medians, feature_ranges
    joblib.dump({"model": model, "features": feats, "units": "cm", "meta": meta or {}, "band_transform": (meta or {}).get("band_transform", "snv"),
                 "feature_medians": feature_medians(table, feats) if table is not None else {},
                 "feature_ranges": feature_ranges(table, feats) if table is not None else {}}, path)
    return path


def load_height_model(path):
    import joblib
    return joblib.load(path)


def apply_height_model(df, saved, offset_cm: float = 0.0, vnir_df=None, spectra=None, folds="all", rgb_df=None, excluded=None):
    """Predict height (cm) for every plot of a feature table with a saved model or a saved ensemble.
    offset_cm is added to the prediction (per-flight correction from reference targets or a few measured plots).
    vnir_df / rgb_df / spectra: the flight's VNIR features, RGB features and spectra table, needed by models that use them."""
    import pandas as pd
    from .ensembles import predict_saved, required_features, fill_missing, range_check, band_transform
    d = prepare_features(df, vnir_df, rgb_df)
    need = required_features(saved, folds); filled = []
    sb = [f for f in need if f.startswith("sb_")]
    if spectra is not None and sb:
        # exactly the bands the model was trained on (the flight's usable-band set may differ, e.g. clipped blue bands),
        # with the transform it was trained with
        sm = spectra_matrix(spectra, excluded=excluded, bands=sb, transform=band_transform(saved)); filled = list(sm.attrs.get("filled_bands", []))
        d = d.merge(sm, left_on="Plot_ID", right_index=True, how="left")
    d, filled_feats, replaced = fill_missing(saved, d, need)
    pred, ok = predict_saved(saved, d, folds)
    pred = pred + offset_cm
    out = pd.DataFrame({"Plot_ID": d["Plot_ID"], "height_pred_cm": pred, "offset_cm": offset_cm, "model_ok": ok.astype(int)})
    if "cth_p95" in d.columns:
        out.insert(1, "cth_p95_cm", d["cth_p95"].values)
    out.attrs["filled_bands"] = filled; out.attrs["filled_features"] = filled_feats; out.attrs["replaced_features"] = replaced
    out.attrs["unusable_features"] = [f for f in need if f in d.columns and not f.startswith("sb_") and d[f].isna().all()]
    out.attrs["missing_features"] = [f for f in dict.fromkeys(need) if f not in d.columns]
    out.attrs["out_of_range"] = range_check(saved, d, [f for f in need if f not in replaced])
    out.attrs["n_range_checked"] = len([f for f in dict.fromkeys(need) if f in saved.get("feature_ranges", {})]) if isinstance(saved, dict) else 0
    return out
