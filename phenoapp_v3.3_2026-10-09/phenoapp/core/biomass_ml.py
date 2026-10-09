"""
Biomass (and generic trait) machine-learning on fused LiDAR v3 + VNIR v3 plot features (PhenoApp v3).

Feature families
  LiDAR_core : canopy-only percentiles, canopy-top metrics, cover, Pgap, LAI proxy, voxel volume, PVI,
               profile area, density, roughness, rumple, normalised intensity, 20 cm gap layers
  VNIR_core  : vegetation-masked red-edge / NDVI-family / water indices, reflectance bands, derivative
               features, spectral PCA scores, vegetation fraction
  Fused_core : both + CHM-weighted indices (index x cover, index x H95, NDRE x PVI)
  Fused_all  : every available v3 column
Learners     : Ridge, PLS, SVR (RBF), GPR, RandomForest, ExtraTrees, XGBoost*, LightGBM*  (*if installed)
Validation   : nested repeated k-fold (feature selection by random-forest importance inside each training fold),
               leave-one-out, or fit-only. Metrics: R2, RMSE, rRMSE, MAE, bias, r, accuracy (100 - MAPE).

Public API
----------
FAMILIES, LEARNERS, feature_family(df, family)
fit_trait_models(df, gt, target_col, families, learners, cv="rkf10", n_select=12, progress_cb=None)
    -> (results list, predictions DataFrame, fitted {(family, learner): (model, feats)})
results_table(results)
save_model / load_model / apply_model
"""

from __future__ import annotations
import os
import numpy as np

from .lidar_features import LIDAR_V3_CORE
from .vnir_features import VNIR_V3_CORE

LIDAR_CORE_BIOMASS = ["H95", "H99", "H_mean", "H50", "H_sd", "H_crr", "cth_p95", "cover_5cm", "Pgap", "LAI_proxy", "vox_volume_m3_per_m2", "PVI_m",
                      "profile_area_m", "canopy_pts_per_m2", "roughness", "rumple", "I_canopy_mean", "I_canopy_p90", "I_canopy_x_cover",
                      "Pgap_below_020cm", "Pgap_below_040cm", "Pgap_below_060cm", "dens_1", "dens_5", "dens_9"]
VNIR_CORE_BIOMASS = VNIR_V3_CORE + ["WDRVI_veg", "NDVI_nb", "NDRE740", "OSAVI", "refl_RE2", "refl_G", "red_depth"]
WEIGHTED = {"NDRE740_veg_x_cover": ("NDRE740_veg", "cover_5cm"), "OSAVI_veg_x_cover": ("OSAVI_veg", "cover_5cm"), "GNDVI_veg_x_cover": ("GNDVI_veg", "cover_5cm"),
            "NDVI_nb_veg_x_cover": ("NDVI_nb_veg", "cover_5cm"), "NDRE740_veg_x_H95": ("NDRE740_veg", "H95"), "OSAVI_veg_x_H95": ("OSAVI_veg", "H95"),
            "REP_veg_x_H95": ("REP_veg", "H95"), "LCI_veg_x_H95": ("LCI_veg", "H95"), "NDRE_x_PVI": ("NDRE740_veg", "PVI_m")}
FAMILIES = ["LiDAR_core", "VNIR_core", "Fused_core", "Fused_all"]
LEARNERS = ["Ridge", "PLS", "SVR_rbf", "GPR", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM"]
_NON_FEATURES = {"Plot_ID", "Plot", "B/R", "Bank", "Row", "Range", "region_mode", "region_area_m2", "area_m2", "qc_ok", "vnir_qc_ok", "red_ok", "ground_source",
                 "n_raw", "n_noise_removed", "ground_cells", "ground_rms_cm", "I_ring_median_raw", "region_px", "frac_in_cube", "veg_ndvi_threshold", "cth_ok"}


def add_weighted(df):
    d = df.copy()
    for k, (a, b) in WEIGHTED.items():
        if a in d.columns and b in d.columns:
            d[k] = d[a] * d[b]
    return d


def _available(cols, d):
    return [c for c in cols if c in d.columns and np.isfinite(d[c].astype(float)).mean() > 0.95 and d[c].astype(float).std() > 0]


def feature_family(df, family, exclude=()):
    d = add_weighted(df)
    d = d.drop(columns=[c for c in exclude if c in d.columns])
    if family == "LiDAR_core":
        return _available(LIDAR_CORE_BIOMASS, d)
    if family == "VNIR_core":
        return _available(VNIR_CORE_BIOMASS, d)
    if family == "Fused_core":
        return _available(LIDAR_CORE_BIOMASS + VNIR_CORE_BIOMASS + list(WEIGHTED), d)
    if family == "Fused_all":
        num = [c for c in d.columns if c not in _NON_FEATURES and not c.endswith(("_valid", "_measured")) and not c.startswith(("pred", "biomass", "height_cm", "ruler", "dm_", "fresh_"))
               and d[c].dtype != object]
        return _available(num, d)
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


def fit_trait_models(df, gt, target_col, families, learners, cv="rkf10", n_select=12, progress_cb=None, scale_y=None):
    """Nested cross-validated fit of `target_col` (from gt, joined on Plot_ID) on the chosen families x learners."""
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import RepeatedKFold, LeaveOneOut
    d = add_weighted(df).merge(gt[["Plot_ID", target_col]].dropna(), on="Plot_ID", how="inner")
    if "qc_ok" in d.columns:
        d = d[d.qc_ok == 1]
    d = d.reset_index(drop=True); y = d[target_col].to_numpy(float)
    if len(y) < 10:
        raise ValueError(f"only {len(y)} plots with features and {target_col}")
    sy = scale_y or float(y.std())
    pred = pd.DataFrame({"Plot_ID": d.Plot_ID, f"{target_col}_measured": y})
    results, fitted = [], {}
    total = len(families) * len(learners); done = 0
    for fam in families:
        feats = feature_family(d, fam, exclude=(target_col,))
        uses_vnir = fam in ("VNIR_core", "Fused_core", "Fused_all", "LiDAR+VNIR_height")
        if uses_vnir and "vnir_qc_ok" in d.columns:
            # plots whose VNIR failed QC (outside the cube, no vegetation, or a non-reflectance cube) are excluded
            # from EVERY family that uses VNIR features, not only from the VNIR-only family
            dm = d[d.vnir_qc_ok == 1]
        else:
            dm = d
        ym = dm[target_col].to_numpy(float); X = dm[feats].astype(float).fillna(dm[feats].astype(float).median()).to_numpy()
        k = n_select if len(feats) > n_select + 3 else None
        splits = list(RepeatedKFold(n_splits=10, n_repeats=5, random_state=0).split(X)) if cv == "rkf10" else (list(LeaveOneOut().split(X)) if cv == "loo" else [])
        for ln in learners:
            done += 1
            if not learner_available(ln) or not feats:
                results.append(dict(family=fam, learner=ln, status="skipped (package missing or no features)")); continue
            model = make_learner(ln, min(len(feats), k or len(feats)), sy)
            full = clone(model).fit(X[:, _select(X, ym, k)] if k else X, ym)
            idx_full = _select(X, ym, k) if k else list(range(len(feats)))
            pf = clone(model).fit(X[:, idx_full], ym).predict(X[:, idx_full]).ravel()
            res = dict(family=fam, learner=ln, status="ok", n=int(len(ym)), n_features=len(feats), features=[feats[i] for i in idx_full], fit=metrics(pf, ym), cv_mode=cv)
            col = f"pred_{fam}_{ln}"; pred.loc[pred.Plot_ID.isin(dm.Plot_ID), col] = pf
            if splits:
                ps = np.zeros(len(ym)); cnt = np.zeros(len(ym)); selc = np.zeros(len(feats))
                for tr, te in splits:
                    idx = _select(X[tr], ym[tr], k); selc[idx] += 1
                    mm = clone(model).fit(X[tr][:, idx], ym[tr]); ps[te] += mm.predict(X[te][:, idx]).ravel(); cnt[te] += 1
                pcv = ps / np.maximum(cnt, 1); res["cv"] = metrics(pcv, ym); res["selection_frequency"] = dict(zip(feats, (selc / len(splits)).round(2).tolist()))
                pred.loc[pred.Plot_ID.isin(dm.Plot_ID), f"predcv_{fam}_{ln}"] = pcv
            fitted[(fam, ln)] = (clone(model).fit(X[:, idx_full], ym), [feats[i] for i in idx_full])
            results.append(res)
            if progress_cb:
                progress_cb(int(100 * done / total), f"{fam} / {ln}" + (f": cv RMSE {res['cv']['RMSE']:.3g}" if "cv" in res else ""))
    return results, pred, fitted


def results_table(results, unit=""):
    lines = [f"{'family':12s} {'learner':12s} {'n':>4s} {'nfeat':>5s} {'cvR2':>6s} {'cvRMSE':>9s} {'rRMSE%':>7s} {'cvMAE':>9s} {'cv r':>5s} {'acc%':>6s} {'fitR2':>6s}"]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"{r['family']:12s} {r['learner']:12s} {r['status']}"); continue
        c = r.get("cv"); f = r["fit"]
        cvs = f"{c['R2']:6.3f} {c['RMSE']:9.3g} {c['rRMSE']:7.1f} {c['MAE']:9.3g} {c['r']:5.2f} {c['accuracy']:6.1f}" if c else " " * 49
        lines.append(f"{r['family']:12s} {r['learner']:12s} {r['n']:4d} {len(r['features']):5d} {cvs} {f['R2']:6.3f}")
    return "\n".join(lines)


def save_model(path, model, feats, meta=None):
    import joblib
    joblib.dump({"model": model, "features": feats, "meta": meta or {}}, path); return path


def load_model(path):
    import joblib
    return joblib.load(path)


def apply_model(df, saved, offset=0.0):
    import pandas as pd
    d = add_weighted(df); feats = saved["features"]
    missing = [f for f in feats if f not in d.columns]
    if missing:
        raise ValueError("metrics table lacks " + ", ".join(missing))
    X = d[feats].astype(float).to_numpy(); ok = np.isfinite(X).all(1); p = np.full(len(d), np.nan)
    if ok.any():
        p[ok] = saved["model"].predict(X[ok]).ravel() + offset
    return pd.DataFrame({"Plot_ID": d.Plot_ID, "predicted": p, "model_ok": ok.astype(int)})
