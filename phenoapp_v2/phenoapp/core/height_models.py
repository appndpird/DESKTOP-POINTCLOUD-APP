"""
Plant-height models: per-plot LiDAR features -> measured plant height.

Validated on Muresk NUE 25NO43 (128 wheat plots, two flights, ruler at maturity;
see Height Prediction Models/ANALYSIS.md):

  cal_linear  : ruler = a + b x cth_p95.  Within a trial this is at the noise
                floor of the ruler (LOOCV 3.8 cm); the reference model.
  struct_ridge: ridge on 7 canopy-structure features (cth_p95, tip thinness,
                cth_cover, cover_frac, roughness, vertical spread, point
                density); best within-flight (3.7 cm) but transfers poorly.
  struct_rf   : random forest on the same 7 features; the model that transfers
                between dates (5.3 cm raw, 4.9 cm with a per-flight offset).
  struct_xgb  : XGBoost on the same 7 features, monotone in cth_p95 (optional;
                needs the xgboost package).

Features come from the Traits tab metrics CSV; the cth_* traits must have
been computed (Canopy-top height row). Heights are modelled in centimetres.

Public API
----------
HEIGHT_MODELS                       ordered dict key -> (label, feature list, learner)
prepare_features(df)                adds tip_thin / vspread, converts m -> cm
fit_height_models(df, gt, keys, cv) -> (results, predictions DataFrame, fitted dict)
save_height_model / load_height_model / apply_height_model
results_table(results)              plain-text table
"""

from __future__ import annotations
import os
import json
import numpy as np

STRUCT_FEATURES = ["cth_p95", "tip_thin", "cth_cover", "cover_frac", "roughness", "vspread", "pt_density"]
HEIGHT_MODELS = {
    "cal_linear":   ("Linear calibration on cth_p95 (reference; ruler = a + b x cth_p95)", ["cth_p95"], "linear"),
    "struct_ridge": ("Ridge on 7 canopy-structure features", STRUCT_FEATURES, "ridge"),
    "struct_rf":    ("Random forest on 7 canopy-structure features (transfers between dates)", STRUCT_FEATURES, "rf"),
    "struct_xgb":   ("XGBoost on 7 canopy-structure features, monotone in cth_p95", STRUCT_FEATURES, "xgb"),
}
_CM_COLS = ("cth_p95", "cth_max", "cth_p90", "cth_p99", "cth_mean", "cth_pt_p99", "h_p99", "h_median", "h_p95", "h_max", "h_mean")
REQUIRED_RAW = ["cth_p95", "cth_max", "cth_cover", "cover_frac", "roughness", "h_p99", "h_median", "pt_density"]


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False


def prepare_features(df):
    """Return a copy with heights in cm and the derived structure features."""
    d = df.copy()
    missing = [c for c in REQUIRED_RAW if c not in d.columns]
    if missing:
        raise ValueError("metrics CSV lacks " + ", ".join(missing) + " - run Compute Traits with the canopy-top "
                         "(cth_*) heights, cover, roughness, point density and percentile heights ticked")
    for c in _CM_COLS:
        if c in d.columns and d[c].abs().max() < 20:        # metres -> cm (metrics are stored in m)
            d[c] = d[c] * 100.0
    d["tip_thin"] = d["cth_max"] - d["cth_p95"]
    d["vspread"] = d["h_p99"] - d["h_median"]
    return d


def _gt_height_cm(gt):
    """Pick the height column of a ground-truth table and return cm."""
    cols = [c for c in gt.columns if c != "Plot_ID" and ("height" in c.lower() or c.lower().startswith("h_"))]
    if not cols:
        cols = [c for c in gt.columns if c != "Plot_ID"]
    if not cols:
        raise ValueError("ground-truth CSV needs Plot_ID and a height column (height_cm or height_m)")
    c = cols[0]
    v = gt[c].astype(float)
    if c.lower().endswith("_m") or (v.dropna().abs().max() < 5):
        v = v * 100.0
    return v, c


def _make(learner, feats):
    from sklearn.pipeline import make_pipeline
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
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), slope=float(np.polyfit(pred, y, 1)[0]) if np.std(pred) > 0 else float("nan"))


def fit_height_models(df, gt, keys, cv: str = "loo", progress_cb=None):
    """Fit + validate the chosen models.

    df   : metrics table (Plot_ID + traits, from the Traits tab)
    gt   : DataFrame Plot_ID + height column (cm, or m if the name ends in _m / values < 5)
    keys : list of HEIGHT_MODELS keys
    cv   : 'loo' | 'kfold5' | 'none'
    Returns (results list, predictions DataFrame, fitted {key: (model, feats)}).
    """
    import pandas as pd
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneOut, KFold
    d = prepare_features(df)
    y_all, gt_col = _gt_height_cm(gt)
    g = pd.DataFrame({"Plot_ID": gt["Plot_ID"], "height_cm": y_all}).dropna()
    m = d.merge(g, on="Plot_ID", how="inner")
    if len(m) < 10:
        raise ValueError(f"only {len(m)} plots have both metrics and a height - need at least 10")
    y = m["height_cm"].to_numpy(float)
    pred_df = pd.DataFrame({"Plot_ID": m["Plot_ID"], "height_measured_cm": y, "cth_p95_cm": m["cth_p95"]})
    results, fitted = [], {}
    splitter = LeaveOneOut() if cv == "loo" else (KFold(5, shuffle=True, random_state=0) if cv == "kfold5" else None)
    for i, k in enumerate(keys):
        label, feats, learner = HEIGHT_MODELS[k]
        if learner == "xgb" and not xgboost_available():
            results.append(dict(key=k, label=label, status="skipped - xgboost not installed")); continue
        X = m[feats].to_numpy(float)
        if not np.isfinite(X).all():
            results.append(dict(key=k, label=label, status="skipped - NaN in features (check cth_ok / QC flags)")); continue
        model = _make(learner, feats)
        fit_model = clone(model).fit(X, y)
        pfit = fit_model.predict(X)
        res = dict(key=k, label=label, n=int(len(y)), features=feats, status="ok", fit=_metrics(pfit, y))
        pred_df[f"pred_{k}"] = pfit
        if splitter is not None:
            pcv = np.full(len(y), np.nan)
            for tr, te in splitter.split(X):
                mm = clone(model).fit(X[tr], y[tr]); pcv[te] = mm.predict(X[te])
            res["cv"] = _metrics(pcv, y); res["cv_mode"] = cv
            pred_df[f"predcv_{k}"] = pcv
        if learner == "linear":
            b = np.polyfit(m["cth_p95"].to_numpy(float), y, 1)
            res["calibration"] = {"intercept_cm": float(b[1]), "slope": float(b[0])}
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


def save_height_model(path, model, feats, meta=None):
    import joblib
    joblib.dump({"model": model, "features": feats, "units": "cm", "meta": meta or {}}, path)
    return path


def load_height_model(path):
    import joblib
    return joblib.load(path)


def apply_height_model(df, saved, offset_cm: float = 0.0):
    """Predict height (cm) for every plot of a metrics table with a saved model.
    offset_cm is added to the prediction (per-flight correction from reference
    targets or a few measured plots)."""
    import pandas as pd
    d = prepare_features(df)
    feats = saved["features"]
    X = d[feats].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    pred = np.full(len(d), np.nan)
    if ok.any():
        pred[ok] = saved["model"].predict(X[ok]) + offset_cm
    out = pd.DataFrame({"Plot_ID": d["Plot_ID"], "cth_p95_cm": d["cth_p95"], "height_pred_cm": pred, "offset_cm": offset_cm,
                        "model_ok": ok.astype(int)})
    return out
