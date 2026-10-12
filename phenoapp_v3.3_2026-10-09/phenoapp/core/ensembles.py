"""
Saved model files -> predictions, for single models and for the top-three mean ensembles (PhenoApp v3.3.2).

A single saved model is {"model", "features", ...}. A mean ensemble is {"type": "mean_ensemble", "members": [{"family",
"learner", "model", "features"}, ...], "nested_cv": {...}, ...}: the three feature families chosen by nested
cross-validation, each fitted on all plots; the prediction is the mean of the members. A plot is predicted when every
input of every member is finite (model_ok = 1).
"""
from __future__ import annotations
import numpy as np


def is_ensemble(saved) -> bool:
    return isinstance(saved, dict) and saved.get("type") == "mean_ensemble" and "members" in saved


def required_features(saved):
    if is_ensemble(saved):
        out = []
        for m in saved["members"]:
            out += [f for f in m["features"] if f not in out]
        return out
    return list(saved["features"])


def predict_saved(saved, d):
    """d: DataFrame with the input columns. Returns (pred array with NaN where inputs are missing, ok bool array)."""
    members = saved["members"] if is_ensemble(saved) else [{"model": saved["model"], "features": saved["features"]}]
    missing = [f for f in required_features(saved) if f not in d.columns]
    if missing:
        raise ValueError("input table lacks " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    ok = np.ones(len(d), bool); preds = []
    for m in members:
        X = d[m["features"]].to_numpy(float); ok &= np.isfinite(X).all(axis=1)
    for m in members:
        p = np.full(len(d), np.nan); X = d[m["features"]].to_numpy(float)
        if ok.any():
            p[ok] = np.asarray(m["model"].predict(X[ok])).ravel()
        preds.append(p)
    return np.mean(preds, axis=0), ok


def describe(saved) -> str:
    if is_ensemble(saved):
        cv = saved.get("nested_cv", {})
        return ("mean of " + ", ".join(f"{m['family']}/{m['learner']}" for m in saved["members"]) +
                (f" (nested CV R2 {cv['R2']:.2f}, RMSE {cv['RMSE']:.3g})" if cv else ""))
    return f"single model, {len(saved['features'])} inputs"
