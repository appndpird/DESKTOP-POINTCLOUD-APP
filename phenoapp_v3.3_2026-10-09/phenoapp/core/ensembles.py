"""
Saved model files -> predictions (PhenoApp v3.3.4).

Three kinds of saved file:
  single model     {"model", "features", ...}
  mean ensemble    {"type": "mean_ensemble", "members": [{"family", "learner", "model", "features"}, ...], "nested_cv": {...}}
                   - the top three models of a category, chosen by nested cross-validation and fitted on all plots
  fold ensembles   {"type": "fold_ensembles", "folds": [{"fold", "test": {R2, RMSE, ...}, "members": [...]}, ...]}
                   - one top-three ensemble per cross-validation fold, each fitted on that fold's training plots, with its
                     score on that fold's test plots

Fold selection (fold ensembles only): "all" (default, average of every fold ensemble), "best3" / "bestN" (the N folds
with the lowest test RMSE) or a list of fold numbers. Ranking folds by their test scores selects on the test sets: a
best-folds prediction is fine to use, but its accuracy must be quoted from the all-folds cross-validation, not from the
chosen folds' scores. A plot is predicted when every input of every member used is finite (model_ok = 1).
"""
from __future__ import annotations
import numpy as np

FOLD_CHOICES = {"all": "all fold models (average of the 10 fold ensembles)", "best3": "best three folds (lowest test RMSE) - optimistic scores",
                "final": "final model fitted on all plots"}


def is_ensemble(saved) -> bool:
    return isinstance(saved, dict) and saved.get("type") == "mean_ensemble" and "members" in saved


def is_fold_ensembles(saved) -> bool:
    return isinstance(saved, dict) and saved.get("type") == "fold_ensembles" and "folds" in saved


def select_folds(saved, folds="all"):
    """list of fold entries of a fold-ensembles file according to the selection"""
    F = saved["folds"]
    if folds in (None, "all", "final"):
        return F
    if isinstance(folds, str) and folds.startswith("best"):
        n = int(folds[4:] or 3)
        return sorted(F, key=lambda f: f["test"]["RMSE"])[:n]
    want = {int(k) for k in folds}
    return [f for f in F if f["fold"] in want]


def _members(saved, folds="all"):
    """flat list of member groups: [[member, ...] per ensemble]"""
    if is_fold_ensembles(saved):
        return [f["members"] for f in select_folds(saved, folds)]
    if is_ensemble(saved):
        return [saved["members"]]
    return [[{"model": saved["model"], "features": saved["features"]}]]


def required_features(saved, folds="all"):
    out = []
    for grp in _members(saved, folds):
        for m in grp:
            out += [f for f in m["features"] if f not in out]
    return out


def predict_saved(saved, d, folds="all"):
    """d: DataFrame with the input columns. Returns (pred with NaN where inputs are missing, ok bool array).
    Ensembles: mean of the members; fold ensembles: mean over the selected folds of each fold's member mean."""
    groups = _members(saved, folds)
    missing = [f for f in required_features(saved, folds) if f not in d.columns]
    if missing:
        raise ValueError("input table lacks " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    ok = np.ones(len(d), bool)
    for grp in groups:
        for m in grp:
            ok &= np.isfinite(d[m["features"]].to_numpy(float)).all(axis=1)
    preds = []
    for grp in groups:
        ps = []
        for m in grp:
            p = np.full(len(d), np.nan); X = d[m["features"]].to_numpy(float)
            if ok.any():
                p[ok] = np.asarray(m["model"].predict(X[ok])).ravel()
            ps.append(p)
        preds.append(np.mean(ps, axis=0))
    return np.mean(preds, axis=0), ok


def fold_table(saved):
    """DataFrame of the fold ensembles: fold, n_test, R2, RMSE, members, best3 flag"""
    import pandas as pd
    if not is_fold_ensembles(saved):
        return None
    best = {f["fold"] for f in select_folds(saved, "best3")}
    return pd.DataFrame([dict(fold=f["fold"], n_test=f["n_test"], R2=round(f["test"]["R2"], 3), RMSE=round(f["test"]["RMSE"], 3),
                              members=" | ".join(f"{m['family']}/{m['learner']}" for m in f["members"]), best3=f["fold"] in best) for f in saved["folds"]])


def describe(saved, folds="all") -> str:
    if is_fold_ensembles(saved):
        sel = select_folds(saved, folds); cv = saved.get("nested_cv", {})
        return (f"fold ensembles: {len(sel)} of {len(saved['folds'])} folds used ({'all' if folds in (None, 'all', 'final') else folds}: "
                + ", ".join(str(f["fold"]) for f in sel) + ")" + (f"; all-folds nested CV R2 {cv['R2']:.2f}, RMSE {cv['RMSE']:.3g}" if cv else ""))
    if is_ensemble(saved):
        cv = saved.get("nested_cv", {})
        return ("mean of " + ", ".join(f"{m['family']}/{m['learner']}" for m in saved["members"]) +
                (f" (nested CV R2 {cv['R2']:.2f}, RMSE {cv['RMSE']:.3g})" if cv else ""))
    return f"single model, {len(saved['features'])} inputs"
