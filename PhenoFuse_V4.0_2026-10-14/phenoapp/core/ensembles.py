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


def fill_missing(saved, d, feats):
    """NaN cells of the needed (non-band) feature columns of d: first the column median over the plots of this flight,
    then - for a feature that is NaN on every plot (e.g. an index that needs a band clipped on this sensor) - the
    training-set median stored in the model under "feature_medians". Flight-specific PCA scores (spec_pc*, whose sign and
    scale differ between extraction runs) are always replaced by the training median. Returns (d, filled, replaced):
    filled = features that took the training median because they were NaN everywhere, replaced = the spec_pc* ones;
    a feature with no stored median stays NaN, so those plots get model_ok = 0."""
    import pandas as pd
    from .pca_families import NON_TRANSFERABLE
    d = d.copy(); filled = []; replaced = []
    have = list(dict.fromkeys(f for f in feats if f in d.columns and not f.startswith("sb_") and pd.api.types.is_numeric_dtype(d[f])))
    if have:
        d[have] = d[have].astype(float).fillna(d[have].astype(float).median())
    med = saved.get("feature_medians", {}) if isinstance(saved, dict) else {}
    for f in have:
        if f in med and med[f] == med[f] and NON_TRANSFERABLE.match(f):
            d[f] = float(med[f]); replaced.append(f)
        elif d[f].isna().all() and f in med and med[f] == med[f]:
            d[f] = float(med[f]); filled.append(f)
    return d, filled, replaced


def feature_medians(table, feats):
    """{feature: median over the training table} for the numeric non-band features present (stored in saved models)."""
    import pandas as pd
    out = {}
    for f in feats:
        if f in table.columns and not f.startswith("sb_") and pd.api.types.is_numeric_dtype(table[f]):
            m = float(table[f].astype(float).median())
            if m == m:
                out[f] = m
    return out


def feature_ranges(table, feats):
    """{feature: [5th, 95th percentile over the training table]} for the numeric non-band features present; stored in
    saved models so a prediction on a flight outside the training range (another growth stage) can be flagged."""
    import pandas as pd
    out = {}
    for f in feats:
        if f in table.columns and not f.startswith("sb_") and pd.api.types.is_numeric_dtype(table[f]):
            v = table[f].astype(float).dropna()
            if len(v) >= 5:
                lo, hi = float(v.quantile(0.05)), float(v.quantile(0.95))
                if lo == lo and hi == hi:
                    out[f] = [lo, hi]
    return out


def range_check(saved, d, feats):
    """Features whose median over the plots of this flight lies outside the model's training 5-95 % range
    ("feature_ranges"): list of (feature, flight_median, lo, hi), largest relative excursion first. Empty when the
    model stores no ranges."""
    import pandas as pd
    rng = saved.get("feature_ranges", {}) if isinstance(saved, dict) else {}
    out = []
    for f in dict.fromkeys(feats):
        if f in rng and f in d.columns and pd.api.types.is_numeric_dtype(d[f]):
            m = float(d[f].astype(float).median()); lo, hi = rng[f]
            if m == m and (m < lo or m > hi):
                w = max(hi - lo, 1e-12); out.append((f, m, lo, hi, (lo - m) / w if m < lo else (m - hi) / w))
    out.sort(key=lambda r: -r[4])
    return [r[:4] for r in out]


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
