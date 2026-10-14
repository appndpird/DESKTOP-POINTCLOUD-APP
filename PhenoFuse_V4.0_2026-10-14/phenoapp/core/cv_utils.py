"""
Cross-validation splits shared by the Height Models and Biomass Models tabs (PhenoFuse 4.0).

cv modes
  rkf10   repeated 10-fold (5 repeats)      - the recommended setting
  kfold5  5-fold
  loo     leave-one-out
  none    fit only (no held-out predictions)

groups
  When several flights of the same trial are pooled, the same plot appears once per flight. Its rows must stay
  together in one fold, otherwise the model sees the test plot's twin from another date during training and the
  held-out score is optimistic. `groups` (e.g. "<trial>:<Plot_ID>") makes every split group-aware; with a single
  flight every plot is its own group and the splits are the plain ones.
"""
from __future__ import annotations
import numpy as np

CV_LABELS = {"rkf10": "Repeated 10-fold (5x) - recommended", "kfold5": "5-fold cross-validation",
             "loo": "Leave-one-out (LOOCV)", "none": "Fit only (in-sample)"}


def make_splits(n: int, cv: str = "rkf10", groups=None, seed: int = 0):
    """List of (train_idx, test_idx) pairs; empty list for cv='none'."""
    if cv == "none" or n < 3:
        return []
    g = np.arange(n) if groups is None else np.asarray(groups)
    ug, ginv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = []
    if cv == "loo":
        for k in range(len(ug)):
            te = np.flatnonzero(ginv == k); tr = np.flatnonzero(ginv != k)
            out.append((tr, te))
        return out
    k_folds = 5 if cv == "kfold5" else 10
    k_folds = min(k_folds, len(ug))
    repeats = 5 if cv == "rkf10" else 1
    for r in range(repeats):
        perm = rng.permutation(len(ug)); fold_of_group = np.empty(len(ug), int); fold_of_group[perm] = np.arange(len(ug)) % k_folds
        f = fold_of_group[ginv]
        for k in range(k_folds):
            te = np.flatnonzero(f == k); tr = np.flatnonzero(f != k)
            if len(te) and len(tr):
                out.append((tr, te))
    return out


def cv_predict(model_factory, X, y, splits, fit_fn=None):
    """Average held-out prediction per row over the splits (rows never held out get NaN)."""
    from sklearn.base import clone
    ps = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in splits:
        m = model_factory() if callable(model_factory) else clone(model_factory)
        m = (fit_fn(m, X[tr], y[tr]) if fit_fn else m.fit(X[tr], y[tr]))
        ps[te] += np.asarray(m.predict(X[te])).ravel(); cnt[te] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, ps / np.maximum(cnt, 1), np.nan)
