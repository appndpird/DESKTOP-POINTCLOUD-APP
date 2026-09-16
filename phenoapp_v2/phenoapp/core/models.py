"""
Biomass & dry-matter models: LiDAR, VNIR, and fusion — with honest
leave-one-out cross-validation.

Design decisions (validated on the 2025 DPIRD Fodder trials, Busselton):

* All calibrations include an INTERCEPT. The zero-intercept `PVI x k`
  model cross-validates below no-skill on dense pasture, because short
  dense swards hold biomass that a through-origin line cannot represent.
* Fresh biomass first. LiDAR + cameras sense the wet standing canopy;
  dry matter = fresh x DM%, and DM% is only observable spectrally (970 nm
  water feature). DM kg predictions are therefore derived as
  fresh_pred x DM%_pred and flagged with wide validation errors.
* Every fit reports LOOCV RMSE / R^2 (never just in-sample), so a model
  that merely memorised 36 plots is exposed immediately.
* PLS is implemented in pure numpy (NIPALS, PLS1) on log10+SNV spectra.
  The fitted model collapses to a single linear coefficient vector, so
  applying it needs no external ML library.

Ground-truth CSV columns (template written by the Biomass tab):
  Plot_ID, fresh_<unit> [, dm_frac 0-1 or dm_<unit>]
where <unit> is kg_ha, t_ha, g_m2, kg_m2 or kg (= kg per plot). Every
target is converted to a density in kg/ha before fitting (see units.py),
and per-plot LiDAR volumes are divided by region_area_m2, so models and
their coefficients transfer between trials with different plot sizes.
Predictions are written in kg/ha and, when the plot area is known, also as
kg per plot (*_kg_plot columns).
"""

from __future__ import annotations
import json
import numpy as np


# ----------------------------------------------------------------------
# Metrics & cross-validation engines
# ----------------------------------------------------------------------
def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Full metric set for a set of (observed, predicted) pairs.

    accuracy_pct = 100 - MAPE ("simple accuracy"): intuitive but sensitive
    to small observed values; rmse_pct (RMSE as % of the observed mean) is
    the more robust headline number.
    """
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    n = len(y)
    if n < 2:
        return {"n": int(n)}
    e = pred - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    bias = float(np.mean(e))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((e ** 2).sum()) / ss if ss > 0 else 0.0
    ymean = float(np.mean(y))
    rmse_pct = 100.0 * rmse / abs(ymean) if ymean else float("nan")
    nz = np.abs(y) > 1e-9
    mape = float(np.mean(np.abs(e[nz] / y[nz])) * 100.0) if nz.any() else float("nan")
    acc = max(0.0, 100.0 - mape) if np.isfinite(mape) else float("nan")
    # Pearson / Spearman / Lin's concordance
    if np.std(y) > 0 and np.std(pred) > 0:
        pear = float(np.corrcoef(pred, y)[0, 1])
        try:
            from scipy.stats import spearmanr
            spear = float(spearmanr(pred, y)[0])
        except Exception:
            spear = float("nan")
        ccc = float(2 * np.cov(pred, y)[0, 1] /
                    (np.var(pred) + np.var(y) + (np.mean(pred) - ymean) ** 2))
    else:
        pear = spear = ccc = float("nan")
    return {"n": int(n), "rmse": rmse, "rmse_pct": rmse_pct, "mae": mae,
            "mape_pct": mape, "accuracy_pct": acc, "bias": bias, "r2": r2,
            "pearson_r": pear, "spearman_rho": spear, "ccc": ccc}


CV_MODES = {
    "loo":    "Leave-one-out (LOOCV) — recommended for <60 plots",
    "kfold5": "5-fold cross-validation",
    "none":   "Fit only (no validation — in-sample, optimistic)",
}


def _folds(n: int, mode: str):
    """Deterministic fold index lists for the chosen CV mode."""
    if mode == "loo":
        return [[i] for i in range(n)]
    if mode == "kfold5":
        k = min(5, n)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        return [sorted(idx[i::k].tolist()) for i in range(k)]
    return []          # "none"


def cv_linear(X: np.ndarray, y: np.ndarray, mode: str = "loo"):
    """Cross-validated predictions for y ~ [1, X]. Returns pred array
    (NaN-free, aligned with y) or None for mode='none'."""
    if mode == "none":
        return None
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    pred = np.zeros(n)
    for test in _folds(n, mode):
        m = np.ones(n, bool); m[test] = False
        beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        pred[test] = A[test] @ beta
    return pred


def loocv_linear(X: np.ndarray, y: np.ndarray):
    """LOOCV predictions for y ~ [1, X]. Returns (pred, rmse, r2)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    pred = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        pred[i] = A[i] @ beta
    e = pred - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((e ** 2).sum()) / ss if ss > 0 else 0.0
    return pred, rmse, r2


def fit_linear(X: np.ndarray, y: np.ndarray):
    """Full-data OLS with intercept. Returns (intercept, coefs)."""
    n = len(y)
    A = np.column_stack([np.ones(n), np.asarray(X, float)])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(beta[0]), [float(b) for b in beta[1:]]


# ----------------------------------------------------------------------
# Pure-numpy PLS1 (NIPALS)
# ----------------------------------------------------------------------
def preprocess_spectra(S: np.ndarray) -> np.ndarray:
    """log10 + SNV (standard normal variate) per spectrum."""
    A = np.log10(np.clip(np.asarray(S, float), 1.0, None))
    mu = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (A - mu) / sd


def _pls1_train(X, y, ncomp):
    """NIPALS PLS1 on centred/scaled data. Returns regression vector + means.

    Model: y_hat = (X - x_mean)/x_std @ b + y_mean
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    x_mean = X.mean(axis=0); x_std = X.std(axis=0); x_std[x_std == 0] = 1.0
    y_mean = float(y.mean())
    Xc = (X - x_mean) / x_std
    yc = y - y_mean

    n, p = Xc.shape
    W = np.zeros((p, ncomp)); P = np.zeros((p, ncomp)); Q = np.zeros(ncomp)
    Xr = Xc.copy(); yr = yc.copy()
    for a in range(ncomp):
        w = Xr.T @ yr
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            W = W[:, :a]; P = P[:, :a]; Q = Q[:a]
            break
        w /= nw
        t = Xr @ w
        tt = float(t @ t)
        if tt < 1e-12:
            W = W[:, :a]; P = P[:, :a]; Q = Q[:a]
            break
        p_load = (Xr.T @ t) / tt
        q = float(yr @ t) / tt
        Xr = Xr - np.outer(t, p_load)
        yr = yr - q * t
        W[:, a] = w; P[:, a] = p_load; Q[a] = q
    if W.shape[1] == 0:
        b = np.zeros(p)
    else:
        # b = W (P^T W)^-1 Q
        M = P.T @ W
        try:
            b = W @ np.linalg.solve(M, Q[:W.shape[1]])
        except np.linalg.LinAlgError:
            b = W @ np.linalg.lstsq(M, Q[:W.shape[1]], rcond=None)[0]
    return b, x_mean, x_std, y_mean


def pls_nested_loocv(S: np.ndarray, y: np.ndarray, max_comp: int = 8):
    """Nested-LOOCV PLS1: inner loop picks n_components per outer fold.

    Returns (pred, rmse, r2, final_model_dict). The final model is trained
    on ALL samples with the median chosen component count, stored as a
    plain linear vector so it can be applied anywhere.
    """
    X = preprocess_spectra(S)
    y = np.asarray(y, float)
    n = len(y)
    max_comp = int(min(max_comp, max(1, n - 3)))
    pred = np.zeros(n)
    chosen = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        Xi, yi = X[m], y[m]
        best_c, best_rmse = 1, np.inf
        for c in range(1, max_comp + 1):
            p2 = np.zeros(n - 1)
            for j in range(n - 1):
                mj = np.ones(n - 1, bool); mj[j] = False
                b, xm, xs, ym = _pls1_train(Xi[mj], yi[mj], c)
                p2[j] = float((Xi[j] - xm) / xs @ b) + ym
            rm = float(np.sqrt(np.mean((p2 - yi) ** 2)))
            if rm < best_rmse:
                best_rmse, best_c = rm, c
        chosen.append(best_c)
        b, xm, xs, ym = _pls1_train(Xi, yi, best_c)
        pred[i] = float((X[i] - xm) / xs @ b) + ym
    e = pred - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((e ** 2).sum()) / ss if ss > 0 else 0.0

    ncomp = int(np.median(chosen))
    b, xm, xs, ym = _pls1_train(X, y, ncomp)
    model = {
        "type": "pls1",
        "ncomp": ncomp,
        "coef": b.tolist(),
        "x_mean": xm.tolist(),
        "x_std": xs.tolist(),
        "y_mean": ym,
        "preprocess": "log10+snv",
    }
    return pred, rmse, r2, model


def pls_cv_predict(S: np.ndarray, y: np.ndarray, mode: str = "loo",
                   max_comp: int = 8):
    """Cross-validated PLS predictions with per-fold inner component
    selection (inner LOO on the training set). None for mode='none'."""
    if mode == "none":
        return None
    X = preprocess_spectra(S)
    y = np.asarray(y, float)
    n = len(y)
    max_comp = int(min(max_comp, max(1, n - 3)))
    pred = np.zeros(n)
    for test in _folds(n, mode):
        m = np.ones(n, bool); m[test] = False
        Xi, yi = X[m], y[m]
        ni = len(yi)
        best_c, best_rmse = 1, np.inf
        for c in range(1, max_comp + 1):
            p2 = np.zeros(ni)
            for j in range(ni):
                mj = np.ones(ni, bool); mj[j] = False
                b, xm, xs, ym = _pls1_train(Xi[mj], yi[mj], c)
                p2[j] = float((Xi[j] - xm) / xs @ b) + ym
            rm = float(np.sqrt(np.mean((p2 - yi) ** 2)))
            if rm < best_rmse:
                best_rmse, best_c = rm, c
        b, xm, xs, ym = _pls1_train(Xi, yi, best_c)
        pred[test] = (X[test] - xm) / xs @ b + ym
    return pred


def apply_pls(model: dict, S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    if model.get("bands_used") is not None:
        S = S[:, np.asarray(model["bands_used"], int)]
    X = preprocess_spectra(S)
    b = np.asarray(model["coef"], float)
    xm = np.asarray(model["x_mean"], float)
    xs = np.asarray(model["x_std"], float)
    return (X - xm) / xs @ b + float(model["y_mean"])


# ----------------------------------------------------------------------
# Ridge & kernel-ridge (pure numpy; hyperparameters by inner CV)
# ----------------------------------------------------------------------
# Feature pool for the multi-feature models: whichever of these exist in
# the merged per-plot table are used.
FUSED_FEATURES = ["h_mean", "h_median", "h_p95", "h_std", "cover_frac",
                  "vol_voxel", "roughness",
                  "NDVI", "NDRE", "WBI", "NDWI970"]

_RIDGE_LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
_KRR_GAMMAS = [0.05, 0.1, 0.3, 1.0]          # on standardized features
_KRR_LAMBDAS = [0.01, 0.1, 1.0]


def _standardize_train(X):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _ridge_train(X, y, lam):
    Xs, mu, sd = _standardize_train(X)
    ym = float(y.mean())
    A = Xs.T @ Xs + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, Xs.T @ (y - ym))
    return b, mu, sd, ym


def _ridge_pick_lam(X, y):
    """Inner-LOO lambda selection."""
    n = len(y)
    best_lam, best = _RIDGE_LAMBDAS[0], np.inf
    for lam in _RIDGE_LAMBDAS:
        errs = []
        for j in range(n):
            m = np.ones(n, bool); m[j] = False
            b, mu, sd, ym = _ridge_train(X[m], y[m], lam)
            errs.append(((X[j] - mu) / sd) @ b + ym - y[j])
        rm = float(np.sqrt(np.mean(np.square(errs))))
        if rm < best:
            best, best_lam = rm, lam
    return best_lam


def ridge_cv_predict(X, y, mode="loo"):
    if mode == "none":
        return None
    n = len(y)
    pred = np.zeros(n)
    for test in _folds(n, mode):
        m = np.ones(n, bool); m[test] = False
        lam = _ridge_pick_lam(X[m], y[m])
        b, mu, sd, ym = _ridge_train(X[m], y[m], lam)
        pred[test] = ((X[test] - mu) / sd) @ b + ym
    return pred


def _krr_train(X, y, gamma, lam):
    Xs, mu, sd = _standardize_train(X)
    ym = float(y.mean())
    d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
    K = np.exp(-gamma * d2)
    alpha = np.linalg.solve(K + lam * np.eye(len(y)), y - ym)
    return alpha, Xs, mu, sd, ym, gamma


def _krr_predict(model, Xnew):
    alpha, Xtr, mu, sd, ym, gamma = model
    Xn = (Xnew - mu) / sd
    d2 = ((Xn[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2) @ alpha + ym


def _krr_pick(X, y):
    n = len(y)
    best, best_gl = np.inf, (_KRR_GAMMAS[0], _KRR_LAMBDAS[0])
    for g in _KRR_GAMMAS:
        for lam in _KRR_LAMBDAS:
            errs = []
            for j in range(n):
                m = np.ones(n, bool); m[j] = False
                mdl = _krr_train(X[m], y[m], g, lam)
                errs.append(_krr_predict(mdl, X[j:j+1])[0] - y[j])
            rm = float(np.sqrt(np.mean(np.square(errs))))
            if rm < best:
                best, best_gl = rm, (g, lam)
    return best_gl


def krr_cv_predict(X, y, mode="loo"):
    if mode == "none":
        return None
    n = len(y)
    pred = np.zeros(n)
    for test in _folds(n, mode):
        m = np.ones(n, bool); m[test] = False
        g, lam = _krr_pick(X[m], y[m])
        mdl = _krr_train(X[m], y[m], g, lam)
        pred[test] = _krr_predict(mdl, X[test])
    return pred


# ----------------------------------------------------------------------
# The model suite
# ----------------------------------------------------------------------
# (key, label, modality, target, predictors) — predictors are metric columns;
# "SPECTRA" means the full VNIR spectrum via PLS.
MODEL_SUITE = [
    ("fresh_lidar",   "Fresh — LiDAR (mean height, intercept)",
     "LiDAR",  "fresh_kg_ha", ["h_mean"]),
    ("fresh_lidar_p95","Fresh — LiDAR (p95 height; better for sparse/row canopies)",
     "LiDAR",  "fresh_kg_ha", ["h_p95"]),
    ("fresh_lidar_hc","Fresh — LiDAR (p95 + cover fraction)",
     "LiDAR",  "fresh_kg_ha", ["h_p95", "cover_frac"]),
    ("fresh_vnir",    "Fresh — VNIR (NDRE)",
     "VNIR",   "fresh_kg_ha", ["NDRE"]),
    ("fresh_vnir_pls","Fresh — VNIR (full-spectrum PLS)",
     "VNIR",   "fresh_kg_ha", "SPECTRA"),
    ("fresh_fusion",  "Fresh — Fusion (NDRE + mean height)",
     "Fusion", "fresh_kg_ha", ["NDRE", "h_mean"]),
    ("dmfrac_vnir",   "DM% — VNIR (WBI water index)",
     "VNIR",   "dm_frac",  ["WBI"]),
    ("dmfrac_vnir_pls","DM% — VNIR (full-spectrum PLS)",
     "VNIR",   "dm_frac",  "SPECTRA"),
    ("dm_lidar",      "DM — LiDAR direct (reference; usually weak)",
     "LiDAR",  "dm_kg_ha", ["h_mean"]),
    ("dm_fusion",     "DM — Fusion direct (NDRE + WBI + mean height)",
     "Fusion", "dm_kg_ha", ["NDRE", "WBI", "h_mean"]),
    # multi-feature machine-learning models (hyperparameters by inner CV;
    # with ~36 calibration plots these only sometimes beat the simple
    # models - the validated columns will tell you)
    ("fresh_ridge",   "Fresh — Ridge regression (all LiDAR+VNIR features)",
     "Fusion", "fresh_kg_ha", "RIDGE"),
    ("fresh_krr",     "Fresh — Kernel ridge RBF (nonlinear, all features)",
     "Fusion", "fresh_kg_ha", "KRR"),
    ("dm_ridge",      "DM — Ridge regression (all features)",
     "Fusion", "dm_kg_ha", "RIDGE"),
]


def fit_model_suite(df, gt, spectra=None, spectra_ids=None,
                    enabled=None, progress_cb=None, cv: str = "loo",
                    gt_unit: str = "auto", basis: str = "auto"):
    """Fit + cross-validate the whole suite.

    df          : merged per-plot DataFrame containing Plot_ID, region_area_m2
                  and any of h_mean, NDRE, WBI (etc.)
    gt          : DataFrame with Plot_ID, fresh_<unit> [, dm_frac / dm_<unit>]
                  in any biomass unit; converted to kg/ha internally
    gt_unit     : 'auto' (from the column name: fresh_kg_ha, fresh_t_ha,
                  fresh_g_m2, fresh_kg_m2, fresh_kg = kg per plot) or an
                  explicit key from phenoapp.core.units.BIOMASS_UNITS
    basis       : 'fresh' (default when a fresh_* column exists), 'dry'
                  (the ground truth is dry matter: every biomass model in
                  the suite is fitted to dm_kg_ha and labelled 'Dry biomass';
                  DM% models and the derived-DM step are skipped), or 'auto'
                  (dry when only dm_* columns are present)
    spectra     : optional (n, bands) array aligned with spectra_ids
    spectra_ids : Plot_ID list matching spectra rows
    enabled     : optional set of model keys to fit
    cv          : "loo" (default) | "kfold5" | "none" — see CV_MODES.
                  Every result carries metrics_cv (validated) AND
                  metrics_fit (in-sample); the gap between them is the
                  overfitting you would not see with fit-only metrics.

    Returns (results, predictions_df).
    """
    import pandas as pd
    from .units import (resolve_unit, to_kg_m2, kg_m2_to_kg_ha,
                        area_normalise, AREA_SCALED_FEATURES)

    # per-plot LiDAR volumes -> per m², so every predictor is a density
    df, _normed = area_normalise(df, sorted(AREA_SCALED_FEATURES)) \
        if "region_area_m2" in df.columns else (df.copy(), [])

    # ground truth -> kg/ha densities named fresh_kg_ha / dm_kg_ha
    gt = gt.copy()
    area_by_plot = (df.set_index("Plot_ID")["region_area_m2"]
                    if "region_area_m2" in df.columns else None)
    gt_units = {}
    for stem in ("fresh", "dm"):
        col = next((c for c in gt.columns
                    if c.lower().startswith(stem) and c.lower() != "dm_frac"
                    and c.lower() != f"{stem}_kg_ha"), None)
        if f"{stem}_kg_ha" in gt.columns:
            gt[f"{stem}_kg_ha"] = pd.to_numeric(gt[f"{stem}_kg_ha"], errors="coerce")
            gt_units[stem] = "kg_ha"
            continue
        if col is None:
            continue
        unit_key = resolve_unit(gt_unit, col, default="kg_plot")
        area = (gt["Plot_ID"].map(area_by_plot).to_numpy(float)
                if area_by_plot is not None else None)
        dens = to_kg_m2(pd.to_numeric(gt[col], errors="coerce").to_numpy(float),
                        unit_key, area)
        gt[f"{stem}_kg_ha"] = kg_m2_to_kg_ha(dens)
        gt_units[stem] = unit_key
    if "dm_frac" in gt.columns:
        gt["dm_frac"] = pd.to_numeric(gt["dm_frac"], errors="coerce")
    if "dm_frac" not in gt.columns and {"dm_kg_ha", "fresh_kg_ha"} <= set(gt.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            gt["dm_frac"] = gt["dm_kg_ha"] / gt["fresh_kg_ha"]
    if "dm_kg_ha" not in gt.columns and {"dm_frac", "fresh_kg_ha"} <= set(gt.columns):
        gt["dm_kg_ha"] = gt["dm_frac"] * gt["fresh_kg_ha"]

    merged = df.merge(gt, on="Plot_ID", how="inner", suffixes=("", "_gt"))
    results = {}
    pred_df = df[["Plot_ID"]].copy()

    spec_lookup = None
    if spectra is not None and spectra_ids is not None:
        spec_lookup = {pid: spectra[i] for i, pid in enumerate(spectra_ids)}

    if basis == "auto":
        basis = "fresh" if "fresh_kg_ha" in gt.columns else ("dry" if "dm_kg_ha" in gt.columns else "fresh")
    todo = [m for m in MODEL_SUITE if (enabled is None or m[0] in enabled)]
    if basis == "dry":
        # re-target the fresh-weight models to dry matter; DM% needs fresh weight, so skip it
        todo = [(k, l.replace("Fresh \u2014 ", "Dry biomass \u2014 ").replace("DM \u2014 ", "Dry biomass \u2014 "),
                 mod, ("dm_kg_ha" if t == "fresh_kg_ha" else t), p)
                for (k, l, mod, t, p) in todo if t != "dm_frac"]
        seen = set(); dedup = []
        for spec in todo:                       # dm_lidar (h_mean) duplicates fresh_lidar once re-targeted
            sig = (spec[3], tuple(spec[4]) if isinstance(spec[4], list) else spec[4])
            if sig in seen:
                continue
            seen.add(sig); dedup.append(spec)
        todo = dedup
    for mi, (key, label, modality, target, predictors) in enumerate(todo):
        if progress_cb:
            progress_cb(int(100 * mi / max(len(todo), 1)), f"Fitting {label}")
        if target not in merged.columns:
            results[key] = {"label": label, "status":
                            f"skipped - ground truth has no '{target}'"}
            continue

        sub = merged.dropna(subset=[target])
        if predictors == "SPECTRA":
            if spec_lookup is None:
                results[key] = {"label": label,
                                "status": "skipped - no VNIR spectra"}
                continue
            sub = sub[sub["Plot_ID"].isin(spec_lookup.keys())]
            S = np.array([spec_lookup[p] for p in sub["Plot_ID"]])
            # drop dead BANDS first (dropout rates are band-dependent in
            # these cubes; short wavelengths are zeroed far more often) —
            # otherwise one dead band would discard every plot
            band_ok = np.isfinite(S).mean(axis=0) >= 0.8 if len(S) else \
                np.zeros(0, bool)
            bands_used = np.where(band_ok)[0]
            S = S[:, band_ok] if len(S) else S
            ok = (np.isfinite(S).all(axis=1)
                  & np.isfinite(sub[target].values)) if S.size else \
                np.zeros(len(sub), bool)
            sub = sub[ok]; S = S[ok]
            if len(sub) < 8 or S.shape[1] < 10:
                results[key] = {"label": label,
                                "status": (f"skipped - only {len(sub)} plots"
                                           f" / {S.shape[1] if S.size else 0}"
                                           " usable bands")}
                continue
            y = sub[target].values.astype(float)
            # final model (ncomp by inner LOO on all data) + chosen-CV metrics
            _, _, _, model = pls_nested_loocv(S, y)
            model.update({"target": target, "label": label,
                          "modality": modality,
                          "bands_used": bands_used.tolist()})
            X0 = preprocess_spectra(S)
            pred_fit = (X0 - np.asarray(model["x_mean"])) / \
                np.asarray(model["x_std"]) @ np.asarray(model["coef"]) + \
                model["y_mean"]
            metrics_fit = regression_metrics(y, pred_fit)
            pred_cv = pls_cv_predict(S, y, cv)
            metrics_cv = (regression_metrics(y, pred_cv)
                          if pred_cv is not None else None)
            head = metrics_cv or metrics_fit
            cv_predictions = (dict(zip([int(p) for p in sub["Plot_ID"]],
                                       [float(v) for v in pred_cv]))
                              if pred_cv is not None else None)
            results[key] = {"label": label, "modality": modality,
                            "target": target, "n": int(len(y)),
                            "loocv_rmse": head["rmse"], "loocv_r2": head["r2"],
                            "metrics_cv": metrics_cv,
                            "metrics_fit": metrics_fit, "cv_mode": cv,
                            "cv_predictions": cv_predictions,
                            "model": model,
                            "status": ("ok" if metrics_cv else
                                       "ok (fit-only - optimistic)")}
            if cv_predictions is not None:
                pred_df[f"predcv_{key}"] = pred_df["Plot_ID"].map(
                    cv_predictions)
            # apply to every plot with a spectrum (same band subset)
            all_ids = [p for p in df["Plot_ID"] if p in spec_lookup]
            Sall = np.array([spec_lookup[p] for p in all_ids])
            Ssub = Sall[:, bands_used]
            good = np.isfinite(Ssub).all(axis=1)
            vals = np.full(len(all_ids), np.nan)
            if good.any():
                vals[good] = apply_pls(model, Sall[good])
            col = pd.Series(vals, index=all_ids)
            pred_df[f"pred_{key}"] = pred_df["Plot_ID"].map(col)
        elif predictors in ("RIDGE", "KRR"):
            feats = [f for f in FUSED_FEATURES if f in sub.columns]
            if len(feats) < 3:
                results[key] = {"label": label, "status":
                                f"skipped - only {len(feats)} features present"}
                continue
            sub2 = sub.dropna(subset=feats + [target])
            if len(sub2) < 10:
                results[key] = {"label": label,
                                "status": f"skipped - only {len(sub2)} plots"}
                continue
            X = sub2[feats].values.astype(float)
            y = sub2[target].values.astype(float)
            if predictors == "RIDGE":
                lam = _ridge_pick_lam(X, y)
                b, mu, sd, ym = _ridge_train(X, y, lam)
                model = {"type": "ridge", "target": target, "label": label,
                         "modality": modality, "features": feats,
                         "lambda": lam, "coef": b.tolist(),
                         "x_mean": mu.tolist(), "x_std": sd.tolist(),
                         "y_mean": ym}
                pred_fit = ((X - mu) / sd) @ b + ym
                pred_cv = ridge_cv_predict(X, y, cv)
            else:
                g, lam = _krr_pick(X, y)
                mdl = _krr_train(X, y, g, lam)
                model = {"type": "krr", "target": target, "label": label,
                         "modality": modality, "features": feats,
                         "gamma": g, "lambda": lam,
                         "alpha": mdl[0].tolist(),
                         "x_train_std": mdl[1].tolist(),
                         "x_mean": mdl[2].tolist(), "x_std": mdl[3].tolist(),
                         "y_mean": mdl[4]}
                pred_fit = _krr_predict(mdl, X)
                pred_cv = krr_cv_predict(X, y, cv)
            metrics_fit = regression_metrics(y, pred_fit)
            metrics_cv = (regression_metrics(y, pred_cv)
                          if pred_cv is not None else None)
            head = metrics_cv or metrics_fit
            cv_predictions = (dict(zip([int(p) for p in sub["Plot_ID"]],
                                       [float(v) for v in pred_cv]))
                              if pred_cv is not None else None)
            results[key] = {"label": label, "modality": modality,
                            "target": target, "n": int(len(y)),
                            "loocv_rmse": head["rmse"], "loocv_r2": head["r2"],
                            "metrics_cv": metrics_cv,
                            "metrics_fit": metrics_fit, "cv_mode": cv,
                            "cv_predictions": cv_predictions,
                            "model": model,
                            "status": ("ok" if metrics_cv else
                                       "ok (fit-only - optimistic)")}
            if cv_predictions is not None:
                pred_df[f"predcv_{key}"] = pred_df["Plot_ID"].map(
                    cv_predictions)
            ok_rows = df.dropna(subset=[f for f in feats if f in df.columns]) \
                if all(f in df.columns for f in feats) else df.iloc[0:0]
            if len(ok_rows):
                Xa = ok_rows[feats].values.astype(float)
                if predictors == "RIDGE":
                    vals = ((Xa - mu) / sd) @ b + ym
                else:
                    vals = _krr_predict(mdl, Xa)
                col = pd.Series(vals, index=ok_rows["Plot_ID"])
                pred_df[f"pred_{key}"] = pred_df["Plot_ID"].map(col)
        else:
            missing = [p for p in predictors if p not in sub.columns]
            if missing:
                results[key] = {"label": label, "status":
                                f"skipped - missing metric(s): {', '.join(missing)}"}
                continue
            sub = sub.dropna(subset=predictors)
            if len(sub) < len(predictors) + 3:
                results[key] = {"label": label,
                                "status": f"skipped - only {len(sub)} plots"}
                continue
            X = sub[predictors].values.astype(float)
            y = sub[target].values.astype(float)
            b0, coefs = fit_linear(X, y)
            model = {"type": "linear", "target": target, "label": label,
                     "modality": modality, "predictors": predictors,
                     "intercept": b0, "coefs": coefs}
            pred_fit = b0 + X @ np.asarray(coefs)
            metrics_fit = regression_metrics(y, pred_fit)
            pred_cv = cv_linear(X, y, cv)
            metrics_cv = (regression_metrics(y, pred_cv)
                          if pred_cv is not None else None)
            head = metrics_cv or metrics_fit
            cv_predictions = (dict(zip([int(p) for p in sub["Plot_ID"]],
                                       [float(v) for v in pred_cv]))
                              if pred_cv is not None else None)
            results[key] = {"label": label, "modality": modality,
                            "target": target, "n": int(len(y)),
                            "loocv_rmse": head["rmse"], "loocv_r2": head["r2"],
                            "metrics_cv": metrics_cv,
                            "metrics_fit": metrics_fit, "cv_mode": cv,
                            "cv_predictions": cv_predictions,
                            "model": model,
                            "status": ("ok" if metrics_cv else
                                       "ok (fit-only - optimistic)")}
            if cv_predictions is not None:
                pred_df[f"predcv_{key}"] = pred_df["Plot_ID"].map(
                    cv_predictions)
            ok_rows = df.dropna(subset=[p for p in predictors])
            vals = b0 + ok_rows[predictors].values.astype(float) @ np.array(coefs)
            col = pd.Series(vals, index=ok_rows["Plot_ID"])
            pred_df[f"pred_{key}"] = pred_df["Plot_ID"].map(col)

    # ---- derived DM kg = fresh_pred x DM%_pred (fresh basis only) ----
    fresh_col = None if basis == "dry" else next((f"pred_{k}" for k in
                      ("fresh_vnir_pls", "fresh_vnir", "fresh_fusion",
                       "fresh_lidar")
                      if f"pred_{k}" in pred_df.columns), None)
    dm_col = None if basis == "dry" else next((f"pred_{k}" for k in ("dmfrac_vnir_pls", "dmfrac_vnir")
                   if f"pred_{k}" in pred_df.columns), None)
    if fresh_col and dm_col:
        pred_df["pred_dm_kg_ha_derived"] = pred_df[fresh_col] * pred_df[dm_col]
        if "dm_kg_ha" in merged.columns:
            chk = merged[["Plot_ID", "dm_kg_ha"]].merge(
                pred_df[["Plot_ID", "pred_dm_kg_ha_derived"]], on="Plot_ID"
            ).dropna()
            if len(chk) >= 3:
                mder = regression_metrics(chk["dm_kg_ha"].values,
                                          chk["pred_dm_kg_ha_derived"].values)
                results["dm_kg_derived"] = {
                    "label": "DM — derived (fresh_pred x DM%_pred)",
                    "modality": "Fusion", "target": "dm_kg_ha",
                    "n": mder["n"],
                    "loocv_rmse": mder["rmse"], "loocv_r2": mder["r2"],
                    "metrics_cv": mder, "metrics_fit": mder, "cv_mode": cv,
                    "status": "ok (errors compound; treat as indicative)",
                }

    # ---- units: tag every result/model, add per-plot kg columns ----
    for key, r in results.items():
        tgt = r.get("target", "")
        unit = "0-1" if tgt == "dm_frac" else ("kg/ha" if tgt.endswith("_kg_ha") else "")
        r["target_unit"] = unit; r["basis"] = basis
        if r.get("model") is not None:
            r["model"]["basis"] = basis
            r["model"]["target_unit"] = unit
            r["model"]["area_normalised"] = list(_normed)
            r["model"]["gt_units"] = gt_units
    if "region_area_m2" in df.columns:
        area = df.set_index("Plot_ID")["region_area_m2"]
        pred_df["region_area_m2"] = pred_df["Plot_ID"].map(area)
        for c in [c for c in pred_df.columns if c.startswith(("pred_", "predcv_"))
                  and "dmfrac" not in c and not c.endswith("_kg_plot")]:
            pred_df[c + "_kg_plot"] = pred_df[c] * pred_df["region_area_m2"] / 10_000.0

    return results, pred_df


def results_table(results: dict, detail: bool = True) -> str:
    """Plain-text summary table of a fit_model_suite() run.

    Validated columns come from the chosen CV mode; 'fit R2' is in-sample
    (always >= the validated R2 — the gap is overfitting).
    """
    cv_mode = next((r.get("cv_mode") for r in results.values()
                    if r.get("cv_mode")), "loo")
    tag = {"loo": "LOOCV", "kfold5": "5-fold", "none": "FIT-ONLY"}.get(cv_mode,
                                                                       cv_mode)
    if not detail:
        lines = [f"{'model':<46}{'n':>4}{tag+' RMSE':>12}{tag+' R2':>10}  status"]
        for r in results.values():
            if r.get("status", "").startswith("ok"):
                lines.append(f"{r['label']:<46}{r['n']:>4}"
                             f"{r['loocv_rmse']:>12.3f}{r['loocv_r2']:>10.3f}"
                             f"  {r['status']}")
            else:
                lines.append(f"{r['label']:<46}{'-':>4}{'-':>12}{'-':>10}"
                             f"  {r.get('status','')}")
        return "\n".join(lines)

    hdr = (f"{'model':<46}{'n':>4}{'R2':>7}{'RMSE':>9}{'RMSE%':>8}"
           f"{'MAE':>8}{'Acc%':>7}{'bias':>8}{'r':>6}{'fitR2':>7}  status")
    lines = [f"validation mode: {CV_MODES.get(cv_mode, cv_mode)}",
             "units: fresh / DM targets and their RMSE, MAE and bias are in "
             "kg/ha (ground truth converted from its own unit); DM% is 0-1",
             hdr]
    for r in results.values():
        if not r.get("status", "").startswith("ok"):
            lines.append(f"{r['label']:<46}{'-':>4}"
                         + " " * 60 + f"  {r.get('status','')}")
            continue
        m = r.get("metrics_cv") or r.get("metrics_fit") or {}
        mf = r.get("metrics_fit") or {}
        def g(d, k, fmt="{:.3f}"):
            v = d.get(k)
            return fmt.format(v) if v is not None and np.isfinite(v) else "-"
        lines.append(
            f"{r['label']:<46}{m.get('n','-'):>4}"
            f"{g(m,'r2'):>7}{g(m,'rmse'):>9}{g(m,'rmse_pct','{:.1f}'):>8}"
            f"{g(m,'mae'):>8}{g(m,'accuracy_pct','{:.1f}'):>7}"
            f"{g(m,'bias','{:+.2f}'):>8}{g(m,'pearson_r','{:.2f}'):>6}"
            f"{g(mf,'r2'):>7}  {r['status']}")
    return "\n".join(lines)


def save_models(results: dict, path: str):
    out = {k: r.get("model") for k, r in results.items() if r.get("model")}
    meta = {k: {kk: r[kk] for kk in ("label", "n", "loocv_rmse", "loocv_r2")
                if kk in r}
            for k, r in results.items() if r.get("status", "").startswith("ok")}
    with open(path, "w") as f:
        json.dump({"models": out, "validation": meta,
                   "target_unit": "kg/ha (dm_frac: 0-1)",
                   "note": ("Predictors listed in each model's "
                            "'area_normalised' are per m² (metric / "
                            "region_area_m2). Multiply kg/ha by "
                            "region_area_m2 / 10000 for kg per plot.")},
                  f, indent=2)


def load_models(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
