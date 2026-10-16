"""
Biomass feature models on the two Muresk flights only, with the FULL usable spectrum (every band valid on >= 95 % of the
plots of both flights, log10 + SNV) next to the LiDAR and VNIR index features. Both flights are pooled (256 plots, stage flag
as an input) and validated with folds grouped by physical plot (both flights of a plot in the same fold; 3 repeats of
stratified-group 10-fold), so a plot never sits in training while its other flight is in the test set. Per-flight scores
of the pooled model are reported as well. The best model of every family (fitted on all 256 plots) is saved for later
fine-tuning / reuse on data where all bands are valid.

Outputs: biomass\muresk_fullspectrum\ model_comparison.csv, per_plot_predictions.csv, best_models.csv, bands_used.csv,
         models\biomass_muresk_fullspectrum_<family>_<learner>.joblib
"""
import os, sys, time, json, numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
try: import xgboost as xgb
except Exception: xgb = None
try: import lightgbm as lgb
except Exception: lgb = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_families import spectra_matrix, pca_preprocessor, pca_learners, pca_blocks, pca_summary, SB_PREFIX
sys.path.insert(0, r"C:\Users\appn\Downloads\PhenoFuse-main\phenoapp_v3_2026-10-06")
from phenoapp.core.biomass_ml import LIDAR_CORE_BIOMASS as LIDAR_CORE, VNIR_CORE_BIOMASS as VNIR_CORE, WEIGHTED, _NON_FEATURES

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--bands", default="recommended", choices=["recommended", "all", "startend"],
                                                help="recommended: bands outside the excluded ranges (0-415, 755-770, 928-962 nm) and valid on >= 95 %% of the plots of both flights (138); "
                                                     "all: every band valid on >= 95 %% of the plots of both flights, excluded ranges included")
ap.add_argument("--datasets", default="muresk_anthesis,muresk_maturity", help="Muresk datasets to pool (default both); 'muresk_maturity' = the plots that have every valid band")
ap.add_argument("--run", default=None, help="output folder / model-name tag (default muresk_fullspectrum | muresk_allbands)")
ap.add_argument("--spectra-file", default="vnir_spectra.csv", help="per-flight spectra table to use (vnir_spectra_o2a.csv = with the oxygen-A bands recomputed from the plot cubes)")
cli = ap.parse_args(); RUN = cli.run or {"recommended": "muresk_fullspectrum", "all": "muresk_allbands", "startend": "muresk_maturity_validplots"}[cli.bands]
EXCL_RANGES = {"recommended": None, "all": (), "startend": ((0.0, 415.0), (928.0, 962.0))}[cli.bands]      # startend: only the first bands and the water-vapour tail dropped
USE = [d.strip() for d in cli.datasets.split(",")]
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; M = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(M, "dataset")
OUT = os.path.join(M, "biomass", RUN); MOD = os.path.join(M, "biomass", "models"); os.makedirs(OUT, exist_ok=True); os.makedirs(MOD, exist_ok=True)
LOGF = open(os.path.join(OUT, "run.log"), "a"); t0 = time.time()
def log(*a):
    s = f"[{time.time()-t0:5.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
BOOK = set(_NON_FEATURES) | {"Plot_ID", "trial", "flight", "stage", "has_ground_truth", "valid_lidar", "valid_vnir", "valid_vnir_lenient", "valid_fused", "vnir_reflectance_ok", "n_usable_bands", "source_flight"}
def add_weighted(d):
    d = d.copy()
    for k, (a, b) in WEIGHTED.items():
        if a in d.columns and b in d.columns: d[k] = d[a] * d[b]
    return d
WCOLS = list(WEIGHTED)

tabs = []; lidar_cols = vnir_cols = None; SBs = {}
FLIGHTS = [(k, s) for k, s in (("2025-09-30", "anthesis"), ("2025-11-21", "maturity")) if f"muresk_{s}" in USE]
for key, stage in FLIGHTS:
    d = pd.read_csv(os.path.join(DS, "muresk", key, "features_all.csv")); L = pd.read_csv(os.path.join(DS, "muresk", key, "features_lidar.csv")); V = pd.read_csv(os.path.join(DS, "muresk", key, "features_vnir.csv"))
    lc = [c for c in L.columns if c not in BOOK and L[c].dtype != object]; vc = [c for c in V.columns if c not in BOOK and V[c].dtype != object and not c.endswith("_valid")]
    lidar_cols = lc if lidar_cols is None else [c for c in lidar_cols if c in lc]; vnir_cols = vc if vnir_cols is None else [c for c in vnir_cols if c in vc]
    d = add_weighted(d[(d.has_ground_truth == 1) & (d.valid_lidar == 1) & d.biomass_kg_ha.notna()].reset_index(drop=True))
    d["stage_flag"] = 1.0 if stage == "maturity" else 0.0; d["dataset"] = f"muresk_{stage}"
    SBs[key] = spectra_matrix(pd.read_csv(os.path.join(DS, "muresk", key, cli.spectra_file)), min_frac=0.95, excluded=EXCL_RANGES)   # bands valid on >= 95 % of the flight's plots (excluded ranges dropped unless --bands all)
    tabs.append(d)
common_bands = sorted(set.intersection(*[set(SBs[k].columns) for k, _ in FLIGHTS]), key=lambda c: float(c[len(SB_PREFIX):]))
log(f"band policy {cli.bands}, datasets {USE}: band columns per flight {[SBs[k].shape[1] for k, _ in FLIGHTS]}, usable on all used flights {len(common_bands)}")
pd.DataFrame({"band_column": common_bands, "wavelength_nm": [float(c[len(SB_PREFIX):]) for c in common_bands]}).to_csv(os.path.join(OUT, "bands_used.csv"), index=False)
P = pd.concat([t.merge(SBs[k][common_bands], left_on="Plot_ID", right_index=True, how="left") for t, (k, _) in zip(tabs, FLIGHTS)], ignore_index=True)
P["group"] = P.Plot_ID.astype(str); y = P.biomass_kg_ha.to_numpy(float)
log(f"pooled Muresk: {len(P)} plot-flights ({(P.dataset == 'muresk_anthesis').sum()} anthesis, {(P.dataset == 'muresk_maturity').sum()} maturity); biomass mean {y.mean():.0f} sd {y.std():.0f}")

def available(cols, d): return [c for c in cols if c in d.columns and np.isfinite(d[c].astype(float)).mean() > 0.95 and d[c].astype(float).std() > 0]
def learners(nfeat):
    L = {"Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "PLS": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(8, nfeat) + 1))}, cv=5, scoring="neg_root_mean_squared_error"),
         "SVR_rbf": GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": [100, 300, 1000, 3000], "svr__epsilon": [50, 150, 300], "svr__gamma": ["scale", 0.03]}, cv=5, scoring="neg_root_mean_squared_error"),
         "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(nfeat) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0)),
         "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1),
         "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1)}
    if xgb is not None: L["XGBoost"] = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0)
    if lgb is not None: L["LightGBM"] = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, verbose=-1)
    return L
def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / y) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), rRMSE=float(np.sqrt(np.mean(e ** 2)) / y.mean() * 100), MAE=float(np.abs(e).mean()),
                bias=float(e.mean()), r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, MAPE=mape, accuracy=100 - mape)
def nested_select(Xtr, ytr, nfeat, k):
    if k is None or k >= nfeat: return list(range(nfeat))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(Xtr, ytr); return list(np.argsort(rf.feature_importances_)[::-1][:k])
REPEATS = 3
splits = [sp for r in range(REPEATS) for sp in StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=r).split(P, P.dataset, groups=P.group)]
def cv_run(X, model, k):
    ps = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in splits:
        idx = nested_select(X[tr], y[tr], X.shape[1], k); m = clone(model).fit(X[tr][:, idx], y[tr]); ps[te] += m.predict(X[te][:, idx]).ravel(); cnt[te] += 1
    return ps / np.maximum(cnt, 1)

SEL_FAMS = {"LiDAR_core": available(LIDAR_CORE, P), "VNIR_core": available(VNIR_CORE, P), "Fused_core": available(LIDAR_CORE + VNIR_CORE + WCOLS, P), "Fused_all": available(lidar_cols + vnir_cols + WCOLS, P)}
lid_all = available(lidar_cols, P); vn_all = available(vnir_cols, P); bands = available(common_bands, P)
PCA_FAMS = {"LiDAR_PCA": (lid_all, [], []), "VNIR_PCA": ([], vn_all, []), "Fused_PCA": (lid_all, vn_all, []), "Fused_PCA_joint": (lid_all, vn_all, []), "VNIR_bands_PCA": ([], [], bands), "Fused_bands_PCA": (lid_all, [], bands)}
rows, preds, best = [], [], []
for fam, cols in SEL_FAMS.items():
    cols2 = cols + ["stage_flag"]; X = P[cols2].fillna(P[cols2].median()).to_numpy(float); k = 12 if len(cols2) > 15 else None
    fam_best = None
    for lname, model in learners(min(len(cols2), k or len(cols2))).items():
        t1 = time.time(); p = cv_run(X, model, k); m = metrics(p, y)
        rows.append(dict(scheme="muresk_pooled_group10fold_x3", dataset="muresk_both", feature_set=fam, n_features=len(cols2), n_selected=k or len(cols2), learner=lname, n=len(y), **m, seconds=round(time.time() - t1)))
        for ds in USE:
            sel = (P.dataset == ds).to_numpy()
            if sel.sum() < 10 or sel.all(): continue
            mm = metrics(p[sel], y[sel]); rows.append(dict(scheme="muresk_pooled_group10fold_x3", dataset=ds, feature_set=fam, n_features=len(cols2), n_selected=k or len(cols2), learner=lname, n=int(sel.sum()), **mm))
        preds.append(pd.DataFrame(dict(feature_set=fam, learner=lname, dataset=P.dataset, Plot_ID=P.Plot_ID, Plot=P.Plot, measured=y, predicted=p)))
        log(f"{fam:16s} {lname:12s} n {len(y)} feats {len(cols2)} R2 {m['R2']:.3f} RMSE {m['RMSE']:.0f} ({time.time()-t1:.0f}s)")
        if fam_best is None or m["RMSE"] < fam_best[0]: fam_best = (m["RMSE"], lname, model, cols2, k, m)
    rmse, lname, model, cols2, k, m = fam_best; X = P[cols2].fillna(P[cols2].median()).to_numpy(float); idx = nested_select(X, y, len(cols2), k); fit = clone(model).fit(X[:, idx], y)
    f = f"biomass_{RUN}_{fam}_{lname}.joblib"; joblib.dump({"model": fit, "features": [cols2[i] for i in idx], "pca_family": False, "target": "biomass_kg_ha", "dataset": "muresk_both_flights", "band_policy": cli.bands, "cv": m, "bands": None}, os.path.join(MOD, f))
    best.append(dict(family=fam, learner=lname, **m, features=len(idx), model_file=f))
for fam, (lc, vc, bc) in PCA_FAMS.items():
    cols = lc + vc + bc
    if len(cols) < 2: continue
    blocks = pca_blocks(fam, len(lc), len(vc), len(bc)); cols2 = cols + ["stage_flag"]; X = P[cols2].fillna(P[cols2].median()).to_numpy(float); pre = pca_preprocessor(blocks)
    fam_best = None
    for lname, model in pca_learners(pre, [100, 300, 1000, 3000], [50, 150, 300]).items():
        if lname == "MLP": continue
        t1 = time.time(); p = cv_run(X, model, None); m = metrics(p, y); fit0 = clone(model).fit(X, y); ps = pca_summary(getattr(fit0, "best_estimator_", fit0).named_steps["pre"])
        rows.append(dict(scheme="muresk_pooled_group10fold_x3", dataset="muresk_both", feature_set=fam, n_features=len(cols2), n_selected=sum(s["n_components"] for s in ps.values()), learner=lname, n=len(y), **m, seconds=round(time.time() - t1)))
        for ds in USE:
            sel = (P.dataset == ds).to_numpy()
            if sel.sum() < 10 or sel.all(): continue
            mm = metrics(p[sel], y[sel]); rows.append(dict(scheme="muresk_pooled_group10fold_x3", dataset=ds, feature_set=fam, n_features=len(cols2), n_selected=0, learner=lname, n=int(sel.sum()), **mm))
        preds.append(pd.DataFrame(dict(feature_set=fam, learner=lname, dataset=P.dataset, Plot_ID=P.Plot_ID, Plot=P.Plot, measured=y, predicted=p)))
        log(f"{fam:16s} {lname:12s} n {len(y)} inputs {len(cols2)} PCs {ps} R2 {m['R2']:.3f} RMSE {m['RMSE']:.0f} ({time.time()-t1:.0f}s)")
        if fam_best is None or m["RMSE"] < fam_best[0]: fam_best = (m["RMSE"], lname, fit0, m, ps)
    rmse, lname, fit0, m, ps = fam_best
    f = f"biomass_{RUN}_{fam}_{lname}.joblib"; joblib.dump({"model": fit0, "features": cols2, "pca_family": True, "pca": ps, "target": "biomass_kg_ha", "dataset": "muresk_both_flights", "band_policy": cli.bands, "cv": m, "bands": bc if bc else None}, os.path.join(MOD, f))
    best.append(dict(family=fam, learner=lname, **m, features=len(cols2), model_file=f))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False); pd.concat(preds, ignore_index=True).to_csv(os.path.join(OUT, "per_plot_predictions.csv"), index=False)
Bst = pd.DataFrame(best).sort_values("RMSE"); Bst.to_csv(os.path.join(OUT, "best_models.csv"), index=False)
json.dump(dict(datasets=USE, n=len(y), folds="StratifiedGroupKFold 10 x 3 repeats, grouped by plot", band_policy=cli.bands, bands_used=len(bands), band_columns=bands, lidar_features=len(lid_all), vnir_index_features=len(vn_all),
               note="pooled Muresk model with a stage flag; " + ("recommended spectrum (excluded ranges dropped)" if cli.bands == "recommended" else "every band valid on >= 95 % of the plots of both flights, excluded ranges included") + "; best model per family saved on all 256 plot-flights"),
          open(os.path.join(OUT, "settings.json"), "w"), indent=1)
log("BEST PER FAMILY:\n" + Bst[["family", "learner", "R2", "RMSE", "rRMSE", "features", "model_file"]].round(3).to_string(index=False)); log("MURESK FULL-SPECTRUM DONE")
