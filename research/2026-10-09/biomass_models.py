"""
Biomass models (fused LiDAR v3.2 + cleaned VNIR v3.0.1 features) on Dataset_2026-10-09, Muresk NUE + AGT NUE 2025.

Datasets  : muresk_anthesis (2025-09-30), muresk_maturity (2025-11-21), agt_anthesis (2025-09-22), agt_maturity (2025-11-13 merged)
Plots     : ground truth present and LiDAR QC ok; VNIR families additionally require vnir_qc_ok = 1 (a non-reflectance cube
            fails every plot, so AGT maturity runs LiDAR-only). Red-based indices are NaN on red-clipped plots (NaN rule);
            a feature is used only when finite on > 95 % of the plots, the rest is median-filled inside the dataset.
Families  : LiDAR_core (v3.2 core 22 + gap layers / densities / PVI), VNIR_core, Fused_core (+ CHM-weighted indices), Fused_all
Learners  : Ridge, PLS, SVR, GPR, RandomForest, ExtraTrees, XGBoost, LightGBM, MLP
Validation: nested repeated 10-fold (5x, RF top-12 selection inside the fold), pooled grouped 10-fold with trial/stage flags,
            cross-trial transfer per stage.
Outputs   : <Models_2026-10-09>/biomass/results/*.csv, models/*.joblib, features/features_<dataset>.csv
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV, GroupKFold
import joblib
try: import xgboost as xgb
except Exception: xgb = None
try: import lightgbm as lgb
except Exception: lgb = None
TOOL = r"C:\Users\appn\Downloads\DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06"; sys.path.insert(0, TOOL)
from phenoapp.core.lidar_features import LIDAR_V3_CORE
from phenoapp.core.biomass_ml import VNIR_CORE_BIOMASS, WEIGHTED
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(ROOT, "dataset")
HERE = os.path.join(ROOT, "biomass"); RES = os.path.join(HERE, "results"); MOD = os.path.join(HERE, "models"); FEA = os.path.join(HERE, "features")
for d in (RES, os.path.join(RES, "figures"), MOD, FEA): os.makedirs(d, exist_ok=True)
t0 = time.time(); LOGF = open(os.path.join(HERE, "biomass_models.log"), "a")
def log(*a):
    s = f"[{time.time()-t0:5.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

SETS_DATA = {"muresk_anthesis": ("muresk", "2025-09-30"), "muresk_maturity": ("muresk", "2025-11-21"), "agt_anthesis": ("agt", "2025-09-22"), "agt_maturity": ("agt", "2025-11-13_merged")}
BOOK = {"Plot_ID", "trial", "flight", "stage", "has_ground_truth", "valid_lidar", "valid_vnir", "valid_vnir_lenient", "valid_fused", "area_m2", "qc_ok", "ground_source",
        "n_raw", "n_noise_removed", "noise_rule", "ground_cells", "ground_rms_cm", "I_ring_median_raw", "region_px", "frac_in_cube", "veg_ndvi_threshold", "red_ok",
        "vnir_reflectance_ok", "vnir_qc_ok", "cth_ok", "n_interior", "interior_area_frac", "source_flight", "height_source", "has_ground_truth"}
LIDAR_CORE = list(dict.fromkeys(LIDAR_V3_CORE + ["PVI_m", "I_canopy_x_cover", "Pgap_below_020cm", "Pgap_below_040cm", "Pgap_below_060cm", "dens_1", "dens_5", "dens_9", "cover_5cm", "Pgap"]))
VNIR_CORE = list(VNIR_CORE_BIOMASS)
def add_weighted(d):
    for k, (a, b) in WEIGHTED.items():
        if a in d.columns and b in d.columns: d[k] = d[a] * d[b]
    return d
WCOLS = list(WEIGHTED)
tables = {}; lidar_all_cols = None; vnir_all_cols = None
for name, (trial, key) in SETS_DATA.items():
    d = pd.read_csv(os.path.join(DS, trial, key, "features_all.csv"))
    src = key if key != "2025-11-13_merged" else "2025-11-13_f1"
    L = pd.read_csv(os.path.join(DS, trial, src, "features_lidar.csv")); V = pd.read_csv(os.path.join(DS, trial, src, "features_vnir.csv"))
    lc = [c for c in L.columns if c not in BOOK and L[c].dtype != object]; vc = [c for c in V.columns if c not in BOOK and V[c].dtype != object and not c.endswith("_valid")]
    lidar_all_cols = lc if lidar_all_cols is None else [c for c in lidar_all_cols if c in lc]; vnir_all_cols = vc if vnir_all_cols is None else [c for c in vnir_all_cols if c in vc]
    d = add_weighted(d[(d.has_ground_truth == 1) & (d.valid_lidar == 1) & d.biomass_kg_ha.notna()].reset_index(drop=True))
    d["trial_flag"] = 1.0 if trial == "agt" else 0.0; d["stage_flag"] = 1.0 if name.endswith("maturity") else 0.0
    # vegetation mean spectra (valid bands only) for the band-PCA families; merged AGT maturity takes each plot's source flight
    from pca_families import spectra_matrix, SB_PREFIX
    srcs = sorted(d.source_flight.unique()) if "source_flight" in d.columns else [key]
    parts = []
    for sk in srcs:
        spf = os.path.join(DS, trial, sk, "vnir_spectra.csv")
        if os.path.exists(spf):
            SP = pd.read_csv(spf)
            if SP.drop(columns=["Plot_ID"]).notna().any().any():
                SB = spectra_matrix(SP); SB = SB.loc[SB.index.isin(d.loc[d.source_flight == sk, "Plot_ID"] if "source_flight" in d.columns else d.Plot_ID)]; parts.append(SB)
    if parts:
        SB = pd.concat(parts); SB = SB.loc[:, SB.notna().mean(0) >= 0.95]; d = d.merge(SB, left_on="Plot_ID", right_index=True, how="left")
    band_cols = [c for c in d.columns if c.startswith(SB_PREFIX)]
    log(f"{name}: spectral band columns for band-PCA: {len(band_cols)}")
    tables[name] = d; d.to_csv(os.path.join(FEA, f"features_{name}.csv"), index=False)
    log(f"{name}: {len(d)} plots with GT + LiDAR QC ok; vnir_qc_ok {int(d.vnir_qc_ok.fillna(0).sum())}; red_ok {int(d.red_ok.fillna(0).sum())}; "
        f"reflectance_ok {int(d.vnir_reflectance_ok.fillna(0).max())}; biomass mean {d.biomass_kg_ha.mean():.0f} sd {d.biomass_kg_ha.std():.0f}")
def available(cols, d):
    return [c for c in cols if c in d.columns and np.isfinite(d[c].astype(float)).mean() > 0.95 and d[c].astype(float).std() > 0]
def learners(nfeat):
    L = {"Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "PLS": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(8, nfeat) + 1))}, cv=5, scoring="neg_root_mean_squared_error"),
         "SVR_rbf": GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": [100, 300, 1000, 3000], "svr__epsilon": [50, 150, 300], "svr__gamma": ["scale", 0.03]}, cv=5, scoring="neg_root_mean_squared_error"),
         "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(nfeat) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0)),
         "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1),
         "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1),
         "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=3000, early_stopping=True, random_state=0))}
    if xgb is not None: L["XGBoost"] = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0)
    if lgb is not None: L["LightGBM"] = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, verbose=-1)
    return L
def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / y) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), rRMSE=float(np.sqrt(np.mean(e ** 2)) / y.mean() * 100), MAE=float(np.abs(e).mean()),
                bias=float(e.mean()), r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, MAPE=mape, accuracy=100 - mape)
def nested_select(Xtr, ytr, feats, k):
    if k is None or k >= len(feats): return list(range(len(feats)))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(Xtr, ytr); return list(np.argsort(rf.feature_importances_)[::-1][:k])
def cv_run(X, y, model, splits, feats, k):
    ps = np.zeros(len(y)); cnt = np.zeros(len(y)); selc = np.zeros(len(feats))
    for tr, te in splits:
        idx = nested_select(X[tr], y[tr], feats, k); selc[idx] += 1
        m = clone(model).fit(X[tr][:, idx], y[tr]); ps[te] += m.predict(X[te][:, idx]).ravel(); cnt[te] += 1
    return ps / np.maximum(cnt, 1), selc
def family_sets(d):
    return {"LiDAR_core": available(LIDAR_CORE, d), "VNIR_core": available(VNIR_CORE, d), "Fused_core": available(LIDAR_CORE + VNIR_CORE + WCOLS, d),
            "Fused_all": available(lidar_all_cols + vnir_all_cols + WCOLS, d)}
from pca_families import pca_preprocessor, pca_learners, pca_blocks, pca_summary, SB_PREFIX
PCA_FAMILIES = ["LiDAR_PCA", "VNIR_PCA", "Fused_PCA", "Fused_PCA_joint", "VNIR_bands_PCA", "Fused_bands_PCA"]
USES_LIDAR = {"LiDAR_PCA", "Fused_PCA", "Fused_PCA_joint", "Fused_bands_PCA"}; USES_VNIR_IDX = {"VNIR_PCA", "Fused_PCA", "Fused_PCA_joint"}; USES_BANDS = {"VNIR_bands_PCA", "Fused_bands_PCA"}
SVR_C_BIO, SVR_EPS_BIO = [100, 300, 1000, 3000], [50, 150, 300]
def pca_inputs(family, d_):
    """(columns, blocks, n_lidar, n_vnir) for a PCA family: all finite LiDAR features / VNIR index features / valid spectral bands"""
    lc = available(lidar_all_cols, d_) if family in USES_LIDAR else []
    vc = available(vnir_all_cols, d_) if family in USES_VNIR_IDX else []
    bc = available([c for c in d_.columns if c.startswith(SB_PREFIX)], d_) if family in USES_BANDS else []
    return lc + vc + bc, pca_blocks(family, len(lc), len(vc), len(bc)), len(lc), len(vc) + len(bc)
pca_rows = []

rows, preds, selrows, skipped = [], [], [], []
for name, d in tables.items():
    for sname, fl_ in family_sets(d).items():
        uses_vnir = sname != "LiDAR_core"
        d_ = d[d.vnir_qc_ok.fillna(0) == 1] if uses_vnir else d
        vn = [c for c in fl_ if c in VNIR_CORE or c in vnir_all_cols or c in WCOLS]
        if uses_vnir and (len(d_) < 20 or (sname != "Fused_all" and not vn) or (sname == "VNIR_core" and not fl_)):
            skipped.append(dict(dataset=name, feature_set=sname, reason=f"{len(d_)} plots with usable VNIR / {len(vn)} VNIR features")); log(f"{name} {sname}: skipped ({skipped[-1]['reason']})"); continue
        if sname == "Fused_all" and uses_vnir and not vn:
            fl_ = available(lidar_all_cols, d_)
        y_ = d_.biomass_kg_ha.to_numpy(float); X = d_[fl_].fillna(d_[fl_].median()).to_numpy(float); k = 12 if len(fl_) > 15 else None
        splits = list(RepeatedKFold(n_splits=10, n_repeats=5, random_state=0).split(X))
        for lname, model in learners(min(len(fl_), k or len(fl_))).items():
            t1 = time.time(); p, selc = cv_run(X, y_, model, splits, fl_, k); m = metrics(p, y_)
            rows.append(dict(scheme="repeated10fold", dataset=name, trial=d_.trial.iloc[0], stage=d_.stage.iloc[0], feature_set=sname, n_features=len(fl_), n_vnir_features=len(vn), n_selected=k or len(fl_), learner=lname, n=len(y_), **m, seconds=round(time.time() - t1)))
            preds.append(pd.DataFrame(dict(scheme="repeated10fold", dataset=name, feature_set=sname, learner=lname, Plot_ID=d_.Plot_ID, Plot=d_.Plot, Variety=d_.Variety, measured=y_, predicted=p)))
            if k: selrows.append(pd.DataFrame(dict(dataset=name, feature_set=sname, learner=lname, feature=fl_, selection_frequency=selc / len(splits))))
            log(f"{name:16s} {sname:11s} {lname:12s} n {len(y_)} feats {len(fl_)} R2 {m['R2']:.3f} RMSE {m['RMSE']:.0f} rRMSE {m['rRMSE']:.1f}% ({time.time()-t1:.0f}s)")
    # ---- PCA families (PCA fitted inside each training fold)
    for fam in PCA_FAMILIES:
        uses_vnir = fam != "LiDAR_PCA"
        d_ = d[d.vnir_qc_ok.fillna(0) == 1] if uses_vnir else d
        cols, blocks, nl, nv = pca_inputs(fam, d_)
        if (uses_vnir and (nv < 2 or len(d_) < 20)) or (fam in USES_LIDAR and nl < 2):
            skipped.append(dict(dataset=name, feature_set=fam, reason=f"{len(d_)} plots, {nl} LiDAR / {nv} VNIR features")); log(f"{name} {fam}: skipped ({skipped[-1]['reason']})"); continue
        y_ = d_.biomass_kg_ha.to_numpy(float); X = d_[cols].fillna(d_[cols].median()).to_numpy(float)
        pre = pca_preprocessor(blocks); splits = list(RepeatedKFold(n_splits=10, n_repeats=5, random_state=0).split(X))
        pre_full = clone(pre).fit(X); ps = pca_summary(pre_full); pca_rows.append(dict(dataset=name, feature_set=fam, n_inputs=len(cols), **{f"{k}_{kk}": v for k, s in ps.items() for kk, v in s.items()}))
        for lname, model in pca_learners(pre, SVR_C_BIO, SVR_EPS_BIO).items():
            t1 = time.time(); p, _ = cv_run(X, y_, model, splits, cols, None); m = metrics(p, y_)
            rows.append(dict(scheme="repeated10fold", dataset=name, trial=d_.trial.iloc[0], stage=d_.stage.iloc[0], feature_set=fam, n_features=len(cols), n_vnir_features=nv,
                             n_selected=sum(s["n_components"] for s in ps.values()), learner=lname, n=len(y_), **m, seconds=round(time.time() - t1)))
            preds.append(pd.DataFrame(dict(scheme="repeated10fold", dataset=name, feature_set=fam, learner=lname, Plot_ID=d_.Plot_ID, Plot=d_.Plot, Variety=d_.Variety, measured=y_, predicted=p)))
            log(f"{name:16s} {fam:15s} {lname:12s} n {len(y_)} inputs {len(cols)} PCs {ps} R2 {m['R2']:.3f} RMSE {m['RMSE']:.0f} rRMSE {m['rRMSE']:.1f}% ({time.time()-t1:.0f}s)")
# pooled (all datasets) with trial/stage flags, grouped 10-fold by trial-plot
P_all = pd.concat(tables.values(), ignore_index=True); P_all["group"] = P_all.trial + "_" + P_all.Plot_ID.astype(str); yP = P_all.biomass_kg_ha.to_numpy(float)
for sname, cols in (("Fused_core+flags", available(LIDAR_CORE + VNIR_CORE + WCOLS, P_all) + ["trial_flag", "stage_flag"]), ("LiDAR_core+flags", available(LIDAR_CORE, P_all) + ["trial_flag", "stage_flag"])):
    X = P_all[cols].fillna(P_all[cols].median()).to_numpy(float); k = 12 if len(cols) > 15 else None
    splits = list(GroupKFold(n_splits=10).split(X, yP, groups=P_all.group))
    for lname, model in learners(min(len(cols), k or len(cols))).items():
        p, _ = cv_run(X, yP, model, splits, cols, k)
        for name in tables:
            sel = ((P_all.trial + "_" + P_all.stage) == name).to_numpy(); m = metrics(p[sel], yP[sel])
            rows.append(dict(scheme="pooled_group10fold", dataset=name, trial=name.split("_")[0], stage=name.split("_")[1], feature_set=sname, n_features=len(cols), n_selected=k or len(cols), learner=lname, n=int(sel.sum()), **m))
        preds.append(pd.DataFrame(dict(scheme="pooled_group10fold", dataset=P_all.trial + "_" + P_all.stage, feature_set=sname, learner=lname, Plot_ID=P_all.Plot_ID, Plot=P_all.Plot, Variety=P_all.Variety, measured=yP, predicted=p)))
        log(f"pooled {sname} {lname}: overall R2 {metrics(p, yP)['R2']:.3f} RMSE {metrics(p, yP)['RMSE']:.0f}")
for fam in ("LiDAR_PCA", "Fused_PCA"):
    Pv = P_all if fam == "LiDAR_PCA" else P_all[P_all.vnir_qc_ok.fillna(0) == 1]
    cols, blocks, nl, nv = pca_inputs(fam, Pv)
    if fam == "Fused_PCA" and nv < 2: continue
    cols2 = cols + ["trial_flag", "stage_flag"]; X = Pv[cols2].fillna(Pv[cols2].median()).to_numpy(float); yv = Pv.biomass_kg_ha.to_numpy(float)
    pre = pca_preprocessor(blocks); splits = list(GroupKFold(n_splits=10).split(X, yv, groups=Pv.group))
    for lname, model in pca_learners(pre, SVR_C_BIO, SVR_EPS_BIO).items():
        p, _ = cv_run(X, yv, model, splits, cols2, None)
        for name in tables:
            sel = ((Pv.trial + "_" + Pv.stage) == name).to_numpy()
            if sel.sum() < 10: continue
            m = metrics(p[sel], yv[sel])
            rows.append(dict(scheme="pooled_group10fold", dataset=name, trial=name.split("_")[0], stage=name.split("_")[1], feature_set=f"{fam}+flags", n_features=len(cols2), n_selected=0, learner=lname, n=int(sel.sum()), **m))
        preds.append(pd.DataFrame(dict(scheme="pooled_group10fold", dataset=Pv.trial + "_" + Pv.stage, feature_set=f"{fam}+flags", learner=lname, Plot_ID=Pv.Plot_ID, Plot=Pv.Plot, Variety=Pv.Variety, measured=yv, predicted=p)))
        log(f"pooled {fam}+flags {lname}: overall R2 {metrics(p, yv)['R2']:.3f} RMSE {metrics(p, yv)['RMSE']:.0f}")
# cross-trial transfer per stage
for stage in ("anthesis", "maturity"):
    for tr, te in (("muresk", "agt"), ("agt", "muresk")):
        dtr = tables[f"{tr}_{stage}"]; dte = tables[f"{te}_{stage}"]
        for sname, base_cols in (("LiDAR_core", LIDAR_CORE), ("Fused_core", LIDAR_CORE + VNIR_CORE + WCOLS)):
            cols = [c for c in available(base_cols, dtr) if c in available(base_cols, dte)]
            if sname == "Fused_core" and not [c for c in cols if c not in LIDAR_CORE]: continue
            Xtr = dtr[cols].fillna(dtr[cols].median()).to_numpy(float); Xte = dte[cols].fillna(dte[cols].median()).to_numpy(float); ytr = dtr.biomass_kg_ha.to_numpy(float); yte = dte.biomass_kg_ha.to_numpy(float)
            idx = nested_select(Xtr, ytr, cols, 12 if len(cols) > 15 else None)
            for lname in ("Ridge", "RandomForest", "XGBoost", "SVR_rbf"):
                if lname not in learners(len(idx)): continue
                m_ = clone(learners(len(idx))[lname]).fit(Xtr[:, idx], ytr); p = m_.predict(Xte[:, idx]).ravel(); m = metrics(p, yte)
                rows.append(dict(scheme="cross_trial", dataset=f"{te}_{stage}", trial=te, stage=stage, feature_set=sname, n_features=len(cols), n_selected=len(idx), learner=lname, n=len(yte), trained_on=f"{tr}_{stage}", **m))
                preds.append(pd.DataFrame(dict(scheme="cross_trial", dataset=f"{te}_{stage}", feature_set=sname, learner=lname, Plot_ID=dte.Plot_ID, Plot=dte.Plot, Variety=dte.Variety, measured=yte, predicted=p)))
                log(f"transfer {tr}->{te} {stage} {sname} {lname}: R2 {m['R2']:.2f} RMSE {m['RMSE']:.0f} bias {m['bias']:+.0f} r {m['r']:.2f}")
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
PP = pd.concat(preds, ignore_index=True); PP["error"] = PP.predicted - PP.measured; PP.to_csv(os.path.join(RES, "per_plot_predictions.csv"), index=False)
(pd.concat(selrows, ignore_index=True) if selrows else pd.DataFrame()).to_csv(os.path.join(RES, "feature_selection_frequency.csv"), index=False)
pd.DataFrame(skipped).to_csv(os.path.join(RES, "skipped_families.csv"), index=False)
# top features per dataset + saved best models
top_rows, best_rows = [], []
for name, d in tables.items():
    fs = family_sets(d); run = R[(R.scheme == "repeated10fold") & (R.dataset == name)]
    if run.empty: continue
    best = run.nsmallest(1, "RMSE").iloc[0]; fam = best.feature_set
    d_ = d[d.vnir_qc_ok.fillna(0) == 1] if fam not in ("LiDAR_core", "LiDAR_PCA") else d; y = d_.biomass_kg_ha.to_numpy(float)
    cols = fs["Fused_all"] if fam not in ("LiDAR_core", "LiDAR_PCA") else fs["LiDAR_core"]
    if fam not in ("LiDAR_core", "LiDAR_PCA") and not [c for c in cols if c not in lidar_all_cols]: cols = available(lidar_all_cols, d_)
    X = d_[cols].fillna(d_[cols].median()).to_numpy(float)
    rf = RandomForestRegressor(n_estimators=800, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y)
    top_rows.append(pd.DataFrame({"dataset": name, "feature": cols, "rf_importance": rf.feature_importances_, "r_with_biomass": [np.corrcoef(d_[c].fillna(d_[c].median()), y)[0, 1] for c in cols]}).sort_values("rf_importance", ascending=False))
    if fam in PCA_FAMILIES:
        pc_cols, blocks, nl, nv = pca_inputs(fam, d_); Xb = d_[pc_cols].fillna(d_[pc_cols].median()).to_numpy(float)
        mb = clone(pca_learners(pca_preprocessor(blocks), SVR_C_BIO, SVR_EPS_BIO)[best.learner]).fit(Xb, y); selb = pc_cols
    else:
        fl_ = fs[fam] if fs[fam] else cols; Xb = d_[fl_].fillna(d_[fl_].median()).to_numpy(float); idx = nested_select(Xb, y, fl_, 12 if len(fl_) > 15 else None); selb = [fl_[i] for i in idx]
        mb = clone(learners(len(idx))[best.learner]).fit(Xb[:, idx], y)
    pth = os.path.join(MOD, f"biomass_{name}_{fam}_{best.learner}.joblib")
    joblib.dump({"model": mb, "features": selb, "pca_family": fam in PCA_FAMILIES, "target": "biomass_kg_ha", "dataset": name, "cv": best.to_dict(), "data": "Dataset_2026-10-09"}, pth)
    best_rows.append(dict(dataset=name, family=fam, learner=best.learner, R2=best.R2, RMSE=best.RMSE, rRMSE=best.rRMSE, n=best.n, features=", ".join(selb), model_file=os.path.basename(pth)))
    for fam2 in PCA_FAMILIES:                                           # also keep the best model of each PCA family
        r2 = run[run.feature_set == fam2]
        if r2.empty: continue
        b2 = r2.nsmallest(1, "RMSE").iloc[0]; d2 = d[d.vnir_qc_ok.fillna(0) == 1] if fam2 != "LiDAR_PCA" else d; y2 = d2.biomass_kg_ha.to_numpy(float)
        pc_cols, blocks, nl, nv = pca_inputs(fam2, d2); X2 = d2[pc_cols].fillna(d2[pc_cols].median()).to_numpy(float)
        m2 = clone(pca_learners(pca_preprocessor(blocks), SVR_C_BIO, SVR_EPS_BIO)[b2.learner]).fit(X2, y2)
        joblib.dump({"model": m2, "features": pc_cols, "pca_family": True, "target": "biomass_kg_ha", "dataset": name, "cv": b2.to_dict(), "data": "Dataset_2026-10-09"}, os.path.join(MOD, f"biomass_{name}_{fam2}_{b2.learner}.joblib"))
pd.DataFrame(pca_rows).to_csv(os.path.join(RES, "pca_components.csv"), index=False)
pd.concat(top_rows).to_csv(os.path.join(RES, "top_features.csv"), index=False); pd.DataFrame(best_rows).to_csv(os.path.join(RES, "best_models.csv"), index=False)
json.dump(dict(LIDAR_CORE=LIDAR_CORE, VNIR_CORE=VNIR_CORE, WEIGHTED=WCOLS, lidar_all=lidar_all_cols, vnir_all=vnir_all_cols), open(os.path.join(RES, "feature_sets.json"), "w"), indent=1)
pd.set_option("display.width", 240)
log("\n" + R[R.scheme == "repeated10fold"].pivot_table(index=["feature_set", "learner"], columns="dataset", values="R2").round(2).to_string())
log("BIOMASS DONE")
