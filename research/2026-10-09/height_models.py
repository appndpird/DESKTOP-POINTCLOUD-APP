"""
Plant-height models (Muresk NUE 25NO43, both flights) on Dataset_2026-10-09: LiDAR v3.2 core (22 features), all LiDAR
features, and LiDAR + cleaned VNIR v3.0.1 (NaN where a band is unusable; VNIR columns median-filled per flight).
Validation as in height_models_v2: nested repeated 10-fold (5 repeats, RF top-10 selection inside the fold), LOOCV for
the top 3 models + the cth_p95 linear reference, leave-one-flight-out transfer (raw and 5-plot offset).
Outputs: <Models_2026-10-09>/height/results/*, models/*.joblib
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
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, GridSearchCV
import joblib
try: import xgboost as xgb
except Exception: xgb = None
try: import lightgbm as lgb
except Exception: lgb = None
TOOL = r"C:\Users\appn\Downloads\DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06"; sys.path.insert(0, TOOL)
from phenoapp.core.lidar_features import LIDAR_V3_CORE
from phenoapp.core.vnir_features import VNIR_V3_CORE
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(ROOT, "dataset")
HERE = os.path.join(ROOT, "height"); RES = os.path.join(HERE, "results"); MOD = os.path.join(HERE, "models")
for d in (RES, os.path.join(RES, "figures"), MOD): os.makedirs(d, exist_ok=True)
t0 = time.time(); LOGF = open(os.path.join(HERE, "height_models.log"), "a")
def log(*a):
    s = f"[{time.time()-t0:5.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

FL = {"2025-09-30": "anthesis", "2025-11-21": "maturity"}
BOOK = {"Plot_ID", "trial", "flight", "stage", "has_ground_truth", "valid_lidar", "valid_vnir", "valid_vnir_lenient", "valid_fused", "area_m2", "qc_ok", "ground_source",
        "n_raw", "n_noise_removed", "noise_rule", "ground_cells", "ground_rms_cm", "I_ring_median_raw", "region_px", "frac_in_cube", "veg_ndvi_threshold", "red_ok",
        "vnir_reflectance_ok", "vnir_qc_ok", "cth_ok", "n_interior", "interior_area_frac"}
data = {}; lidar_cols = None
for fl in FL:
    d = pd.read_csv(os.path.join(DS, "muresk", fl, "features_all.csv"))
    L = pd.read_csv(os.path.join(DS, "muresk", fl, "features_lidar.csv")); V = pd.read_csv(os.path.join(DS, "muresk", fl, "features_vnir.csv"))
    d = d[(d.has_ground_truth == 1) & (d.valid_lidar == 1) & d.height_cm.notna()].reset_index(drop=True)
    for c in [c for c in d.columns if c.startswith(("H", "cth_", "tip_thin", "roughness", "Hall", "profile_area")) and c not in ("H_cv", "H_skew", "H_kurt", "H_crr")]:
        if d[c].dtype != object: d[c] = d[c] * 100.0          # m -> cm
    vcols = [c for c in V.columns if c not in BOOK and c != "Plot_ID" and not c.endswith("_valid") and V[c].dtype != object]
    nan_frac = d[vcols].isna().mean()
    d[vcols] = d[vcols].fillna(d[vcols].median())             # NaN rule cells -> flight median (VNIR columns only)
    from pca_families import spectra_matrix, SB_PREFIX
    SB = spectra_matrix(pd.read_csv(os.path.join(DS, "muresk", fl, "vnir_spectra.csv")))   # valid spectral bands, log + SNV, for the band-PCA families
    d = d.merge(SB, left_on="Plot_ID", right_index=True, how="left")
    d["flight"] = fl; d["stage_flag"] = 1.0 if fl == "2025-11-21" else 0.0; data[fl] = d
    lc = [c for c in L.columns if c not in BOOK and c != "Plot_ID" and L[c].dtype != object]
    lidar_cols = lc if lidar_cols is None else [c for c in lidar_cols if c in lc]
    vnir_cols = vcols if "vnir_cols" not in dir() else [c for c in vnir_cols if c in vcols]
    log(f"{fl}: {len(d)} plots; VNIR columns with NaN (filled): {int((nan_frac > 0).sum())}, max NaN share {nan_frac.max():.2f}")
lidar_all = [c for c in lidar_cols if all(np.isfinite(data[f][c]).all() and data[f][c].std() > 0 for f in FL)]
lidar_core = [c for c in LIDAR_V3_CORE if c in lidar_all]
vnir_core = [c for c in VNIR_V3_CORE if all(c in data[f].columns and np.isfinite(data[f][c]).all() and data[f][c].std() > 0 for f in FL)]
vnir_all = [c for c in vnir_cols if all(np.isfinite(data[f][c]).all() and data[f][c].std() > 0 for f in FL)]
bands_all = [c for c in data["2025-09-30"].columns if c.startswith(SB_PREFIX) and all(c in data[f].columns and np.isfinite(data[f][c]).all() and data[f][c].std() > 0 for f in FL)]
SETS = {"LiDAR_core": lidar_core, "LiDAR_all": lidar_all, "LiDAR+VNIR": lidar_core + vnir_core}
# PCA families: PCA fitted inside each training fold on all finite LiDAR features / VNIR index features / valid spectral bands
from pca_families import pca_preprocessor, pca_learners, pca_summary
PCA_SETS = {"LiDAR_PCA": [("lidar", lidar_all)], "LiDAR+VNIR_PCA": [("lidar", lidar_all), ("vnir", vnir_all)], "LiDAR+VNIR_PCA_joint": [("joint", lidar_all + vnir_all)],
            "VNIR_bands_PCA": [("bands", bands_all)], "LiDAR+VNIR_bands_PCA": [("lidar", lidar_all), ("bands", bands_all)]}
SVR_C_H, SVR_EPS_H = [1, 3, 10, 30, 100], [0.5, 1.0, 2.0]
def pca_cols_blocks(fname):
    cols, blocks, off = [], [], 0
    for n_, cs in PCA_SETS[fname]: blocks.append((n_, list(range(off, off + len(cs))))); cols += cs; off += len(cs)
    return cols, blocks
ALL_SETS = {**SETS, **{f: pca_cols_blocks(f)[0] for f in PCA_SETS}}
def learner_dict(fname, nfeat):
    return pca_learners(pca_preprocessor(pca_cols_blocks(fname)[1]), SVR_C_H, SVR_EPS_H) if fname in PCA_SETS else learners(nfeat)
def k_for(fname, feats): return None if fname in PCA_SETS else (10 if len(feats) > 12 else None)
log("feature sets:", {k: len(v) for k, v in ALL_SETS.items()}, "| lidar_core:", lidar_core, "| vnir_core:", vnir_core, "| vnir_all:", len(vnir_all), "| valid spectral bands:", len(bands_all))
json.dump({k: v for k, v in ALL_SETS.items()}, open(os.path.join(RES, "feature_sets.json"), "w"), indent=1)
pca_rows = []
for fl, d in data.items():
    for f in PCA_SETS:
        cols, blocks = pca_cols_blocks(f); ps = pca_summary(clone(pca_preprocessor(blocks)).fit(d[cols].to_numpy(float)))
        pca_rows.append(dict(flight=fl, feature_set=f, n_inputs=len(cols), **{f"{k}_{kk}": v for k, s in ps.items() for kk, v in s.items()}))
pd.DataFrame(pca_rows).to_csv(os.path.join(RES, "pca_components.csv"), index=False)

def learners(nfeat):
    L = {"Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "PLS": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(8, nfeat) + 1))}, cv=5, scoring="neg_root_mean_squared_error"),
         "SVR_rbf": GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": [1, 3, 10, 30, 100], "svr__epsilon": [0.5, 1.0, 2.0], "svr__gamma": ["scale", 0.03, 0.1]}, cv=5, scoring="neg_root_mean_squared_error"),
         "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(length_scale=np.ones(nfeat) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=2, random_state=0)),
         "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1),
         "ExtraTrees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1),
         "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=3000, early_stopping=True, random_state=0))}
    if xgb is not None: L["XGBoost"] = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, n_jobs=4, verbosity=0)
    if lgb is not None: L["LightGBM"] = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, verbose=-1)
    return L
def metrics(p, y):
    e = p - y; mape = float(np.mean(np.abs(e) / y) * 100)
    return dict(R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.abs(e).mean()), bias=float(e.mean()),
                r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, MAPE=mape, accuracy=100 - mape, within5cm=float(np.mean(np.abs(e) <= 5) * 100))
def nested_select(Xtr, ytr, feats, k):
    if k is None or k >= len(feats): return list(range(len(feats)))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    return list(np.argsort(rf.feature_importances_)[::-1][:k])
def cv_run(X, y, model, splitter, feats, k=None):
    pred_sum = np.zeros(len(y)); cnt = np.zeros(len(y)); selcount = np.zeros(len(feats))
    for tr, te in splitter.split(X):
        idx = nested_select(X[tr], y[tr], feats, k); selcount[idx] += 1
        m = clone(model).fit(X[tr][:, idx], y[tr]); pred_sum[te] += m.predict(X[te][:, idx]).ravel(); cnt[te] += 1
    return pred_sum / np.maximum(cnt, 1), selcount

rows, preds, imps = [], [], []
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
for fl, d in data.items():
    y = d.height_cm.to_numpy(float)
    for sname, feats in ALL_SETS.items():
        X = d[feats].to_numpy(float); k = k_for(sname, feats)
        for lname, model in learner_dict(sname, min(len(feats), k or len(feats))).items():
            t1 = time.time(); p, selc = cv_run(X, y, model, rkf, feats, k); m = metrics(p, y)
            rows.append(dict(scheme="repeated10fold", flight=fl, stage=FL[fl], feature_set=sname, n_features=len(feats), n_selected=k or len(feats), learner=lname, n=len(y), **m, seconds=round(time.time() - t1)))
            preds.append(pd.DataFrame(dict(scheme="repeated10fold", flight=fl, feature_set=sname, learner=lname, Plot_ID=d.Plot_ID, Plot=d.Plot, Range=d.Range, Variety=d.Variety, measured=y, predicted=p)))
            if k: imps.append(pd.DataFrame(dict(flight=fl, feature_set=sname, learner=lname, feature=feats, selection_frequency=selc / 50)))
            log(f"{fl} {sname:11s} {lname:12s} R2 {m['R2']:.3f} RMSE {m['RMSE']:.2f} acc {m['accuracy']:.1f}% ({time.time()-t1:.0f}s)")
R = pd.DataFrame(rows)
for fl, d in data.items():
    y = d.height_cm.to_numpy(float); top = R[(R.flight == fl) & (R.scheme == "repeated10fold")].nsmallest(3, "RMSE")
    for _, t in top.iterrows():
        feats = ALL_SETS[t.feature_set]; X = d[feats].to_numpy(float); k = k_for(t.feature_set, feats)
        p, _ = cv_run(X, y, learner_dict(t.feature_set, min(len(feats), k or len(feats)))[t.learner], LeaveOneOut(), feats, k); m = metrics(p, y)
        rows.append(dict(scheme="LOOCV", flight=fl, stage=FL[fl], feature_set=t.feature_set, n_features=len(feats), n_selected=k or len(feats), learner=t.learner, n=len(y), **m))
        preds.append(pd.DataFrame(dict(scheme="LOOCV", flight=fl, feature_set=t.feature_set, learner=t.learner, Plot_ID=d.Plot_ID, Plot=d.Plot, Range=d.Range, Variety=d.Variety, measured=y, predicted=p)))
        log(f"LOOCV {fl} {t.feature_set} {t.learner}: R2 {m['R2']:.3f} RMSE {m['RMSE']:.2f}")
    X = d[["cth_p95"]].to_numpy(float); p, _ = cv_run(X, y, make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-6])), LeaveOneOut(), ["cth_p95"]); m = metrics(p, y)
    rows.append(dict(scheme="LOOCV", flight=fl, stage=FL[fl], feature_set="cth_p95_only", n_features=1, n_selected=1, learner="Linear", n=len(y), **m))
    preds.append(pd.DataFrame(dict(scheme="LOOCV", flight=fl, feature_set="cth_p95_only", learner="Linear", Plot_ID=d.Plot_ID, Plot=d.Plot, Range=d.Range, Variety=d.Variety, measured=y, predicted=p)))
rng = np.random.default_rng(0)
for tr_fl, te_fl in (("2025-09-30", "2025-11-21"), ("2025-11-21", "2025-09-30")):
    ytr = data[tr_fl].height_cm.to_numpy(float); yte = data[te_fl].height_cm.to_numpy(float)
    for sname, feats in ALL_SETS.items():
        Xtr = data[tr_fl][feats].to_numpy(float); Xte = data[te_fl][feats].to_numpy(float); k = k_for(sname, feats)
        idx = nested_select(Xtr, ytr, feats, k); LD = learner_dict(sname, len(idx))
        for lname in ("Ridge", "RandomForest", "XGBoost", "SVR_rbf", "GPR"):
            if lname not in LD: continue
            m_ = clone(LD[lname]).fit(Xtr[:, idx], ytr); p = m_.predict(Xte[:, idx]).ravel(); m = metrics(p, yte)
            rm = []
            for _ in range(30):
                kk = rng.choice(len(yte), 5, replace=False); off = np.mean(yte[kk] - p[kk]); rest = np.setdiff1d(np.arange(len(yte)), kk); rm.append(np.sqrt(np.mean((p[rest] + off - yte[rest]) ** 2)))
            rows.append(dict(scheme="leave_flight_out_raw", flight=te_fl, stage=FL[te_fl], feature_set=sname, n_features=len(feats), n_selected=len(idx), learner=lname, n=len(yte), trained_on=tr_fl, **m))
            rows.append(dict(scheme="leave_flight_out_5plot_offset", flight=te_fl, stage=FL[te_fl], feature_set=sname, n_features=len(feats), n_selected=len(idx), learner=lname, n=len(yte), trained_on=tr_fl, RMSE=float(np.mean(rm)), r=m["r"]))
            preds.append(pd.DataFrame(dict(scheme="leave_flight_out_raw", flight=te_fl, feature_set=sname, learner=lname, Plot_ID=data[te_fl].Plot_ID, Plot=data[te_fl].Plot, Range=data[te_fl].Range, Variety=data[te_fl].Variety, measured=yte, predicted=p)))
            log(f"transfer {tr_fl}->{te_fl} {sname} {lname}: raw RMSE {m['RMSE']:.2f} bias {m['bias']:+.1f} | 5-plot offset {np.mean(rm):.2f}")
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
P = pd.concat(preds, ignore_index=True); P["error"] = P.predicted - P.measured; P.to_csv(os.path.join(RES, "per_plot_predictions.csv"), index=False)
if imps: pd.concat(imps).to_csv(os.path.join(RES, "feature_selection_frequency.csv"), index=False)
best_rows = []
for fl, d in data.items():
    y = d.height_cm.to_numpy(float)
    for fam in ALL_SETS:
        sub = R[(R.scheme == "repeated10fold") & (R.flight == fl) & (R.feature_set == fam)].nsmallest(1, "RMSE").iloc[0]
        feats = ALL_SETS[fam]; X = d[feats].to_numpy(float); k = k_for(fam, feats); idx = nested_select(X, y, feats, k); sel = [feats[i] for i in idx]
        m_ = clone(learner_dict(fam, len(idx))[sub.learner]).fit(X[:, idx], y)
        p = os.path.join(MOD, f"height_{fam.replace('+', '_')}_{sub.learner}_{fl}.joblib"); joblib.dump({"model": m_, "features": sel, "pca_family": fam in PCA_SETS, "units": "cm", "flight": fl, "cv": sub.to_dict(), "dataset": "Dataset_2026-10-09"}, p)
        best_rows.append(dict(flight=fl, family=fam, learner=sub.learner, R2=sub.R2, RMSE=sub.RMSE, accuracy=sub.accuracy, features=", ".join(sel), model_file=os.path.basename(p)))
        if fam not in PCA_SETS:
            rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y)
            pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}).sort_values("importance", ascending=False).to_csv(os.path.join(RES, f"rf_importance_{fam.replace('+', '_')}_{fl}.csv"), index=False)
pd.DataFrame(best_rows).to_csv(os.path.join(RES, "best_models.csv"), index=False)
pd.set_option("display.width", 220)
log("\n" + R[R.scheme == "repeated10fold"].pivot_table(index=["feature_set", "learner"], columns="flight", values=["R2", "RMSE"]).round(3).to_string())
log("HEIGHT DONE")
