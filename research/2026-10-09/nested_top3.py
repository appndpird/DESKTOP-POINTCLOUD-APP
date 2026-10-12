"""
Nested selection of the top-three feature families and their mean ensemble, for biomass (per dataset) and height
(maturity flight, the only flight with a ruler measurement).

Outer loop: 10-fold CV (shuffled, seed 0) -> honest score of the whole procedure "pick the top three, average them".
Inner loop (on the outer training plots only): 5-fold CV of every candidate (feature family x learner); the best learner of
each family is kept, the three families with the lowest inner RMSE are fitted on the outer training plots and their mean
predicts the outer test plots. Feature selection (selected families) and PCA (PCA families) stay inside every fit.
Final ensemble: the same selection on all plots (inner 5-fold x 2 repeats), the three members fitted on all plots, saved as
<target>_<dataset>_top3_ensemble.joblib = {"type": "mean_ensemble", "members": [{family, learner, model, features}], ...}.

Learners in the candidate set: Ridge, PLS (selected families), GPR, RandomForest, ExtraTrees, LightGBM (the fast, stable
ones; SVR / XGBoost / MLP were never in a top three of the full runs).
Outputs: <target>\results\nested_top3_<dataset>.csv (outer-fold choices + metrics), nested_top3_summary.csv; models\*_top3_ensemble.joblib
"""
import os, sys, json, time, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
try: import lightgbm as lgb
except Exception: lgb = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_families import pca_preprocessor, spectra_matrix, SB_PREFIX
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(ROOT, "dataset")
def _opt(name):
    """value of --name <value> on the command line (None when absent)"""
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else None
PLOTS = _opt("--plots"); RUNTAG = _opt("--tag") or ""          # restrict to a plot list (dataset, Plot_ID) and suffix every output
_skip = {PLOTS, RUNTAG} - {None, ""}
TARGETS = [a for a in sys.argv[1:] if not a.startswith("--") and a not in _skip] or ["height", "biomass"]
PLOTLIST = pd.read_csv(PLOTS)[["dataset", "Plot_ID"]] if PLOTS else None
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s] " + " ".join(str(x) for x in a), flush=True)

def sel_learners(nfeat):
    L = {"Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "PLS": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression(scale=False)), {"plsregression__n_components": list(range(1, min(8, nfeat) + 1))}, cv=5, scoring="neg_root_mean_squared_error"),
         "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(nfeat) * 3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=1, random_state=0)),
         "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1),
         "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1)}
    if lgb is not None: L["LightGBM"] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=0, verbose=-1)
    return L
def pca_learners(pre):
    def pipe(est): return Pipeline([("pre", clone(pre)), ("m", est)])
    L = {"Ridge": pipe(RidgeCV(alphas=np.logspace(-2, 3, 30))),
         "GPR": pipe(GaussianProcessRegressor(ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(1.0), normalize_y=True, n_restarts_optimizer=1, random_state=0)),
         "RandomForest": pipe(RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_features=0.7, random_state=0, n_jobs=-1)),
         "ExtraTrees": pipe(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.8, random_state=0, n_jobs=-1))}
    if lgb is not None: L["LightGBM"] = pipe(lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, random_state=0, verbose=-1))
    return L
def rf_select(X, y, k):
    if k is None or k >= X.shape[1]: return np.arange(X.shape[1])
    return np.argsort(RandomForestRegressor(n_estimators=200, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y).feature_importances_)[::-1][:k]
def metrics(p, y):
    e = p - y; return dict(R2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()), bias=float(e.mean()),
                           r=float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else np.nan, rRMSE=float(np.sqrt((e ** 2).mean()) / y.mean() * 100))

class Candidate:
    """one feature family: columns, (optional) PCA blocks, k for RF selection; fit/predict on a row subset"""
    def __init__(self, name, cols, blocks=None, k=None): self.name, self.cols, self.blocks, self.k = name, cols, blocks, k
    def learners(self): return pca_learners(pca_preprocessor(self.blocks)) if self.blocks else sel_learners(min(len(self.cols), self.k or len(self.cols)))
    def fit(self, model, X, y):
        idx = rf_select(X, y, self.k); return clone(model).fit(X[:, idx], y), idx
def inner_scores(cands, Xs, y, splits):
    """RMSE of every (family, learner) on inner splits -> {(family, learner): rmse}"""
    out = {}
    for c in cands:
        X = Xs[c.name]
        for ln, model in c.learners().items():
            p = np.zeros(len(y)); n = np.zeros(len(y))
            for tr, te in splits:
                m, idx = c.fit(model, X[tr], y[tr]); p[te] += np.asarray(m.predict(X[te][:, idx])).ravel(); n[te] += 1
            out[(c.name, ln)] = float(np.sqrt(((p / np.maximum(n, 1) - y) ** 2).mean()))
    return out

def top3(sc, fams):
    """three best (family, learner) combinations within a category"""
    return sorted([(k, v) for k, v in sc.items() if k[0] in fams], key=lambda kv: kv[1])[:3]

EACH = "--each" in sys.argv          # score every individual model on the reporting folds instead of the nested selection
if EACH: TARGETS = [t for t in TARGETS if not t.startswith("--")] or ["height", "biomass"]

def each_model(target, ds, d, cands, cats, ycol, out_res):
    """held-out predictions of every (family, learner) on the same outer folds as the nested run (KFold(10, seed 0))"""
    y = d[ycol].to_numpy(float); fold = np.zeros(len(y), int); outer = list(KFold(10, shuffle=True, random_state=0).split(y))
    for k, (_, te) in enumerate(outer, 1): fold[te] = k
    cat_of = {f: c for c, fs in cats.items() for f in fs}; rows = []
    for c in cands:
        X = d[c.cols].to_numpy(float)
        for ln, model in c.learners().items():
            p = np.zeros(len(y)); t1 = time.time()
            for tr, te in outer:
                m, idx = c.fit(model, X[tr], y[tr]); p[te] = np.asarray(m.predict(X[te][:, idx])).ravel()
            rows.append(pd.DataFrame(dict(target=target, dataset=ds, category=cat_of.get(c.name, "?"), family=c.name, learner=ln, Plot_ID=d.Plot_ID, fold=fold, measured=y, predicted=p)))
            log(f"{target} {ds} {c.name}/{ln}: RMSE {np.sqrt(((p - y) ** 2).mean()):.1f} ({time.time()-t1:.0f}s)")
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(out_res, f"fold_each_model_{ds}{RUNTAG}.csv"), index=False)
    return []

def run(target, ds, d, cands, cats, ycol, out_res, out_mod, meta):
    if EACH: return each_model(target, ds, d, cands, cats, ycol, out_res)
    """cats: {category: [families]} - top three (family x learner) per category, chosen in the inner loop"""
    cats = {k: [f for f in v if any(c.name == f for c in cands)] for k, v in cats.items()}; cats = {k: v for k, v in cats.items() if v}
    y = d[ycol].to_numpy(float); Xs = {c.name: d[c.cols].to_numpy(float) for c in cands}; get = {c.name: c for c in cands}
    rows = []; pred = {k: np.zeros(len(y)) for k in cats}; outer = list(KFold(10, shuffle=True, random_state=0).split(y)); fold_models = {}
    for f, (tr, te) in enumerate(outer):
        sc = inner_scores(cands, {k: v[tr] for k, v in Xs.items()}, y[tr], list(KFold(5, shuffle=True, random_state=f).split(tr)))
        for cat, fams in cats.items():
            top = top3(sc, fams); ps = []; fmembers = []
            for (fam, ln), r in top:
                c = get[fam]; m, idx = c.fit(c.learners()[ln], Xs[fam][tr], y[tr]); ps.append(np.asarray(m.predict(Xs[fam][te][:, idx])).ravel())
                fmembers.append(dict(family=fam, learner=ln, inner_rmse=r, model=m, features=[c.cols[i] for i in idx]))
            pred[cat][te] = np.mean(ps, 0)
            rows.append(dict(target=target, dataset=ds, category=cat, outer_fold=f + 1, n_test=len(te), top3=" | ".join(f"{fam}/{ln} ({r:.1f})" for (fam, ln), r in top), **metrics(pred[cat][te], y[te])))
            fold_models.setdefault(cat, []).append(dict(fold=f + 1, n_train=len(tr), n_test=len(te), test=metrics(pred[cat][te], y[te]), members=fmembers))
        log(f"{target} {ds} fold {f+1}/10 done: " + "; ".join(f"{r['category']} {r['top3']}" for r in rows[-len(cats):]))
    pd.DataFrame(rows).to_csv(os.path.join(out_res, f"nested_top3_{ds}{RUNTAG}.csv"), index=False)
    pd.DataFrame(dict(Plot_ID=d.Plot_ID, measured=y, **{f"pred_{k}": v for k, v in pred.items()})).to_csv(os.path.join(out_res, f"nested_top3_predictions_{ds}{RUNTAG}.csv"), index=False)
    # final ensembles on all plots
    sc = inner_scores(cands, Xs, y, list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=0).split(y)))
    out = []
    for cat, fams in cats.items():
        m = metrics(pred[cat], y); freq = pd.Series([p.split(" (")[0] for r in rows if r["category"] == cat for p in r["top3"].split(" | ")]).value_counts().to_dict()
        members = []
        for (fam, ln), r in top3(sc, fams):
            c = get[fam]; mod, idx = c.fit(c.learners()[ln], Xs[fam], y)
            members.append(dict(family=fam, learner=ln, inner_rmse=r, model=mod, features=[c.cols[i] for i in idx]))
        fn = f"{target}_{ds}_{cat}_top3_ensemble{RUNTAG}.joblib"
        joblib.dump(dict(type="mean_ensemble", target=ycol, dataset=ds, category=cat, members=members, nested_cv=m, outer_fold_choices=freq, **meta), os.path.join(out_mod, fn))
        log(f"{target} {ds} {cat}: NESTED top-3 R2 {m['R2']:.3f} RMSE {m['RMSE']:.1f} | saved {fn}: {[(x['family'], x['learner']) for x in members]}")
        ffn = f"{target}_{ds}_{cat}_fold_ensembles{RUNTAG}.joblib"      # the 10 outer-fold ensembles (each fitted on its training plots) + their test scores
        joblib.dump(dict(type="fold_ensembles", target=ycol, dataset=ds, category=cat, folds=fold_models[cat], nested_cv=m,
                         note="fold k = top-three ensemble chosen and fitted on the training plots of outer fold k; 'test' = its score on fold k's test plots. "
                              "Ranking folds by these scores and keeping the best ones selects on the test sets, so the scores of a best-folds ensemble are optimistic.", **meta),
                    os.path.join(out_mod, ffn))
        out.append(dict(target=target, dataset=ds, category=cat, n=len(y), **m, final_members=" | ".join(f"{x['family']}/{x['learner']}" for x in members), outer_fold_choices=json.dumps(freq), model_file=fn))
    return out

summary = []
if "height" in TARGETS:
    RES = os.path.join(ROOT, "height", "results"); MOD = os.path.join(ROOT, "height", "models"); FS = json.load(open(os.path.join(RES, "feature_sets.json")))
    d = pd.read_csv(os.path.join(DS, "muresk", "2025-11-21", "features_all.csv")); V = pd.read_csv(os.path.join(DS, "muresk", "2025-11-21", "features_vnir.csv"))
    d = d[(d.has_ground_truth == 1) & (d.valid_lidar == 1) & d.height_cm.notna()].reset_index(drop=True)
    for c in [c for c in d.columns if c.startswith(("H", "cth_", "tip_thin", "roughness", "Hall", "profile_area")) and c not in ("H_cv", "H_skew", "H_kurt", "H_crr")]:
        if d[c].dtype != object: d[c] = d[c] * 100.0
    vc = [c for c in V.columns if c in d.columns and c != "Plot_ID" and d[c].dtype != object]; d[vc] = d[vc].fillna(d[vc].median())
    d = d.merge(spectra_matrix(pd.read_csv(os.path.join(DS, "muresk", "2025-11-21", "vnir_spectra.csv"))), left_on="Plot_ID", right_index=True, how="left")
    lid = FS["LiDAR_all"]; vn = [c for c in FS["LiDAR+VNIR_PCA"] if c not in lid]; bands = FS["VNIR_bands_PCA"]
    def blk(*parts):
        out, o = [], 0
        for n_, cs in parts: out.append((n_, list(range(o, o + len(cs))))); o += len(cs)
        return out
    cands = [Candidate("LiDAR_core", FS["LiDAR_core"], k=10), Candidate("LiDAR_all", lid, k=10), Candidate("LiDAR+VNIR", FS["LiDAR+VNIR"], k=10),
             Candidate("LiDAR_PCA", lid, blk(("lidar", lid))), Candidate("LiDAR+VNIR_PCA", lid + vn, blk(("lidar", lid), ("vnir", vn))),
             Candidate("LiDAR+VNIR_PCA_joint", lid + vn, blk(("joint", lid + vn))), Candidate("VNIR_bands_PCA", bands, blk(("bands", bands))),
             Candidate("LiDAR+VNIR_bands_PCA", lid + bands, blk(("lidar", lid), ("bands", bands)))]
    for c in cands: d[c.cols] = d[c.cols].fillna(d[c.cols].median())
    HCATS = {"lidar": ["LiDAR_core", "LiDAR_all", "LiDAR_PCA"], "vnir": ["VNIR_bands_PCA"], "fused": ["LiDAR+VNIR", "LiDAR+VNIR_PCA", "LiDAR+VNIR_PCA_joint", "LiDAR+VNIR_bands_PCA"]}
    summary += run("height", "muresk_maturity", d, cands, HCATS, "height_cm", RES, MOD,
                       dict(input_prep="research height_models.py: H*, cth_*, tip_thin, roughness, Hall*, profile_area* columns m -> cm; VNIR NaN -> flight median; bands = spectra_matrix(vnir_spectra.csv)",
                            flight="2025-11-21 (ruler at maturity)"))
    if not EACH: pd.DataFrame([s for s in summary if s["target"] == "height"]).to_csv(os.path.join(RES, f"nested_top3_summary{RUNTAG}.csv"), index=False)
if "biomass" in TARGETS:
    RES = os.path.join(ROOT, "biomass", "results"); MOD = os.path.join(ROOT, "biomass", "models"); FS = json.load(open(os.path.join(RES, "feature_sets.json")))
    def available(cols, d): return [c for c in cols if c in d.columns and np.isfinite(d[c].astype(float)).mean() > 0.95 and d[c].astype(float).std() > 0]
    bsum = []
    for ds in ("muresk_anthesis", "muresk_maturity", "agt_anthesis", "agt_maturity"):
        d = pd.read_csv(os.path.join(ROOT, "biomass", "features", f"features_{ds}.csv"))
        if PLOTLIST is not None:
            d = d.merge(PLOTLIST[PLOTLIST.dataset == ds][["Plot_ID"]], on="Plot_ID").reset_index(drop=True)
            if len(d) < 20: log(f"biomass {ds}: {len(d)} plots in {os.path.basename(PLOTS)} - skipped"); continue
        has_vnir = (d.vnir_qc_ok.fillna(0) == 1).sum() >= 20
        if has_vnir: d = d[d.vnir_qc_ok.fillna(0) == 1].reset_index(drop=True)
        W = FS["WEIGHTED"] if isinstance(FS["WEIGHTED"], list) else list(FS["WEIGHTED"])
        lc, vcr = available(FS["LIDAR_CORE"], d), available(FS["VNIR_CORE"], d); la, va = available(FS["lidar_all"], d), available(FS["vnir_all"], d)
        bands = available([c for c in d.columns if c.startswith(SB_PREFIX)], d); wc = available(W, d)
        def blk(*parts):
            out, o = [], 0
            for n_, cs in parts: out.append((n_, list(range(o, o + len(cs))))); o += len(cs)
            return out
        cands = [Candidate("LiDAR_core", lc, k=12 if len(lc) > 15 else None), Candidate("LiDAR_PCA", la, blk(("lidar", la)))]
        if has_vnir and len(vcr) >= 2:
            fa = available(FS["lidar_all"] + FS["vnir_all"] + W, d)
            cands += [Candidate("VNIR_core", vcr, k=12 if len(vcr) > 15 else None), Candidate("Fused_core", lc + vcr + wc, k=12), Candidate("Fused_all", fa, k=12),
                      Candidate("VNIR_PCA", va, blk(("vnir", va))), Candidate("Fused_PCA", la + va, blk(("lidar", la), ("vnir", va))), Candidate("Fused_PCA_joint", la + va, blk(("joint", la + va)))]
            if len(bands) >= 2: cands += [Candidate("VNIR_bands_PCA", bands, blk(("bands", bands))), Candidate("Fused_bands_PCA", la + bands, blk(("lidar", la), ("bands", bands)))]
        for c in cands: d[c.cols] = d[c.cols].fillna(d[c.cols].median())
        log(f"biomass {ds}: {len(d)} plots, {len(cands)} candidate families")
        BCATS = {"lidar": ["LiDAR_core", "LiDAR_PCA"], "vnir": ["VNIR_core", "VNIR_PCA", "VNIR_bands_PCA"], "fused": ["Fused_core", "Fused_all", "Fused_PCA", "Fused_PCA_joint", "Fused_bands_PCA"]}
        bsum += run("biomass", ds, d, cands, BCATS, "biomass_kg_ha", RES, MOD, dict(input_prep="biomass\\features\\features_<dataset>.csv columns as stored (NaN -> dataset median)"))
        if not EACH: pd.DataFrame(bsum).to_csv(os.path.join(RES, f"nested_top3_summary{RUNTAG}.csv"), index=False)
log("NESTED TOP-3 DONE")
