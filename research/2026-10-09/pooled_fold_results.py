"""
Reporting tables (rule of 12 Oct 2026):
 - HEIGHT: Muresk maturity flight only (the only flight with a ruler), per fold and overall, features (nested top 3) and network.
 - BIOMASS: overall results only, pooled over all trials and stages, computed on the TEST SET of each fold and over all folds,
   for features (nested top 3 per category) and the network, per category (LiDAR only; VNIR only and fused on the plots with
   usable VNIR, i.e. AGT maturity excluded because its cubes are radiance).
   Feature models are fitted per dataset; fold k's pooled test set = the union of fold k of every dataset (KFold(10, seed 0)
   per dataset, as in nested_top3.py). Network folds are global (StratifiedGroupKFold(10, seed 0), stratified by dataset,
   grouped by plot). Only plots present in both approaches are used, so features and network are scored on the same plots.
Outputs: results\report_height_folds.csv, results\report_biomass_pooled_folds.csv, results\report_summary.csv
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"; OUT = os.path.join(R, "results"); os.makedirs(OUT, exist_ok=True)
CAT = {"lidar_only": "lidar", "vnir_only": "vnir", "two_stream": "fused"}

def met(p, y):
    e = p - y; ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), R2=float(1 - (e ** 2).sum() / ss), RMSE=float(np.sqrt((e ** 2).mean())), rRMSE=float(np.sqrt((e ** 2).mean()) / y.mean() * 100),
                MAE=float(np.abs(e).mean()), bias=float(e.mean()), r=float(np.corrcoef(p, y)[0, 1]))

def feature_preds(target):
    """long table: dataset, Plot_ID, category, measured, predicted, fold (feature models, nested top 3)"""
    rows = []
    for f in sorted(os.listdir(os.path.join(R, target, "results"))):
        if not (f.startswith("nested_top3_predictions_") and f.endswith(".csv")) or "__" in f: continue      # "__rule*" = valid-plot runs, handled in valid_plot_results.py
        ds = f[len("nested_top3_predictions_"):-4]; P = pd.read_csv(os.path.join(R, target, "results", f))
        fold = np.zeros(len(P), int)
        for k, (_, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(P), 1): fold[te] = k
        for c in [c for c in P.columns if c.startswith("pred_")]:
            rows.append(pd.DataFrame(dict(dataset=ds, Plot_ID=P.Plot_ID, category=c[5:], measured=P.measured, predicted=P[c], fold=fold)))
    return pd.concat(rows, ignore_index=True)

def net_preds(target, files, dsets=None):
    tcol = "biomass_kg_ha" if target == "biomass" else "height_cm"
    idx = pd.read_csv(os.path.join(R, "two_stream", "dl_data", "index.csv")); idx = idx[idx[tcol].notna()].reset_index(drop=True)
    if dsets: idx = idx[idx.dataset.isin(dsets)].reset_index(drop=True)
    groups = (idx.trial.astype(str) + "_" + idx.Plot_ID.astype(str)).to_numpy(); fold = np.zeros(len(idx), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(idx, idx.dataset, groups), 1): fold[te] = k
    key = idx[["dataset", "Plot_ID"]].assign(fold=fold)
    P = pd.concat([pd.read_csv(os.path.join(R, "two_stream", "results", f)) for f in files if os.path.exists(os.path.join(R, "two_stream", "results", f))], ignore_index=True)
    P = P.drop_duplicates(["variant", "dataset", "Plot_ID"], keep="last").merge(key, on=["dataset", "Plot_ID"])
    return P.assign(category=P.variant.map(CAT))[["dataset", "Plot_ID", "category", "measured", "predicted", "fold"]]

def table(F, N, label, restrict=None, cats=("lidar", "vnir", "fused")):
    """per-fold + overall metrics for features (F) and network (N) on the plots common to both, per category;
    restrict: DataFrame (dataset, Plot_ID) limiting the plots (e.g. the plots with usable VNIR)"""
    out = []
    for cat in cats:
        f = F[F.category == cat]; n = N[N.category == cat]
        if f.empty or n.empty: continue
        common = f[["dataset", "Plot_ID"]].merge(n[["dataset", "Plot_ID"]]).drop_duplicates()
        if restrict is not None: common = common.merge(restrict, on=["dataset", "Plot_ID"])
        for appr, D in (("features (nested top 3)", f), ("deep learning (two-stream net)", n)):
            D = D.merge(common, on=["dataset", "Plot_ID"])
            for k in range(1, 11):
                s = D[D.fold == k]; out.append(dict(target=label, approach=appr, category=cat, fold=k, datasets=s.dataset.nunique(), **met(s.predicted.to_numpy(float), s.measured.to_numpy(float))))
            out.append(dict(target=label, approach=appr, category=cat, fold="all", datasets=D.dataset.nunique(), **met(D.predicted.to_numpy(float), D.measured.to_numpy(float))))
    return pd.DataFrame(out)

H = table(feature_preds("height"), net_preds("height", ["dl_height_commonbands_grouped_maturity_per_plot_predictions.csv"], ["muresk_maturity"]), "height, Muresk maturity")
FB = feature_preds("biomass"); NB = net_preds("biomass", ["dl_biomass_commonbands_grouped_per_plot_predictions.csv"])
VPLOTS = FB[FB.category == "fused"][["dataset", "Plot_ID"]].drop_duplicates()          # plots with usable VNIR (AGT maturity excluded)
B = pd.concat([table(FB, NB, "biomass, pooled, plots with usable VNIR (3 datasets)", restrict=VPLOTS),
               table(FB, NB, "biomass, pooled, all plots (4 datasets)", cats=("lidar",))], ignore_index=True)
H.to_csv(os.path.join(OUT, "report_height_folds.csv"), index=False); B.to_csv(os.path.join(OUT, "report_biomass_pooled_folds.csv"), index=False)

# ---- which flights make up the test set of each fold (plots per flight; network folds, which the feature folds match in size)
FLIGHT = {"muresk_anthesis": "Muresk 2025-09-30 (anthesis)", "muresk_maturity": "Muresk 2025-11-21 (maturity)", "agt_anthesis": "AGT 2025-09-22 (anthesis)", "agt_maturity": "AGT 2025-11-13 (maturity, flights 1+2 merged)"}
comp = []
for label, N_, restrict in (("height, Muresk maturity", net_preds("height", ["dl_height_commonbands_grouped_maturity_per_plot_predictions.csv"], ["muresk_maturity"]), None),
                            ("biomass, pooled, plots with usable VNIR (3 datasets)", NB, VPLOTS), ("biomass, pooled, all plots (4 datasets)", NB, None)):
    n_ = N_[N_.category == N_.category.iloc[0]][["dataset", "Plot_ID", "fold"]].drop_duplicates()
    if restrict is not None: n_ = n_.merge(restrict, on=["dataset", "Plot_ID"])
    t = n_.assign(flight=n_.dataset.map(FLIGHT)).pivot_table(index="flight", columns="fold", values="Plot_ID", aggfunc="count", fill_value=0)
    t["all folds"] = t.sum(axis=1); t.loc["total"] = t.sum(); t = t.reset_index(); t.insert(0, "test set", label); comp.append(t)
pd.concat(comp, ignore_index=True).to_csv(os.path.join(OUT, "report_testset_composition.csv"), index=False)
S = []
for T in (H, B):
    for (tg, ap, ca), g in T.groupby(["target", "approach", "category"], sort=False):
        fo = g[g.fold != "all"]; al = g[g.fold == "all"].iloc[0]
        S.append(dict(target=tg, approach=ap, category=ca, plots=int(al.n), test_plots_per_fold=round(fo.n.mean(), 1), R2_overall=al.R2, RMSE_overall=al.RMSE, rRMSE_overall=al.rRMSE,
                      R2_fold_mean=fo.R2.mean(), R2_fold_sd=fo.R2.std(), RMSE_fold_mean=fo.RMSE.mean(), RMSE_fold_sd=fo.RMSE.std()))
S = pd.DataFrame(S); S.to_csv(os.path.join(OUT, "report_summary.csv"), index=False)

# ---- every individual feature model on the same test folds (fold_each_model_<ds>.csv from nested_top3.py --each)
def each(target):
    fs = [os.path.join(R, target, "results", f) for f in os.listdir(os.path.join(R, target, "results")) if f.startswith("fold_each_model_") and "__" not in f]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
EM = []
for target, label, restrict in (("height", "height, Muresk maturity", None), ("biomass", "biomass, pooled, plots with usable VNIR (3 datasets)", VPLOTS), ("biomass", "biomass, pooled, all plots (4 datasets)", "all")):
    E = each(target)
    if E is None: continue
    if target == "height": E = E[E.dataset == "muresk_maturity"]
    for (cat, fam, ln), g in E.groupby(["category", "family", "learner"]):
        if restrict is not None and not isinstance(restrict, str): g = g.merge(restrict, on=["dataset", "Plot_ID"])
        if isinstance(restrict, str) and g.dataset.nunique() < 4: continue          # all-plots group: only families available on all 4 datasets
        if g.empty: continue
        for k in range(1, 11):
            s = g[g.fold == k]; EM.append(dict(target=label, category=cat, family=fam, learner=ln, fold=k, **met(s.predicted.to_numpy(float), s.measured.to_numpy(float))))
        EM.append(dict(target=label, category=cat, family=fam, learner=ln, fold="all", **met(g.predicted.to_numpy(float), g.measured.to_numpy(float))))
if EM:
    EM = pd.DataFrame(EM); EM.to_csv(os.path.join(OUT, "report_each_model_folds.csv"), index=False)
    A = EM[EM.fold == "all"][["target", "category", "family", "learner", "n", "R2", "RMSE", "rRMSE"]]
    Fm = EM[EM.fold != "all"].groupby(["target", "category", "family", "learner"]).agg(RMSE_fold_mean=("RMSE", "mean"), RMSE_fold_sd=("RMSE", "std"), R2_fold_mean=("R2", "mean"), R2_fold_sd=("R2", "std")).reset_index()
    A.merge(Fm).sort_values(["target", "category", "RMSE"]).to_csv(os.path.join(OUT, "report_each_model_summary.csv"), index=False)
    print("each-model tables:", len(A), "models")
pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))
