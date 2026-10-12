"""
Fold-by-fold held-out results for every case: feature models (nested top three per category: LiDAR only, VNIR only, fused)
and the two-stream network (variants lidar_only, vnir_only, two_stream), for height (maturity flight) and biomass (4 datasets).

Feature models: the outer folds of nested_top3.py (10-fold per dataset, seed 0) - metrics per fold from nested_top3_<ds>.csv
(recomputed from the per-plot predictions so every fold has R2, RMSE, rRMSE, MAE).
Network: the folds of dl_twostream.py with --bands common and folds grouped by plot (StratifiedGroupKFold(10, shuffle, seed 0)
on the index rows, stratified by dataset), rebuilt here exactly and applied to the saved per-plot predictions.
Outputs: results\fold_results_long.csv (one row per target x dataset x approach x category x fold) and
results\fold_results_summary.csv (mean, SD, min, max over the 10 folds + the all-plots value).
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"
OUT = os.path.join(R, "results"); os.makedirs(OUT, exist_ok=True)
CAT = {"lidar_only": "lidar", "vnir_only": "vnir", "two_stream": "fused"}

def met(p, y):
    e = p - y; ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), R2=float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan, RMSE=float(np.sqrt((e ** 2).mean())), rRMSE=float(np.sqrt((e ** 2).mean()) / y.mean() * 100), MAE=float(np.abs(e).mean()), bias=float(e.mean()))

rows = []
# ---- feature models: nested top three per category
for target, folder in (("height", "height"), ("biomass", "biomass")):
    for f in sorted(os.listdir(os.path.join(R, folder, "results"))):
        if not (f.startswith("nested_top3_predictions_") and f.endswith(".csv")): continue
        ds = f[len("nested_top3_predictions_"):-4]; P = pd.read_csv(os.path.join(R, folder, "results", f)); y = P.measured.to_numpy(float)
        folds = list(KFold(10, shuffle=True, random_state=0).split(y))
        for col in [c for c in P.columns if c.startswith("pred_")]:
            p = P[col].to_numpy(float); cat = col[5:]
            for k, (_, te) in enumerate(folds, 1): rows.append(dict(target=target, dataset=ds, approach="features (nested top 3)", category=cat, fold=k, **met(p[te], y[te])))
            rows.append(dict(target=target, dataset=ds, approach="features (nested top 3)", category=cat, fold="all", **met(p, y)))
# ---- network: grouped common-band runs
idx_all = pd.read_csv(os.path.join(R, "two_stream", "dl_data", "index.csv"))
for target, tag, dsets in (("biomass", "_commonbands_grouped", None), ("height", "_commonbands_grouped_maturity", ["muresk_maturity"])):
    pp = os.path.join(R, "two_stream", "results", f"dl_{target}{tag}_per_plot_predictions.csv")
    if not os.path.exists(pp): continue
    tcol = "biomass_kg_ha" if target == "biomass" else "height_cm"
    idx = idx_all[idx_all[tcol].notna()].reset_index(drop=True)
    if dsets: idx = idx[idx.dataset.isin(dsets)].reset_index(drop=True)
    groups = (idx.trial.astype(str) + "_" + idx.Plot_ID.astype(str)).to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0).split(idx, idx.dataset, groups))
    fold_of = np.zeros(len(idx), int)
    for k, (_, te) in enumerate(folds, 1): fold_of[te] = k
    key = idx[["dataset", "Plot_ID"]].assign(fold=fold_of)
    P = pd.read_csv(pp)
    for variant, g in P.groupby("variant"):
        g = g.merge(key, on=["dataset", "Plot_ID"], how="inner")
        for ds, gd in g.groupby("dataset"):
            for k in range(1, 11):
                s = gd[gd.fold == k]
                if len(s) >= 3: rows.append(dict(target=target, dataset=ds, approach="deep learning (two-stream net)", category=CAT[variant], fold=k, **met(s.predicted.to_numpy(float), s.measured.to_numpy(float))))
            rows.append(dict(target=target, dataset=ds, approach="deep learning (two-stream net)", category=CAT[variant], fold="all", **met(gd.predicted.to_numpy(float), gd.measured.to_numpy(float))))
L = pd.DataFrame(rows); L.to_csv(os.path.join(OUT, "fold_results_long.csv"), index=False)
F = L[L.fold != "all"]; A = L[L.fold == "all"].set_index(["target", "dataset", "approach", "category"])
S = F.groupby(["target", "dataset", "approach", "category"]).agg(folds=("fold", "count"), plots_per_fold=("n", "mean"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"), R2_min=("R2", "min"), R2_max=("R2", "max"),
                                                              RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), rRMSE_mean=("rRMSE", "mean"))
S["R2_all_plots"] = A.R2; S["RMSE_all_plots"] = A.RMSE; S = S.reset_index()
S.to_csv(os.path.join(OUT, "fold_results_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string(index=False))
