"""
Biomass on valid-band plots only (rule A / rule B), overall results pooled over all flights, on each fold's test set.
 rule A: every common band (111) valid on >= 50 % of the plot's pixels, reflectance cube, VNIR QC ok  -> dataset\valid_plots_ruleA.csv
 rule B: every common band valid on >= 95 % of the plot's pixels, plot fully inside the cube, QC ok  -> dataset\valid_plots_ruleB.csv
Feature models: nested_top3.py --plots ... --tag __ruleX (top three per category, per flight; fold k = union of fold k of
every flight). Network: dl_twostream.py --plots ... --tag __ruleX (variants vnir_only, two_stream; folds grouped by plot).
VNIR-only and fused are scored on the plots common to both approaches; LiDAR-only (features) is given as the reference.
Outputs: results\valid_plots_folds.csv, valid_plots_summary.csv, valid_plots_composition.csv
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"; OUT = os.path.join(R, "results")
CAT = {"lidar_only": "lidar", "vnir_only": "vnir", "two_stream": "fused"}
FLIGHT = {"muresk_anthesis": "Muresk 2025-09-30 (anthesis)", "muresk_maturity": "Muresk 2025-11-21 (maturity)", "agt_anthesis": "AGT 2025-09-22 (anthesis)", "agt_maturity": "AGT 2025-11-13 (maturity)"}

def met(p, y):
    e = p - y; ss = ((y - y.mean()) ** 2).sum()
    return dict(n=len(y), R2=float(1 - (e ** 2).sum() / ss), RMSE=float(np.sqrt((e ** 2).mean())), rRMSE=float(np.sqrt((e ** 2).mean()) / y.mean() * 100), MAE=float(np.abs(e).mean()), bias=float(e.mean()))

rows, comp = [], []
for rule in ("A", "B"):
    tag = f"__rule{rule}"; plots = pd.read_csv(os.path.join(R, "dataset", f"valid_plots_rule{rule}.csv"))
    F = []
    for f in sorted(os.listdir(os.path.join(R, "biomass", "results"))):
        if f.startswith("nested_top3_predictions_") and f.endswith(f"{tag}.csv"):
            ds = f[len("nested_top3_predictions_"):-len(f"{tag}.csv")]; P = pd.read_csv(os.path.join(R, "biomass", "results", f)); fold = np.zeros(len(P), int)
            for k, (_, te) in enumerate(KFold(10, shuffle=True, random_state=0).split(P), 1): fold[te] = k
            for c in [c for c in P.columns if c.startswith("pred_")]: F.append(pd.DataFrame(dict(dataset=ds, Plot_ID=P.Plot_ID, category=c[5:], measured=P.measured, predicted=P[c], fold=fold)))
    F = pd.concat(F, ignore_index=True) if F else pd.DataFrame(columns=["dataset", "Plot_ID", "category", "measured", "predicted", "fold"])
    N = pd.DataFrame(columns=F.columns); pp = os.path.join(R, "two_stream", "results", f"dl_biomass_commonbands{tag}_per_plot_predictions.csv")
    if os.path.exists(pp):
        idx = pd.read_csv(os.path.join(R, "two_stream", "dl_data", "index.csv")); idx = idx[idx.biomass_kg_ha.notna()].merge(plots, on=["dataset", "Plot_ID"]).reset_index(drop=True)
        g = (idx.trial.astype(str) + "_" + idx.Plot_ID.astype(str)).to_numpy(); fold = np.zeros(len(idx), int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(idx, idx.dataset, g), 1): fold[te] = k
        P = pd.read_csv(pp).merge(idx[["dataset", "Plot_ID"]].assign(fold=fold), on=["dataset", "Plot_ID"]); N = P.assign(category=P.variant.map(CAT))[F.columns.tolist()]
        t = idx.assign(fold=fold, flight=idx.dataset.map(FLIGHT)).pivot_table(index="flight", columns="fold", values="Plot_ID", aggfunc="count", fill_value=0)
        t["all folds"] = t.sum(axis=1); t.loc["total"] = t.sum(); t = t.reset_index(); t.insert(0, "rule", rule); comp.append(t)
    for cat in ("lidar", "vnir", "fused"):
        f = F[F.category == cat]; n = N[N.category == cat]
        sets = [("features (nested top 3)", f)] + ([("deep learning (two-stream net)", n)] if not n.empty else [])
        common = f[["dataset", "Plot_ID"]] if n.empty else f[["dataset", "Plot_ID"]].merge(n[["dataset", "Plot_ID"]])
        for appr, D in sets:
            if D.empty: continue
            D = D.merge(common.drop_duplicates(), on=["dataset", "Plot_ID"])
            for k in range(1, 11):
                s = D[D.fold == k]
                if len(s) >= 3: rows.append(dict(rule=rule, approach=appr, category=cat, fold=k, flights=s.dataset.nunique(), **met(s.predicted.to_numpy(float), s.measured.to_numpy(float))))
            rows.append(dict(rule=rule, approach=appr, category=cat, fold="all", flights=D.dataset.nunique(), **met(D.predicted.to_numpy(float), D.measured.to_numpy(float))))
L = pd.DataFrame(rows); L.to_csv(os.path.join(OUT, "valid_plots_folds.csv"), index=False)
S = []
for (r_, a, c), g in L.groupby(["rule", "approach", "category"], sort=False):
    fo = g[g.fold.astype(str) != "all"]; al = g[g.fold.astype(str) == "all"].iloc[0]
    S.append(dict(rule=r_, category=c, approach=a, plots=int(al.n), flights=int(al.flights), R2_overall=al.R2, RMSE_overall=al.RMSE, rRMSE_overall=al.rRMSE, RMSE_fold_mean=fo.RMSE.mean(), RMSE_fold_sd=fo.RMSE.std(), R2_fold_mean=fo.R2.mean(), R2_fold_sd=fo.R2.std()))
S = pd.DataFrame(S); S.to_csv(os.path.join(OUT, "valid_plots_summary.csv"), index=False)
if comp: pd.concat(comp, ignore_index=True).to_csv(os.path.join(OUT, "valid_plots_composition.csv"), index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
