"""Top three feature families per dataset (best learner each, by held-out RMSE), the mean of their held-out predictions,
and the noise level of the RMSE (bootstrap SE). Writes biomass\results\top3_ensemble.csv and height\results\top3_ensemble.csv."""
import os, numpy as np, pandas as pd
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"

def met(p, y):
    e = p - y
    return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt((e ** 2).mean())

def boot_se(p, y, n=1000):
    rng = np.random.default_rng(0)
    return float(np.std([met(p[i], y[i])[1] for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]))

for lab, folder, key in (("BIOMASS", "biomass", "dataset"), ("HEIGHT", "height", "flight")):
    t = pd.read_csv(os.path.join(R, folder, "results", "model_comparison.csv"))
    P = pd.read_csv(os.path.join(R, folder, "results", "per_plot_predictions.csv"))
    t = t[t.scheme == "repeated10fold"]; P = P[P.scheme == "repeated10fold"]
    if lab == "HEIGHT":       # maturity flight only (the only flight with a ruler measurement)
        t = t[t[key].astype(str) == "2025-11-21"]; P = P[P[key].astype(str) == "2025-11-21"]
    rows = []
    for ds, g in t.groupby(key):
        best = g.loc[g.groupby("feature_set").RMSE.idxmin()].nsmallest(3, "RMSE")
        preds = []
        for r in best.itertuples():
            q = P[(P[key].astype(str) == str(ds)) & (P.feature_set == r.feature_set) & (P.learner == r.learner)]
            preds.append(q.groupby("Plot_ID").agg(predicted=("predicted", "mean"), measured=("measured", "first")))
        idx = preds[0].index; y = preds[0].measured.to_numpy(float)
        E = np.mean([q.predicted.reindex(idx).to_numpy(float) for q in preds], 0)
        r2, rm = met(E, y); se = boot_se(preds[0].predicted.to_numpy(float), y)
        print(f"{lab} {ds}:")
        for k, r in enumerate(best.itertuples(), 1):
            print(f"   {k}. {r.feature_set:20s} {r.learner:12s} R2 {r.R2:.3f}  RMSE {r.RMSE:.1f}")
            rows.append(dict(dataset=ds, rank=k, feature_set=r.feature_set, learner=r.learner, R2=round(r.R2, 3), RMSE=round(r.RMSE, 1)))
        print(f"   mean of the three:          R2 {r2:.3f}  RMSE {rm:.1f}   | bootstrap SE of the top RMSE {se:.1f}; spread of the top three {best.RMSE.max() - best.RMSE.min():.1f}")
        rows.append(dict(dataset=ds, rank="ensemble", feature_set="mean of top 3", learner="", R2=round(r2, 3), RMSE=round(rm, 1), rmse_bootstrap_se_top=round(se, 1)))
    pd.DataFrame(rows).to_csv(os.path.join(R, folder, "results", "top3_ensemble.csv"), index=False)
