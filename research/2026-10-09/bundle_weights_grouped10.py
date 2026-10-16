"""Bundle all 10 fold models of the grouped common-band runs (two_stream, lidar_only, vnir_only) into the tool, with
fold_scores.csv = each fold model's score on its own test plots (all datasets of the fold pooled), so the runner and the
Deep Models tab can use all folds or the best three."""
import os, glob, shutil, torch, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"; SRC = os.path.join(R, "two_stream", "models")
TOOL = r"C:\Users\appn\Downloads\PhenoFuse-main\phenoapp_v3_2026-10-06"; DST = os.path.join(TOOL, "phenoapp", "assets", "dl_models")
idx_all = pd.read_csv(os.path.join(R, "two_stream", "dl_data", "index.csv")); rows = []; n = 0
for f in glob.glob(os.path.join(DST, "*.pt")): os.unlink(f)
for target, tag, dsets in (("biomass", "commonbands_grouped", None), ("height", "commonbands_grouped_maturity", ["muresk_maturity"])):
    tcol = "biomass_kg_ha" if target == "biomass" else "height_cm"
    idx = idx_all[idx_all[tcol].notna()].reset_index(drop=True)
    if dsets: idx = idx[idx.dataset.isin(dsets)].reset_index(drop=True)
    g = (idx.trial.astype(str) + "_" + idx.Plot_ID.astype(str)).to_numpy(); fold = np.zeros(len(idx), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(idx, idx.dataset, g)): fold[te] = k
    P = pd.read_csv(os.path.join(R, "two_stream", "results", f"dl_{target}_{tag}_per_plot_predictions.csv")).merge(idx[["dataset", "Plot_ID"]].assign(fold_index=fold), on=["dataset", "Plot_ID"])
    for variant in ("two_stream", "lidar_only", "vnir_only"):
        for k in range(10):
            s = os.path.join(SRC, f"dl_{target}_{tag}_{variant}_fold{k}.pt")
            if not os.path.exists(s): continue
            sd = torch.load(s, map_location="cpu"); torch.save({kk: (v.half() if v.is_floating_point() else v) for kk, v in sd.items()}, os.path.join(DST, f"{variant}_{target}_fold{k}.pt")); n += 1
            q = P[(P.variant == variant) & (P.fold_index == k)]
            if len(q):
                e = q.predicted - q.measured
                rows.append(dict(variant=variant, target=target, fold_index=k, fold=k + 1, n_test=len(q), RMSE=float(np.sqrt((e ** 2).mean())), R2=float(1 - (e ** 2).sum() / ((q.measured - q.measured.mean()) ** 2).sum())))
FS = pd.DataFrame(rows); FS.to_csv(os.path.join(DST, "fold_scores.csv"), index=False)
FS["best3"] = FS.groupby(["variant", "target"]).RMSE.rank(method="first") <= 3
print("bundled", n, "fold files;", round(sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)) / 1e6, 1), "MB")
print(FS.round(3).to_string(index=False))
