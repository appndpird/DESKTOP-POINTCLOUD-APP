"""
Anthesis-flight plant height: PREDICTED, not validated.

Muresk has one ruler height per plot, measured at maturity (2025-11-21). Error metrics (RMSE, R2 ...) are therefore
reported on the maturity flight only. For the anthesis flight (2025-09-30) this script applies the maturity-flight model
to the anthesis-flight features and reports the predicted height per plot together with the difference
"maturity ruler minus predicted anthesis height" (growth / lodging between the two dates), plus the same difference on the
raw canopy-top LiDAR height (cth_p95) as a model-free cross-check.

Outputs (height/results): anthesis_predicted_height.csv, anthesis_height_difference_summary.json,
figures/height_anthesis_prediction.png
"""
import os, json, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"
M = os.path.join(B, "Biomass_Height_2026-10-09"); HP = os.path.join(M, "height"); FIG = os.path.join(M, "figures")
DS = os.path.join(M, "dataset", "muresk")
best = pd.read_csv(os.path.join(HP, "results", "best_models.csv"))
# LiDAR-only maturity model: the VNIR features of a senesced canopy (maturity) and a green canopy (anthesis) differ by
# more than 100 standard deviations (red-edge indices), so a model that uses them cannot be applied across the two dates;
# LiDAR canopy structure is comparable between dates.
cand = best[(best.flight.astype(str) == "2025-11-21") & best.family.isin(["LiDAR_core", "LiDAR_all", "LiDAR_PCA"])].sort_values("RMSE")
bm = cand.iloc[0]
model = joblib.load(os.path.join(HP, "models", bm.model_file)) if os.path.exists(os.path.join(HP, "models", bm.model_file)) else joblib.load(bm.model_file)
feats = model["features"] if isinstance(model, dict) else None; est = model["model"] if isinstance(model, dict) else model
A = pd.read_csv(os.path.join(DS, "2025-09-30", "features_all.csv")); Mt = pd.read_csv(os.path.join(DS, "2025-11-21", "features_all.csv"))
G = pd.read_csv(os.path.join(DS, "2025-11-21", "ground_truth.csv"))[["Plot_ID", "Plot", "Variety", "N_treatment", "height_cm"]].rename(columns={"height_cm": "ruler_maturity_cm"})
# same preparation as height_models.py: height-type columns m -> cm, VNIR NaN cells (NaN rule) -> flight median
def prep(df):
    d = df.copy()
    for c in [c for c in d.columns if c.startswith(("H", "cth_", "tip_thin", "roughness", "Hall", "profile_area")) and c not in ("H_cv", "H_skew", "H_kurt", "H_crr")]:
        if d[c].dtype != object: d[c] = d[c] * 100.0
    vc = [c for c in feats if c in d.columns and d[c].dtype != object]
    d[vc] = d[vc].fillna(d[vc].median())
    return d.set_index("Plot_ID")
XA = prep(A); XM = prep(Mt)
missing = [f for f in feats if f not in XA.columns]
if missing: raise SystemExit("anthesis features lack " + ", ".join(missing[:8]))
ok = XA[feats].notna().all(axis=1) & (XA.valid_lidar == 1)
pa = pd.Series(np.nan, index=XA.index); pa[ok] = est.predict(XA.loc[ok, feats].to_numpy(float))
okm = XM[feats].notna().all(axis=1) & (XM.valid_lidar == 1)
pm = pd.Series(np.nan, index=XM.index); pm[okm] = est.predict(XM.loc[okm, feats].to_numpy(float))
out = G.set_index("Plot_ID").copy()
out["pred_anthesis_cm"] = pa.reindex(out.index); out["pred_maturity_cm"] = pm.reindex(out.index)
out["cth_p95_anthesis_cm"] = (XA.cth_p95 * (100 if XA.cth_p95.abs().max() < 20 else 1)).reindex(out.index)
out["cth_p95_maturity_cm"] = (XM.cth_p95 * (100 if XM.cth_p95.abs().max() < 20 else 1)).reindex(out.index)
out["ruler_minus_pred_anthesis_cm"] = out.ruler_maturity_cm - out.pred_anthesis_cm
out["cth_p95_maturity_minus_anthesis_cm"] = out.cth_p95_maturity_cm - out.cth_p95_anthesis_cm
out["pred_maturity_minus_anthesis_cm"] = out.pred_maturity_cm - out.pred_anthesis_cm
# model-free alternative: the maturity-flight linear calibration ruler = a + b x cth_p95 applied to the anthesis cth_p95
mm = out.dropna(subset=["cth_p95_maturity_cm", "ruler_maturity_cm"]); b1, b0 = np.polyfit(mm.cth_p95_maturity_cm, mm.ruler_maturity_cm, 1)
out["pred_anthesis_cth_calibration_cm"] = b0 + b1 * out.cth_p95_anthesis_cm
out.reset_index().to_csv(os.path.join(HP, "results", "anthesis_predicted_height.csv"), index=False)
def s(v): v = v.dropna(); return dict(n=int(len(v)), mean=round(float(v.mean()), 2), sd=round(float(v.std()), 2), median=round(float(v.median()), 2), p10=round(float(v.quantile(.1)), 2), p90=round(float(v.quantile(.9)), 2))
summ = dict(model=dict(feature_set=bm["family"], learner=bm.learner, trained_on="2025-11-21 (maturity flight, ruler at maturity)", maturity_cv=dict(R2=round(float(bm.R2), 3), RMSE=round(float(bm.RMSE), 2)), file=bm.model_file),
            predicted_anthesis_height_cm=s(out.pred_anthesis_cm), ruler_maturity_cm=s(out.ruler_maturity_cm),
            ruler_maturity_minus_predicted_anthesis_cm=s(out.ruler_minus_pred_anthesis_cm),
            cth_p95_maturity_minus_anthesis_cm=s(out.cth_p95_maturity_minus_anthesis_cm),
            predicted_maturity_minus_predicted_anthesis_cm=s(out.pred_maturity_minus_anthesis_cm),
            cth_calibration=dict(fitted_on="2025-11-21", intercept_cm=round(float(b0), 2), slope=round(float(b1), 3), predicted_anthesis_cm=s(out.pred_anthesis_cth_calibration_cm),
                                 ruler_maturity_minus_predicted_anthesis_cm=s(out.ruler_maturity_cm - out.pred_anthesis_cth_calibration_cm)),
            why_lidar_only="VNIR features of a senesced canopy (maturity) and a green canopy (anthesis) differ by >100 sd (red-edge indices); a maturity model using them extrapolates to nonsense on the anthesis flight (joint PCA ridge gave a mean of -33 cm), so only LiDAR-structure models are applied across dates.",
            by_N_treatment={str(k): s(v) for k, v in out.groupby("N_treatment").ruler_minus_pred_anthesis_cm} if "N_treatment" in out.columns else {},
            note="No height was measured at anthesis, so the anthesis values are predictions and the differences are not error metrics.")
json.dump(summ, open(os.path.join(HP, "results", "anthesis_height_difference_summary.json"), "w"), indent=1)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=110)
axs[0].scatter(out.pred_anthesis_cm, out.ruler_maturity_cm, s=18, c="#1f6f4a"); lo, hi = 60, 105; axs[0].plot([lo, hi], [lo, hi], "k--", lw=1); axs[0].set_xlim(lo, hi); axs[0].set_ylim(lo, hi)
axs[0].set_xlabel("predicted height, anthesis flight (cm)"); axs[0].set_ylabel("ruler height at maturity (cm)"); axs[0].set_title("Anthesis prediction vs maturity ruler (not a validation)")
axs[1].hist(out.ruler_minus_pred_anthesis_cm.dropna(), bins=20, color="#1f6f4a", alpha=.85); axs[1].axvline(0, color="k", lw=1)
axs[1].set_xlabel("ruler at maturity - predicted anthesis height (cm)"); axs[1].set_title(f"difference: mean {summ['ruler_maturity_minus_predicted_anthesis_cm']['mean']} cm, sd {summ['ruler_maturity_minus_predicted_anthesis_cm']['sd']} cm")
axs[2].hist(out.cth_p95_maturity_minus_anthesis_cm.dropna(), bins=20, color="#4a6fa5", alpha=.85); axs[2].axvline(0, color="k", lw=1)
axs[2].set_xlabel("LiDAR canopy-top P95: maturity - anthesis (cm)"); axs[2].set_title(f"model-free cross-check: mean {summ['cth_p95_maturity_minus_anthesis_cm']['mean']} cm")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "height_anthesis_prediction.png")); plt.close(fig)
print(json.dumps(summ, indent=1))
