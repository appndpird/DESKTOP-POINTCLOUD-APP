"""Combined height + biomass report (figures PNG + self-contained HTML) for Models_2026-10-09 from the results CSVs."""
import os, io, base64, json, glob, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(ROOT, "dataset"); M = ROOT
HP = os.path.join(M, "height"); BM = os.path.join(M, "biomass"); DL = os.path.join(M, "two_stream")
FIG = os.path.join(M, "figures"); os.makedirs(FIG, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
def metr(p, y):
    e = p - y; mape = np.mean(np.abs(e) / y) * 100
    return dict(R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.abs(e).mean(), r=np.corrcoef(p, y)[0, 1], acc=100 - mape, n=len(y))
def scatter(ax, p, y, title, unit, color=C1):
    m = metr(p, y); lo, hi = min(p.min(), y.min()), max(p.max(), y.max()); pad = (hi - lo) * 0.05
    ax.scatter(p, y, s=16, color=color, alpha=0.75, edgecolor="white", linewidth=0.5); ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
    b = np.polyfit(p, y, 1); xx = np.array([lo, hi]); ax.plot(xx, np.polyval(b, xx), color="#444", lw=1)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_xlabel(f"predicted ({unit})"); ax.set_ylabel(f"measured ({unit})"); ax.grid(alpha=.3)
    ax.set_title(title, fontsize=9); ax.text(0.03, 0.97, f"R² {m['R2']:.2f}  r {m['r']:.2f}\nRMSE {m['RMSE']:.2f}  MAE {m['MAE']:.2f}\n100-MAPE {m['acc']:.1f}%  n {m['n']}", transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))
    return m
def b64(path): return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
def table_html(df, floatfmt="{:.2f}"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return "<div class='tw'>" + d.to_html(index=False, escape=False, border=0) + "</div>"
figs = {}
def img(k): return f"<img src='{b64(figs[k])}'>" if k in figs else ""

# ------------------------------------------------------------ data quality (dataset summary)
Q = pd.read_csv(os.path.join(DS, "dataset_summary.csv"))
qcols = [c for c in ("trial", "flight", "stage", "plots", "gt_plots", "lidar_qc_ok", "vnir_reflectance_ok", "vnir_qc_ok", "red_ok", "gt_valid_lidar", "gt_valid_vnir", "gt_valid_fused", "usable_band_cells_pct") if c in Q.columns]
# ------------------------------------------------------------ height
H = Rl = Rt = full_h = None
hp = os.path.join(HP, "results", "model_comparison.csv")
if os.path.exists(hp):
    R = pd.read_csv(hp); P = pd.read_csv(os.path.join(HP, "results", "per_plot_predictions.csv"))
    sets = json.load(open(os.path.join(HP, "results", "feature_sets.json"))); hrows = []
    HFAMS = [f for f in ("LiDAR_core", "LiDAR_all", "LiDAR+VNIR", "LiDAR_PCA", "LiDAR+VNIR_PCA", "LiDAR+VNIR_PCA_joint", "VNIR_bands_PCA", "LiDAR+VNIR_bands_PCA") if (R.feature_set == f).any()]
    HCOL = {"LiDAR_core": C1, "LiDAR_all": C3, "LiDAR+VNIR": C2, "LiDAR_PCA": "#7a5cc7", "LiDAR+VNIR_PCA": "#c7407a", "LiDAR+VNIR_PCA_joint": "#8a6d3b", "VNIR_bands_PCA": "#e07be0", "LiDAR+VNIR_bands_PCA": "#2f9e9e"}
    # Error metrics are reported on the MATURITY flight only: Muresk has one ruler height per plot, measured at maturity.
    # The anthesis flight is reported as a predicted height (height_anthesis_prediction.py), not as an error score.
    HFL = ["2025-11-21"]
    R = R[R.flight.isin(HFL) | R.scheme.str.startswith("leave_flight_out")]; P = P[P.flight.isin(HFL)]
    for fl in HFL:
        A = R[(R.scheme == "repeated10fold") & (R.flight == fl)]
        fig, axs = plt.subplots(1, len(HFAMS), figsize=(5 * len(HFAMS), 4.8), dpi=110)
        for ax, fam in zip(np.atleast_1d(axs), HFAMS):
            b = A[A.feature_set == fam].nsmallest(1, "RMSE").iloc[0]
            d = P[(P.scheme == "repeated10fold") & (P.flight == fl) & (P.feature_set == fam) & (P.learner == b.learner)]
            m = scatter(ax, d.predicted.values, d.measured.values, f"{fl} {fam}: {b.learner} (repeated 10-fold)", "cm", HCOL[fam])
            hrows.append(dict(flight=fl, feature_set=fam, best_learner=b.learner, n_features=int(b.n_features), **{k: v for k, v in m.items() if k != "n"}))
        fig.tight_layout(); p = os.path.join(FIG, f"height_scatter_{fl}.png"); fig.savefig(p); plt.close(fig); figs[f"h_{fl}"] = p
    fig, axs = plt.subplots(1, len(HFL), figsize=(8 * len(HFL), 5), dpi=110)
    for ax, fl in zip(np.atleast_1d(axs), HFL):
        A = R[(R.scheme == "repeated10fold") & (R.flight == fl)].pivot(index="learner", columns="feature_set", values="RMSE")[HFAMS]
        A.plot.bar(ax=ax, color=[HCOL[f] for f in HFAMS], width=0.85); ax.set_ylabel("RMSE cm (repeated 10-fold)"); ax.set_title(f"Height, {fl} (maturity flight, ruler at maturity)"); ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); p = os.path.join(FIG, "height_rmse_by_model.png"); fig.savefig(p); plt.close(fig); figs["h_bar"] = p
    fig, axs = plt.subplots(1, len(HFL), figsize=(7 * len(HFL), 5), dpi=110)
    for ax, fl in zip(np.atleast_1d(axs), HFL):
        imp = pd.read_csv(os.path.join(HP, "results", f"rf_importance_LiDAR_VNIR_{fl}.csv")).head(15)
        ax.barh(imp.feature, imp.importance, color=[C2 if f.endswith(("_veg", "vnir")) or f.startswith(("refl", "spec_pc", "red_")) else C1 for f in imp.feature]); ax.invert_yaxis(); ax.set_title(f"RF importance, LiDAR+VNIR set, {fl} (blue LiDAR, orange VNIR)"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "height_importance.png"); fig.savefig(p); plt.close(fig); figs["h_imp"] = p
    H = pd.DataFrame(hrows)
    Rl = R[(R.scheme == "LOOCV") & R.flight.isin(HFL)][["flight", "feature_set", "learner", "R2", "RMSE", "MAE", "r", "accuracy"]]
    Rt = None   # train-one-flight / test-the-other is not reported: the anthesis flight has no height measurement of its own
    full_h = R[(R.scheme == "repeated10fold") & R.flight.isin(HFL)][["flight", "feature_set", "learner", "n_features", "R2", "RMSE", "MAE", "r", "accuracy", "within5cm"]].sort_values(["flight", "RMSE"])
    H.to_csv(os.path.join(HP, "results", "summary_best_per_family.csv"), index=False)
    # anthesis flight: predicted height + difference to the maturity ruler (height_anthesis_prediction.py)
    HA = None; hap = os.path.join(HP, "results", "anthesis_height_difference_summary.json")
    if os.path.exists(hap):
        HA = json.load(open(hap)); figs["h_anth"] = os.path.join(FIG, "height_anthesis_prediction.png")
# ------------------------------------------------------------ biomass
Bt = cross = pooled = skipped = None; DSETS = ["muresk_anthesis", "muresk_maturity", "agt_anthesis", "agt_maturity"]
bp = os.path.join(BM, "results", "model_comparison.csv")
if os.path.exists(bp):
    RB = pd.read_csv(bp); PB = pd.read_csv(os.path.join(BM, "results", "per_plot_predictions.csv")); TOP = pd.read_csv(os.path.join(BM, "results", "top_features.csv"))
    sk = os.path.join(BM, "results", "skipped_families.csv"); skipped = pd.read_csv(sk) if os.path.exists(sk) and os.path.getsize(sk) > 5 else None
    brows = []
    BFAMS = ("LiDAR_core", "VNIR_core", "Fused_core", "Fused_all", "LiDAR_PCA", "VNIR_PCA", "Fused_PCA", "Fused_PCA_joint", "VNIR_bands_PCA", "Fused_bands_PCA")
    BCOL = {"LiDAR_core": C1, "VNIR_core": C2, "Fused_core": C3, "Fused_all": C4, "LiDAR_PCA": "#7a5cc7", "VNIR_PCA": "#e07be0", "Fused_PCA": "#c7407a", "Fused_PCA_joint": "#8a6d3b", "VNIR_bands_PCA": "#d4a017", "Fused_bands_PCA": "#2f9e9e"}
    for ds in DSETS:
        A = RB[(RB.scheme == "repeated10fold") & (RB.dataset == ds)]
        fams = [f for f in BFAMS if (A.feature_set == f).any()]
        if not fams: continue
        fig, axs = plt.subplots(1, len(fams), figsize=(5 * len(fams), 4.8), dpi=110)
        for ax, fam in zip(np.atleast_1d(axs), fams):
            col = BCOL[fam]
            b = A[A.feature_set == fam].nsmallest(1, "RMSE").iloc[0]
            d = PB[(PB.scheme == "repeated10fold") & (PB.dataset == ds) & (PB.feature_set == fam) & (PB.learner == b.learner)]
            m = scatter(ax, d.predicted.values, d.measured.values, f"{ds} {fam}: {b.learner}", "kg/ha", col)
            brows.append(dict(dataset=ds, feature_set=fam, best_learner=b.learner, n_features=int(b.n_features), **{k: v for k, v in m.items() if k != "n"}, n=int(m["n"])))
        fig.tight_layout(); p = os.path.join(FIG, f"biomass_scatter_{ds}.png"); fig.savefig(p); plt.close(fig); figs[f"b_{ds}"] = p
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=110)
    for ax, ds in zip(axs.ravel(), DSETS):
        A = RB[(RB.scheme == "repeated10fold") & (RB.dataset == ds)]
        if A.empty: ax.set_title(f"Biomass, {ds}: not run"); continue
        A = A.pivot(index="learner", columns="feature_set", values="R2"); A = A[[c for c in BFAMS if c in A.columns]]
        A.plot.bar(ax=ax, color=[BCOL[c] for c in A.columns], width=0.85); ax.set_ylabel("R² (repeated 10-fold)"); ax.set_title(f"Biomass, {ds}"); ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); p = os.path.join(FIG, "biomass_r2_by_model.png"); fig.savefig(p); plt.close(fig); figs["b_bar"] = p
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=110)
    for ax, ds in zip(axs.ravel(), DSETS):
        t = TOP[TOP.dataset == ds].head(15)
        if t.empty: continue
        ax.barh(t.feature, t.rf_importance, color=[C2 if (f.endswith(("_veg", "_vnir")) or f.startswith(("refl", "spec_pc", "red_", "NDRE", "NDVI", "OSAVI", "GNDVI", "REP", "LCI", "MTCI", "CIRE", "kNDVI", "NIRv", "WDRVI", "PRI", "WBI"))) else C1 for f in t.feature])
        ax.invert_yaxis(); ax.set_title(f"Top features, {ds} (RF importance; blue LiDAR, orange VNIR)"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "biomass_top_features.png"); fig.savefig(p); plt.close(fig); figs["b_imp"] = p
    Bt = pd.DataFrame(brows); Bt.to_csv(os.path.join(BM, "results", "summary_best_per_family.csv"), index=False)
    cross = RB[RB.scheme == "cross_trial"][["dataset", "trained_on", "feature_set", "learner", "R2", "RMSE", "rRMSE", "bias", "r"]].sort_values(["dataset", "RMSE"])
    pooled = RB[RB.scheme == "pooled_group10fold"].pivot_table(index=["feature_set", "learner"], columns="dataset", values="R2").round(2)
# ------------------------------------------------------------ two-stream
dl_rows = None; DLH = None
dlp = os.path.join(DL, "results", "dl_biomass_comparison.csv")
if os.path.exists(dlp):
    DLc = pd.read_csv(dlp); DLP = pd.read_csv(os.path.join(DL, "results", "dl_biomass_per_plot_predictions.csv"))
    vs = [v for v in ("two_stream", "lidar_only", "vnir_only") if (DLP.variant == v).any()]
    fig, axs = plt.subplots(1, len(vs), figsize=(5 * len(vs), 4.8), dpi=110)
    for ax, v in zip(np.atleast_1d(axs), vs):
        d = DLP[(DLP.variant == v) & DLP.predicted.notna()]; scatter(ax, d.predicted.values, d.measured.values, f"Network {v}: 10-fold, all datasets pooled", "kg/ha", {"two_stream": C3, "lidar_only": C1, "vnir_only": C2}[v])
    fig.tight_layout(); p = os.path.join(FIG, "biomass_dl_scatter.png"); fig.savefig(p); plt.close(fig); figs["b_dl"] = p
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), dpi=110)
    for ax, ds in zip(axs, DSETS):
        d = DLP[(DLP.variant == "two_stream") & (DLP.dataset == ds) & DLP.predicted.notna()]
        if len(d): scatter(ax, d.predicted.values, d.measured.values, f"two_stream, {ds} (held-out folds)", "kg/ha", C3)
    fig.tight_layout(); p = os.path.join(FIG, "biomass_dl_scatter_per_dataset.png"); fig.savefig(p); plt.close(fig); figs["b_dl2"] = p
    dl_rows = DLc[["variant", "dataset", "n", "R2", "RMSE", "rRMSE", "MAE", "r", "accuracy"]]
dlh = os.path.join(DL, "results", "dl_height_comparison.csv")
if os.path.exists(dlh): DLH = pd.read_csv(dlh); DLH = DLH[DLH.dataset == "muresk_maturity"][["variant", "dataset", "n", "R2", "RMSE", "MAE", "r", "accuracy"]]
# common-band run (--bands common): the same network with the fixed 111-band list for every plot
DLCB = DLHCB = None
dlcb = os.path.join(DL, "results", "dl_biomass_commonbands_comparison.csv")
if os.path.exists(dlcb): DLCB = pd.read_csv(dlcb)[["variant", "dataset", "n", "R2", "RMSE", "rRMSE", "MAE", "r", "accuracy"]]
dlhcb = os.path.join(DL, "results", "dl_height_commonbands_comparison.csv")
if os.path.exists(dlhcb): DLHCB = pd.read_csv(dlhcb); DLHCB = DLHCB[DLHCB.dataset == "muresk_maturity"][["variant", "dataset", "n", "R2", "RMSE", "MAE", "r", "accuracy"]]
# grouped-fold reruns (folds keep both flights of a plot together; height on the maturity flight only): the valid two-stream scores
DLG = DLHG = None
dlg = os.path.join(DL, "results", "dl_biomass_commonbands_grouped_comparison.csv")
if os.path.exists(dlg): DLG = pd.read_csv(dlg)[["variant", "dataset", "n", "R2", "RMSE", "rRMSE", "MAE", "r", "accuracy"]]
dlhg = os.path.join(DL, "results", "dl_height_commonbands_grouped_maturity_comparison.csv")
if os.path.exists(dlhg): DLHG = pd.read_csv(dlhg)[["variant", "dataset", "n", "R2", "RMSE", "MAE", "r", "accuracy"]]
DLMF = DLMA = None   # Muresk only: the 138 recommended bands (fixed list, dataset\vnir_muresk_bands.csv) / every usable band per plot incl. the excluded ranges; grouped folds
dlmf = os.path.join(DL, "results", "dl_biomass_commonbands_muresk_fullbands_comparison.csv")
if os.path.exists(dlmf): DLMF = pd.read_csv(dlmf)[["variant", "dataset", "n", "R2", "RMSE", "rRMSE", "MAE", "r", "accuracy"]]
dlma = os.path.join(DL, "results", "dl_biomass_commonbands_muresk_validbands_comparison.csv")
if os.path.exists(dlma): DLMA = pd.read_csv(dlma)[["variant", "dataset", "n", "R2", "RMSE", "rRMSE", "MAE", "r", "accuracy"]]
ncb = None
cbp = os.path.join(M, "dataset", "vnir_common_bands.csv")
if os.path.exists(cbp): ncb = int(pd.read_csv(cbp).common_95.sum())

css = """<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:0 auto;padding:20px;color:#1b1b1b;line-height:1.45}h1{font-size:1.7rem}h2{margin-top:36px;border-bottom:2px solid #ddd;padding-bottom:4px}h3{margin-top:22px}
table{border-collapse:collapse;font-size:.82rem;font-variant-numeric:tabular-nums}th,td{padding:4px 8px;border-bottom:1px solid #e3e3e3;text-align:right}th:first-child,td:first-child,td:nth-child(2),th:nth-child(2){text-align:left}th{background:#f3f4f2;position:sticky;top:0}
.tw{overflow-x:auto;margin:8px 0 16px}img{max-width:100%;height:auto;border:1px solid #e3e3e3;border-radius:6px;margin:6px 0}.note{background:#f6f8f5;border-left:4px solid #1f6f4a;padding:8px 14px;margin:12px 0}.warn{background:#fff4f2;border-left:4px solid #b3261e;padding:8px 14px;margin:12px 0}.small{color:#666;font-size:.85rem}</style>"""
html = [f"<title>Height and Biomass Models 2026-10-09</title>{css}<h1>Plant height and biomass models from GOBI LiDAR + VNIR — Muresk and AGT NUE trials 2025 (rebuild of 9 October 2026)</h1>",
        f"<p class='small'>Built {time.strftime('%Y-%m-%d %H:%M')} on Dataset_2026-10-09 with PhenoApp v3.3 (v3.2 LiDAR pipeline: gap noise rule, plot-local ground, canopy classification, interior gap metrics; v3.0.1 VNIR: NaN rule for unusable bands, flight-level reflectance check, red-clipping censoring). "
        "Validation: nested repeated 10-fold CV (feature selection inside each fold), LOOCV for the top height models, leave-one-flight-out and cross-trial transfer; the two-stream network uses 10-fold CV stratified by dataset.</p>",
        "<div class='warn'><b>Data limits carried into this rebuild.</b> AGT 2025-11-13 (maturity): both VNIR orthomosaics are at-sensor radiance, not reflectance, so every VNIR value is NaN and the dataset is modelled LiDAR-only. AGT 2025-09-22 (anthesis): the 670 nm band is clipped to zero on about half of the plot pixels; red-based indices are NaN on plots with red_ok = 0 and are dropped from the models when missing on more than 5 % of plots. "
        "Muresk: blue bands clipped (not used by the core features). The 'accuracy' column is 100 - MAPE and reflects the spread of the target, not model skill.</div>",
        "<h2>1. Dataset</h2>", table_html(Q[qcols], "{:.1f}"),
        "<p>plots = grid polygons; gt_plots = with ground truth; gt_valid_* = ground-truth plots passing LiDAR QC / strict VNIR QC (vnir_qc_ok and red_ok) / both. usable_band_cells_pct = share of (plot x band) cells with a usable band in the original orthomosaic.</p>",
        "<h2>2. Methods: feature families and validation</h2>",
        "<p><b>Features.</b> LiDAR (PhenoApp v3.2): per plot, points inside the polygon plus a 0.25-1.0 m alley ring; noise flagged with the gap rule (points more than 30 cm above the 99.9th percentile), a robust ground plane fitted to the alley ring (25 cm cells, 5th percentile) and refined with in-plot points within 8 cm, heights normalised to it, canopy = h > 10 cm. "
        "Features are computed on canopy points only: height percentiles (H50 to H99.9, top-N mean), spread and shape, 10 density layers, canopy cover on 5 cm and 2 cm grids, voxel volume, gap fraction Pgap and 20 cm gap layers, LAI proxy, profile area, canopy-top P90/P95/P99, rumple, roughness, normalised intensity; v3.2 adds the same gap metrics on the 20 cm-trimmed interior (the refit polygons include alley soil along the long sides) and ground visibility on a 2 cm grid. "
        "VNIR (PhenoApp v3.0.1): over the sampling region (polygon inset 10 cm), 30 narrow-band indices evaluated per pixel and averaged over all valid pixels and over vegetation pixels (NDVI threshold per flight), broad-band reflectances, red-edge derivative features, Guyot red-edge position, and five PCA scores of the vegetation spectra. A band is usable for a plot only when it is non-zero on at least 50 % of the region pixels, lies outside the excluded ranges (below 415 nm, 755-770 nm, 928-962 nm) and comes from a reflectance cube; an index is usable only when every band it needs is usable on at least 50 % of the pixels. Anything else is NaN and never filled with a number.</p>",
        "<p><b>Feature families.</b> <i>LiDAR_core</i>: the 22 v3.2 core features (biomass adds PVI, intensity x cover, three gap layers, three density layers, cover_5cm, Pgap: 32). <i>LiDAR_all</i>: every finite, non-constant LiDAR feature (73). <i>VNIR_core</i>: 22 cleaned index / reflectance / derivative / PCA-score features (biomass: 29). <i>LiDAR+VNIR / Fused_core</i>: both core sets, for biomass also nine canopy-weighted indices (index x cover, index x H95, NDRE x PVI). <i>Fused_all</i>: all finite LiDAR and VNIR features. "
        "In these families a random forest ranks the features inside every training fold and the top 10 (height) or 12 (biomass) are passed to the learner, so the held-out plots never influence the selection.</p>",
        "<p><b>PCA families</b> (new in this rebuild). Instead of selecting features, the inputs are standardised and projected on principal components that keep 95 % of their variance; the scaler and the PCA are fitted inside every training fold and applied to the held-out fold, so the components are never computed from test plots. "
        "<i>LiDAR_PCA</i>: one PCA on the 73 LiDAR features. <i>VNIR_PCA</i>: one PCA on all usable VNIR index features. <i>Fused_PCA</i>: a separate PCA per modality, scores concatenated (late fusion), so the LiDAR structure and the spectral information keep their own components. <i>Fused_PCA_joint</i>: a single PCA on the concatenated LiDAR and VNIR features (early fusion). "
        "<i>VNIR_bands_PCA</i>: PCA on the vegetation mean spectrum itself rather than on indices: only the bands that are usable on at least 95 % of the plots enter (the clipped blue region, the oxygen and water-vapour bands and, at AGT anthesis, the red absorption well are left out), each plot's spectrum is log-transformed and normalised by its own mean and standard deviation (SNV) so that components describe spectral shape rather than brightness, and the remaining NaN cells are replaced by the band median before the PCA. <i>Fused_bands_PCA</i>: the LiDAR PCA block plus the band PCA block. "
        "The components kept per block are listed in pca_components.csv in the height and biomass results folders. The same learners are used on the components (ridge, SVR, Gaussian process, random forest, extra trees, MLP, XGBoost, LightGBM).</p>",
        "<p><b>Plots entering the models.</b> A plot needs ground truth and LiDAR QC; families that use VNIR additionally need vnir_qc_ok = 1 (plot inside the cube, vegetation present, reflectance cube). A feature column is used only when finite on more than 95 % of the plots of a dataset, so red-based indices drop out of the AGT anthesis models automatically; the remaining NaN cells are median-filled inside the dataset. AGT maturity has no usable VNIR and is modelled with the LiDAR families only.</p>",
        "<p><b>Validation.</b> Nested repeated 10-fold cross-validation (5 repeats, selection or PCA inside each fold) is the headline figure; leave-one-out for the top height models and the cth_p95 linear reference; leave-one-flight-out transfer for height; a pooled model with trial and stage flags (grouped 10-fold by plot) and cross-trial transfer for biomass. R² and RMSE are computed on the held-out predictions; 'accuracy' is 100 minus the mean absolute percentage error and reflects the spread of the target as much as model skill.</p>"]
for lab, folder in (("height", HP), ("biomass", BM)):
    pcf = os.path.join(folder, "results", "pca_components.csv")
    if os.path.exists(pcf):
        html += [f"<h3>Principal components kept per block ({lab} models, fitted on all plots for reference)</h3>", table_html(pd.read_csv(pcf), "{:.3f}")]
# ------------------------------------------------------------ RESULTS TO REPORT (rule of 12 Oct 2026; pooled_fold_results.py)
rsp = os.path.join(M, "results", "report_summary.csv")
if os.path.exists(rsp):
    RS = pd.read_csv(rsp); HF = pd.read_csv(os.path.join(M, "results", "report_height_folds.csv")); BF = pd.read_csv(os.path.join(M, "results", "report_biomass_pooled_folds.csv"))
    def summ(df):
        d = df.copy(); d["R2 (all test plots)"] = d.R2_overall.round(2); d["RMSE (all test plots)"] = d.RMSE_overall.round(1); d["rRMSE %"] = d.rRMSE_overall.round(1)
        d["RMSE per fold, mean ± SD"] = d.RMSE_fold_mean.round(1).astype(str) + " ± " + d.RMSE_fold_sd.round(1).astype(str)
        d["R² per fold, mean ± SD"] = d.R2_fold_mean.round(2).astype(str) + " ± " + d.R2_fold_sd.round(2).astype(str)
        return d[["target", "category", "approach", "plots", "test_plots_per_fold", "R2 (all test plots)", "RMSE (all test plots)", "rRMSE %", "RMSE per fold, mean ± SD", "R² per fold, mean ± SD"]]
    def per_fold(df):
        d = df[df.fold.astype(str) != "all"].copy(); d["fold"] = d.fold.astype(int); d["cell"] = d.R2.round(2).astype(str) + " / " + d.RMSE.round(1).astype(str)
        return d.pivot_table(index=["target", "category", "approach"], columns="fold", values="cell", aggfunc="first").reset_index()
    html += ["<h2>3. Results to report (overall; height on Muresk maturity)</h2>",
             "<div class='note'><b>Reporting rules.</b> Height: Muresk maturity flight only, the only flight with a ruler measurement. Biomass: one overall result pooled over all trials and stages, "
             "computed on the test set of each of the 10 folds (every plot is in the test set of exactly one fold; folds grouped by plot) and over all test plots together. "
             "Feature models = the average of the top three models of the category, chosen inside each training fold (nested). Deep learning = the two-stream network, variant LiDAR only / VNIR only / two-stream. "
             "LiDAR only, VNIR only and fused are compared on the same plots: the 506 plot-flights with usable VNIR (Muresk anthesis and maturity, AGT anthesis; AGT maturity has radiance cubes); "
             "the LiDAR-only result over all 758 plot-flights is given in addition. Pooled R² mixes trials and stages with different mean biomass and is therefore higher on the 758 plots than on the 506; "
             "compare categories within the same row group, and use RMSE / rRMSE across groups. "
             "The overall results use exactly the same test plots for features and deep learning. Fold k holds the same number of plots of each flight for both approaches "
             "(table 'Flights in the test set'), but not the same individual plots, because the feature models were cross-validated within each flight and the network across all flights; "
             "fold-by-fold values are therefore compared within an approach, the overall values across approaches.</div>",
             "<h3>Overall results (all test plots of the 10 folds together)</h3>", table_html(summ(RS)),
             ("<h3>Flights in the test set (plots per flight in each fold's test set)</h3>" + table_html(pd.read_csv(os.path.join(M, "results", "report_testset_composition.csv")))) if os.path.exists(os.path.join(M, "results", "report_testset_composition.csv")) else "",
             "<h3>Height, Muresk maturity: test set of each fold (R² / RMSE cm, 12-13 plots per fold)</h3>", table_html(per_fold(HF)),
             "<h3>Biomass, pooled over all trials: test set of each fold (R² / RMSE kg/ha, about 51 plots per fold on the VNIR plots, 76 on all plots)</h3>", table_html(per_fold(BF))]
    # fold models and the "best three folds" option (feature fold ensembles: nested_top3_<ds>.csv; network: fold_scores.csv of the bundled ensembles)
    fm = []
    for t in ("height", "biomass"):
        for f in sorted(os.listdir(os.path.join(M, t, "results"))):
            if f.startswith("nested_top3_") and f.endswith(".csv") and not f.startswith(("nested_top3_predictions", "nested_top3_summary")) and "__" not in f:
                if t == "height" and "muresk_maturity" not in f: continue
                q = pd.read_csv(os.path.join(M, t, "results", f))
                fm.append(q.assign(approach="features (fold ensemble: top three chosen on the fold's training plots)", fold=q.outer_fold)[["target", "dataset", "category", "approach", "fold", "n_test", "R2", "RMSE"]])
    fsp = r"C:\Users\appn\Downloads\PhenoFuse-main\phenoapp_v3_2026-10-06\phenoapp\assets\dl_models\fold_scores.csv"
    if os.path.exists(fsp):
        q = pd.read_csv(fsp); q["dataset"] = np.where(q.target == "height", "muresk_maturity", "all four datasets pooled")
        fm.append(q.assign(category=q.variant.map({"lidar_only": "lidar", "vnir_only": "vnir", "two_stream": "fused"}), approach="deep learning (fold model)")[["target", "dataset", "category", "approach", "fold", "n_test", "R2", "RMSE"]])
    if fm:
        FM = pd.concat(fm, ignore_index=True); FM["fold"] = FM.fold.astype(int)
        FM["best3"] = FM.groupby(["target", "dataset", "category", "approach"]).RMSE.rank(method="first") <= 3
        FM["cell"] = FM.RMSE.round(1).astype(str) + np.where(FM.best3, " *", "")
        fpv = FM.pivot_table(index=["target", "dataset", "category", "approach"], columns="fold", values="cell", aggfunc="first").reset_index()
        html += ["<h3>Fold models and the 'best three folds' option</h3>",
                 "<p>Every fold has its own model: for the feature models, the top-three ensemble chosen and fitted on that fold's training plots (per flight); for the network, the fold model. "
                 "Values are the test RMSE of each fold model on its own test plots; <b>*</b> marks the three best folds of each case. The app (Height Models / Biomass ML: *_fold_ensembles.joblib; "
                 "Deep Models: fold_scores.csv) predicts with all fold models (default) or with the best three. Ranking folds by their test score selects on the test sets, so the best three look "
                 "better than they are; their scores are not an independent accuracy estimate, and the accuracy to quote remains the all-folds result above.</p>",
                 table_html(fpv)]
    emp = os.path.join(M, "results", "report_each_model_folds.csv")
    if os.path.exists(emp):
        EMF = pd.read_csv(emp); EMS = pd.read_csv(os.path.join(M, "results", "report_each_model_summary.csv"))
        EMS["RMSE per fold, mean ± SD"] = EMS.RMSE_fold_mean.round(1).astype(str) + " ± " + EMS.RMSE_fold_sd.round(1).astype(str)
        d_ = EMF[EMF.fold.astype(str) != "all"].copy(); d_["fold"] = d_.fold.astype(int); d_["cell"] = d_.R2.round(2).astype(str) + " / " + d_.RMSE.round(1).astype(str)
        pv = d_.pivot_table(index=["target", "category", "family", "learner"], columns="fold", values="cell", aggfunc="first").reset_index()
        html += ["<h3>Every individual feature model on the same test folds</h3>",
                 "<p>Each feature family x learner scored on the same 10 test folds as the tables above (no selection involved, so these are the components the top-three ensembles choose from). "
                 "Sorted by RMSE within each target and category. The network variants are the deep-learning rows of the tables above.</p>",
                 table_html(EMS[["target", "category", "family", "learner", "n", "R2", "RMSE", "rRMSE", "RMSE per fold, mean ± SD"]].round(3)),
                 "<h4>Per fold (R² / RMSE)</h4>", table_html(pv)]
# ------------------------------------------------------------ VALID-BAND PLOTS ONLY (rule A / rule B; valid_plot_results.py)
vps = os.path.join(M, "results", "valid_plots_summary.csv")
if os.path.exists(vps):
    VS = pd.read_csv(vps); VF = pd.read_csv(os.path.join(M, "results", "valid_plots_folds.csv"))
    VS["RMSE per fold, mean ± SD"] = VS.RMSE_fold_mean.round(1).astype(str) + " ± " + VS.RMSE_fold_sd.round(1).astype(str)
    vf_ = VF[VF.fold.astype(str) != "all"].copy(); vf_["fold"] = vf_.fold.astype(int); vf_["cell"] = vf_.R2.round(2).astype(str) + " / " + vf_.RMSE.round(1).astype(str)
    vpv = vf_.pivot_table(index=["rule", "category", "approach"], columns="fold", values="cell", aggfunc="first").reset_index()
    vcp = os.path.join(M, "results", "valid_plots_composition.csv")
    html += ["<h2>4. Biomass on valid-band plots only (rule A and rule B)</h2>",
             "<div class='note'><b>Plot selection.</b> Only plots whose spectra have no band problem enter these runs. "
             "<b>Rule A:</b> every one of the 111 common bands valid (not clipped) on at least 50 % of the plot's pixels, reflectance cube, VNIR QC passed: 497 plot-flights "
             "(AGT anthesis 243, Muresk anthesis 128, Muresk maturity 126). <b>Rule B:</b> every common band valid on at least 95 % of the plot's pixels and the plot entirely inside the cube: "
             "209 plot-flights, Muresk only (anthesis 83, maturity 126); no AGT plot qualifies (red bands partly clipped at anthesis, radiance cubes at maturity). "
             "Models were trained and tested on these plots only: feature models = top three per category by nested selection, per flight; network = VNIR-only and two-stream variants on the same plots, folds grouped by plot. "
             "Overall results pooled over the flights of the rule, on the test set of each of the 10 folds; features and network on the same plots. LiDAR only (features) is shown as the reference. "
             "Plot lists: dataset\\valid_plots_ruleA.csv, valid_plots_ruleB.csv; saved ensembles: biomass\\models\\*_top3_ensemble__ruleA/B.joblib.</div>",
             "<h3>Overall results</h3>", table_html(VS[["rule", "category", "approach", "plots", "flights", "R2_overall", "RMSE_overall", "rRMSE_overall", "RMSE per fold, mean ± SD"]].round(3)),
             ("<h3>Flights in the test set (plots per flight in each fold)</h3>" + table_html(pd.read_csv(vcp))) if os.path.exists(vcp) else "",
             "<h3>Test set of each fold (R² / RMSE kg/ha)</h3>", table_html(vpv)]
if H is not None:
    hs = HA["ruler_maturity_minus_predicted_anthesis_cm"] if HA else None; hc = HA["cth_p95_maturity_minus_anthesis_cm"] if HA else None; hp_ = HA["predicted_anthesis_height_cm"] if HA else None
    html += ["<h2>Appendix A. Plant height, Muresk maturity flight: model development detail (ground truth = ruler at maturity, n = 128)</h2>",
             "<div class='note'><b>What is validated and what is predicted.</b> Muresk has one ruler height per plot, measured at maturity, so RMSE, R² and the other error metrics are reported for the maturity flight only. "
             "The anthesis flight (2025-09-30) has no height measurement; its height is <i>predicted</i> with the maturity-flight model and reported below as a predicted height with the difference to the maturity ruler, not as an error score. The AGT trial has no height ground truth.</div>",
             img("h_bar"), "<h3>Best model per feature family (held-out predictions, repeated 10-fold, maturity flight)</h3>", table_html(H),
             ("<h3>Anthesis flight: predicted height and the maturity-anthesis difference</h3>"
              f"<p>Model applied to the anthesis-flight features: {HA['model']['feature_set']} / {HA['model']['learner']}, trained on the maturity flight (held-out R² {HA['model']['maturity_cv']['R2']}, RMSE {HA['model']['maturity_cv']['RMSE']} cm). "
              f"Predicted anthesis height: mean {hp_['mean']} cm, sd {hp_['sd']} cm (n {hp_['n']}). Ruler at maturity minus predicted anthesis height: mean {hs['mean']} cm, sd {hs['sd']} cm, median {hs['median']} cm, 10th-90th percentile {hs['p10']} to {hs['p90']} cm. "
              f"Model-free cross-check on the LiDAR canopy-top height (cth_p95, maturity minus anthesis): mean {hc['mean']} cm, sd {hc['sd']} cm. A difference near zero means the crop had reached its final height by the anthesis flight; a negative LiDAR difference at maturity reflects lodging and bending of the senesced canopy. These are differences between dates, not errors.</p>"
              + img("h_anth")) if HA else "",
             img("h_2025-09-30"), img("h_2025-11-21"), "<h3>Leave-one-out for the top models and the cth_p95 reference</h3>", table_html(Rl),
             "<h3>Which features matter</h3>", img("h_imp"), "<h3>All height models (maturity flight)</h3>", table_html(full_h)]
else:
    html += ["<h2>3. Plant height</h2><p>not run</p>"]
if Bt is not None:
    html += ["<h2>Appendix B. Above-ground biomass per trial and flight: development detail (not the reported results)</h2>", img("b_bar"), "<h3>Best model per feature family and dataset (repeated 10-fold)</h3>", table_html(Bt),
             ("<p class='small'>Families not run: " + "; ".join(f"{r.dataset} {r.feature_set} ({r.reason})" for _, r in skipped.iterrows()) + "</p>") if skipped is not None and len(skipped) else "",
             "".join(img(f"b_{ds}") for ds in DSETS), "<h3>Top features (random-forest importance on all features, per dataset)</h3>", img("b_imp"),
             "<h3>Pooled model across trials and stages (grouped 10-fold), R² per dataset</h3>", table_html(pooled.reset_index()),
             "<h3>Cross-trial transfer (train one trial, test the other, same stage)</h3>", table_html(cross)]
    # Muresk-only, both flights pooled with a stage flag, folds grouped by plot (biomass_muresk_fullspectrum.py): recommended bands and all bands
    for run, title, note in (("muresk_fullspectrum", "both flights, all 128 + 128 ground-truth plots, every valid band: 138 bands (469-754, 772-926, 965-1000 nm) - the first bands, the water-vapour tail and the oxygen-A bands are not carried by the spectra tables, and the blue is clipped on the anthesis flight", "biomass\\models\\biomass_muresk_fullspectrum_*.joblib"),):
        mfs = os.path.join(BM, run, "best_models.csv")
        if not os.path.exists(mfs): html += [f"<h3>Muresk only, {title}</h3><p>in progress</p>"]; continue
        MF = pd.read_csv(mfs); MFc = pd.read_csv(os.path.join(BM, run, "model_comparison.csv")); MFs = json.load(open(os.path.join(BM, run, "settings.json")))
        per = MFc[MFc.dataset != "muresk_both"].merge(MF[["family", "learner"]].rename(columns={"family": "feature_set"}), on=["feature_set", "learner"]).pivot_table(index="feature_set", columns="dataset", values=["R2", "RMSE"]).round(2)
        html += [f"<h3>Muresk only, {title} (both flights pooled, stage flag, folds grouped by plot, 3 x 10-fold)</h3>",
                 f"<p>Inputs: {MFs['lidar_features']} LiDAR features, {MFs['vnir_index_features']} VNIR index features and the mean vegetation spectrum over {MFs['bands_used']} bands (log10, SNV). "
                 f"The best model of each family is saved for later fine-tuning on data where every band is valid ({note}); the per-dataset models of section 4 are kept unchanged. Scores are on the pooled 256 plot-flights (R² partly reflects the anthesis-maturity difference in mean biomass) and per flight within the same folds.</p>",
                 table_html(MF[["family", "learner", "R2", "RMSE", "rRMSE", "r", "features"]].round(3)), "<h4>Per flight, best learner of each family</h4>", table_html(per.reset_index())]
else:
    html += ["<h2>4. Above-ground biomass</h2><p>not run</p>"]
# ------------------------------------------------------------ reporting models: top three per category, nested selection
NT = []
for t in ("height", "biomass"):
    p_ = os.path.join(M, t, "results", "nested_top3_summary.csv")
    if os.path.exists(p_): NT.append(pd.read_csv(p_))
if NT:
    NT = pd.concat(NT, ignore_index=True)
    piv = NT.assign(cell=NT.R2.round(2).astype(str) + " / " + NT.RMSE.round(1).astype(str)).pivot_table(index=["target", "dataset"], columns="category", values="cell", aggfunc="first").reindex(columns=["lidar", "vnir", "fused"]).fillna("-")
    html += ["<h2>Appendix C. Top three per category per trial and flight (nested selection); saved ensembles</h2>",
             "<div class='note'><b>How these numbers are made.</b> Outer 10-fold cross-validation; inside each outer training set a 5-fold inner loop scores every feature family x learner, "
             "and the three best combinations of each category are fitted on the outer training plots and averaged on the outer test plots. The choice of the three never sees the plots it is scored on, "
             "so these are the honest scores of the procedure 'take the top three of the category and average them', to be quoted in reports. The same selection on all plots gives the saved ensembles "
             "(height\\models and biomass\\models, <code>&lt;target&gt;_&lt;dataset&gt;_&lt;category&gt;_top3_ensemble.joblib</code>), which the Height Models and Biomass ML tabs apply directly. "
             "Height: maturity flight only (ruler at maturity). AGT maturity: LiDAR only (radiance cubes).</div>",
             "<h3>R² / RMSE (kg/ha for biomass, cm for height)</h3>", table_html(piv.reset_index()),
             "<h3>Members of the saved ensembles</h3>", table_html(NT[["target", "dataset", "category", "n", "R2", "RMSE", "final_members", "model_file"]].round(3))]
html += ["<h2>Appendix D. Two-stream deep network per trial and flight (sparse 3D CNN + spectral transformer, late fusion): runs and band policies</h2>",
         "<p>Inputs from the same dataset; the spectral stream never sees an unusable band (per-plot band mask from the NaN rule, clipped pixels and excluded ranges are masked and the availability mask is given to the network). 10-fold CV stratified by dataset, 100 epochs per fold (AdamW, OneCycle schedule, log-biomass target). "
         "Training-time data augmentation, applied on the fly to every training plot: point cloud randomly rotated by 180° and mirrored across the row axis, 0-30 % of the points dropped at random, 1 cm xy jitter; spectral pixels bootstrap-resampled within the plot and scaled by a random brightness factor of 0.9-1.1. "
         "Nothing is augmented at test time, and the feature-based models use no augmentation (their regularisation is the nested selection, PCA and the ridge / PLS penalties). The pooled R² is inflated by the between-dataset differences in mean biomass; the per-dataset rows are the comparable numbers.</p>",
         img("b_dl"), img("b_dl2"), table_html(dl_rows) if dl_rows is not None else "<p>not run (torch / spconv unavailable at build time)</p>",
         "<div class='warn'><b>Fold caveat for the runs of 9-10 October.</b> Their 10 folds were stratified by dataset but not grouped by physical plot: in 246 of 256 height test cases and 708 of 760 biomass test cases the same plot's other flight was in the training set. "
         "For height both flights share one ruler value per plot, so those height scores are optimistic and are superseded by the grouped rerun on the maturity flight below; for biomass the two stages have different cuts (r = 0.33 between a plot's anthesis and maturity biomass), so the effect is small, and the grouped rerun below replaces them where available.</div>",
         (("<h3>Grouped-fold reruns (both flights of a plot in the same fold; common bands): the valid two-stream scores</h3>" + table_html(DLG)) if DLG is not None else "<p>Grouped-fold biomass rerun: in progress.</p>"),
         (("<h4>Height, maturity flight only, grouped folds, common bands</h4>" + table_html(DLHG)) if DLHG is not None else "<p>Grouped-fold height rerun (maturity flight only): in progress.</p>"),
         (("<h4>Muresk only, the 138 recommended Muresk bands (469-754, 772-926, 965-1000 nm; fixed list), both flights, grouped folds: the full-spectrum network saved for later fine-tuning</h4>" + table_html(DLMF)) if DLMF is not None else "<p>Muresk recommended-band two-stream run: in progress.</p>"),
         (("<h4>Muresk, 128 + 128 plots, the 142 valid bands (only 0-415 and 928-962 nm excluded; 469-926, 965-1000 nm), grouped folds</h4>" + table_html(DLMA)) if DLMA is not None else "<p>Muresk valid-bands two-stream run (142 bands): queued after the recommended-band run.</p>"),
         ("<h3>Same network for height (Muresk, maturity flight; per-plot bands, ungrouped folds - superseded)</h3>" + table_html(DLH)) if DLH is not None else "",
         (f"<h3>Common-band run: the same network with one fixed band list for every flight</h3><p>Band policy 'common': the {ncb} bands usable on at least 95 % of the ground-truth plots of every reflectance flight "
          "(522-645, 691-754, 772-926, 965-1000 nm; dataset\\vnir_common_bands.csv) are the spectral input of every plot, so no flight is represented by a different band set and the network cannot use band availability as a trial fingerprint. "
          "Blue and the red absorption well are dropped for all plots; clipped pixels are still ignored; the radiance flight stays LiDAR-only. Per-plot run above = most data per plot; common-band run = identical inputs across flights.</p>"
          + table_html(DLCB) + (("<h4>Height, common bands (maturity flight; ungrouped folds - superseded by the grouped rerun above)</h4>" + table_html(DLHCB)) if DLHCB is not None else "")) if DLCB is not None else "",
         "<h2>Files</h2><p class='small'>Everything is under Biomass Experiment\\Biomass_Height_2026-10-09: dataset\\ (per trial / flight: ground_truth.csv, features_lidar.csv, features_vnir.csv, vnir_spectra.csv, vnir_band_usable.csv, features_all.csv, lidar_classified\\, vnir\\), "
         "height\\ and biomass\\ (results\\, models\\), two_stream\\ (dl_data\\, results\\, models\\), figures\\, scripts\\ (build_dataset.py, dl_prepare.py, height_models.py, biomass_models.py, pca_families.py, dl_twostream.py, report_builder.py, grid_pyshp.py, vnir_envi.py) and README.md.</p>"]
page = "\n".join(html)
out = os.path.join(M, "REPORT_height_biomass_2026-10-09.html"); open(out, "w", encoding="utf-8").write(page)
print("report written:", out, round(len(page) / 1e6, 2), "MB")
