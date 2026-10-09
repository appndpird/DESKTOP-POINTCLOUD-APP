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
    for fl in ("2025-09-30", "2025-11-21"):
        A = R[(R.scheme == "repeated10fold") & (R.flight == fl)]
        fig, axs = plt.subplots(1, len(HFAMS), figsize=(5 * len(HFAMS), 4.8), dpi=110)
        for ax, fam in zip(np.atleast_1d(axs), HFAMS):
            b = A[A.feature_set == fam].nsmallest(1, "RMSE").iloc[0]
            d = P[(P.scheme == "repeated10fold") & (P.flight == fl) & (P.feature_set == fam) & (P.learner == b.learner)]
            m = scatter(ax, d.predicted.values, d.measured.values, f"{fl} {fam}: {b.learner} (repeated 10-fold)", "cm", HCOL[fam])
            hrows.append(dict(flight=fl, feature_set=fam, best_learner=b.learner, n_features=int(b.n_features), **{k: v for k, v in m.items() if k != "n"}))
        fig.tight_layout(); p = os.path.join(FIG, f"height_scatter_{fl}.png"); fig.savefig(p); plt.close(fig); figs[f"h_{fl}"] = p
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), dpi=110)
    for ax, fl in zip(axs, ("2025-09-30", "2025-11-21")):
        A = R[(R.scheme == "repeated10fold") & (R.flight == fl)].pivot(index="learner", columns="feature_set", values="RMSE")[HFAMS]
        A.plot.bar(ax=ax, color=[HCOL[f] for f in HFAMS], width=0.85); ax.set_ylabel("RMSE cm (repeated 10-fold)"); ax.set_title(f"Height, {fl}"); ax.grid(axis="y", alpha=.3); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); p = os.path.join(FIG, "height_rmse_by_model.png"); fig.savefig(p); plt.close(fig); figs["h_bar"] = p
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
    for ax, fl in zip(axs, ("2025-09-30", "2025-11-21")):
        imp = pd.read_csv(os.path.join(HP, "results", f"rf_importance_LiDAR_VNIR_{fl}.csv")).head(15)
        ax.barh(imp.feature, imp.importance, color=[C2 if f.endswith(("_veg", "vnir")) or f.startswith(("refl", "spec_pc", "red_")) else C1 for f in imp.feature]); ax.invert_yaxis(); ax.set_title(f"RF importance, LiDAR+VNIR set, {fl} (blue LiDAR, orange VNIR)"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "height_importance.png"); fig.savefig(p); plt.close(fig); figs["h_imp"] = p
    H = pd.DataFrame(hrows)
    Rl = R[R.scheme == "LOOCV"][["flight", "feature_set", "learner", "R2", "RMSE", "MAE", "r", "accuracy"]]
    Rt = R[R.scheme.str.startswith("leave_flight_out")].pivot_table(index=["feature_set", "learner"], columns=["scheme", "flight"], values="RMSE").round(2)
    full_h = R[R.scheme == "repeated10fold"][["flight", "feature_set", "learner", "n_features", "R2", "RMSE", "MAE", "r", "accuracy", "within5cm"]].sort_values(["flight", "RMSE"])
    H.to_csv(os.path.join(HP, "results", "summary_best_per_family.csv"), index=False)
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
if os.path.exists(dlh): DLH = pd.read_csv(dlh)[["variant", "dataset", "n", "R2", "RMSE", "MAE", "r", "accuracy"]]

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
if H is not None:
    html += ["<h2>2. Plant height (Muresk, ruler at maturity, n = 128 per flight)</h2>", img("h_bar"), "<h3>Best model per feature family (held-out predictions, repeated 10-fold)</h3>", table_html(H),
             img("h_2025-09-30"), img("h_2025-11-21"), "<h3>Leave-one-out for the top models and the cth_p95 reference</h3>", table_html(Rl),
             "<h3>Transfer between dates (train one flight, test the other): RMSE cm</h3>", table_html(Rt.reset_index()), "<h3>Which features matter</h3>", img("h_imp"), "<h3>All height models</h3>", table_html(full_h)]
else:
    html += ["<h2>2. Plant height</h2><p>not run</p>"]
if Bt is not None:
    html += ["<h2>3. Above-ground biomass (kg/ha, dry)</h2>", img("b_bar"), "<h3>Best model per feature family and dataset (repeated 10-fold)</h3>", table_html(Bt),
             ("<p class='small'>Families not run: " + "; ".join(f"{r.dataset} {r.feature_set} ({r.reason})" for _, r in skipped.iterrows()) + "</p>") if skipped is not None and len(skipped) else "",
             "".join(img(f"b_{ds}") for ds in DSETS), "<h3>Top features (random-forest importance on all features, per dataset)</h3>", img("b_imp"),
             "<h3>Pooled model across trials and stages (grouped 10-fold), R² per dataset</h3>", table_html(pooled.reset_index()),
             "<h3>Cross-trial transfer (train one trial, test the other, same stage)</h3>", table_html(cross)]
else:
    html += ["<h2>3. Above-ground biomass</h2><p>not run</p>"]
html += ["<h2>4. Two-stream deep network (sparse 3D CNN on the classified plot cloud + spectral transformer on VNIR pixels, late fusion)</h2>",
         "<p>Inputs from the same dataset; the spectral stream never sees an unusable band (per-plot band mask from the NaN rule, clipped pixels and excluded ranges are masked and the availability mask is given to the network). 10-fold CV stratified by dataset. The pooled R² is inflated by the between-dataset differences in mean biomass; the per-dataset rows are the comparable numbers.</p>",
         img("b_dl"), img("b_dl2"), table_html(dl_rows) if dl_rows is not None else "<p>not run (torch / spconv unavailable at build time)</p>",
         ("<h3>Same network for height (Muresk)</h3>" + table_html(DLH)) if DLH is not None else "",
         "<h2>5. Files</h2><p class='small'>Everything is under Biomass Experiment\\Biomass_Height_2026-10-09: dataset\\ (per trial / flight: ground_truth.csv, features_lidar.csv, features_vnir.csv, vnir_spectra.csv, vnir_band_usable.csv, features_all.csv, lidar_classified\\, vnir\\), "
         "height\\ and biomass\\ (results\\, models\\), two_stream\\ (dl_data\\, results\\, models\\), figures\\, scripts\\ (build_dataset.py, dl_prepare.py, height_models.py, biomass_models.py, pca_families.py, dl_twostream.py, report_builder.py, grid_pyshp.py, vnir_envi.py) and README.md.</p>"]
page = "\n".join(html)
out = os.path.join(M, "REPORT_height_biomass_2026-10-09.html"); open(out, "w", encoding="utf-8").write(page)
print("report written:", out, round(len(page) / 1e6, 2), "MB")
