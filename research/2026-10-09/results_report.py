"""
Scenario-wise results report (results\RESULTS_REPORT.html): for every scenario the data used, the methods (feature models
with PCA spelled out, deep learning), the overall results on all test plots, the fold-by-fold test results, the models the
nested selection chose, and predicted-vs-measured plots - for features and deep learning, per category
(LiDAR only, VNIR only, fused). Reads only the result tables written by the pipeline; re-run after any rerun.
"""
import os, io, base64, json, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"; RES = os.path.join(R, "results")
CATN = {"lidar": "LiDAR only", "vnir": "VNIR only", "fused": "Fused (LiDAR + VNIR)"}
APN = {"features (nested top 3)": "Features", "deep learning (two-stream net)": "Deep learning"}
FL = {"muresk_anthesis": "Muresk 2025-09-30 (anthesis)", "muresk_maturity": "Muresk 2025-11-21 (maturity)", "agt_anthesis": "AGT 2025-09-22 (anthesis)", "agt_maturity": "AGT 2025-11-13 (maturity, flights 1 + 2)"}
C_FEAT, C_DL = "#1f6f4a", "#b5651d"

def rd(p): return pd.read_csv(p) if os.path.exists(p) else None
def img(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return f"<img src='data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}'>"
SOIL, CAN, GEST, MODE = "#854F0B", "#639922", "#D85A30", "#7F77DD"

def fig_ground_topview():
    """Schematic top view: 3 x 3 plots, the alley ring of the centre plot and its 25 cm cells (not to scale)."""
    w, l, cxs, cys = 1.5, 4.0, [0.0, 2.3, 4.6], [0.0, 5.0, 10.0]
    tx, ty = 2.3, 5.0
    def dist(x, y, cx, cy): return np.hypot(np.maximum(np.abs(x - cx) - w / 2, 0), np.maximum(np.abs(y - cy) - l / 2, 0))
    g = np.arange(-1.5, 6.6, 0.25) + 0.125; gy = np.arange(-2.5, 12.6, 0.25) + 0.125; X, Y = np.meshgrid(g, gy)
    near_any = np.zeros_like(X, bool)
    for cx in cxs:
        for cy in cys: near_any |= dist(X, Y, cx, cy) <= 0.25
    ring = (dist(X, Y, tx, ty) <= 1.0) & ~near_any
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    for cx in cxs:
        for cy in cys:
            hit = (cx, cy) == (tx, ty)
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - l / 2), w, l, fc=CAN, alpha=0.55 if hit else 0.22, ec="#3B6D11", lw=1.4 if hit else 0.5))
    for x, y in zip(X[ring], Y[ring]):
        ax.add_patch(plt.Rectangle((x - 0.11, y - 0.11), 0.22, 0.22, fc=GEST, ec="none", alpha=0.85))
    ax.text(tx, ty, "plot", ha="center", va="center", fontsize=9, color="#173404")
    ax.set_xlim(-1.3, 5.9); ax.set_ylim(-2.6, 12.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("alley ring of the centre plot: 0.25-1.0 m from its edge,\nleaving out anything within 0.25 m of any plot", fontsize=8, color=GEST)
    ax.set_title("Top view: where the ground of one plot comes from\n(each orange square = one 25 cm cell; schematic)", fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    return img(fig)

def fig_ground_side():
    """Side view of a column of plots on sloping ground: one flat level per column (Jimenez-Berni et al. 2018) vs a plane per plot (ours)."""
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    gnd = lambda x: 0.04 * x
    plots = [(0.5, 2.0), (2.6, 4.1), (4.7, 6.2)]
    for k, ax in enumerate(axs):
        xs = np.linspace(0, 6.7, 50); ax.plot(xs, gnd(xs), color=SOIL, lw=2, label="soil surface")
        for a, b in plots:
            ax.fill([a, a, b, b], [gnd(a), gnd(a) + 0.8, gnd(b) + 0.82, gnd(b)], color=CAN, alpha=0.35, ec="#3B6D11", lw=0.5)
        if k == 0:
            lev = gnd(xs).mean(); ax.plot(xs, np.full_like(xs, lev), color=GEST, lw=2, ls="--", label="ground estimate: one level for the column")
            ax.set_title("A. Jimenez-Berni et al. 2018: peak of the column's height histogram = one flat ground level", fontsize=9, loc="left")
            ax.annotate("ground set too high here\n(heights too small)", (1.2, lev), (0.05, 0.55), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
            ax.annotate("ground set too low here\n(heights too large)", (5.4, lev), (4.6, 0.55), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
        else:
            for a, b in plots:
                xx = np.array([a - 0.35, b + 0.35]); ax.plot(xx, gnd(xx) - 0.01, color=GEST, lw=2, ls="--")
                for xc in (a - 0.3, a - 0.15, b + 0.15, b + 0.3):
                    ax.plot(xc, gnd(xc) - 0.01, "s", color=GEST, ms=4)
            ax.plot([], [], color=GEST, lw=2, ls="--", marker="s", ms=4, label="ground estimate: a tilted plane per plot from its alley cells")
            ax.set_title("B. Ours: each plot gets its own tilted plane, fitted to the alley cells around it", fontsize=9, loc="left")
        ax.set_ylim(-0.15, 1.5); ax.set_ylabel("elevation (m)"); ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)
    axs[1].set_xlabel("distance along the column of plots (m); slope exaggerated, schematic")
    fig.tight_layout(); return img(fig)

def fig_ground_hist():
    """Soil returns in one alley cell: histogram mode (Jimenez-Berni) vs 5th percentile (ours)."""
    rng = np.random.default_rng(0); sd = 2.5; z = rng.normal(0, sd, 4000)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.hist(z, bins=40, color="#B4B2A9", ec="white", lw=0.4)
    p5 = np.percentile(z, 5)
    ax.axvline(0, color=MODE, lw=2, label="mode (histogram peak): Jimenez-Berni et al.")
    ax.axvline(p5, color=GEST, lw=2, ls="--", label="5th percentile of the cell: ours")
    yl = ax.get_ylim()[1]; ax.annotate("", (p5, yl * 0.5), (0, yl * 0.5), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(p5 - 0.4, yl * 0.62, f"offset ≈ 1.6 × scatter\n({abs(p5):.1f} cm at {sd} cm scatter)", ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("elevation of the soil returns in one 25 cm alley cell, relative to the mean soil surface (cm)"); ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, loc="upper right"); fig.tight_layout(); return img(fig)

def tbl(df, fmt=None):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].map(lambda v: "" if pd.isna(v) else (fmt or "{:.2f}").format(v) if abs(v) < 100 else f"{v:,.0f}")
    return "<div class='tw'>" + df.to_html(index=False, border=0, escape=False) + "</div>"
def met(p, y):
    e = p - y; return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), float(np.sqrt((e ** 2).mean()))

# ------------------------------------------------------------------ inputs
SUM = rd(os.path.join(RES, "report_summary.csv")); HF = rd(os.path.join(RES, "report_height_folds.csv")); BF = rd(os.path.join(RES, "report_biomass_pooled_folds.csv"))
COMP = rd(os.path.join(RES, "report_testset_composition.csv")); VS = rd(os.path.join(RES, "valid_plots_summary.csv")); VF = rd(os.path.join(RES, "valid_plots_folds.csv")); VC = rd(os.path.join(RES, "valid_plots_composition.csv"))
PCA_B = rd(os.path.join(R, "biomass", "results", "pca_components.csv")); PCA_H = rd(os.path.join(R, "height", "results", "pca_components.csv"))
EACH = rd(os.path.join(RES, "report_each_model_summary.csv"))

def scenario_rows(kind, key):
    """overall rows and fold rows of a scenario as uniform tables: category, approach, fold, n, R2, RMSE, rRMSE"""
    if kind == "main":
        S = SUM[SUM.target == key]; F = pd.concat([HF, BF], ignore_index=True); F = F[F.target == key]
        S = S.rename(columns={"R2_overall": "R2", "RMSE_overall": "RMSE", "rRMSE_overall": "rRMSE"})
    else:
        if VS is None: return None, None
        S = VS[VS.rule == key].rename(columns={"R2_overall": "R2", "RMSE_overall": "RMSE", "rRMSE_overall": "rRMSE"}); F = VF[VF.rule == key]
    return S, F

def selection(datasets, tag, cats):
    """how often each family / learner was among the top three across folds and flights, with PCA flagged"""
    rows = []
    for ds in datasets:
        for t in ("biomass", "height"):
            p = os.path.join(R, t, "results", f"nested_top3_{ds}{tag}.csv")
            if not os.path.exists(p) or (t == "height" and tag): continue
            q = pd.read_csv(p); q = q[q.target == ("height" if "height" in SCEN_T else "biomass")] if "target" in q.columns else q
            for r in q.itertuples():
                if r.category not in cats: continue
                for m in str(r.top3).split(" | "):
                    fam_ln = m.split(" (")[0]; fam, ln = fam_ln.split("/")
                    rows.append(dict(category=r.category, family=fam, learner=ln, flight=ds))
    if not rows: return None
    D = pd.DataFrame(rows); n_fold = D.groupby(["category", "flight"]).size() / 3
    out = D.groupby(["category", "family", "learner"]).size().rename("times chosen").reset_index()
    tot = D.groupby("category").size() / 3
    out["share of fold ensembles"] = out.apply(lambda r: f"{100 * r['times chosen'] / tot[r.category]:.0f} %", axis=1)
    out["uses PCA"] = out.family.str.contains("PCA").map({True: "yes", False: "no"})
    out["category"] = out.category.map(CATN)
    return out.sort_values(["category", "times chosen"], ascending=[True, False])

def pca_table(target, datasets):
    P = PCA_H if target == "height" else PCA_B
    if P is None: return None
    key = "flight" if target == "height" else "dataset"
    P = P[P[key].astype(str).isin(["2025-11-21"] if target == "height" else datasets)].copy()
    def blk(r):
        out = []
        for b, nm in (("lidar", "LiDAR"), ("vnir", "VNIR indices"), ("joint", "LiDAR + VNIR together"), ("bands", "spectral bands")):
            n = r.get(f"{b}_n_components")
            if pd.notna(n): out.append(f"{nm}: {int(n)} PCs ({100 * r.get(f'{b}_explained_variance', np.nan):.0f} % var.)")
        return "; ".join(out)
    P["components kept (fitted on all plots, for reference)"] = P.apply(blk, axis=1)
    P = P.rename(columns={key: "flight", "feature_set": "PCA family", "n_inputs": "input features"})
    P["flight"] = P.flight.map(lambda v: FL.get(v, "Muresk 2025-11-21 (maturity)" if str(v) == "2025-11-21" else v))
    return P[["flight", "PCA family", "input features", "components kept (fitted on all plots, for reference)"]]

def preds(target, datasets, tag, dl_file, plots=None):
    """pooled predictions of the scenario per category: features and DL"""
    F = []
    for ds in datasets:
        p = os.path.join(R, target, "results", f"nested_top3_predictions_{ds}{tag}.csv")
        if os.path.exists(p):
            q = pd.read_csv(p)
            for c in [c for c in q.columns if c.startswith("pred_")]: F.append(pd.DataFrame(dict(dataset=ds, Plot_ID=q.Plot_ID, category=c[5:], approach="Features", measured=q.measured, predicted=q[c])))
    D = pd.concat(F, ignore_index=True) if F else pd.DataFrame()
    p = os.path.join(R, "two_stream", "results", dl_file)
    if os.path.exists(p):
        q = pd.read_csv(p); q = q[q.dataset.isin(datasets)]
        q = q.assign(category=q.variant.map({"lidar_only": "lidar", "vnir_only": "vnir", "two_stream": "fused"}), approach="Deep learning")[["dataset", "Plot_ID", "category", "approach", "measured", "predicted"]]
        D = pd.concat([D, q], ignore_index=True)
    if plots is not None and len(D): D = D.merge(plots, on=["dataset", "Plot_ID"])
    return D

def fig_overall(S, unit):
    cats = [c for c in ("lidar", "vnir", "fused") if c in set(S.category)]; x = np.arange(len(cats)); w = 0.36
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (ap, col) in enumerate((("Features", C_FEAT), ("Deep learning", C_DL))):
        s = S[S.ap == ap].set_index("category").reindex(cats)
        axs[0].bar(x + (k - 0.5) * w, s.RMSE, w, color=col, label=ap); axs[1].bar(x + (k - 0.5) * w, s.R2, w, color=col, label=ap)
    for ax, t in zip(axs, (f"RMSE on all test plots ({unit})", "R² on all test plots")):
        ax.set_xticks(x); ax.set_xticklabels([CATN[c] for c in cats], fontsize=8); ax.set_title(t, fontsize=10); ax.grid(axis="y", alpha=.3)
    axs[1].axhline(0, color="k", lw=.6); axs[0].legend(fontsize=8)
    return img(fig)

def fig_folds(F, unit):
    cats = [c for c in ("lidar", "vnir", "fused") if c in set(F.category)]
    fig, axs = plt.subplots(1, len(cats), figsize=(4.2 * len(cats), 3.2), squeeze=False)
    for ax, c in zip(axs[0], cats):
        for ap, col in (("Features", C_FEAT), ("Deep learning", C_DL)):
            s = F[(F.category == c) & (F.ap == ap)].sort_values("fold")
            if len(s): ax.plot(s.fold, s.RMSE, "o-", color=col, label=ap, lw=1.4, ms=4)
        ax.set_title(CATN[c], fontsize=10); ax.set_xlabel("fold"); ax.set_xticks(range(1, 11)); ax.grid(alpha=.3)
    axs[0][0].set_ylabel(f"test RMSE ({unit})"); axs[0][0].legend(fontsize=8)
    return img(fig)

def fig_scatter(D, unit):
    cats = [c for c in ("lidar", "vnir", "fused") if c in set(D.category)]
    fig, axs = plt.subplots(1, len(cats), figsize=(4.2 * len(cats), 3.8), squeeze=False)
    lo, hi = np.nanpercentile(D.measured, 1), np.nanpercentile(D.measured, 99); pad = (hi - lo) * .15; lo, hi = lo - pad, hi + pad
    for ax, c in zip(axs[0], cats):
        for ap, col in (("Features", C_FEAT), ("Deep learning", C_DL)):
            s = D[(D.category == c) & (D.approach == ap)]
            if len(s):
                r2, rm = met(s.predicted.to_numpy(float), s.measured.to_numpy(float))
                ax.scatter(s.predicted, s.measured, s=9, alpha=.55, color=col, label=f"{ap}: R² {r2:.2f}, RMSE {rm:,.1f}")
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title(CATN[c], fontsize=10)
        ax.set_xlabel(f"predicted ({unit})"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)
    axs[0][0].set_ylabel(f"measured ({unit})")
    return img(fig)

# ------------------------------------------------------------------ scenarios
VPLOTS = None
for ds in ("muresk_anthesis", "muresk_maturity", "agt_anthesis"):
    p = os.path.join(R, "biomass", "results", f"nested_top3_predictions_{ds}.csv")
    if os.path.exists(p): VPLOTS = pd.concat([VPLOTS, pd.read_csv(p)[["Plot_ID"]].assign(dataset=ds)]) if VPLOTS is not None else pd.read_csv(p)[["Plot_ID"]].assign(dataset=ds)
SCEN = [
    dict(id="S1", title="Plant height - Muresk maturity flight", target="height", unit="cm", kind="main", key="height, Muresk maturity", datasets=["muresk_maturity"], tag="",
         dl="dl_height_commonbands_grouped_maturity_per_plot_predictions.csv", plots=None, comp=("main", "height, Muresk maturity"),
         data="128 wheat plots of the Muresk NUE trial flown on 2025-11-21 (maturity). Ground truth: one ruler height per plot, measured at maturity - the only flight with a height measured on the same occasion, so the only one scored. "
              "Inputs per plot: LiDAR v3.2 features of the classified plot cloud, VNIR v3.0.1 features and the mean vegetation spectrum (bands valid on at least 95 % of the plots)."),
    dict(id="S2", title="Above-ground biomass - all flights, LiDAR only", target="biomass", unit="kg/ha", kind="main", key="biomass, pooled, all plots (4 datasets)", datasets=["muresk_anthesis", "muresk_maturity", "agt_anthesis", "agt_maturity"], tag="",
         dl="dl_biomass_commonbands_grouped_per_plot_predictions.csv", plots=None, comp=("main", "biomass, pooled, all plots (4 datasets)"),
         data="758 plot-flights with a dry-biomass cut and a usable LiDAR cloud: Muresk anthesis 128, Muresk maturity 126, AGT anthesis 252, AGT maturity 252 (flights 1 + 2 merged). "
              "Only LiDAR can be compared on all four flights, because the AGT maturity cubes are radiance, not reflectance."),
    dict(id="S3", title="Above-ground biomass - flights with usable VNIR (LiDAR only, VNIR only, fused)", target="biomass", unit="kg/ha", kind="main", key="biomass, pooled, plots with usable VNIR (3 datasets)",
         datasets=["muresk_anthesis", "muresk_maturity", "agt_anthesis"], tag="", dl="dl_biomass_commonbands_grouped_per_plot_predictions.csv", plots="vnir", comp=("main", "biomass, pooled, plots with usable VNIR (3 datasets)"),
         data="506 plot-flights whose VNIR passed the quality check: Muresk anthesis 128, Muresk maturity 126, AGT anthesis 252. All three categories are scored on exactly these plots, so they can be compared directly. "
              "On AGT anthesis part of the red band is clipped; red-based indices are NaN there and leave the models (NaN rule)."),
    dict(id="S4", title="Above-ground biomass - valid-band plots only, rule A", target="biomass", unit="kg/ha", kind="rule", key="A", datasets=["muresk_anthesis", "muresk_maturity", "agt_anthesis"], tag="__ruleA",
         dl="dl_biomass_commonbands__ruleA_per_plot_predictions.csv", plots=None, comp=("rule", "A"),
         data="497 plot-flights in which every one of the 111 common bands is unclipped on at least 50 % of the plot's pixels (reflectance cube, VNIR QC passed): Muresk anthesis 128, Muresk maturity 126, AGT anthesis 243. Models trained and tested on these plots only."),
    dict(id="S5", title="Above-ground biomass - valid-band plots only, rule B (strict)", target="biomass", unit="kg/ha", kind="rule", key="B", datasets=["muresk_anthesis", "muresk_maturity"], tag="__ruleB",
         dl="dl_biomass_commonbands__ruleB_per_plot_predictions.csv", plots=None, comp=("rule", "B"),
         data="209 plot-flights in which every common band is unclipped on at least 95 % of the plot's pixels and the plot lies entirely inside the cube: Muresk anthesis 83, Muresk maturity 126. No AGT plot qualifies (partly clipped red bands at anthesis, radiance at maturity)."),
]

css = """<style>
:root{--ink:#1b1b1b;--muted:#5f6b66;--line:#e2e6e3;--soft:#f5f7f5;--accent:#1f6f4a;--warm:#b5651d}
body{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:0 auto;padding:24px 20px;color:var(--ink);line-height:1.5;background:#fff}
h1{font-size:1.7rem;margin-bottom:4px}h2{margin-top:44px;padding-bottom:6px;border-bottom:2px solid var(--line)}h3{margin-top:26px;font-size:1.08rem}
table{border-collapse:collapse;font-size:.82rem;font-variant-numeric:tabular-nums}th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right}
th{background:var(--soft);text-align:center}td:first-child,th:first-child{text-align:left}
.tw{overflow-x:auto;margin:8px 0 16px}img{max-width:100%;height:auto;margin:6px 0}
.box{background:var(--soft);border-left:4px solid var(--accent);padding:10px 16px;margin:14px 0}.warn{border-left-color:var(--warm)}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}.kpi div{background:var(--soft);padding:10px 14px;border-radius:8px;min-width:200px}
.kpi b{display:block;font-size:1.25rem}.small{color:var(--muted);font-size:.86rem}nav a{margin-right:14px}
</style>"""
H = [f"<title>Results - height and biomass models</title>{css}<h1>Results: plant height and above-ground biomass from UAV LiDAR and VNIR</h1>",
     f"<p class='small'>Built {time.strftime('%Y-%m-%d %H:%M')} from Biomass_Height_2026-10-09\\results and the model folders. Features = feature-based machine learning; Deep learning = two-stream network. "
     "All numbers are on held-out test plots.</p>",
     "<nav>" + " ".join(f"<a href='#{s['id']}'>{s['id']}</a>" for s in SCEN) + " <a href='#preprocessing'>Preprocessing</a> <a href='#methods'>Methods</a> <a href='#summary'>Summary</a></nav>"]

# ---- summary across scenarios
srows = []
for s in SCEN:
    S, F = scenario_rows(s["kind"], s["key"])
    if S is None or S.empty: continue
    for r in S.itertuples():
        srows.append({"scenario": f"{s['id']} {s['title'].split(' - ')[-1]}", "category": CATN[r.category], "approach": APN.get(r.approach, r.approach), "plots": int(r.plots), "R²": r.R2, f"RMSE": r.RMSE, "rRMSE %": r.rRMSE})
H += ["<h2 id='summary'>Summary of all scenarios</h2>",
      "<div class='box'><b>How to read the numbers.</b> R², RMSE and rRMSE (RMSE as % of the mean) are computed on all test plots of the 10 cross-validation folds together; every plot is a test plot exactly once and was never seen by the model that predicted it. "
      "Height is scored only on the flight with a ruler measurement. Biomass is one overall result pooled over the flights of the scenario. R² depends on how different the pooled flights are: "
      "scenario S2 includes AGT maturity, whose mean biomass is much higher, so its R² is larger at about the same rRMSE - compare R² within a scenario and rRMSE across scenarios.</div>",
      tbl(pd.DataFrame(srows))]

# ---- preprocessing and quality control (code: PhenoApp core/qc.py, lidar_features.py, vnir_features.py, spectral_indices.py)
H += ["<h2 id='preprocessing'>Data preprocessing and quality control</h2>",
      "<p>Both modalities are cleaned per plot before any feature is computed. Flagged points and unusable spectral values are kept out of every feature; "
      "missing values are never filled in.</p>",
      "<h3>LiDAR</h3>",
      "<p><b>Flight-level checks</b> (reported, not altered): coordinate reference system present; points more than 10 m outside the 1st-99th height percentile band; "
      "share of single returns (the Ouster cloud is 100 % single-return, so return-based metrics are not used); ground classes, scan angles, flight-line IDs and GPS time encoding; "
      "with surveyed GCPs, the vertical consistency of the cloud between GCP pairs.</p>",
      "<p><b>Per plot</b> (the plot polygon plus a 0.25-1.0 m alley ring, neighbouring plots excluded):</p><ol>"
      "<li><b>Noise</b> - points more than 30 cm above the plot's 99.9th height percentile are flagged as noise (class 7). They are marked, not deleted, and every metric ignores them.</li>"
      "<li><b>Ground</b> - a robust plane through the alley ring (5th percentile of height in 25 cm cells), refined with in-plot points within ±8 cm; points with |h| ≤ 8 cm are ground (class 2).</li>"
      "<li><b>Height normalisation</b> - h = z − plane(x, y).</li>"
      "<li><b>Canopy</b> - h > 10 cm is canopy (class 5); 8-10 cm is low vegetation (class 3).</li>"
      "<li><b>Features</b> on canopy points only: height percentiles, canopy-top P90/P95/P99 (5 cm cell maxima, 15 cm edge trim), density layers, cover, voxel volume, "
      "gap fraction and LAI proxy −ln(Pgap)/0.5 (also on the 20 cm-trimmed plot interior, because the polygons include alley soil), rumple, roughness, intensity normalised by the alley ground.</li></ol>",
      "<div class='box warn'><b>Why not statistical outlier removal?</b> On the validation flight (about 30,000 points/m²) every statistical-outlier variant removed sparse canopy tips (heads and awns; 0.3-1.5 % of points, "
      "half of them in the top 5 % of the canopy) and lowered the correlation of canopy-top P95 with the ruler height from r 0.70 (unfiltered) to 0.68-0.66. "
      "The gap rule flagged a single point in the whole trial.</div>",
      "<h3>The LiDAR rules explained</h3>",
      "<p><b>1. Gross-outlier check (whole flight, report only).</b> The 1st (p1) and 99th (p99) elevation percentiles of all points are computed, and points below p1 − 10 m or above p99 + 10 m are counted. "
      "Percentiles are used because the minimum and maximum are themselves set by the outliers (one bird return at +80 m makes the maximum useless as a reference). "
      "A 10 m margin catches only gross errors: birds, dust or insects in the air, multipath returns below the ground, sensor glitches. "
      "Grade: PASS below 0.01 % of the points, WARN below 0.1 %, FAIL otherwise (the flight needs cleaning or checking first).</p>",
      "<p><b>2. Noise rule ('gap', per plot).</b> The 99.9th elevation percentile (p99.9) of the plot's points is computed, and any point more than 30 cm above it is flagged as noise (class 7). "
      "At about 30,000 points/m² a plot holds a few hundred thousand points, so p99.9 is effectively the top of the canopy, with only a few hundred points above it. "
      "Real heads and awns form a continuous layer at that level; a point 30 cm above the canopy top with empty air in between cannot be plant material. "
      "A neighbour-distance (statistical outlier) filter instead flags wherever points are sparse, which in a cereal canopy is exactly the real tips of heads and awns. "
      "Flagged points stay in the file (class 7, withheld flag in the written LAS) so they can be inspected; every height, cover and gap metric skips them.</p>",
      "<p><b>3. Ground plane (per plot).</b> The ground under a closed canopy is invisible, so it is estimated from the alleys around the plot:</p><ol>"
      "<li><b>Alley ring</b> - the area between 0.25 m and 1.0 m from the plot edge, minus anything within 0.25 m of any plot (neighbours included): bare alley soil without overhanging canopy.</li>"
      "<li><b>25 cm cells, 5th percentile</b> - each 25 × 25 cm cell of the ring is represented by the 5th percentile of its elevations, i.e. the soil surface. "
      "Not the minimum (a single low multipath point would pull it down) and not the mean (weeds and stubble would lift it).</li>"
      "<li><b>Robust plane</b> - z = c<sub>0</sub> + c<sub>1</sub>x + c<sub>2</sub>y fitted to the cell values by least squares; cells more than 2.5 robust standard deviations (from the median absolute deviation) "
      "off the plane are rejected and the plane refitted, up to 4 times, which removes cells with a weed, a clod or a wheel rut. A plane is enough because a plot is only a few metres long.</li>"
      "<li><b>Refinement</b> - points inside the plot within ±8 cm of the ring plane (soil seen through canopy gaps) are added and the plane refitted; the refined plane is kept only if its RMS misfit is below 6 cm, otherwise the ring-only plane is used.</li>"
      "<li><b>Fallback</b> - with 200 ring points or fewer, the ground is a flat level at the 1st percentile of the plot's own points.</li></ol>"
      "<p class='small'>The sidecar JSON of each classified plot LAS records the ground source (ring, ring + in-plot, fallback), the plane coefficients and the RMS misfit.</p>",
      "<div class='box'><b>In plain words.</b> Plant height is the top of the crop minus the soil under it. Under a dense canopy the laser hardly ever reaches the soil, "
      "so the soil level under a plot has to be borrowed from somewhere the laser does see soil: the alleys right around that plot. "
      "The method collects the alley soil around the plot (orange squares below), fits a flat but tilted sheet through it (the plane), "
      "slides that sheet under the plot, and measures every point's height from the sheet. Each plot gets its own sheet, so a field that slopes is handled plot by plot.</div>",
      fig_ground_topview(),
      "<h3>Comparison with the published method (Jimenez-Berni et al. 2018)</h3>",
      "<p>Jimenez-Berni et al. (2018, <i>Frontiers in Plant Science</i> 9:237; ground buggy LiDAR on wheat) also take the ground from the bare soil between plots: "
      "the peak of the height histogram of a whole column of plots gives one ground level for that column, and plant height is a high percentile (0.955) of the plot's points minus that level. "
      "Our method keeps the idea (ground from the alleys, height = high percentile minus ground) and changes the estimator: one tilted plane per plot instead of one flat level per column (panel B below), "
      "built from the 5th percentile of 25 cm alley cells with outlier rejection instead of a histogram peak.</p>",
      fig_ground_side(),
      "<p>The two estimators can also differ on perfectly flat ground. A histogram peak sits in the middle of the soil returns; a 5th percentile sits about 1.6 times the scatter of the soil returns below the middle. "
      "With the 1.7-3.5 cm scatter measured on these flights this could put the ring plane roughly 2-5 cm below the mean soil surface (heights slightly larger); "
      "the refinement with soil points inside the plot probably cancels part of it. <b>This offset is reasoned, not measured</b>; a constant offset does not change correlations or trained models, "
      "but it does change absolute heights and their bias against the ruler. A direct comparison with the histogram-peak method on these flights is planned.</p>",
      fig_ground_hist(),
      "<p><b>4. Height normalisation.</b> h = z − plane(x, y), computed for every point with the plane value at that point's own position, so on a sloping plot each point is measured against the ground beneath it, not one average level. "
      "Example: ground plane 245.32 m at a spot, head at 246.17 m there → h = 0.85 m. h is written as the <i>HeightAboveGround</i> attribute of the classified LAS.</p>",
      "<p><b>5. Classes from height above ground.</b></p>",
      tbl(pd.DataFrame([
          {"h": "|h| ≤ 8 cm", "class": "2 ground", "meaning": "soil, within the tolerance of the ground fit"},
          {"h": "8 cm < h ≤ 10 cm", "class": "3 low vegetation", "meaning": "buffer zone where soil and plant cannot be told apart"},
          {"h": "h > 10 cm", "class": "5 canopy", "meaning": "the crop; used by every canopy metric"},
          {"h": "h < −8 cm", "class": "1 unclassified", "meaning": "below the ground plane (multipath, ruts)"},
          {"h": "flagged in step 2", "class": "7 noise", "meaning": "ignored by all metrics"}])),
      "<p><b>Why 8 cm:</b> it covers the scatter of real soil around a plane (clods, furrows, sowing rows; fit RMS typically a few cm), the ranging noise of the sensor and small plane-fit errors; "
      "a tighter value would push genuine soil into 'vegetation'. <b>Why 10 cm:</b> the 8 cm tolerance plus a 2 cm buffer, so only points clearly above the soil count as crop; "
      "the 8-10 cm band holds the ambiguous points and is left out of both soil and canopy. Thresholds of 5-15 cm are typical in crop LiDAR studies.</p>",
      "<div class='box warn'><b>Consequences to keep in mind.</b><ul>"
      "<li>Gap fraction (ground points ÷ all points) slightly overestimates soil: leaves or lodged plants within 8 cm of the soil count as ground. The bias is usually small at anthesis and maturity.</li>"
      "<li>Crops shorter than about 10 cm (early growth stages) would be mostly lost from the canopy metrics; the thresholds suit these flights but would need lowering for seedling-stage work.</li>"
      "<li>A single plane cannot follow raised beds or strong micro-relief inside a plot; heights would then be biased by the bed height. This is not an issue on these trials.</li></ul></div>",
      "<h3>VNIR</h3><ol>"
      "<li><b>Is the flight usable?</b> A cube fails the reflectance check when its median NDVI is below 0.05, or when it is a float cube without a scale factor and NIR above 2 (still radiance). "
      "AGT maturity fails this check and has no VNIR features.</li>"
      "<li><b>Pixel cleaning</b> - a value of 0 is a negative reflectance clipped to zero by the processing software and is treated as missing, not as zero. "
      "Three wavelength ranges are never used: 0-415 nm (sensor edge, weak signal), 755-770 nm (O<sub>2</sub>-A absorption) and 928-962 nm (water-vapour absorption).</li>"
      "<li><b>Soil mask per flight</b> - NDVI from 800 and 670 nm, split with Otsu's method; threshold bounded to 0.15-0.50; switched off when a senesced canopy shows no soil/crop contrast.</li>"
      "<li><b>Per-plot rules</b> - a band or index is NaN for a plot when it is valid on fewer than 50 % of the plot's pixels (otherwise the mean would come only from the brighter surviving pixels and be biased high). "
      "When 670 nm is valid on 50 % of the pixels or fewer, every index using a red band (620-690 nm) is NaN (<i>red_ok</i> = 0). "
      "Plot QC (<i>vnir_qc_ok</i>): more than 90 % of the plot inside the cube, vegetation cover above 0.2, reflectance cube.</li></ol>",
      "<h3>Valid and invalid bands</h3><p>Band validity is decided at three levels, each building on the previous one:</p>",
      tbl(pd.DataFrame([
          {"level": "Band in a plot", "rule": "outside the three excluded ranges and valid on ≥ 50 % of the plot's pixels", "result": "per-plot spectra tables (NaN otherwise)"},
          {"level": "Band for the dataset (common bands)", "rule": "usable on ≥ 95 % of the plots of every reflectance flight", "result": "111 bands: 522-645, 691-754, 772-926, 965-1000 nm (networks, spectral PCA families)"},
          {"level": "Band for the Muresk-only runs", "rule": "the same 95 % rule on the two Muresk flights", "result": "138 bands: 469-754, 772-926, 965-1000 nm (142 with the four O2-A bands)"},
          {"level": "Plot (valid-band scenarios)", "rule": "rule A: all 111 bands valid on ≥ 50 % of pixels + plot QC + reflectance; rule B: ≥ 95 % of pixels and ≥ 95 % of the plot inside the cube",
           "result": "rule A 497 plot-flights (S4); rule B 209, Muresk only (S5)"}])),
      "<p class='small'>The common list lacks 415-522 nm (blue) and 645-691 nm (red) compared with the Muresk list because those bands are clipped to zero on many AGT anthesis plots - a property of the data, not a design choice.</p>",
      "<h3>Comparison with the literature</h3>",
      tbl(pd.DataFrame([
          {"step": "Noise → ground → height normalisation → canopy threshold → plot percentile metrics", "status": "standard", "note": "usual sequence in crop LiDAR (e.g. Madec et al. 2017; ten Harkel et al. 2020)"},
          {"step": "Canopy threshold 10 cm", "status": "standard", "note": "5-15 cm is typical"},
          {"step": "Gap fraction and Beer-Lambert LAI proxy (G = 0.5)", "status": "standard", "note": "e.g. Jimenez-Berni et al. 2018 (wheat)"},
          {"step": "Removing sensor-edge, O2-A and water-vapour bands", "status": "standard", "note": "the usual hyperspectral bad-band list"},
          {"step": "NDVI soil mask with Otsu threshold", "status": "standard", "note": "Otsu 1979"},
          {"step": "log reflectance + SNV before spectral PCA", "status": "standard", "note": "common spectroscopy preprocessing (SNV: Barnes et al. 1989)"},
          {"step": "Ground from a plot-local alley plane", "status": "adapted", "note": "same idea as Jimenez-Berni et al. 2018 (ground from bare soil between plots, one level per column); ours fits a tilted plane per plot. Other studies use a ground filter (CSF, PMF, SMRF) or a bare-soil flight"},
          {"step": "Gap noise rule instead of statistical outlier removal", "status": "adapted", "note": "SOR is the usual choice; replaced after our own test (box above)"},
          {"step": "Interior gap metrics", "status": "adapted", "note": "from our diagnosis that the polygons include alley soil"},
          {"step": "50 % per-plot and 95 % per-dataset band thresholds", "status": "own choice", "note": "reasonable conventions; rarely specified in papers, which often average all pixels or drop bands by inspection"},
          {"step": "Clipped zeros treated as missing; red-based indices censored", "status": "own choice", "note": "sound but rarely discussed"},
          {"step": "Savitzky-Golay smoothing; BRDF/illumination correction between flights", "status": "not applied", "note": "common in the literature; red-edge derivatives here use the raw mean spectrum"},
      ])),
      ]

# ---- methods
H += ["<h2 id='methods'>Methods in brief</h2>",
      "<h3>Feature-based models</h3><p>Per plot, LiDAR v3.2 features (canopy-only height percentiles, canopy-top heights, cover, gap fraction on the plot interior, voxel volume, profile area, rumple, roughness, normalised intensity; about 75-85 columns) "
      "and VNIR v3.0.1 features (vegetation-masked spectral indices, reflectance in broad bands, red-edge shape, spectral PCA scores; about 130 columns; unusable bands and the indices built on them are NaN and leave the model), "
      "plus the mean vegetation spectrum (log reflectance, standardised per plot) over the bands valid on at least 95 % of the plots. These are grouped into feature families per category:</p>"
      "<ul><li><b>LiDAR only:</b> LiDAR core (22-32 selected features, random-forest selection of the top 10-12 inside each training fold), LiDAR all (height only), and <b>LiDAR PCA</b>.</li>"
      "<li><b>VNIR only:</b> VNIR core (selected indices), <b>VNIR PCA</b> (all VNIR features) and <b>VNIR bands PCA</b> (the spectrum itself).</li>"
      "<li><b>Fused:</b> fused core, fused all (selection), <b>Fused PCA</b> (one PCA per modality, components joined), <b>Fused PCA joint</b> (one PCA on LiDAR + VNIR together) and <b>Fused bands PCA</b> (LiDAR PCA + spectrum PCA).</li></ul>"
      "<p><b>PCA.</b> In every family marked PCA, each block of features is standardised and reduced to the principal components that keep 95 % of its variance; the scaler and the PCA are fitted inside each training fold only, so a test plot never influences them, "
      "and the learner sees only the components. Typically about 80 LiDAR features become 10-12 components, about 130 VNIR features 9-18, and the spectrum 2-7. The tables 'Models chosen' below show, per scenario, how often a PCA family was among the chosen models.</p>"
      "<p><b>Learners</b> per family: ridge, PLS, Gaussian process, random forest, extra trees, LightGBM. <b>Top three per category, nested:</b> inside each outer training fold a 5-fold inner loop scores every family x learner of the category, "
      "and the three best are fitted on the outer training plots and averaged on the outer test plots. Biomass feature models are fitted per flight; their test predictions are pooled for the overall result.</p>",
      "<h3>Deep learning</h3><p>Two-stream network: a sparse 3D convolutional network on the classified plot point cloud (2 cm voxels; height above ground, normalised intensity, return number) and a transformer on 512 vegetation pixel spectra per plot "
      "(fixed common band list, log reflectance standardised per pixel, band-availability mask), joined with a trial/stage embedding into a regression head. Variants: LiDAR only, VNIR only, two-stream (fused). "
      "100 epochs per fold, AdamW, one-cycle learning rate, training-time augmentation (rotation, mirroring, point dropout, jitter, pixel bootstrap, brightness). One network is trained on all flights of a scenario together.</p>",
      "<h3>Validation</h3><p>10-fold cross-validation, folds grouped by plot (both flights of a plot always fall in the same fold) and stratified by flight. The overall result uses all test plots; the fold tables show each fold's own test set. "
      "Fold-wise R² is shown for completeness: with 13 test plots per fold (height, Muresk) it is unstable, so read the fold RMSE.</p>"]

for s in SCEN:
    SCEN_T = s["target"]
    S, F = scenario_rows(s["kind"], s["key"])
    H += [f"<h2 id='{s['id']}'>{s['id']}. {s['title']}</h2>", f"<h3>Data</h3><p>{s['data']}</p>"]
    if S is None or S.empty:
        H += ["<p class='small'>Results not available yet.</p>"]; continue
    S = S.assign(ap=S.approach.map(APN)); F = F.assign(ap=F.approach.map(APN)); F = F[F.fold.astype(str) != "all"].copy(); F["fold"] = F.fold.astype(int)
    cm = COMP if s["comp"][0] == "main" else VC
    if cm is not None:
        cc = cm[cm[cm.columns[0]].astype(str) == s["comp"][1]].drop(columns=[cm.columns[0]])
        if len(cc): H += ["<p class='small'>Test plots per flight in each fold (every plot is tested exactly once):</p>", tbl(cc)]
    best = S.loc[S.RMSE.idxmin()]
    H += ["<h3>Overall results (all test plots)</h3>",
          f"<div class='kpi'><div>Best: <b>{CATN[best.category]}, {best.ap}</b>R² {best.R2:.2f} · RMSE {best.RMSE:,.2f} {s['unit']} · rRMSE {best.rRMSE:.1f} %</div>"
          f"<div>Plots<b>{int(best.plots)}</b>{len(s['datasets'])} flight(s)</div></div>",
          tbl(S.assign(category=S.category.map(CATN), **{"RMSE per fold (mean ± SD)": S.RMSE_fold_mean.map(lambda v: f"{v:,.2f}") + " ± " + S.RMSE_fold_sd.map(lambda v: f"{v:,.2f}")})
              [["category", "ap", "plots", "R2", "RMSE", "rRMSE", "RMSE per fold (mean ± SD)"]].rename(columns={"ap": "approach", "R2": "R²", "RMSE": f"RMSE ({s['unit']})", "rRMSE": "rRMSE %"})),
          fig_overall(S, s["unit"])]
    fw = F.assign(cell=F.R2.map(lambda v: f"{v:.2f}") + " / " + F.RMSE.map(lambda v: f"{v:,.1f}"))
    piv = fw.pivot_table(index=["category", "ap"], columns="fold", values="cell", aggfunc="first").reset_index()
    piv["category"] = piv.category.map(CATN); piv = piv.rename(columns={"ap": "approach"})
    H += ["<h3>Fold-wise results (each fold's test set: R² / RMSE)</h3>", tbl(piv), fig_folds(F, s["unit"])]
    D = preds(s["target"], s["datasets"], s["tag"], s["dl"], VPLOTS if s["plots"] == "vnir" else None)
    if len(D): H += ["<h3>Predicted against measured (all test plots)</h3>", fig_scatter(D, s["unit"])]
    sel = selection(s["datasets"], s["tag"], set(S.category))
    if sel is not None:
        H += ["<h3>Models chosen by the nested selection (features)</h3>",
              "<p>Each fold and flight picks the three best family x learner combinations of the category on its training plots; the table counts how often each was picked. "
              "'Uses PCA' = the family reduces its inputs with PCA as described in Methods.</p>", tbl(sel)]
    if s["kind"] == "main":
        pt = pca_table(s["target"], s["datasets"])
        if pt is not None and len(pt):
            cats = set(S.category)
            keep = {"lidar": ["LiDAR_PCA"], "vnir": ["VNIR_PCA", "VNIR_bands_PCA"], "fused": ["Fused_PCA", "Fused_PCA_joint", "Fused_bands_PCA", "LiDAR+VNIR_PCA", "LiDAR+VNIR_PCA_joint", "LiDAR+VNIR_bands_PCA"]}
            fams = [f for c in cats for f in keep[c]] + (["LiDAR_PCA"] if s["target"] == "height" else [])
            pt = pt[pt["PCA family"].isin(fams)]
            if len(pt): H += ["<h3>PCA in this scenario: inputs and components</h3>", tbl(pt)]
    H += ["<h3>Deep learning in this scenario</h3><p>" + (
        "One network per variant trained on the 128 plots (10 folds). " if s["target"] == "height" else
        f"One network per variant trained on all {int(best.plots)} plot-flights of the scenario together (10 folds, grouped by plot). ") +
        "The LiDAR-only network uses the point cloud only; the VNIR-only network the pixel spectra only; the two-stream network both. Variants that are not available for a scenario are omitted.</p>"]

H += ["<h2>Files</h2><p class='small'>Tables behind this report (Biomass_Height_2026-10-09\\results): report_summary.csv, report_height_folds.csv, report_biomass_pooled_folds.csv, report_testset_composition.csv, "
      "valid_plots_summary.csv, valid_plots_folds.csv, valid_plots_composition.csv, report_each_model_summary.csv / _folds.csv (every individual feature model on the same folds). "
      "Saved models: height\\models and biomass\\models (*_top3_ensemble*.joblib = final ensembles, *_fold_ensembles*.joblib = one ensemble per fold), two_stream\\models (network fold models). "
      "Full development detail: REPORT_height_biomass_2026-10-09.html.</p>"]
out = os.path.join(RES, "RESULTS_REPORT.html"); open(out, "w", encoding="utf-8").write("\n".join(H)); print("written", out, round(os.path.getsize(out) / 1e6, 2), "MB")
