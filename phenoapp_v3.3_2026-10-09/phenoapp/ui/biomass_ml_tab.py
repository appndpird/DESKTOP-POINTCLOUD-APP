"""
Biomass ML tab (v3): fused LiDAR v3 + VNIR v3 features -> biomass (or any per-plot trait) with nested
cross-validation, saved predictions and models, and application of a saved model to another flight.

Inputs : the Traits-tab metrics CSV (with the LiDAR v3 columns), the VNIR v3 features CSV (VNIR Spectral tab),
         a ground-truth CSV (Plot_ID + target column, e.g. biomass_kg_ha).
Outputs next to the metrics CSV: <metrics>_<target>_ml_predictions.csv, _ml_results.json,
         _ml_model_<family>_<learner>.joblib, _ml_top_features.csv
"""

from __future__ import annotations
import os
import json
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QFileDialog,
                             QMessageBox, QLineEdit, QTextEdit, QCheckBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox)

from phenoapp.core.project import state
from phenoapp.core.biomass_ml import (FAMILIES, LEARNERS, fit_trait_models, results_table, save_model, load_model, apply_model,
                                      learner_available, feature_family, add_weighted)


def _load_inputs(metrics_csv, vnir_csv):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    if vnir_csv and os.path.exists(vnir_csv):
        v = pd.read_csv(vnir_csv); v = v[[c for c in v.columns if c == "Plot_ID" or c not in df.columns]]
        df = df.merge(v, on="Plot_ID", how="left")
    return df


class _Worker(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)

    def __init__(self, metrics_csv, vnir_csv, gt_csv, target, families, learners, cv, n_select):
        super().__init__(); self._a = (metrics_csv, vnir_csv, gt_csv, target, families, learners, cv, n_select)

    def run(self):
        try:
            import pandas as pd
            metrics_csv, vnir_csv, gt_csv, target, families, learners, cv, n_select = self._a
            df = _load_inputs(metrics_csv, vnir_csv); gt = pd.read_csv(gt_csv)
            if target not in gt.columns:
                raise RuntimeError(f"ground-truth CSV has no column '{target}' (columns: {', '.join(gt.columns)})")
            results, pred, fitted = fit_trait_models(df, gt, target, families, learners, cv=cv, n_select=n_select,
                                                     progress_cb=lambda p, m: self.progress.emit(min(98, p), m))
            base = os.path.splitext(metrics_csv)[0] + f"_{target}"
            pred.to_csv(base + "_ml_predictions.csv", index=False)
            saved = {}
            for (fam, ln), (model, feats) in fitted.items():
                p = base + f"_ml_model_{fam}_{ln}.joblib"; save_model(p, model, feats, meta={"target": target, "family": fam, "learner": ln, "cv": cv}); saved[f"{fam}/{ln}"] = os.path.basename(p)
            with open(base + "_ml_results.json", "w") as f:
                json.dump({"target": target, "cv": cv, "n_select": n_select, "results": results, "models": saved}, f, indent=1, default=float)
            # top features: RF importance on the fused_all family
            try:
                from sklearn.ensemble import RandomForestRegressor
                d = add_weighted(df).merge(gt[["Plot_ID", target]].dropna(), on="Plot_ID"); d = d[d.get("qc_ok", 1) == 1]
                cols = feature_family(d, "Fused_all", exclude=(target,)); X = d[cols].astype(float).fillna(d[cols].astype(float).median()).to_numpy(); y = d[target].to_numpy(float)
                rf = RandomForestRegressor(n_estimators=800, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y)
                top = pd.DataFrame({"feature": cols, "rf_importance": rf.feature_importances_, "r_with_target": [np.corrcoef(X[:, i], y)[0, 1] for i in range(len(cols))]}).sort_values("rf_importance", ascending=False)
                top.to_csv(base + "_ml_top_features.csv", index=False); toptxt = "\n\nTop 12 features (RF importance, all v3 features):\n" + top.head(12).round(3).to_string(index=False)
            except Exception as e:
                toptxt = f"\n(top-feature ranking failed: {e})"
            ok = [r for r in results if r.get("status") == "ok" and r.get("cv")]
            best = min(ok, key=lambda r: r["cv"]["RMSE"]) if ok else None
            txt = results_table(results) + toptxt
            if best:
                txt += f"\n\nBest by held-out RMSE: {best['family']} / {best['learner']}: R2 {best['cv']['R2']:.3f}, RMSE {best['cv']['RMSE']:.3g}, accuracy {best['cv']['accuracy']:.1f}%\nFeatures: {', '.join(best['features'])}"
            txt += f"\n\nPredictions -> {base}_ml_predictions.csv\nResults -> {base}_ml_results.json\nModels -> {len(saved)} joblib files"
            self.progress.emit(100, "done"); self.done_ok.emit(txt)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class BiomassMLTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Biomass ML (v3): fused LiDAR + VNIR features, nested cross-validation</h2>"))
        v.addWidget(QLabel("Needs the Traits-tab metrics CSV computed with 'LiDAR v3 features' and the VNIR v3 features CSV from the VNIR Spectral tab. "
                           "Ground truth: CSV with Plot_ID and the target column (default biomass_kg_ha; any numeric per-plot trait works)."))
        g = QGroupBox("Inputs"); f = QFormLayout(g)
        self.ed_metrics = QLineEdit(); self.ed_vnir = QLineEdit(); self.ed_gt = QLineEdit(); self.ed_target = QLineEdit("biomass_kg_ha")
        for lab, ed, filt in (("Metrics CSV (LiDAR v3):", self.ed_metrics, "CSV (*.csv)"), ("VNIR v3 features CSV:", self.ed_vnir, "CSV (*.csv)"), ("Ground-truth CSV:", self.ed_gt, "CSV (*.csv)")):
            row = QHBoxLayout(); b = QPushButton("Browse..."); b.clicked.connect(lambda _, e=ed, ft=filt: self._pick(e, ft)); row.addWidget(ed); row.addWidget(b); f.addRow(lab, row)
        f.addRow("Target column:", self.ed_target)
        v.addWidget(g)
        g2 = QGroupBox("Feature families, learners, validation"); f2 = QFormLayout(g2)
        fr = QHBoxLayout(); self.cb_fam = {}
        for fam in FAMILIES:
            cb = QCheckBox(fam); cb.setChecked(fam in ("LiDAR_core", "Fused_core")); self.cb_fam[fam] = cb; fr.addWidget(cb)
        f2.addRow("Families:", fr)
        lr = QHBoxLayout(); self.cb_lrn = {}
        for ln in LEARNERS:
            cb = QCheckBox(ln); av = learner_available(ln); cb.setChecked(av and ln in ("Ridge", "PLS", "RandomForest", "XGBoost")); cb.setEnabled(av)
            if not av: cb.setText(ln + " (not installed)")
            self.cb_lrn[ln] = cb; lr.addWidget(cb)
        f2.addRow("Learners:", lr)
        self.cb_cv = QComboBox(); self.cb_cv.addItems(["Repeated 10-fold (5x), nested feature selection - recommended", "Leave-one-out", "Fit only"])
        self.sp_sel = QSpinBox(); self.sp_sel.setRange(3, 60); self.sp_sel.setValue(12); self.sp_sel.setToolTip("Features kept per training fold (random-forest importance inside the fold); families with fewer features use all.")
        row = QHBoxLayout(); row.addWidget(self.cb_cv, stretch=3); row.addWidget(QLabel("features kept:")); row.addWidget(self.sp_sel); f2.addRow("Validation:", row)
        self.btn = QPushButton("Fit, validate and save"); self.btn.clicked.connect(self._run); f2.addRow("", self.btn)
        v.addWidget(g2)
        g3 = QGroupBox("Apply a saved model to this flight"); f3 = QFormLayout(g3)
        row = QHBoxLayout(); self.ed_model = QLineEdit(); b = QPushButton("Browse..."); b.clicked.connect(lambda: self._pick(self.ed_model, "joblib (*.joblib)")); row.addWidget(self.ed_model); row.addWidget(b); f3.addRow("Model:", row)
        self.sp_off = QDoubleSpinBox(); self.sp_off.setRange(-1e6, 1e6); self.sp_off.setDecimals(1); self.btn_apply = QPushButton("Predict every plot"); self.btn_apply.clicked.connect(self._apply)
        row = QHBoxLayout(); row.addWidget(self.sp_off); row.addWidget(self.btn_apply); row.addStretch(); f3.addRow("Offset (target units):", row)
        v.addWidget(g3)
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); v.addWidget(self.out, stretch=1)

    def showEvent(self, ev):
        super().showEvent(ev); s = state()
        if not self.ed_metrics.text().strip() and s.out_csv: self.ed_metrics.setText(s.out_csv)
        if not self.ed_vnir.text().strip() and s.out_csv:
            cand = os.path.splitext(s.out_csv)[0] + "_vnir_v3.csv"
            if os.path.exists(cand): self.ed_vnir.setText(cand)

    def _pick(self, ed, filt):
        p, _ = QFileDialog.getOpenFileName(self, "Pick file", "", filt)
        if p: ed.setText(p)

    def _run(self):
        m, g = self.ed_metrics.text().strip(), self.ed_gt.text().strip()
        if not (m and os.path.exists(m)): QMessageBox.warning(self, "No metrics", "Pick the metrics CSV."); return
        if not (g and os.path.exists(g)): QMessageBox.warning(self, "No ground truth", "Pick the ground-truth CSV."); return
        fams = [k for k, cb in self.cb_fam.items() if cb.isChecked()]; lrn = [k for k, cb in self.cb_lrn.items() if cb.isChecked() and cb.isEnabled()]
        if not fams or not lrn: QMessageBox.warning(self, "Nothing selected", "Tick at least one family and one learner."); return
        cv = ("rkf10", "loo", "none")[self.cb_cv.currentIndex()]
        self.btn.setEnabled(False); self.progress.setValue(0)
        self._w = _Worker(m, self.ed_vnir.text().strip(), g, self.ed_target.text().strip() or "biomass_kg_ha", fams, lrn, cv, self.sp_sel.value())
        self._w.progress.connect(lambda p, s: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {s}")))
        self._w.done_ok.connect(lambda t: (self.btn.setEnabled(True), self.out.setPlainText(t)))
        self._w.error.connect(lambda e: (self.btn.setEnabled(True), self.out.append("\nERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._w.start()

    def _apply(self):
        m, p = self.ed_metrics.text().strip(), self.ed_model.text().strip()
        if not (m and os.path.exists(m) and p and os.path.exists(p)): QMessageBox.warning(self, "Missing input", "Pick the metrics CSV and a saved model."); return
        try:
            df = _load_inputs(m, self.ed_vnir.text().strip()); saved = load_model(p); out = apply_model(df, saved, float(self.sp_off.value()))
            tgt = saved.get("meta", {}).get("target", "target"); outp = os.path.splitext(m)[0] + f"_{tgt}_applied_{os.path.splitext(os.path.basename(p))[0].split('_ml_model_')[-1]}.csv"
            out.to_csv(outp, index=False); v = out.predicted.dropna()
            self.out.setPlainText(f"Applied {os.path.basename(p)} (features: {', '.join(saved['features'])})\n{int(out.model_ok.sum())}/{len(out)} plots predicted; mean {v.mean():.4g}, SD {v.std():.4g}\n\nPredictions -> {outp}")
        except Exception as e:
            import traceback; QMessageBox.critical(self, "Apply failed", f"{e}\n\n{traceback.format_exc()}")
