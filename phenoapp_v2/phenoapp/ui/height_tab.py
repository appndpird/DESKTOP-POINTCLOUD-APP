"""
Height Models tab.

Trains and validates plant-height models on the per-plot LiDAR features of
the Traits tab (the canopy-top cth_* traits must be computed) against a
measured-height CSV, saves every prediction and every fitted model to disk,
and applies a saved model to another flight with an optional per-flight
offset.

Outputs next to the metrics CSV:
  <metrics>_height_predictions.csv   Plot_ID, measured, pred_<model> (full fit),
                                     predcv_<model> (held-out)
  <metrics>_height_models.json       validation metrics, calibration line, features
  <metrics>_height_model_<key>.joblib fitted model (joblib dict: model, features, units)
  <metrics>_height_applied_<key>.csv predictions of a saved model on this flight
"""

from __future__ import annotations
import os
import json
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QProgressBar, QFileDialog, QMessageBox, QLineEdit, QTextEdit, QCheckBox,
    QFormLayout, QComboBox, QDoubleSpinBox
)

from phenoapp.core.project import state
from phenoapp.core import load_grid
from phenoapp.core.height_models import (HEIGHT_MODELS, fit_height_models, results_table, save_height_model,
                                         load_height_model, apply_height_model, xgboost_available)


class _FitWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, metrics_csv, gt_csv, keys, cv):
        super().__init__()
        self._a = (metrics_csv, gt_csv, keys, cv)

    def run(self):
        try:
            import pandas as pd
            metrics_csv, gt_csv, keys, cv = self._a
            df = pd.read_csv(metrics_csv); gt = pd.read_csv(gt_csv)
            if "Plot_ID" not in gt.columns:
                raise RuntimeError("ground-truth CSV must have a Plot_ID column and a height column (height_cm)")
            self.progress.emit(2, "fitting...")
            results, pred, fitted = fit_height_models(df, gt, keys, cv=cv,
                                                      progress_cb=lambda p, m: self.progress.emit(int(2 + p * 0.9), m))
            base = os.path.splitext(metrics_csv)[0]
            pred_csv = base + "_height_predictions.csv"; pred.to_csv(pred_csv, index=False)
            saved = {}
            for k, (model, feats) in fitted.items():
                p = base + f"_height_model_{k}.joblib"
                save_height_model(p, model, feats, meta={"metrics_csv": metrics_csv, "ground_truth": gt_csv, "cv": cv,
                                                         "results": [r for r in results if r.get("key") == k]})
                saved[k] = p
            meta = {"metrics_csv": metrics_csv, "ground_truth": gt_csv, "cv": cv, "n_plots": int(len(pred)),
                    "results": results, "models": saved, "predictions": pred_csv}
            with open(base + "_height_models.json", "w") as f:
                json.dump(meta, f, indent=1, default=float)
            ok = [r for r in results if r.get("status") == "ok" and r.get("cv")]
            best = min(ok, key=lambda r: r["cv"]["rmse"]) if ok else None
            txt = results_table(results)
            txt += (f"\n\nColumn guide: cv* = every plot predicted by a model that never saw it ({cv}); fit* = in-sample. "
                    "Compare cvRMSE with the SD of the measured heights: equal means no skill. Slope near 1 means the "
                    "model tracks plot differences one for one.")
            if best:
                txt += f"\n\nBest by held-out RMSE: {best['label']} ({best['cv']['rmse']:.2f} cm)."
            cal = next((r for r in results if r.get("calibration")), None)
            if cal:
                c = cal["calibration"]; txt += f"\nCalibration line: height_cm = {c['intercept_cm']:.2f} + {c['slope']:.3f} x cth_p95_cm"
            txt += f"\n\nPredictions -> {pred_csv}\nModels -> " + ", ".join(os.path.basename(p) for p in saved.values()) + f"\nSummary -> {base}_height_models.json"
            self.progress.emit(100, "done")
            self.done_ok.emit(txt)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class HeightTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Plant-height models from LiDAR features</h2>"))
        v.addWidget(QLabel(
            "Uses the per-plot metrics of the Traits tab (compute the canopy-top cth_* heights, cover fraction, roughness, "
            "point density and the percentile heights first). Fits the chosen models to measured plant height, validates them "
            "with every plot held out, and saves predictions and models to disk. On Muresk NUE 2025 the linear calibration on "
            "cth_p95 was already at the ruler's noise floor (3.8 cm); the random forest is the one to carry to another date."))

        g = QGroupBox("Inputs"); f = QFormLayout(g)
        row = QHBoxLayout(); self.ed_metrics = QLineEdit()
        b = QPushButton("Browse..."); b.clicked.connect(self._pick_metrics)
        row.addWidget(self.ed_metrics); row.addWidget(b); f.addRow("Metrics CSV:", row)
        row2 = QHBoxLayout(); self.ed_gt = QLineEdit()
        b2 = QPushButton("Browse..."); b2.clicked.connect(self._pick_gt)
        b3 = QPushButton("Save template..."); b3.clicked.connect(self._save_template)
        row2.addWidget(self.ed_gt); row2.addWidget(b2); row2.addWidget(b3); f.addRow("Measured height CSV:", row2)
        self.ed_gt.setToolTip("Plot_ID, height_cm (or height_m). One row per measured plot; unmeasured plots may be left out.")
        v.addWidget(g)

        g2 = QGroupBox("Models and validation"); f2 = QFormLayout(g2)
        self.cbs = {}
        for k, (label, feats, learner) in HEIGHT_MODELS.items():
            cb = QCheckBox(label); cb.setChecked(True)
            cb.setToolTip("features: " + ", ".join(feats))
            if learner == "xgb" and not xgboost_available():
                cb.setChecked(False); cb.setEnabled(False); cb.setText(label + "  (xgboost not installed)")
            self.cbs[k] = cb; f2.addRow("", cb)
        self.cb_cv = QComboBox()
        self.cb_cv.addItems(["Leave-one-out (LOOCV) - recommended for <200 plots", "5-fold cross-validation", "Fit only (in-sample)"])
        f2.addRow("Validation:", self.cb_cv)
        self.btn_fit = QPushButton("Fit, validate and save models"); self.btn_fit.clicked.connect(self._run_fit)
        f2.addRow("", self.btn_fit)
        v.addWidget(g2)

        g3 = QGroupBox("Apply a saved model to this flight"); f3 = QFormLayout(g3)
        row3 = QHBoxLayout(); self.ed_model = QLineEdit()
        b4 = QPushButton("Browse..."); b4.clicked.connect(self._pick_model)
        row3.addWidget(self.ed_model); row3.addWidget(b4); f3.addRow("Saved model (.joblib):", row3)
        self.sp_offset = QDoubleSpinBox(); self.sp_offset.setRange(-100, 100); self.sp_offset.setDecimals(1); self.sp_offset.setSuffix(" cm")
        self.sp_offset.setToolTip("Per-flight correction added to every prediction: from reference targets "
                                  "(cth_*_cal columns / _target_calibration.json) or from a few measured plots of this flight.")
        self.btn_apply = QPushButton("Predict heights for every plot"); self.btn_apply.clicked.connect(self._run_apply)
        rowo = QHBoxLayout(); rowo.addWidget(self.sp_offset); rowo.addWidget(self.btn_apply); rowo.addStretch()
        f3.addRow("Offset:", rowo)
        v.addWidget(g3)

        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); v.addWidget(self.out, stretch=1)

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        s = state()
        if not self.ed_metrics.text().strip() and s.out_csv:
            self.ed_metrics.setText(s.out_csv)

    def _pick_metrics(self):
        p, _ = QFileDialog.getOpenFileName(self, "Metrics CSV (Traits tab output)", "", "CSV (*.csv)")
        if p: self.ed_metrics.setText(p)

    def _pick_gt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Measured height CSV (Plot_ID, height_cm)", "", "CSV (*.csv)")
        if p: self.ed_gt.setText(p)

    def _pick_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Saved height model", "", "joblib (*.joblib)")
        if p: self.ed_model.setText(p)

    def _save_template(self):
        import pandas as pd
        s = state(); ids = None
        try:
            m = self.ed_metrics.text().strip() or s.out_csv
            if m and os.path.exists(m):
                ids = pd.read_csv(m)["Plot_ID"].tolist()
            else:
                gp = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
                if gp and os.path.exists(gp):
                    ids = load_grid(gp, target_crs=s.work_crs)["Plot_ID"].tolist()
        except Exception:
            ids = None
        if not ids:
            QMessageBox.warning(self, "No plots", "Load a project / compute traits first."); return
        p, _ = QFileDialog.getSaveFileName(self, "Save height template",
                                           os.path.join(os.path.dirname(s.out_csv or ""), "plant_height_ground_truth.csv"),
                                           "CSV (*.csv)")
        if not p: return
        pd.DataFrame({"Plot_ID": ids, "height_cm": [""] * len(ids)}).to_csv(p, index=False)
        self.ed_gt.setText(p)
        QMessageBox.information(self, "Template saved", f"{p}\n\nFill height_cm (plant height in cm, ideally the mean of 5-10 plants measured within a few days of the flight).")

    # ------------------------------------------------------------------
    def _run_fit(self):
        m = self.ed_metrics.text().strip(); g = self.ed_gt.text().strip()
        if not m or not os.path.exists(m):
            QMessageBox.warning(self, "No metrics", "Pick the Traits-tab metrics CSV."); return
        if not g or not os.path.exists(g):
            QMessageBox.warning(self, "No ground truth", "Pick the measured-height CSV."); return
        keys = [k for k, cb in self.cbs.items() if cb.isChecked() and cb.isEnabled()]
        if not keys:
            QMessageBox.warning(self, "No model", "Tick at least one model."); return
        cv = ("loo", "kfold5", "none")[self.cb_cv.currentIndex()]
        self.btn_fit.setEnabled(False); self.progress.setValue(0)
        self._w = _FitWorker(m, g, keys, cv)
        self._w.progress.connect(self._on_prog); self._w.done_ok.connect(self._on_done); self._w.error.connect(self._on_err)
        self._w.start()

    def _run_apply(self):
        m = self.ed_metrics.text().strip(); p = self.ed_model.text().strip()
        if not m or not os.path.exists(m):
            QMessageBox.warning(self, "No metrics", "Pick the metrics CSV of the flight to predict."); return
        if not p or not os.path.exists(p):
            QMessageBox.warning(self, "No model", "Pick a saved .joblib model."); return
        try:
            import pandas as pd
            saved = load_height_model(p)
            out = apply_height_model(pd.read_csv(m), saved, offset_cm=float(self.sp_offset.value()))
            key = os.path.splitext(os.path.basename(p))[0].split("height_model_")[-1]
            outp = os.path.splitext(m)[0] + f"_height_applied_{key}.csv"; out.to_csv(outp, index=False)
            v = out.height_pred_cm.dropna()
            self.out.setPlainText(f"Applied {os.path.basename(p)} (features: {', '.join(saved['features'])}) with offset {self.sp_offset.value():+.1f} cm\n"
                                  f"{int(out.model_ok.sum())}/{len(out)} plots predicted; height mean {v.mean():.1f} cm, SD {v.std():.1f}, min {v.min():.1f}, max {v.max():.1f}\n\n"
                                  f"Predictions -> {outp}")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Apply failed", f"{e}\n\n{traceback.format_exc()}")

    def _on_prog(self, p, m):
        self.progress.setValue(p); self.progress.setFormat(f"{p}% - {m}")

    def _on_done(self, msg):
        self.btn_fit.setEnabled(True); self.out.setPlainText(msg)

    def _on_err(self, err):
        self.btn_fit.setEnabled(True); self.out.append(f"\nERROR: {err}")
        QMessageBox.critical(self, "Failed", err.split("\n")[0])
