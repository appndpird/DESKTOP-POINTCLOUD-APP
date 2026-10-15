"""
Shared "use a pretrained model" and "train from scratch" panels of the Biomass Models and Height Models tabs
(PhenoFuse 4.0). kind = "biomass" (any per-plot target, biomass units handled) or "height" (plant height in cm).

Training writes one run folder under <root>/Models/<Biomass models | Height models>/<run>/:
    predictions.csv, results.json, run.json, one .joblib per model, top3_ensemble.joblib (mean of the three best by
    held-out RMSE, members refitted on all plots), top_features.csv (biomass)
<root> is the flight folder (single-flight training) or the dataset folder.
"""
from __future__ import annotations
import os
import json
import time
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QMessageBox, QLineEdit, QTextEdit,
                             QCheckBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QGridLayout, QScrollArea)

from phenoapp.core.project import state
from phenoapp.core.units import AUTO_LABEL, UNIT_LABELS, UNIT_KEYS
from phenoapp.core.cv_utils import CV_LABELS
from phenoapp.core.model_runs import new_run_dir, list_models, top3_ensemble, save_run_meta
from phenoapp.ui.common import TrainingSource, InferenceSource, ModalityBox, file_row

def _isnum(series):
    from pandas.api.types import is_numeric_dtype
    return bool(is_numeric_dtype(series))



def _bundled_models_dir(kind):
    from phenoapp.ui.about_tab import _resource_path
    p = _resource_path(os.path.join("assets", "pretrained", "Height models" if kind == "height" else "Biomass models"))
    return p if p and os.path.isdir(p) else ""


def _basename_key(path):
    return os.path.splitext(os.path.basename(path))[0]


# ======================================================================
class _TrainWorker(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)

    def __init__(self, kind, table, root, opts):
        super().__init__(); self.kind, self.t, self.root, self.o = kind, table, root, opts

    def run(self):
        try:
            import pandas as pd
            t, o = self.t, self.o
            df, gt = t["df"], t["gt"]
            run_dir = new_run_dir(self.root, self.kind, o["target"] if self.kind == "biomass" else "height", o["modalities"], o.get("run_name", ""))
            cb = lambda p, m: self.progress.emit(min(98, int(p)), m)
            if self.kind == "biomass":
                from phenoapp.core.biomass_ml import fit_trait_models, results_table, save_model, ground_truth_kg_ha, valid_band_plots, PLOT_RULES, add_weighted, feature_family
                target = o["target"]
                if target not in gt.columns:
                    raise RuntimeError(f"ground truth has no column '{target}' (columns: {', '.join(c for c in gt.columns if c != 'Plot_ID')})")
                area = df.set_index("Plot_ID")["region_area_m2"] if "region_area_m2" in df.columns else None
                gt2, tcol, unit_label = ground_truth_kg_ha(gt, target, o["gt_unit"], area)
                plot_ids = None; rule_txt = ""
                if o["plot_rule"] in ("A", "B"):
                    if t.get("valid_fraction") is None:
                        raise RuntimeError("the valid-band plot rule needs the band valid-fraction table next to the VNIR features (written by the VNIR Spectral Features tab)")
                    vdf = df[["Plot_ID"] + [c for c in ("vnir_qc_ok", "vnir_reflectance_ok", "frac_in_cube") if c in df.columns]]
                    plot_ids = valid_band_plots(vdf, t["valid_fraction"], rule=o["plot_rule"])
                    rule_txt = f"\nPlots: {PLOT_RULES[o['plot_rule']]} -> {len(plot_ids)} plots kept"
                results, pred, fitted = fit_trait_models(df, gt2, tcol, o["families"], o["learners"], cv=o["cv"], n_select=o["n_select"], progress_cb=cb,
                                                         vnir_cols=t["vnir_cols"], spectra=t["spectra"], plot_ids=plot_ids, groups=t["groups"], rgb_cols=t["rgb_cols"],
                                                         vnir_mode=o["vnir_mode"], modalities=o["modalities"], excluded=t.get("excluded"),
                                                         band_transform=o["band_transform"])
                pred.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
                saved = {}; train_tab = add_weighted(df)
                for (fam, ln), (model, feats) in fitted.items():
                    fn = f"{fam}_{ln}".replace("[", "_").replace("]", "").replace("/", "_") + ".joblib"
                    save_model(os.path.join(run_dir, fn), model, feats, meta={"target": tcol, "family": fam, "learner": ln, "cv": o["cv"], "modalities": o["modalities"], "vnir_mode": o["vnir_mode"],
                                                                              "band_transform": o["band_transform"]}, table=train_tab)
                    saved[f"{fam}/{ln}"] = fn
                ens = top3_ensemble(results, fitted, key_fn=lambda r: (r["family"], r["learner"]), rmse_key="RMSE", table=train_tab, band_transform=o["band_transform"])
                if ens:
                    import joblib
                    ens["meta"] = {"target": tcol, "kind": "biomass", "modalities": o["modalities"], "vnir_mode": o["vnir_mode"], "cv": o["cv"], "band_transform": o["band_transform"]}
                    joblib.dump(ens, os.path.join(run_dir, "top3_ensemble.joblib"))
                with open(os.path.join(run_dir, "results.json"), "w") as f:
                    json.dump({"target": tcol, "unit": unit_label, "cv": o["cv"], "n_select": o["n_select"], "results": results, "models": saved,
                               "flights": t["flights"], "modalities": o["modalities"], "vnir_mode": o["vnir_mode"], "band_transform": o["band_transform"]}, f, indent=1, default=float)
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    d = add_weighted(df).merge(gt2[["Plot_ID", tcol]].dropna(), on="Plot_ID")
                    cols = feature_family(d, "Fused_all", exclude=(tcol,), vnir_cols=t["vnir_cols"], rgb_cols=t["rgb_cols"], vnir_mode=o["vnir_mode"], modalities=o["modalities"])
                    X = d[cols].astype(float).fillna(d[cols].astype(float).median()).to_numpy(); y = d[tcol].to_numpy(float)
                    rf = RandomForestRegressor(n_estimators=800, min_samples_leaf=4, random_state=0, n_jobs=-1).fit(X, y)
                    top = pd.DataFrame({"feature": cols, "rf_importance": rf.feature_importances_, "r_with_target": [np.corrcoef(X[:, i], y)[0, 1] for i in range(len(cols))]}).sort_values("rf_importance", ascending=False)
                    top.to_csv(os.path.join(run_dir, "top_features.csv"), index=False); toptxt = "\n\nTop 12 features (RF importance, all features of the ticked modalities):\n" + top.head(12).round(3).to_string(index=False)
                except Exception as e:
                    toptxt = f"\n(top-feature ranking failed: {e})"
                ok = [r for r in results if r.get("status") == "ok" and r.get("cv")]
                best = min(ok, key=lambda r: r["cv"]["RMSE"]) if ok else None
                txt = f"Target: {tcol}" + (f" ({unit_label})" if unit_label else "") + f"; flights: {', '.join(t['flights'])}; modalities: {'+'.join(o['modalities'])}; VNIR input: {o['vnir_mode']}" + (f" (bands: {o['band_transform']})" if o['vnir_mode'] != 'indices' else "") + f"; validation: {CV_LABELS[o['cv']]}\n\n"
                txt += results_table(results) + toptxt
                if best:
                    txt += f"\n\nBest by held-out RMSE: {best['family']} / {best['learner']}: R2 {best['cv']['R2']:.3f}, RMSE {best['cv']['RMSE']:.3g}, accuracy {best['cv']['accuracy']:.1f}%\nFeatures: {', '.join(best['features'])}"
                    if best.get("calibration"):
                        txt += f"\nCalibration: {best['calibration']['equation']}"
                cals = [r for r in results if r.get("status") == "ok" and r.get("calibration")]
                if cals:
                    txt += "\n\nLinear calibrations:\n" + "\n".join(f"  {r['calibration']['equation']}   (cv R2 {r['cv']['R2']:.2f})" if r.get("cv") else f"  {r['calibration']['equation']}" for r in cals)
                if ens:
                    txt += "\n\nTop-three mean ensemble saved (top3_ensemble.joblib): " + ", ".join(f"{m['family']}/{m['learner']}" for m in ens["members"])
                txt += rule_txt
            else:
                from phenoapp.core.height_models import fit_height_models, results_table, save_height_model
                if gt is None or len(gt) == 0:
                    raise RuntimeError("no ground truth")
                results, pred, fitted = fit_height_models(df, gt, o["keys"], cv=o["cv"], progress_cb=cb, groups=t["groups"], spectra=t["spectra"],
                                                          vnir_cols=t["vnir_cols"], rgb_cols=t["rgb_cols"], vnir_mode=o["vnir_mode"], excluded=t.get("excluded"),
                                                          band_transform=o["band_transform"])
                pred.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
                saved = {}
                from phenoapp.core.height_models import prepare_features
                train_tab = prepare_features(df, None, None)
                for k, (model, feats) in fitted.items():
                    fn = f"height_model_{k}.joblib"
                    save_height_model(os.path.join(run_dir, fn), model, feats, meta={"cv": o["cv"], "modalities": o["modalities"], "vnir_mode": o["vnir_mode"],
                                                                                     "band_transform": o["band_transform"],
                                                                                     "results": [r for r in results if r.get("key") == k]}, table=train_tab)
                    saved[k] = fn
                ens = top3_ensemble(results, fitted, key_fn=lambda r: r["key"], rmse_key="rmse", table=train_tab, band_transform=o["band_transform"])
                if ens:
                    import joblib
                    ens["meta"] = {"kind": "height", "units": "cm", "modalities": o["modalities"], "vnir_mode": o["vnir_mode"], "cv": o["cv"], "band_transform": o["band_transform"]}
                    joblib.dump(ens, os.path.join(run_dir, "top3_ensemble.joblib"))
                with open(os.path.join(run_dir, "results.json"), "w") as f:
                    json.dump({"cv": o["cv"], "n_plots": int(len(pred)), "results": results, "models": saved, "flights": t["flights"], "modalities": o["modalities"], "vnir_mode": o["vnir_mode"], "band_transform": o["band_transform"]}, f, indent=1, default=float)
                ok = [r for r in results if r.get("status") == "ok" and r.get("cv")]
                best = min(ok, key=lambda r: r["cv"]["rmse"]) if ok else None
                txt = f"Flights: {', '.join(t['flights'])}; modalities: {'+'.join(o['modalities'])}; VNIR input: {o['vnir_mode']}" + (f" (bands: {o['band_transform']})" if o['vnir_mode'] != 'indices' else "") + f"; validation: {CV_LABELS[o['cv']]}\n\n" + results_table(results)
                txt += (f"\n\nColumn guide: cv* = every plot predicted by a model that never saw it; fit* = in-sample. Compare cvRMSE with the SD of the measured "
                        "heights: equal means no skill. Slope near 1 means the model tracks plot differences one for one.")
                if best:
                    txt += f"\n\nBest by held-out RMSE: {best['label']} ({best['cv']['rmse']:.2f} cm)."
                for r in results:
                    if r.get("calibration"):
                        c = r["calibration"]; txt += f"\nCalibration line ({r['key']}): height_cm = {c['intercept_cm']:.2f} + {c['slope']:.3f} x {c['predictor']}_cm"
                if ens:
                    txt += "\n\nTop-three mean ensemble saved (top3_ensemble.joblib): " + ", ".join(m["family"] for m in ens["members"])
            save_run_meta(run_dir, kind=self.kind, flights=t["flights"], modalities=o["modalities"], vnir_mode=o["vnir_mode"], band_transform=o["band_transform"], cv=o["cv"],
                          n_rows=int(len(df)), options={k: v for k, v in o.items() if k not in ("families", "learners", "keys")}, families=o.get("families"), learners=o.get("learners"), keys=o.get("keys"))
            txt += f"\n\nRun folder -> {run_dir}\n  predictions.csv, results.json, run.json, {len(saved)} model files" + (", top3_ensemble.joblib" if ens else "")
            self.progress.emit(100, "done"); self.done_ok.emit(txt)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ======================================================================
class TrainPanel(QWidget):
    def __init__(self, kind, parent=None):
        super().__init__(parent); self.kind = kind; self._build()

    def _build(self):
        v = QVBoxLayout(self)
        self.src = TrainingSource(self, gt_hint=("Plot_ID + biomass column (any unit: biomass_kg_ha, fresh_t_ha, dm_g_m2, fresh_kg ...)" if self.kind == "biomass"
                                                   else "Plot_ID + height_cm (or height_m)"))
        self.src.ed_gt.textChanged.connect(self._gt_changed)
        v.addWidget(self.src)
        g = QGroupBox("Modalities and models"); f = QFormLayout(g)
        self.mod = ModalityBox(self, on_change=self._refresh_models); f.addRow("Modalities:", self.mod)
        if self.kind == "biomass":
            row = QHBoxLayout(); self.cb_target = QComboBox(); self.cb_target.setEditable(True); self.cb_target.setMinimumWidth(180); self.cb_target.addItem("biomass_kg_ha")
            self.cb_unit = QComboBox(); self.cb_unit.addItem(AUTO_LABEL)
            for k in UNIT_KEYS:
                self.cb_unit.addItem(UNIT_LABELS[k])
            self.cb_unit.setToolTip("Unit of the target column. Auto-detect reads it from the column name (biomass_kg_ha, fresh_t_ha, dm_g_m2, fresh_kg = kg per plot...). "
                                    "Everything is converted to kg/ha; per-plot weights need region_area_m2 from the LiDAR table. Non-biomass targets are left as they are.")
            self.cb_basis = QComboBox(); self.cb_basis.addItems(["Target as chosen", "Fresh weight (pick the first fresh_* column)", "Dry matter (pick the first dm_* / dry column)"])
            self.cb_basis.currentIndexChanged.connect(self._basis)
            row.addWidget(QLabel("target column:")); row.addWidget(self.cb_target, stretch=2); row.addWidget(QLabel("unit:")); row.addWidget(self.cb_unit, stretch=2); row.addWidget(self.cb_basis, stretch=2)
            f.addRow("Ground truth:", row)
        self.model_box = QWidget(); self.model_grid = QGridLayout(self.model_box); self.model_grid.setContentsMargins(0, 0, 0, 0); self.cbs = {}
        f.addRow("Families:" if self.kind == "biomass" else "Models:", self.model_box)
        if self.kind == "biomass":
            from phenoapp.core.biomass_ml import LEARNERS, learner_available, PLOT_RULES
            lr = QHBoxLayout(); self.cb_lrn = {}
            for ln in LEARNERS:
                cb = QCheckBox(ln); av = learner_available(ln); cb.setChecked(av and ln in ("Ridge", "PLS", "RandomForest", "XGBoost")); cb.setEnabled(av)
                if not av: cb.setText(ln + " (not installed)")
                self.cb_lrn[ln] = cb; lr.addWidget(cb)
            lr.addStretch(); f.addRow("Learners:", lr)
            self.cb_plots = QComboBox(); self.cb_plots.addItems([PLOT_RULES["all"], PLOT_RULES["A"], PLOT_RULES["B"]])
            self.cb_plots.setToolTip("Which plots enter the models. The valid-band rules keep only plots whose spectra have no band problem (VNIR band valid-fraction table needed).")
            f.addRow("Plots:", self.cb_plots)
        self.cb_cv = QComboBox()
        for k in ("rkf10", "kfold5", "loo", "none"):
            self.cb_cv.addItem(CV_LABELS[k], k)
        self.sp_sel = QSpinBox(); self.sp_sel.setRange(3, 60); self.sp_sel.setValue(12); self.sp_sel.setToolTip("Features kept per training fold (random-forest importance inside the fold); families with fewer features use all.")
        row = QHBoxLayout(); row.addWidget(self.cb_cv, stretch=3)
        if self.kind == "biomass":
            row.addWidget(QLabel("features kept:")); row.addWidget(self.sp_sel)
        f.addRow("Validation:", row)
        self.ed_run = QLineEdit(); self.ed_run.setPlaceholderText("optional run name (folder: <date_time>_<name>_<modalities> under Models/)")
        f.addRow("Run name:", self.ed_run)
        self.lbl_out = QLabel(""); f.addRow("Models go to:", self.lbl_out)
        v.addWidget(g)
        row = QHBoxLayout(); self.btn = QPushButton("Fit, validate and save models"); self.btn.clicked.connect(self._run); row.addWidget(self.btn); row.addStretch(); v.addLayout(row)
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); self.out.setMinimumHeight(220); v.addWidget(self.out, stretch=1)
        self._refresh_models()

    def _refresh_models(self):
        mods = self.mod.modalities(); mode = self.mod.vnir_mode()
        checked = {k for k, cb in self.cbs.items() if cb.isChecked()}
        for cb in self.cbs.values():
            cb.setParent(None)
        self.cbs = {}
        if self.kind == "biomass":
            from phenoapp.core.biomass_ml import families_for, FAMILY_TIPS
            items = [(f, f, FAMILY_TIPS.get(f, "")) for f in families_for(mods, mode)]
            default = {"Simple_linear", "LiDAR_core", "VNIR_core", "RGB_core", "Fused_core", "Fused3_core", "LiDAR_PCA", "Fused_PCA", "Fused3_PCA"}
        else:
            from phenoapp.core.height_models import models_for, HEIGHT_MODELS, DEFAULT_HEIGHT_KEYS, xgboost_available
            items = [(k, HEIGHT_MODELS[k][0], "features: " + (", ".join(HEIGHT_MODELS[k][1]) if not isinstance(HEIGHT_MODELS[k][1], str) else HEIGHT_MODELS[k][1])) for k in models_for(mods)]
            default = set(DEFAULT_HEIGHT_KEYS)
        ncol = 3 if self.kind == "biomass" else 2
        for i, (key, label, tip) in enumerate(items):
            cb = QCheckBox(label); cb.setToolTip(tip); cb.setChecked(key in checked or (not checked and key in default))
            if self.kind == "height" and key == "struct_xgb":
                from phenoapp.core.height_models import xgboost_available
                if not xgboost_available():
                    cb.setChecked(False); cb.setEnabled(False); cb.setText(label + "  (xgboost not installed)")
            self.cbs[key] = cb; self.model_grid.addWidget(cb, i // ncol, i % ncol)

    def _basis(self, i):
        if i == 0: return
        want = "fresh" if i == 1 else "dm"
        for j in range(self.cb_target.count()):
            c = self.cb_target.itemText(j).lower()
            if c.startswith(want) or (want == "dm" and ("dry" in c or c.startswith("dm_"))):
                self.cb_target.setCurrentIndex(j); return

    def _gt_changed(self, p):
        if self.kind != "biomass" or not (p and os.path.exists(p)):
            return
        try:
            import pandas as pd
            from phenoapp.core.dataset import read_csv
            cols = [c for c in read_csv(p, nrows=5).columns if c not in ("Plot_ID", "Plot", "flight", "Variety", "B/R")]
            cur = self.cb_target.currentText(); self.cb_target.clear(); self.cb_target.addItems(cols)
            if cur in cols: self.cb_target.setCurrentText(cur)
            elif cols: self.cb_target.setCurrentIndex(0)
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev); self.src.sync_from_state(); self._update_out()

    def _update_out(self):
        s = state()
        root = self.src.ed_dataset.text().strip() if self.src.is_dataset() else (s.proj_dir or "")
        self.lbl_out.setText(os.path.join(root, "Models", "Biomass models" if self.kind == "biomass" else "Height models") if root else "(load a project or pick a dataset)")

    def _run(self):
        mods = self.mod.modalities()
        if not mods:
            QMessageBox.warning(self, "No modality", "Tick at least one modality."); return
        keys = [k for k, cb in self.cbs.items() if cb.isChecked() and cb.isEnabled()]
        if not keys:
            QMessageBox.warning(self, "Nothing selected", "Tick at least one model / family."); return
        try:
            t, root = self.src.load(mods, gt_required=True)
        except Exception as e:
            QMessageBox.warning(self, "Training data", str(e)); return
        o = dict(modalities=mods, vnir_mode=self.mod.vnir_mode(), band_transform=self.mod.band_transform(), cv=self.cb_cv.currentData(), n_select=self.sp_sel.value(), run_name=self.ed_run.text().strip())
        if self.kind == "biomass":
            lrn = [k for k, cb in self.cb_lrn.items() if cb.isChecked() and cb.isEnabled()]
            if not lrn:
                QMessageBox.warning(self, "No learner", "Tick at least one learner."); return
            i = self.cb_unit.currentIndex()
            o.update(families=keys, learners=lrn, target=self.cb_target.currentText().strip() or "biomass_kg_ha", gt_unit=("auto" if i <= 0 else UNIT_KEYS[i - 1]),
                     plot_rule=("all", "A", "B")[self.cb_plots.currentIndex()])
        else:
            o.update(keys=keys)
        self._update_out(); self.btn.setEnabled(False); self.progress.setValue(0); self.out.setPlainText("training...")
        self._w = _TrainWorker(self.kind, t, root, o)
        self._w.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._w.done_ok.connect(lambda txt: (self.btn.setEnabled(True), self.out.setPlainText(txt)))
        self._w.error.connect(lambda e: (self.btn.setEnabled(True), self.out.append("\nERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._w.start()


# ======================================================================
class PretrainedPanel(QWidget):
    def __init__(self, kind, parent=None):
        super().__init__(parent); self.kind = kind; self._build()

    def _build(self):
        v = QVBoxLayout(self)
        g = QGroupBox("Pretrained model"); f = QFormLayout(g)
        row = QHBoxLayout(); self.cb_known = QComboBox(); self.cb_known.setMinimumWidth(320)
        b = QPushButton("Refresh list"); b.clicked.connect(self.refresh_models); row.addWidget(self.cb_known, stretch=1); row.addWidget(b)
        self.cb_known.currentIndexChanged.connect(lambda i: self.ed_model.setText(self.cb_known.itemData(i) or ""))
        f.addRow("Known models:", row)
        self.ed_model = QLineEdit(); f.addRow("Model file (.joblib):", file_row(self, self.ed_model, "Saved model", "joblib (*.joblib)"))
        self.lbl_model = QLabel(""); self.lbl_model.setWordWrap(True); f.addRow("", self.lbl_model)
        self.ed_model.textChanged.connect(self._describe)
        from phenoapp.core.ensembles import FOLD_CHOICES
        self.cb_folds = QComboBox(); self.cb_folds.addItems([FOLD_CHOICES["all"], FOLD_CHOICES["best3"]])
        self.cb_folds.setToolTip("Only for *_fold_ensembles.joblib files: average all fold ensembles, or the three folds with the lowest test RMSE (optimistic).")
        f.addRow("Fold models:", self.cb_folds)
        self.sp_off = QDoubleSpinBox(); self.sp_off.setRange(-1e6, 1e6); self.sp_off.setDecimals(2)
        self.sp_off.setToolTip("Per-flight correction added to every prediction (from reference targets or a few measured plots of this flight).")
        f.addRow("Offset (target units):" if self.kind == "biomass" else "Offset (cm):", self.sp_off)
        v.addWidget(g)
        self.src = InferenceSource(self); v.addWidget(self.src)
        g2 = QGroupBox("Check against a measured CSV (optional)"); f2 = QFormLayout(g2)
        self.ed_gt = QLineEdit(); self.ed_gt.setPlaceholderText("Plot_ID + measured column: the predictions are scored against it")
        f2.addRow("Measured CSV:", file_row(self, self.ed_gt, "Measured values")); v.addWidget(g2)
        row = QHBoxLayout(); self.btn = QPushButton("Predict every plot"); self.btn.clicked.connect(self._run); row.addWidget(self.btn); row.addStretch(); v.addLayout(row)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); self.out.setMinimumHeight(200); v.addWidget(self.out, stretch=1)

    def showEvent(self, ev):
        super().showEvent(ev); self.src.sync_from_state()
        if self.cb_known.count() == 0:
            self.refresh_models()

    def refresh_models(self):
        s = state(); cur = self.ed_model.text().strip(); self.cb_known.blockSignals(True); self.cb_known.clear()
        self.cb_known.addItem("(pick a file below)", "")
        seen = set()
        for label, root in (("bundled", _bundled_models_dir(self.kind)), ("this flight", s.proj_dir), ("dataset", s.dataset_dir)):
            if not root:
                continue
            files = []
            if label == "bundled":
                files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.endswith(".joblib")]
            else:
                files = list_models(root, self.kind)
            for p in files:
                if p in seen: continue
                seen.add(p); run = os.path.basename(os.path.dirname(p))
                self.cb_known.addItem(f"[{label}] {run}/{os.path.basename(p)}" if label != "bundled" else f"[bundled] {os.path.basename(p)}", p)
        self.cb_known.blockSignals(False)
        if cur:
            i = self.cb_known.findData(cur)
            if i >= 0: self.cb_known.setCurrentIndex(i)

    def _describe(self, p):
        if not (p and os.path.exists(p)):
            self.lbl_model.setText(""); return
        try:
            import joblib
            from phenoapp.core.ensembles import describe, required_features
            saved = joblib.load(p); meta = saved.get("meta", {}) if isinstance(saved, dict) else {}
            need = required_features(saved)
            from phenoapp.core.ensembles import band_transform
            bt = f"; bands: {band_transform(saved)}" if any(f.startswith("sb_") for f in need) else ""
            self.lbl_model.setText(f"{describe(saved)}; target {meta.get('target', 'height_cm' if self.kind == 'height' else '?')}; "
                                   f"modalities {'+'.join(meta.get('modalities', [])) or '?'}{bt}; {len(need)} input features, e.g. {', '.join(need[:8])}")
        except Exception as e:
            self.lbl_model.setText(f"cannot read: {e}")

    def _run(self):
        p = self.ed_model.text().strip()
        if not (p and os.path.exists(p)):
            QMessageBox.warning(self, "No model", "Pick a saved .joblib model."); return
        try:
            import pandas as pd
            from phenoapp.core.ensembles import describe, fold_table, is_fold_ensembles, required_features
            t, out_dir, flight = self.src.load()
            df = t["df"]
            if self.kind == "biomass":
                from phenoapp.core.biomass_ml import load_model, apply_model
                saved = load_model(p)
            else:
                from phenoapp.core.height_models import load_height_model, apply_height_model
                saved = load_height_model(p)
            folds = ("all", "best3")[self.cb_folds.currentIndex()] if is_fold_ensembles(saved) else "all"
            if self.kind == "biomass":
                out = apply_model(df, saved, float(self.sp_off.value()), spectra=t["spectra"], folds=folds, excluded=t.get("excluded"))
                tgt = saved.get("meta", {}).get("target", "predicted"); col = "predicted"
            else:
                out = apply_height_model(df, saved, offset_cm=float(self.sp_off.value()), spectra=t["spectra"], folds=folds, excluded=t.get("excluded"))
                tgt = "height_cm"; col = "height_pred_cm"
            for c in ("Plot_ID_orig", "flight"):
                if c in df.columns: out.insert(1, c, df[c].values)
            os.makedirs(out_dir, exist_ok=True)
            outp = os.path.join(out_dir, f"{flight}_{tgt}_pred_{_basename_key(p)}" + (f"_{folds}" if is_fold_ensembles(saved) else "") + ".csv")
            txt = f"Applied {os.path.basename(p)} ({describe(saved, folds)}) to {flight} with offset {self.sp_off.value():+g}\n"
            v = out[col].dropna(); txt += f"{int(out.model_ok.sum())}/{len(out)} plots predicted; mean {v.mean():.4g}, SD {v.std():.4g}, min {v.min():.4g}, max {v.max():.4g}\n"
            # missing = not in the prepared table (derived columns such as the weighted VNIR x cover features, tip_thin / vspread
            # and the reflectance bands are built inside apply_*; a band model without a spectra table shows its sb_ bands here)
            missing = list(getattr(out, "attrs", {}).get("missing_features", []))
            if missing:
                txt += f"WARNING: {len(missing)} input feature(s) missing from this flight's tables: {', '.join(missing[:8])}\n"
            attrs = getattr(out, "attrs", {})
            filled = list(attrs.get("filled_bands", []))
            if filled:
                txt += (f"WARNING: {len(filled)} of the model's reflectance bands are unusable on every plot of this flight "
                        f"({filled[0][3:]}-{filled[-1][3:]} nm, e.g. clipped by the sensor); they were set to the SNV mean (0), "
                        f"so predictions lean on the remaining bands.\n")
            ff = list(attrs.get("filled_features", []))
            if ff:
                txt += (f"WARNING: {len(ff)} input feature(s) are NaN on every plot of this flight (an index that needs a clipped band, "
                        f"e.g. the 445 nm blue band): {', '.join(ff[:8])}; filled with the model's training-set median.\n")
            rp = list(attrs.get("replaced_features", []))
            if rp:
                txt += (f"NOTE: {len(rp)} flight-specific spectral PCA score(s) ({', '.join(rp[:5])}) were replaced by the model's training "
                        f"median: their sign and scale change between extraction runs, so they cannot be transferred to another flight.\n")
            oor = list(attrs.get("out_of_range", [])); nchk = int(attrs.get("n_range_checked", 0) or 0)
            if oor and nchk:
                meta = saved.get("meta", {}) if isinstance(saved, dict) else {}
                src = ", ".join(str(meta[k]) for k in ("dataset", "flight") if meta.get(k))
                ex = "; ".join(f"{f}: flight {m:.3g} vs training {lo:.3g}-{hi:.3g}" for f, m, lo, hi in oor[:4])
                sev = "WARNING" if len(oor) >= max(3, 0.2 * nchk) else "Note"
                txt += (f"{sev}: {len(oor)} of {nchk} input features lie outside the model's training range (5-95 %){' - trained on ' + src if src else ''}: "
                        f"{ex}. The flight is probably at a different growth stage (or sensor setting) than the training flight, so these "
                        f"predictions are extrapolations; prefer a model trained at the same stage, or train from scratch on this flight.\n")
            uf = [f for f in attrs.get("unusable_features", []) if f not in ff]
            if uf:
                txt += (f"ERROR: {len(uf)} input feature(s) are NaN on every plot of this flight and the model stores no training median "
                        f"for them, so no plot could be predicted: {', '.join(uf[:8])}. Re-extract the VNIR features with the band policy "
                        f"used for training, or use a model that does not need them.\n")
            g = self.ed_gt.text().strip()
            if g and os.path.exists(g):
                from phenoapp.core.dataset import read_csv
                gt = read_csv(g); mc = next((c for c in gt.columns if c not in ("Plot_ID", "Plot", "flight") and _isnum(gt[c])), None)
                if mc:
                    key = "Plot_ID_orig" if "Plot_ID_orig" in out.columns else "Plot_ID"
                    mm = out.merge(gt[["Plot_ID", mc]].rename(columns={"Plot_ID": key}), on=key).dropna(subset=[col, mc])
                    if len(mm) >= 3:
                        from phenoapp.core.biomass_ml import metrics
                        m = metrics(mm[col].to_numpy(float), mm[mc].to_numpy(float))
                        out = out.merge(gt[["Plot_ID", mc]].rename(columns={"Plot_ID": key, mc: f"{mc}_measured"}), on=key, how="left")
                        txt += f"vs measured '{mc}' ({len(mm)} plots): R2 {m['R2']:.3f}, RMSE {m['RMSE']:.3g}, bias {m['bias']:+.3g}, r {m['r']:.2f}, accuracy {m['accuracy']:.1f}%\n"
            out.to_csv(outp, index=False); txt += f"\nPredictions -> {outp}"
            ft = fold_table(saved)
            if ft is not None:
                txt += "\n\nFold ensembles (test score of each fold; * = best three):\n" + ft.assign(best3=ft.best3.map({True: "*", False: ""})).to_string(index=False)
            self.out.setPlainText(txt)
        except Exception as e:
            import traceback; QMessageBox.critical(self, "Predict failed", f"{e}\n\n{traceback.format_exc()}")
