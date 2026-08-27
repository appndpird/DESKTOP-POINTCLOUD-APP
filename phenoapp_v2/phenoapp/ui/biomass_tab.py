"""
Biomass & Fusion tab.

Workflow:
  1. Compute LiDAR traits first (Traits tab) -> plot metrics CSV.
  2. (optional) Point at the VNIR hyperspectral orthomosaic (.bin ENVI) and
     click "Compute VNIR indices + spectra" -> per-plot NDVI/NDRE/WBI + full
     mean spectra, sampled over the SAME region as the LiDAR traits.
  3. Save/fill the ground-truth template (fresh_kg, dm_frac or dm_kg).
  4. "Fit & validate models" -> the whole suite (LiDAR / VNIR / fusion,
     fresh + DM) is fitted with leave-one-out cross-validation and ranked.
  5. Predictions for every plot are written next to the metrics CSV; models
     are saved as JSON for reuse on another flight of the same trial.

Reading the results (validated on the 2025 DPIRD Fodder trials):
  * Fresh biomass is the primary sensor product. On dense multi-species
    pasture expect VNIR (NDRE/PLS) >> LiDAR height; fusion adds little.
  * DM% comes from the 970 nm water feature (WBI / PLS).
  * DM kg direct models are shown for honesty but are usually weak from a
    single flight; the derived product (fresh_pred x DM%_pred) compounds
    errors. Treat DM kg as indicative unless LOOCV says otherwise.
"""

from __future__ import annotations
import os
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QProgressBar, QFileDialog, QMessageBox, QLineEdit, QTextEdit, QCheckBox,
    QFormLayout, QComboBox
)

from phenoapp.core.project import state
from phenoapp.core import load_grid, plot_region, VNIRCube
from phenoapp.core.models import (fit_model_suite, results_table, save_models,
                                  MODEL_SUITE)


class _VNIRWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok  = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, vnir_path, grid_path, work_crs,
                 region_mode, band_width, out_csv, out_npz):
        super().__init__()
        self._a = (vnir_path, grid_path, work_crs, region_mode,
                   band_width, out_csv, out_npz)

    def run(self):
        try:
            (vnir_path, grid_path, work_crs, region_mode,
             band_width, out_csv, out_npz) = self._a
            self.progress.emit(2, "Opening VNIR cube...")
            cube = VNIRCube(vnir_path)
            plots = load_grid(grid_path, target_crs=work_crs)
            regions = [plot_region(g, mode=region_mode,
                                   band_width=band_width)
                       for g in plots.geometry]
            self.progress.emit(5, f"{len(cube.wavelengths)} bands "
                               f"{cube.wavelengths[0]:.0f}-"
                               f"{cube.wavelengths[-1]:.0f} nm")

            def cb1(p, m): self.progress.emit(5 + int(p * 0.25), m)
            df = cube.plot_table(plots, regions, progress_cb=cb1)
            df.to_csv(out_csv, index=False)

            def cb2(p, m): self.progress.emit(30 + int(p * 0.68), m)
            spectra, npx = cube.plot_spectra(plots, regions, progress_cb=cb2)
            np.savez_compressed(out_npz, spectra=spectra,
                                plot_ids=np.asarray(df["Plot_ID"]),
                                wavelengths=cube.wavelengths, n_px=npx)
            cube.close()

            low = df[df["vnir_coverage"] < 0.5]["Plot_ID"].tolist()
            msg = f"VNIR table: {out_csv}\nSpectra: {out_npz}"
            if low:
                msg += (f"\nWARNING: plots {low} have <50% valid VNIR pixels "
                        "(cube cropped or striped there) - treat their "
                        "spectral values with caution.")
            self.progress.emit(100, "VNIR done")
            self.done_ok.emit(msg)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _FitWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok  = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, metrics_csv, vnir_csv, spectra_npz, gt_csv,
                 out_pred_csv, out_models_json, cv_mode="loo", enabled=None):
        super().__init__()
        self._a = (metrics_csv, vnir_csv, spectra_npz, gt_csv,
                   out_pred_csv, out_models_json, cv_mode, enabled)

    def run(self):
        try:
            import pandas as pd
            (metrics_csv, vnir_csv, spectra_npz, gt_csv,
             out_pred_csv, out_models_json, cv_mode, enabled) = self._a

            df = pd.read_csv(metrics_csv)
            if vnir_csv and os.path.exists(vnir_csv):
                vn = pd.read_csv(vnir_csv)
                df = df.merge(vn, on="Plot_ID", how="left")
            spectra = spectra_ids = None
            if spectra_npz and os.path.exists(spectra_npz):
                z = np.load(spectra_npz, allow_pickle=False)
                spectra = z["spectra"]; spectra_ids = list(z["plot_ids"])
            gt = pd.read_csv(gt_csv)
            if "Plot_ID" not in gt.columns or "fresh_kg" not in gt.columns:
                raise RuntimeError(
                    "Ground-truth CSV must have Plot_ID and fresh_kg columns "
                    "(plus optional dm_frac or dm_kg).")

            def cb(p, m): self.progress.emit(int(p * 0.95), m)
            results, pred = fit_model_suite(df, gt, spectra, spectra_ids,
                                            enabled=enabled,
                                            progress_cb=cb, cv=cv_mode)
            pred.to_csv(out_pred_csv, index=False)
            save_models(results, out_models_json)

            txt = results_table(results)
            txt += (f"\n\nPredictions -> {out_pred_csv}"
                    f"\nModels JSON -> {out_models_json}"
                    "\n\nColumn guide: R2/RMSE/RMSE%/MAE/Acc%/bias/r are from "
                    "the chosen validation (every plot predicted by a model "
                    "that never saw it); Acc% = 100 - MAPE ('simple "
                    "accuracy'); fitR2 is in-sample - the gap between fitR2 "
                    "and R2 is overfitting. Compare RMSE to the ground-truth "
                    "std-dev: equal means no skill.")
            self.progress.emit(100, "Fitting complete")
            self.done_ok.emit(txt)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class BiomassTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Biomass &amp; dry matter — LiDAR, VNIR, fusion</h2>"))
        v.addWidget(QLabel(
            "Fresh biomass is the primary sensor product; DM% comes from the "
            "970 nm water feature; DM kg is derived. Every model is "
            "leave-one-out cross-validated so you see honest accuracy."))

        # ---- VNIR section ----
        vbox = QGroupBox("VNIR hyperspectral orthomosaic (ENVI .bin + .hdr)")
        fform = QFormLayout(vbox)
        row = QHBoxLayout()
        self.ed_vnir = QLineEdit()
        b = QPushButton("Browse..."); b.clicked.connect(self._pick_vnir)
        row.addWidget(self.ed_vnir); row.addWidget(b)
        fform.addRow("VNIR cube:", row)
        self.btn_vnir = QPushButton("Compute VNIR indices + spectra per plot")
        self.btn_vnir.clicked.connect(self._run_vnir)
        self.btn_vnir.setToolTip(
            "Computes NDVI, NDRE (fresh-biomass index), WBI/NDWI970 "
            "(moisture) and the full mean spectrum per plot, over the same "
            "sampling region as the LiDAR traits (Traits tab setting).")
        fform.addRow("", self.btn_vnir)
        v.addWidget(vbox)

        # ---- Ground truth + fitting ----
        gbox = QGroupBox("Calibration against ground truth")
        gform = QFormLayout(gbox)
        row2 = QHBoxLayout()
        self.ed_gt = QLineEdit()
        b2 = QPushButton("Browse..."); b2.clicked.connect(self._pick_gt)
        row2.addWidget(self.ed_gt); row2.addWidget(b2)
        gform.addRow("Ground-truth CSV:", row2)

        # Modality selection: which sensor(s) the fitted models may use.
        mod_row = QHBoxLayout()
        self.cb_mod_lidar = QCheckBox("LiDAR (structure)")
        self.cb_mod_vnir = QCheckBox("VNIR (spectral)")
        self.cb_mod_fusion = QCheckBox("Fusion / multi-feature (both)")
        for c in (self.cb_mod_lidar, self.cb_mod_vnir, self.cb_mod_fusion):
            c.setChecked(True)
            mod_row.addWidget(c)
        mod_row.addStretch()
        self.cb_mod_lidar.setToolTip(
            "Height/cover/volume models from the point cloud. Best on "
            "short, sparse or row-structured swards (e.g. Warner Glen).")
        self.cb_mod_vnir.setToolTip(
            "Spectral models (NDRE, water indices, full-spectrum PLS). "
            "Best on dense closed swards (e.g. Busselton); also the only "
            "route to DM%.")
        self.cb_mod_fusion.setToolTip(
            "Models that may combine both sensors, incl. ridge and kernel-"
            "ridge over all features. The validation table shows whether "
            "fusion actually beats the single-sensor models on your data.")
        gform.addRow("Modalities:", mod_row)

        self.cb_cv = QComboBox()
        self.cb_cv.addItems([
            "Leave-one-out (LOOCV) — recommended for <60 plots",
            "5-fold cross-validation — faster, near-identical verdicts",
            "Fit only — in-sample metrics (optimistic; for exploration only)",
        ])
        self.cb_cv.setToolTip(
            "How accuracy is measured.\n\n"
            "LOOCV / 5-fold: each plot is predicted by a model that never "
            "saw it — honest accuracy for new data.\n\n"
            "Fit only: the model is scored on the plots it was trained on. "
            "Always looks better than reality; the in-sample R2 is also "
            "shown as 'fitR2' in every mode so you can see the gap.")
        gform.addRow("Validation:", self.cb_cv)

        row3 = QHBoxLayout()
        self.btn_tmpl = QPushButton("Save ground-truth template...")
        self.btn_tmpl.clicked.connect(self._save_template)
        self.btn_fit = QPushButton("Fit && validate all models")
        self.btn_fit.clicked.connect(self._run_fit)
        row3.addWidget(self.btn_tmpl); row3.addWidget(self.btn_fit)
        row3.addStretch()
        gform.addRow("", row3)
        v.addWidget(gbox)

        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        v.addWidget(self.progress)

        self.out = QTextEdit(); self.out.setReadOnly(True)
        self.out.setFontFamily("Consolas")
        v.addWidget(self.out, stretch=1)

    # ------------------------------------------------------------------
    def _pick_vnir(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Pick VNIR ENVI cube (.bin)", "",
            "ENVI cubes (*.bin *.img *.dat);;All files (*)")
        if p:
            self.ed_vnir.setText(p)
            state().vnir_path = p

    def _pick_gt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Pick ground-truth CSV (Plot_ID, fresh_kg[, dm_frac|dm_kg])",
            "", "CSV files (*.csv);;All files (*)")
        if p:
            self.ed_gt.setText(p)

    def _save_template(self):
        import pandas as pd
        s = state()
        plot_ids = None
        try:
            if s.out_csv and os.path.exists(s.out_csv):
                plot_ids = pd.read_csv(s.out_csv)["Plot_ID"].tolist()
            else:
                gp = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
                if gp and os.path.exists(gp):
                    plot_ids = load_grid(gp, target_crs=s.work_crs)["Plot_ID"].tolist()
        except Exception:
            plot_ids = None
        if not plot_ids:
            QMessageBox.warning(self, "No plots",
                                "Load a project / grid first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ground-truth template",
            os.path.join(os.path.dirname(s.out_csv or ""),
                         "biomass_ground_truth.csv"),
            "CSV files (*.csv)")
        if not path:
            return
        pd.DataFrame({"Plot_ID": plot_ids,
                      "fresh_kg": [""] * len(plot_ids),
                      "dm_frac": [""] * len(plot_ids),
                      "dm_kg": [""] * len(plot_ids)}).to_csv(path, index=False)
        QMessageBox.information(self, "Template saved",
            f"{path}\n\nFill fresh_kg (field-weighed wet strip weight per "
            "plot, kg) and either dm_frac (0-1, from oven-dried subsample) "
            "or dm_kg. Leave unsampled plots blank.")

    # ------------------------------------------------------------------
    def _run_vnir(self):
        s = state()
        vnir = self.ed_vnir.text().strip() or s.vnir_path
        if not vnir or not os.path.exists(vnir):
            QMessageBox.warning(self, "No VNIR cube",
                                "Pick the ENVI .bin file first.")
            return
        grid_path = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
        if not grid_path or not os.path.exists(grid_path):
            QMessageBox.warning(self, "No grid", "Load/align a grid first.")
            return
        s.vnir_path = vnir
        s.derive_default_paths()
        self.btn_vnir.setEnabled(False)
        self._wv = _VNIRWorker(vnir, grid_path, s.work_crs,
                               s.region_mode, s.band_width,
                               s.vnir_csv, s.vnir_spectra)
        self._wv.progress.connect(self._on_prog)
        self._wv.done_ok.connect(self._on_vnir_done)
        self._wv.error.connect(self._on_err)
        self._wv.start()

    def _run_fit(self):
        s = state()
        if not s.out_csv or not os.path.exists(s.out_csv):
            QMessageBox.warning(self, "No metrics",
                "Run Compute Traits first (needs h_mean - tick Mean height).")
            return
        gt = self.ed_gt.text().strip()
        if not gt or not os.path.exists(gt):
            QMessageBox.warning(self, "No ground truth",
                                "Pick the filled ground-truth CSV.")
            return
        base = os.path.splitext(s.out_csv)[0]
        cv_mode = ("loo", "kfold5", "none")[self.cb_cv.currentIndex()]
        mods = set()
        if self.cb_mod_lidar.isChecked():  mods.add("LiDAR")
        if self.cb_mod_vnir.isChecked():   mods.add("VNIR")
        if self.cb_mod_fusion.isChecked(): mods.add("Fusion")
        if not mods:
            QMessageBox.warning(self, "No modality",
                                "Tick at least one modality.")
            return
        enabled = {k for k, _, modality, _, _ in MODEL_SUITE
                   if modality in mods}
        self.btn_fit.setEnabled(False)
        self._wf = _FitWorker(s.out_csv, s.vnir_csv, s.vnir_spectra, gt,
                              base + "_biomass_predictions.csv",
                              base + "_biomass_models.json", cv_mode,
                              enabled)
        self._wf.progress.connect(self._on_prog)
        self._wf.done_ok.connect(self._on_fit_done)
        self._wf.error.connect(self._on_err)
        self._wf.start()

    # ------------------------------------------------------------------
    def _on_prog(self, p, m):
        self.progress.setValue(p)
        self.progress.setFormat(f"{p}% — {m}")

    def _on_vnir_done(self, msg):
        self.btn_vnir.setEnabled(True)
        self.out.append(msg + "\n")

    def _on_fit_done(self, msg):
        self.btn_fit.setEnabled(True)
        self.out.setPlainText(msg)

    def _on_err(self, err):
        self.btn_vnir.setEnabled(True)
        self.btn_fit.setEnabled(True)
        self.out.append(f"\nERROR: {err}")
        QMessageBox.critical(self, "Failed", err.split("\n")[0])
