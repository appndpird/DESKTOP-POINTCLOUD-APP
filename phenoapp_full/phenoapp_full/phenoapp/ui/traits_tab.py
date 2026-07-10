"""
Traits tab: choose physical traits to compute, run extraction, save CSV.
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QPushButton,
    QLabel, QProgressBar, QFileDialog, QMessageBox, QFormLayout,
    QDoubleSpinBox, QGridLayout, QScrollArea, QTextEdit, QSizePolicy,
    QComboBox
)


# Suggested biomass coefficients from UAV-lidar phenotyping literature.
# Multiply biomass_pvi (m^3) by k to get biomass in kg per plot.
BIOMASS_K_PRESETS = {
    "Custom (use the spinbox)": None,
    "None - output PVI in m^3 (k = 1.0)":         1.0,
    "Wheat - dry matter (~k = 0.28)":            0.28,
    "Barley - dry matter (~k = 0.25)":           0.25,
    "Fodder grass - fresh weight (~k = 0.10)":   0.10,
    "Pasture - fresh weight (~k = 0.10)":        0.10,
}

from phenoapp.core.project import state
from phenoapp.core import (
    LASManager, load_grid, extract_all_plots,
    TRAITS_CATALOG, TRAIT_KEYS,
    fit_biomass_multi, apply_biomass_multi, BIOMASS_MODEL_PREDICTORS,
)


class _ExtractWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok  = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self._kw = kwargs
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            kw = self._kw
            mgr = LASManager(kw["las_path"], kw["work_crs"], kw["use_smrf"])
            self.progress.emit(2, "Loading LAS...")
            def cb(p, m): self.progress.emit(int(2 + p * 0.18), m)
            mgr.load(progress_cb=cb)

            self.progress.emit(20, "Loading grid...")
            plots = load_grid(kw["grid_path"], target_crs=kw["work_crs"])
            self.progress.emit(25, f"{len(plots)} plots ready")

            def pcb(p, m):
                self.progress.emit(int(25 + p * 0.74), m)

            df = extract_all_plots(
                mgr, plots,
                out_dir=kw["out_dir"], out_csv=kw["out_csv"],
                selected_traits=kw["selected"],
                write_las=kw["write_las"],
                height_cut=kw["h_cut"], voxel=kw["vox"],
                biomass_k=kw.get("biomass_k", 1.0),
                progress_cb=pcb, cancel_flag=lambda: self._cancel,
            )
            self.progress.emit(100, "Done")

            empty = int((df["n_points"] == 0).sum()) if "n_points" in df else 0
            non = int(len(df) - empty)
            med = 0
            if non:
                med = int(df.loc[df["n_points"] > 0, "n_points"].median())
            self.done_ok.emit({"non": non, "empty": empty, "total": len(df),
                               "median_pts": med, "out_csv": kw["out_csv"],
                               "out_dir": kw["out_dir"]})
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class TraitsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._checks = {}      # key -> QCheckBox
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Compute physical traits per plot</h2>"))
        v.addWidget(QLabel(
            "Pick which traits you want. Recommended defaults are pre-selected. "
            "See the Guidance tab for trait definitions and methods."))

        # ---- Trait selection ----
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        gl = QGridLayout(inner)

        # Group by category
        by_group = {}
        for key, group, label, desc in TRAITS_CATALOG:
            by_group.setdefault(group, []).append((key, label, desc))

        # Recommended defaults
        defaults = {"h_p95", "h_p99", "h_mean", "h_std", "cover_frac",
                    "vol_voxel", "biomass_pvi", "biomass_kg",
                    "lodging_angle", "roughness",
                    "h_ground_p1", "h_max_local", "h_p95_local",
                    "n_points", "n_canopy"}

        col = 0
        for group, items in by_group.items():
            box = QGroupBox(group)
            f = QVBoxLayout(box)
            for key, label, desc in items:
                cb = QCheckBox(label)
                cb.setToolTip(desc)
                if key in defaults:
                    cb.setChecked(True)
                self._checks[key] = cb
                f.addWidget(cb)
            f.addStretch()
            gl.addWidget(box, 0, col)
            col += 1
        v.addWidget(scroll, stretch=1)

        # ---- Quick selectors ----
        h = QHBoxLayout()
        b1 = QPushButton("Select All");          b1.clicked.connect(self._sel_all)
        b2 = QPushButton("Recommended Defaults"); b2.clicked.connect(self._defaults)
        b3 = QPushButton("Deselect All");        b3.clicked.connect(self._desel_all)
        h.addWidget(b1); h.addWidget(b2); h.addWidget(b3); h.addStretch()
        v.addLayout(h)

        # ---- Parameters ----
        params = QGroupBox("Parameters")
        f = QFormLayout(params)
        self.dsb_hcut = QDoubleSpinBox(); self.dsb_hcut.setRange(0.0, 5.0); self.dsb_hcut.setValue(0.15); self.dsb_hcut.setSuffix(" m")
        self.dsb_vox  = QDoubleSpinBox(); self.dsb_vox.setRange(0.01, 1.0); self.dsb_vox.setValue(0.05); self.dsb_vox.setSuffix(" m")

        # Biomass calibration: preset crop dropdown + manual k spinbox +
        # "Fit from CSV..." button (least-squares fit using ground-truth weights).
        bio_row = QHBoxLayout()
        self.cb_biomass_preset = QComboBox()
        self.cb_biomass_preset.addItems(list(BIOMASS_K_PRESETS.keys()))
        self.cb_biomass_preset.currentTextChanged.connect(self._on_biomass_preset)
        self.dsb_biomass_k = QDoubleSpinBox()
        self.dsb_biomass_k.setRange(0.0, 1000.0)
        self.dsb_biomass_k.setDecimals(4)
        self.dsb_biomass_k.setSingleStep(0.01)
        self.dsb_biomass_k.setValue(1.0)
        self.dsb_biomass_k.setSuffix(" kg/m^3 PVI")
        self.btn_fit_k = QPushButton("Fit k from CSV...")
        self.btn_fit_k.clicked.connect(self._on_fit_k)
        self.btn_fit_k.setToolTip(
            "Fit k by least squares from a ground-truth CSV.\n"
            "Required columns: Plot_ID, biomass_kg.\n"
            "Run Compute Traits first so the metrics CSV (with biomass_pvi) exists.")
        bio_row.addWidget(self.cb_biomass_preset, stretch=2)
        bio_row.addWidget(self.dsb_biomass_k,     stretch=1)
        bio_row.addWidget(self.btn_fit_k)

        # Multi-metric biomass model (multiple regression on LiDAR predictors)
        # and a one-click ground-truth CSV template.
        model_row = QHBoxLayout()
        self.btn_template = QPushButton("Save biomass template CSV...")
        self.btn_template.clicked.connect(self._on_save_template)
        self.btn_template.setToolTip(
            "Write a ready-to-fill ground-truth CSV (Plot_ID, biomass_kg) "
            "pre-populated with your plot IDs, so you can enter harvested "
            "weights and feed it back to the fitters.")
        self.btn_fit_multi = QPushButton("Fit multi-metric model from CSV...")
        self.btn_fit_multi.clicked.connect(self._on_fit_multi)
        self.btn_fit_multi.setToolTip(
            "Fit a multiple-regression biomass model\n"
            "  biomass_kg = b0 + b1*h_p95 + b2*cover_frac + b3*vol_voxel\n"
            "from a ground-truth CSV (Plot_ID, biomass_kg), then write a\n"
            "biomass_pred_kg column for every plot. Usually more accurate\n"
            "than the single-k PVI model. Tick h_p95, cover_frac and\n"
            "vol_voxel and run Compute Traits first.")
        model_row.addWidget(self.btn_template)
        model_row.addWidget(self.btn_fit_multi)
        model_row.addStretch()

        self.cb_writelas = QCheckBox("Also write per-plot LAS files")
        self.cb_writelas.setChecked(True)
        f.addRow("Canopy height cutoff:", self.dsb_hcut)
        f.addRow("Voxel size for volume:", self.dsb_vox)
        f.addRow("Biomass calibration k:", bio_row)
        f.addRow("Multi-metric biomass:", model_row)
        f.addRow("", self.cb_writelas)
        v.addWidget(params)

        # ---- Run + progress ----
        h2 = QHBoxLayout()
        self.btn_run = QPushButton("▶  Compute Traits")
        self.btn_run.clicked.connect(self._run)
        h2.addWidget(self.btn_run); h2.addStretch()
        v.addLayout(h2)

        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        v.addWidget(self.progress)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        v.addWidget(self.log)

    def _on_biomass_preset(self, label):
        val = BIOMASS_K_PRESETS.get(label)
        if val is not None:
            self.dsb_biomass_k.setValue(val)

    def _on_fit_k(self):
        """Fit k by least squares from a user-supplied ground-truth CSV."""
        from phenoapp.core import fit_biomass_k
        s = state()
        if not s.out_csv or not os.path.exists(s.out_csv):
            QMessageBox.warning(self, "No metrics CSV yet",
                "Run Compute Traits first so the metrics CSV exists "
                "(it must contain a biomass_pvi column).")
            return
        gt_path, _ = QFileDialog.getOpenFileName(
            self, "Pick ground-truth CSV (must have Plot_ID + biomass_kg columns)",
            os.path.dirname(s.out_csv or ""), "CSV files (*.csv);;All files (*)")
        if not gt_path:
            return
        try:
            r = fit_biomass_k(s.out_csv, gt_path)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return
        self.dsb_biomass_k.setValue(r["k"])
        self.cb_biomass_preset.setCurrentIndex(0)   # "Custom"
        QMessageBox.information(self, "Fit complete",
            f"Fitted k = {r['k']:.4f} kg per m^3 PVI\n"
            f"n = {r['n']} plots matched\n"
            f"R^2 = {r['r2']:.3f}\n"
            f"RMSE = {r['rmse']:.3f} kg\n"
            f"PVI range:  {r['pvi_range'][0]:.2f} .. {r['pvi_range'][1]:.2f}\n"
            f"GT range:   {r['gt_range'][0]:.2f} .. {r['gt_range'][1]:.2f}\n\n"
            f"Click Compute Traits to apply k to all plots.")

    def _on_save_template(self):
        """Write a ground-truth CSV template pre-filled with plot IDs."""
        import pandas as pd
        s = state()
        # Prefer real plot IDs from the metrics CSV or the grid; fall back to a
        # short numbered stub so the user always gets a usable file.
        plot_ids = None
        try:
            if s.out_csv and os.path.exists(s.out_csv):
                plot_ids = pd.read_csv(s.out_csv)["Plot_ID"].tolist()
            else:
                grid_path = (s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid))
                             else s.grid_path)
                if grid_path and os.path.exists(grid_path):
                    plot_ids = load_grid(grid_path, target_crs=s.work_crs)["Plot_ID"].tolist()
        except Exception:
            plot_ids = None
        if not plot_ids:
            plot_ids = list(range(1, 11))

        start_dir = os.path.dirname(s.out_csv) if s.out_csv else os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save biomass ground-truth template",
            os.path.join(start_dir, "biomass_ground_truth_template.csv"),
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            pd.DataFrame({"Plot_ID": plot_ids, "biomass_kg": [""] * len(plot_ids)}) \
                .to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Template saved",
            f"Template written to:\n{path}\n\n"
            f"{len(plot_ids)} plot rows.\n\n"
            "Fill the 'biomass_kg' column with your harvested (cut-and-weigh) "
            "weight for each plot, leave rows you didn't sample blank, then use "
            "'Fit k from CSV...' or 'Fit multi-metric model from CSV...'.")

    def _on_fit_multi(self):
        """Fit a multiple-regression biomass model and write predictions."""
        s = state()
        if not s.out_csv or not os.path.exists(s.out_csv):
            QMessageBox.warning(self, "No metrics CSV yet",
                "Run Compute Traits first (tick h_p95, cover_frac and vol_voxel) "
                "so the metrics CSV with those predictor columns exists.")
            return
        gt_path, _ = QFileDialog.getOpenFileName(
            self, "Pick ground-truth CSV (Plot_ID + biomass_kg columns)",
            os.path.dirname(s.out_csv or ""), "CSV files (*.csv);;All files (*)")
        if not gt_path:
            return
        try:
            model = fit_biomass_multi(s.out_csv, gt_path,
                                      predictors=BIOMASS_MODEL_PREDICTORS)
            n_pred = apply_biomass_multi(s.out_csv, model)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return
        QMessageBox.information(self, "Multi-metric model fitted",
            f"{model['equation']}\n\n"
            f"n = {model['n']} plots matched\n"
            f"R^2 = {model['r2']:.3f}   (adjusted R^2 = {model['adj_r2']:.3f})\n"
            f"RMSE = {model['rmse']:.3f} kg\n\n"
            f"Wrote 'biomass_pred_kg' for {n_pred} plots into:\n{s.out_csv}\n\n"
            "Open the Statistics tab to visualize biomass_pred_kg.")

    def _sel_all(self):
        for cb in self._checks.values(): cb.setChecked(True)
    def _desel_all(self):
        for cb in self._checks.values(): cb.setChecked(False)
    def _defaults(self):
        defaults = {"h_p95", "h_p99", "h_mean", "h_std", "cover_frac",
                    "vol_voxel", "biomass_pvi", "biomass_kg",
                    "lodging_angle", "roughness",
                    "h_ground_p1", "h_max_local", "h_p95_local",
                    "n_points", "n_canopy"}
        for k, cb in self._checks.items():
            cb.setChecked(k in defaults)

    def _run(self):
        s = state()
        if not s.las_path or not os.path.exists(s.las_path):
            QMessageBox.warning(self, "No LAS", "Load a project on the Project tab first.")
            return
        grid_path = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
        if not grid_path or not os.path.exists(grid_path):
            QMessageBox.warning(self, "No grid",
                "Save an aligned grid (Edit tab) or load one on the Project tab.")
            return
        selected = [k for k, cb in self._checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Pick something", "Select at least one trait.")
            return

        # ensure output paths
        s.derive_default_paths()

        kw = dict(
            las_path=s.las_path, work_crs=s.work_crs, use_smrf=s.use_smrf,
            grid_path=grid_path, out_dir=s.out_dir, out_csv=s.out_csv,
            selected=selected, write_las=self.cb_writelas.isChecked(),
            h_cut=self.dsb_hcut.value(), vox=self.dsb_vox.value(),
            biomass_k=self.dsb_biomass_k.value(),
        )
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self._w = _ExtractWorker(kw)
        self._w.progress.connect(self._on_progress)
        self._w.done_ok.connect(self._on_done)
        self._w.error.connect(self._on_error)
        self._w.start()

    def _on_progress(self, p, msg):
        self.progress.setValue(p)
        self.progress.setFormat(f"{p}% — {msg}")
        self.log.append(f"[{p:3d}%] {msg}")

    def _on_done(self, r):
        self.btn_run.setEnabled(True)
        self.log.append("\n--- DONE ---")
        self.log.append(f"CSV : {r['out_csv']}")
        self.log.append(f"LAS dir: {r['out_dir']}")
        self.log.append(f"Plots with points: {r['non']}/{r['total']}")
        self.log.append(f"Empty plots      : {r['empty']}")
        self.log.append(f"Median pts/plot  : {r['median_pts']:,}")
        QMessageBox.information(self, "Complete",
            f"Traits CSV written to:\n{r['out_csv']}\n\n"
            f"{r['non']}/{r['total']} plots had points "
            f"(median {r['median_pts']:,} pts/plot).\n\n"
            "Switch to the Statistics tab to visualize.")

    def _on_error(self, err):
        self.btn_run.setEnabled(True)
        self.log.append(f"\nERROR: {err}")
        QMessageBox.critical(self, "Failed", err)
