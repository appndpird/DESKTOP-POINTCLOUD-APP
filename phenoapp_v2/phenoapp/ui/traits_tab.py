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
from phenoapp.core.units import AUTO_LABEL, UNIT_LABELS, UNIT_KEYS
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
                ground_mode=kw.get("ground_mode", "hag"),
                region_mode=kw.get("region_mode", "whole"),
                band_width=kw.get("band_width", 0.5),
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
                    "vol_voxel", "biomass_pvi", "biomass_kg", "biomass_kg_ha",
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

        # Ground reference. "Exterior surface" fits a smooth ground model from
        # points OUTSIDE the plots (alleys/tracks) — validated best for dense
        # pasture where the laser never reaches soil inside the plots.
        self.cb_ground = QComboBox()
        self.cb_ground.addItems([
            "SMRF / cached HAG (project setting)",
            "Exterior ground surface — recommended for pasture",
        ])
        self.cb_ground.setToolTip(
            "How height-above-ground is computed.\n\n"
            "SMRF/HAG: PDAL SMRF ground classification (good for row crops "
            "with visible soil between rows).\n\n"
            "Exterior surface: fits a smooth surface to ground observed "
            "OUTSIDE the plot polygons (mowed alleys, tracks) and evaluates "
            "it under the plots. On dense closed swards SMRF and per-plot "
            "baselines ride the canopy bottom and can invert biomass "
            "relationships; the exterior surface avoids that entirely.")

        # Sampling region. Central band matches a mower-strip ground truth and
        # avoids plot-edge effects; validated better than whole-plot means.
        region_row = QHBoxLayout()
        self.cb_region = QComboBox()
        self.cb_region.addItems(["Whole plot (10 cm inset)",
                                 "Central band (recommended)"])
        self.dsb_band = QDoubleSpinBox()
        self.dsb_band.setRange(0.2, 5.0); self.dsb_band.setValue(0.5)
        self.dsb_band.setSingleStep(0.1); self.dsb_band.setSuffix(" m wide")
        self.cb_region.setToolTip(
            "Where inside each plot the traits are computed.\n\n"
            "Central band: a band along the plot's long axis. Matches "
            "cut-and-weigh strip ground truth and avoids edge overhang. "
            "Do NOT cherry-pick 'best-looking' areas — ground truth is cut "
            "indiscriminately, so sample representatively.")
        region_row.addWidget(self.cb_region, stretch=2)
        region_row.addWidget(self.dsb_band, stretch=1)

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
            "Fit k (kg per m^3 of canopy) by least squares from a ground-truth CSV.\n"
            "Required columns: Plot_ID and a biomass column in ANY unit -\n"
            "kg/ha, t/ha, g/m^2, kg/m^2 or kg per plot (choose the unit on the\n"
            "right, or name the column biomass_kg_ha / biomass_kg and use Auto).\n"
            "The fit is per m^2, so k does not depend on plot size.\n"
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
            "Write a ready-to-fill ground-truth CSV (Plot_ID, biomass_kg_ha) "
            "pre-populated with your plot IDs. Enter biomass in kg/ha, or "
            "rename the column to the unit you have (biomass_t_ha, "
            "biomass_g_m2, biomass_kg for kg per plot) and feed it back to "
            "the fitters.")
        self.btn_fit_multi = QPushButton("Fit multi-metric model from CSV...")
        self.btn_fit_multi.clicked.connect(self._on_fit_multi)
        self.btn_fit_multi.setToolTip(
            "Fit a multiple-regression biomass model\n"
            "  biomass_kg_ha = b0 + b1*h_p95 + b2*cover_frac + b3*vol_voxel/m^2\n"
            "from a ground-truth CSV (Plot_ID + biomass in any unit), then\n"
            "write biomass_pred_kg_ha and biomass_pred_kg for every plot.\n"
            "Per-plot volumes are divided by plot area so the model is a\n"
            "density and transfers between trials. Tick h_p95, cover_frac\n"
            "and vol_voxel and run Compute Traits first.")
        self.cb_gt_unit = QComboBox()
        self.cb_gt_unit.addItem(AUTO_LABEL)
        for k in UNIT_KEYS:
            self.cb_gt_unit.addItem(UNIT_LABELS[k])
        self.cb_gt_unit.setToolTip(
            "Unit of the biomass column in your ground-truth CSV.\n"
            "Auto-detect reads it from the column name (biomass_kg_ha, "
            "biomass_t_ha, biomass_g_m2, biomass_kg_m2, biomass_kg = kg per "
            "plot). Pick a unit explicitly if the column is just 'biomass'.\n"
            "Both fitters convert to kg/m^2 internally, so k and the model "
            "coefficients are independent of plot size.")
        model_row.addWidget(QLabel("Ground-truth unit:"))
        model_row.addWidget(self.cb_gt_unit, stretch=1)
        model_row.addWidget(self.btn_template)
        model_row.addWidget(self.btn_fit_multi)
        model_row.addStretch()

        self.cb_writelas = QCheckBox("Also write per-plot LAS files")
        self.cb_writelas.setChecked(True)
        f.addRow("Canopy height cutoff:", self.dsb_hcut)
        f.addRow("Voxel size for volume:", self.dsb_vox)
        f.addRow("Ground reference:", self.cb_ground)
        f.addRow("Sampling region:", region_row)
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

    def _gt_unit(self) -> str:
        i = self.cb_gt_unit.currentIndex()
        return "auto" if i <= 0 else UNIT_KEYS[i - 1]

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
            self, "Pick ground-truth CSV (Plot_ID + biomass column, any unit)",
            os.path.dirname(s.out_csv or ""), "CSV files (*.csv);;All files (*)")
        if not gt_path:
            return
        try:
            r = fit_biomass_k(s.out_csv, gt_path, gt_unit=self._gt_unit())
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return
        self.dsb_biomass_k.setValue(r["k"])
        self.cb_biomass_preset.setCurrentIndex(0)   # "Custom"
        QMessageBox.information(self, "Fit complete",
            f"Fitted k = {r['k']:.4f} kg per m^3 of canopy\n"
            f"(biomass kg/m^2 = k x cover_frac x h_p95; independent of plot size)\n\n"
            f"Ground truth: column '{r['gt_col']}' read as {r['unit_label']}\n"
            f"n = {r['n']} plots matched\n"
            f"R^2 = {r['r2']:.3f}\n"
            f"RMSE = {r['rmse_kg_ha']:.0f} kg/ha   (leave-one-out {r['loocv_rmse_kg_ha']:.0f} kg/ha)\n"
            f"Canopy depth range: {r['depth_range'][0]:.3f} .. {r['depth_range'][1]:.3f} m\n"
            f"GT range: {r['gt_range_kg_ha'][0]:.0f} .. {r['gt_range_kg_ha'][1]:.0f} kg/ha\n"
            f"Plot region area: {r['area_range_m2'][0]:.2f} .. {r['area_range_m2'][1]:.2f} m^2\n\n"
            f"Click Compute Traits to apply k to all plots "
            f"(biomass_kg per plot and biomass_kg_ha).")

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
            pd.DataFrame({"Plot_ID": plot_ids, "biomass_kg_ha": [""] * len(plot_ids)}) \
                .to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Template saved",
            f"Template written to:\n{path}\n\n"
            f"{len(plot_ids)} plot rows.\n\n"
            "Fill the 'biomass_kg_ha' column with biomass in kg/ha (quadrat "
            "cut scaled to a hectare). If your data are in another unit, "
            "rename the column: biomass_t_ha, biomass_g_m2, biomass_kg_m2, or "
            "biomass_kg for a whole-plot weight in kg - or pick the unit in "
            "the 'Ground-truth unit' box. Leave rows you didn't sample blank, "
            "then use 'Fit k from CSV...' or 'Fit multi-metric model from CSV...'.")

    def _on_fit_multi(self):
        """Fit a multiple-regression biomass model and write predictions."""
        s = state()
        if not s.out_csv or not os.path.exists(s.out_csv):
            QMessageBox.warning(self, "No metrics CSV yet",
                "Run Compute Traits first (tick h_p95, cover_frac and vol_voxel) "
                "so the metrics CSV with those predictor columns exists.")
            return
        gt_path, _ = QFileDialog.getOpenFileName(
            self, "Pick ground-truth CSV (Plot_ID + biomass column, any unit)",
            os.path.dirname(s.out_csv or ""), "CSV files (*.csv);;All files (*)")
        if not gt_path:
            return
        try:
            model = fit_biomass_multi(s.out_csv, gt_path,
                                      predictors=BIOMASS_MODEL_PREDICTORS,
                                      gt_unit=self._gt_unit())
            n_pred = apply_biomass_multi(s.out_csv, model)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            return
        QMessageBox.information(self, "Multi-metric model fitted",
            f"{model['equation']}\n"
            f"(predictors marked /m^2 are divided by plot area; model is a density)\n\n"
            f"Ground truth: column '{model['gt_col']}' read as {UNIT_LABELS[model['gt_unit']]}\n"
            f"n = {model['n']} plots matched\n"
            f"R^2 = {model['r2']:.3f}   (adjusted R^2 = {model['adj_r2']:.3f})\n"
            f"RMSE = {model['rmse']:.0f} kg/ha\n"
            f"Leave-one-out: RMSE = {model['loocv_rmse']:.0f} kg/ha, R^2 = {model['loocv_r2']:.3f}\n\n"
            f"Wrote 'biomass_pred_kg_ha' and 'biomass_pred_kg' for {n_pred} plots into:\n{s.out_csv}\n\n"
            "Open the Statistics tab to visualize biomass_pred_kg_ha.")

    def _sel_all(self):
        for cb in self._checks.values(): cb.setChecked(True)
    def _desel_all(self):
        for cb in self._checks.values(): cb.setChecked(False)
    def _defaults(self):
        defaults = {"h_p95", "h_p99", "h_mean", "h_std", "cover_frac",
                    "vol_voxel", "biomass_pvi", "biomass_kg", "biomass_kg_ha",
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

        ground_mode = ("exterior" if self.cb_ground.currentIndex() == 1
                       else "hag")
        region_mode = ("central" if self.cb_region.currentIndex() == 1
                       else "whole")
        # persist choices in the project state for the Biomass tab
        s.ground_mode = ground_mode
        s.region_mode = region_mode
        s.band_width = self.dsb_band.value()

        kw = dict(
            las_path=s.las_path, work_crs=s.work_crs, use_smrf=s.use_smrf,
            grid_path=grid_path, out_dir=s.out_dir, out_csv=s.out_csv,
            selected=selected, write_las=self.cb_writelas.isChecked(),
            h_cut=self.dsb_hcut.value(), vox=self.dsb_vox.value(),
            biomass_k=self.dsb_biomass_k.value(),
            ground_mode=ground_mode, region_mode=region_mode,
            band_width=self.dsb_band.value(),
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
