"""
Project tab (PhenoFuse 4.0): one flight = LAS + grid + optional canopy raster, VNIR cube and RGB orthomosaic.
Also: SMRF tickbox, working CRS, flight / trial / stage names (used by the Dataset tab), project save/load.
Picking a different LAS moves every output path to the new flight's folder.
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QFileDialog, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox, QSizePolicy, QDoubleSpinBox, QComboBox
)

from phenoapp.core import (
    LASManager, build_canopy_raster, build_chm, load_grid, GRID_FILE_FILTER,
    quick_summary,
)
from phenoapp.core.project import state


class _LoadWorker(QThread):
    progress  = pyqtSignal(int, str)
    done_ok   = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, las_path, work_crs, use_smrf, canopy_tif,
                 canopy_hcut: float = 0.0, force_canopy: bool = False):
        super().__init__()
        self.las_path = las_path
        self.work_crs = work_crs
        self.use_smrf = use_smrf
        self.canopy_tif = canopy_tif
        self.canopy_hcut = canopy_hcut
        self.force_canopy = force_canopy

    def run(self):
        try:
            mgr = LASManager(self.las_path, self.work_crs, self.use_smrf)
            def cb(p, m): self.progress.emit(p, m)
            mgr.load(progress_cb=cb)
            self.progress.emit(100, "Building canopy raster...")
            build_canopy_raster(mgr, self.canopy_tif, progress_cb=cb,
                                height_cutoff=self.canopy_hcut,
                                force=self.force_canopy)
            if mgr.hag is not None:
                chm_tif = os.path.splitext(self.las_path)[0] + ".chm.tif"
                self.progress.emit(100, "Building Canopy Height Model...")
                build_chm(mgr, chm_tif, progress_cb=cb)
            self.done_ok.emit({"summary": quick_summary(mgr)})
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class ProjectTab(QWidget):
    project_ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._sync_from_state()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        gb_in = QGroupBox("Inputs of this flight")
        form = QFormLayout(gb_in)

        self.las_le = QLineEdit(); self.las_le.setReadOnly(True)
        btn_las = QPushButton("Browse..."); btn_las.clicked.connect(self._pick_las)
        form.addRow("Point cloud (.las/.laz):", self._h(self.las_le, btn_las))

        self.grid_le = QLineEdit(); self.grid_le.setReadOnly(True)
        btn_grid = QPushButton("Browse..."); btn_grid.clicked.connect(self._pick_grid)
        btn_no_grid = QPushButton("No grid yet - generate"); btn_no_grid.clicked.connect(self._no_grid)
        form.addRow("Grid file (.shp/.gpkg/.geojson):", self._h(self.grid_le, btn_grid, btn_no_grid))

        self.canopy_le = QLineEdit(); self.canopy_le.setReadOnly(True)
        btn_canopy = QPushButton("Browse..."); btn_canopy.clicked.connect(self._pick_canopy)
        form.addRow("Canopy raster (.tif, optional):", self._h(self.canopy_le, btn_canopy))

        self.vnir_le = QLineEdit(); self.vnir_le.setReadOnly(True)
        btn_vnir = QPushButton("Browse..."); btn_vnir.clicked.connect(self._pick_vnir)
        form.addRow("VNIR cube (ENVI .bin, optional):", self._h(self.vnir_le, btn_vnir))

        self.ortho_le = QLineEdit(); self.ortho_le.setReadOnly(True)
        btn_ortho = QPushButton("Browse..."); btn_ortho.clicked.connect(self._pick_ortho)
        form.addRow("RGB orthomosaic (.tif, optional):", self._h(self.ortho_le, btn_ortho))

        self.gt_le = QLineEdit(); self.gt_le.setReadOnly(True)
        btn_gt = QPushButton("Browse..."); btn_gt.clicked.connect(self._pick_gt)
        btn_gt_clear = QPushButton("Clear"); btn_gt_clear.clicked.connect(lambda: (self.gt_le.setText(""), setattr(state(), "gt_csv", "")))
        form.addRow("Ground-truth CSV (optional):", self._h(self.gt_le, btn_gt, btn_gt_clear))

        self.crs_le = QLineEdit("EPSG:7850")
        form.addRow("Working CRS (GDA2020 / MGA zone 50 = EPSG:7850):", self.crs_le)

        self.smrf_cb = QCheckBox("Use proper ground classification (SMRF) - slower but more accurate heights")
        self.smrf_cb.setChecked(False)
        form.addRow("", self.smrf_cb)

        canopy_row = QHBoxLayout()
        self.dsb_canopy_hcut = QDoubleSpinBox()
        self.dsb_canopy_hcut.setRange(0.0, 5.0); self.dsb_canopy_hcut.setDecimals(2); self.dsb_canopy_hcut.setSingleStep(0.05); self.dsb_canopy_hcut.setValue(0.0); self.dsb_canopy_hcut.setSuffix(" m")
        self.dsb_canopy_hcut.setToolTip("HAG threshold for the canopy raster. 0.0 includes ground points (best for sparse / young crops); raise to filter ground out.")
        self.btn_rebuild_canopy = QPushButton("Rebuild canopy raster"); self.btn_rebuild_canopy.clicked.connect(self._rebuild_canopy)
        canopy_row.addWidget(self.dsb_canopy_hcut); canopy_row.addWidget(self.btn_rebuild_canopy); canopy_row.addStretch()
        form.addRow("Canopy raster min height:", canopy_row)
        root.addWidget(gb_in)

        gb_fl = QGroupBox("Flight identity (used by the Dataset and modelling tabs)")
        ff = QFormLayout(gb_fl)
        self.flight_le = QLineEdit(); self.flight_le.setPlaceholderText("e.g. 20250930_NUE_I_DPIRD_Muresk_F_Gobi (derived from the LAS name)")
        self.trial_le = QLineEdit(); self.trial_le.setPlaceholderText("trial name shared by all flights of the same plots, e.g. NUE_I_DPIRD_Muresk_F_Gobi")
        self.stage_cb = QComboBox(); self.stage_cb.setEditable(True); self.stage_cb.addItems(["anthesis", "maturity", "tillering", "stem elongation", "booting", "grain filling", "other"])
        self.flight_le.editingFinished.connect(lambda: setattr(state(), "flight_name", self.flight_le.text().strip()))
        self.trial_le.editingFinished.connect(lambda: setattr(state(), "trial_name", self.trial_le.text().strip()))
        self.stage_cb.currentTextChanged.connect(lambda t: setattr(state(), "stage", t.strip()))
        ff.addRow("Flight name:", self.flight_le); ff.addRow("Trial:", self.trial_le); ff.addRow("Growth stage:", self.stage_cb)
        self.lbl_out = QLabel(""); self.lbl_out.setWordWrap(True); ff.addRow("Outputs go to:", self.lbl_out)
        root.addWidget(gb_fl)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load project (build canopy if needed)"); self.btn_load.clicked.connect(self._do_load); btn_row.addWidget(self.btn_load)
        self.btn_save_proj = QPushButton("Save project"); self.btn_save_proj.clicked.connect(self._save_project); btn_row.addWidget(self.btn_save_proj)
        self.btn_open_proj = QPushButton("Open project"); self.btn_open_proj.clicked.connect(self._open_project); btn_row.addWidget(self.btn_open_proj)
        btn_row.addStretch(); root.addLayout(btn_row)

        self.progress = QProgressBar(); self.progress.setTextVisible(True); root.addWidget(self.progress)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setPlaceholderText("Progress and status messages will appear here...")
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); root.addWidget(self.log, stretch=1)

    def _h(self, *widgets):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        for ww in widgets:
            h.addWidget(ww)
        return w

    # ------------------------------------------------------------------
    def _pick_las(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select LAS / LAZ", state().las_path or os.path.expanduser("~"), "Point clouds (*.las *.laz);;All files (*)")
        if path:
            s = state(); s.set_las(path)
            self._sync_from_state()
            self._log(f"Flight folder: {s.proj_dir}\n  LiDAR -> {s.out_dir}\n  VNIR  -> {s.plots_vnir_dir}\n  RGB   -> {s.plots_rgb_dir}\n  Models -> {s.models_dir}")
            self._guess_siblings()

    def _guess_siblings(self):
        """Suggest the VNIR cube and RGB orthomosaic that sit next to the LAS."""
        s = state()
        if not s.proj_dir or not os.path.isdir(s.proj_dir):
            return
        files = sorted(os.listdir(s.proj_dir))
        if not s.vnir_path:
            for fn in files:
                if fn.lower().endswith(".bin") and "vnir" in fn.lower():
                    s.vnir_path = os.path.join(s.proj_dir, fn); break
        if not s.ortho_tif:
            for fn in files:
                lo = fn.lower()
                if lo.endswith((".tif", ".tiff")) and "rgb" in lo and "canopy" not in lo and ".rgb." not in lo and "ortho" in lo:
                    s.ortho_tif = os.path.join(s.proj_dir, fn); break
        self._sync_from_state()

    def _pick_grid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select grid file", state().grid_path or os.path.expanduser("~"), GRID_FILE_FILTER)
        if path:
            self.grid_le.setText(path); state().grid_path = path

    def _no_grid(self):
        QMessageBox.information(self, "Generate a grid", "Switch to the 'Generate Grid' tab to create one by clicking 4 corners on the canopy image.\n\nLoad your point cloud here first, then go there.")

    def _pick_canopy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pre-built canopy raster", state().canopy_tif or os.path.expanduser("~"), "GeoTIFF (*.tif *.tiff);;All files (*)")
        if path:
            self.canopy_le.setText(path); state().canopy_tif = path

    def _pick_vnir(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select VNIR ENVI cube (.bin)", state().vnir_path or state().proj_dir or os.path.expanduser("~"), "ENVI cubes (*.bin *.img *.dat);;All files (*)")
        if path:
            self.vnir_le.setText(path); state().vnir_path = path

    def _pick_ortho(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select RGB orthomosaic", state().ortho_tif or state().proj_dir or os.path.expanduser("~"), "GeoTIFF (*.tif *.tiff);;All files (*)")
        if path:
            self.ortho_le.setText(path); state().ortho_tif = path

    def _pick_gt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ground-truth CSV (Plot_ID + biomass / height columns)", state().gt_csv or state().proj_dir or os.path.expanduser("~"), "CSV (*.csv);;All files (*)")
        if path:
            self.gt_le.setText(path); state().gt_csv = path

    # ------------------------------------------------------------------
    def _do_load(self):
        las = self.las_le.text().strip()
        if not las or not os.path.exists(las):
            QMessageBox.warning(self, "Missing LAS", "Pick a valid LAS file first.")
            return
        s = state()
        if las != s.las_path:
            s.set_las(las)
        s.work_crs = self.crs_le.text().strip() or "EPSG:7850"
        s.use_smrf = self.smrf_cb.isChecked()
        s.canopy_tif = self.canopy_le.text().strip() or s.canopy_tif
        s.grid_path = self.grid_le.text().strip()
        s.flight_name = self.flight_le.text().strip() or s.flight_name
        s.trial_name = self.trial_le.text().strip(); s.stage = self.stage_cb.currentText().strip()
        s.derive_default_paths()
        self._sync_from_state()
        self._log_clear(); self._log("Starting load..."); self.progress.setValue(0); self.btn_load.setEnabled(False)
        self._worker = _LoadWorker(s.las_path, s.work_crs, s.use_smrf, s.canopy_tif, canopy_hcut=self.dsb_canopy_hcut.value(), force_canopy=False)
        self._worker.progress.connect(self._on_progress); self._worker.done_ok.connect(self._on_done); self._worker.error.connect(self._on_error)
        self._worker.start()

    def _rebuild_canopy(self):
        s = state()
        if not s.las_path or not os.path.exists(s.las_path):
            QMessageBox.warning(self, "No project", "Load a project first (LAS + grid + canopy)."); return
        if not s.canopy_tif:
            s.derive_default_paths()
        self.progress.setValue(0); self.log.append(f"[rebuild] Rebuilding canopy raster with min height {self.dsb_canopy_hcut.value()} m ...")
        self.btn_rebuild_canopy.setEnabled(False)
        self._worker = _LoadWorker(s.las_path, s.work_crs, s.use_smrf, s.canopy_tif, canopy_hcut=self.dsb_canopy_hcut.value(), force_canopy=True)
        self._worker.progress.connect(self._on_progress); self._worker.done_ok.connect(self._on_rebuild_done); self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_rebuild_done(self, summary):
        self.btn_rebuild_canopy.setEnabled(True); self.log.append("[rebuild] Done. Switch tabs or click Reload to see the updated canopy.")
        self.progress.setFormat("Canopy raster rebuilt"); self.project_ready.emit()

    def _on_progress(self, pct, msg):
        self.progress.setValue(int(pct)); self.progress.setFormat(f"{int(pct)}% - {msg}"); self._log(f"[{pct:3d}%] {msg}")

    def _on_done(self, info):
        self.progress.setValue(100); self.btn_load.setEnabled(True)
        s = info.get("summary", {}); self._log("\n--- LOADED ---")
        for k, v in s.items():
            self._log(f"{k:>10}: {v}")
        state().las_loaded = True; state().canopy_built = True
        self.project_ready.emit()
        QMessageBox.information(self, "Loaded", "Project loaded successfully. You can now switch to other tabs.")

    def _on_error(self, err):
        self.btn_load.setEnabled(True); self.btn_rebuild_canopy.setEnabled(True); self._log(f"\nERROR: {err}")
        QMessageBox.critical(self, "Load failed", err)

    # ------------------------------------------------------------------
    def _save_project(self):
        s = state()
        path, _ = QFileDialog.getSaveFileName(self, "Save project", s.project_file or os.path.join(s.proj_dir or "", f"{s.flight_name or 'project'}.phenoproj"),
                                              "Phenotyping project (*.phenoproj);;All files (*)")
        if not path: return
        s.grid_path = self.grid_le.text().strip(); s.canopy_tif = self.canopy_le.text().strip(); s.work_crs = self.crs_le.text().strip(); s.use_smrf = self.smrf_cb.isChecked()
        s.flight_name = self.flight_le.text().strip() or s.flight_name; s.trial_name = self.trial_le.text().strip(); s.stage = self.stage_cb.currentText().strip()
        try:
            s.save(path); self._log(f"Saved project to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", state().project_file or os.path.expanduser("~"), "Phenotyping project (*.phenoproj *.json);;All files (*)")
        if not path: return
        try:
            from phenoapp.core.project import set_state, ProjectState
            st = ProjectState.load(path); st.derive_default_paths(); set_state(st)
            self._sync_from_state(); self._log(f"Loaded project from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))

    # ------------------------------------------------------------------
    def _sync_from_state(self):
        s = state()
        self.las_le.setText(s.las_path); self.grid_le.setText(s.grid_path); self.canopy_le.setText(s.canopy_tif)
        self.vnir_le.setText(s.vnir_path); self.ortho_le.setText(s.ortho_tif); self.gt_le.setText(s.gt_csv)
        self.crs_le.setText(s.work_crs); self.smrf_cb.setChecked(s.use_smrf)
        self.flight_le.setText(s.flight_name); self.trial_le.setText(s.trial_name); self.stage_cb.setCurrentText(s.stage or "anthesis")
        self.lbl_out.setText((f"{s.out_dir}  |  {s.plots_vnir_dir}  |  {s.plots_rgb_dir}  |  {s.models_dir}") if s.las_path else "(pick a LAS file)")

    def showEvent(self, ev):
        super().showEvent(ev); self._sync_from_state()

    def _log(self, msg: str):
        self.log.append(msg)

    def _log_clear(self):
        self.log.clear()
