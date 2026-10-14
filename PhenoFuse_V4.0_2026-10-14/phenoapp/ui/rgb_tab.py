"""
RGB Features tab (PhenoFuse 4.0): per-plot crops of the RGB orthomosaic with the same grid as the LiDAR and VNIR
tabs, and per-plot RGB features (rgb_features.csv). Outputs go to <flight folder>/plots_rgb/.
"""
from __future__ import annotations
import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QMessageBox, QLineEdit, QTextEdit,
                             QCheckBox, QFormLayout, QComboBox, QHBoxLayout, QDoubleSpinBox)

from phenoapp.core.project import state
from phenoapp.core import load_grid
from phenoapp.ui.common import file_row


class _Worker(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)

    def __init__(self, ortho, grid, crs, out_dir, out_csv, write_tifs, region_mode, band_width):
        super().__init__(); self._a = (ortho, grid, crs, out_dir, out_csv, write_tifs, region_mode, band_width)

    def run(self):
        try:
            from phenoapp.core.rgb_features import extract_rgb_plots
            ortho, grid, crs, out_dir, out_csv, write_tifs, region_mode, band_width = self._a
            plots = load_grid(grid, target_crs=crs)
            df, idx = extract_rgb_plots(ortho, plots, out_dir, write_tifs=write_tifs, region_mode=region_mode, band_width=band_width,
                                        progress_cb=lambda p, m: self.progress.emit(p, m))
            os.makedirs(os.path.dirname(out_csv), exist_ok=True); df.to_csv(out_csv, index=False)
            ok = int(df.get("rgb_qc_ok", 0).sum()) if "rgb_qc_ok" in df else 0
            msg = (f"RGB features: {df.shape[1]} columns for {len(df)} plots -> {out_csv}\n"
                   f"Plots passing RGB QC (>= 90 % of the region inside the orthomosaic): {ok}/{len(df)}\n")
            if "rgb_veg_threshold" in df:
                msg += f"Vegetation mask: ExG > {float(df.rgb_veg_threshold.iloc[0]):.3f} (Otsu on the whole flight, bounded 0.02-0.25); median RGB cover {df.rgb_cover.median():.2f}\n"
            if write_tifs:
                msg += f"Per-plot crops: {len(idx)} GeoTIFFs (exact pixel copies, lossless) -> {out_dir}\nIndex -> {os.path.join(out_dir, 'plots_rgb_index.csv')}"
            self.progress.emit(100, "RGB done"); self.done_ok.emit(msg)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class RGBTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>RGB Features: per-plot orthomosaic crops and RGB indices</h2>"))
        v.addWidget(QLabel("Uses the same plot grid as the LiDAR and VNIR tabs. Every plot is cut from the RGB orthomosaic on the orthomosaic's own pixel "
                           "grid (no resampling; dtype, CRS and pixel values identical to the source; lossless DEFLATE; polygon as an internal mask) "
                           "into <flight folder>/plots_rgb/, and rgb_features.csv holds the per-plot RGB indices (ExG, ExGR, GLI, VARI, NGRDI, "
                           "MGRVI, RGBVI, CIVE, TGI, VEG, COM, GCC; all pixels and vegetation pixels), RGB cover, HSV, texture and QC flags."))
        g = QGroupBox("Inputs"); f = QFormLayout(g)
        self.ed_ortho = QLineEdit(); f.addRow("RGB orthomosaic (GeoTIFF):", file_row(self, self.ed_ortho, "RGB orthomosaic", "GeoTIFF (*.tif *.tiff);;All files (*)", on_change=self._set_ortho))
        self.lbl_grid = QLabel(""); f.addRow("Plot grid:", self.lbl_grid)
        self.lbl_region = QLabel(""); f.addRow("Sampling region (features):", self.lbl_region)
        self.cb_tifs = QCheckBox("Write the per-plot RGB crops (GeoTIFF per plot)"); self.cb_tifs.setChecked(True); f.addRow("", self.cb_tifs)
        self.ed_out = QLineEdit(); f.addRow("Output folder:", file_row(self, self.ed_out, "Output folder", directory=True))
        v.addWidget(g)
        row = QHBoxLayout(); self.btn = QPushButton("Extract RGB plots + compute RGB features"); self.btn.clicked.connect(self._run); row.addWidget(self.btn); row.addStretch(); v.addLayout(row)
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); v.addWidget(self.out, stretch=1)

    def _set_ortho(self, p):
        state().ortho_tif = p

    def showEvent(self, ev):
        super().showEvent(ev); s = state()
        if not self.ed_ortho.text().strip() and s.ortho_tif:
            self.ed_ortho.setText(s.ortho_tif)
        elif not self.ed_ortho.text().strip() and s.proj_dir:
            # guess: an RGB orthomosaic next to the LAS
            for fn in sorted(os.listdir(s.proj_dir)) if os.path.isdir(s.proj_dir) else []:
                if fn.lower().endswith((".tif", ".tiff")) and "rgb" in fn.lower() and "canopy" not in fn.lower() and ".rgb." not in fn.lower():
                    self.ed_ortho.setText(os.path.join(s.proj_dir, fn)); break
        if s.las_path:
            s.derive_default_paths()
        if not self.ed_out.text().strip() and s.plots_rgb_dir:
            self.ed_out.setText(s.plots_rgb_dir)
        self.lbl_grid.setText(s.grid_file() or "(none - load / align a grid first)")
        self.lbl_region.setText(("central band %.2f m" % s.band_width if s.region_mode == "central" else "whole plot (10 cm inset)") + "  - set on the LiDAR Features tab")

    def _run(self):
        s = state(); ortho = self.ed_ortho.text().strip()
        if not ortho or not os.path.exists(ortho):
            QMessageBox.warning(self, "No orthomosaic", "Pick the RGB orthomosaic GeoTIFF."); return
        grid = s.grid_file()
        if not grid or not os.path.exists(grid):
            QMessageBox.warning(self, "No grid", "Load / align a plot grid first (Project or Edit tab)."); return
        s.ortho_tif = ortho
        out_dir = self.ed_out.text().strip() or s.plots_rgb_dir or os.path.join(os.path.dirname(ortho), "plots_rgb")
        s.plots_rgb_dir = out_dir; s.rgb_csv = os.path.join(out_dir, "rgb_features.csv")
        self.btn.setEnabled(False); self.progress.setValue(0)
        self._w = _Worker(ortho, grid, s.work_crs, out_dir, s.rgb_csv, self.cb_tifs.isChecked(), s.region_mode, s.band_width)
        self._w.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._w.done_ok.connect(lambda t: (self.btn.setEnabled(True), self.out.setPlainText(t)))
        self._w.error.connect(lambda e: (self.btn.setEnabled(True), self.out.append("\nERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._w.start()
