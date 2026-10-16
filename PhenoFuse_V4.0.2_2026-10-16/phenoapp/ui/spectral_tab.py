"""
VNIR Spectral tab.

Two sub-tabs:

  Indices  - pick spectral indices by category from the bundled Awesome
             Spectral Indices catalogue (+ PhenoApp narrow-band extras),
             compute them per plot over the sampling region chosen on the
             Traits tab (per-pixel then averaged, or on the plot mean
             spectrum), optionally write index rasters (GeoTIFF) for QGIS.
             Output: <project>/<las>_spectral_indices.csv (+ _qc.csv with
             per-index sd and valid-pixel fraction, + _provenance.json).

  View     - look at the cube: whole orthomosaic (decimated), selected plots
             at full resolution, or per-plot cube files as a gallery; as a
             CIR / true-colour / custom composite, a single wavelength, or
             any catalogue index. Click a pixel to see its spectrum.
"""

from __future__ import annotations
import os
import re
import json
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QProgressBar, QFileDialog, QMessageBox, QLineEdit, QTextEdit, QCheckBox,
    QFormLayout, QComboBox, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QSplitter, QListWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox,
    QHeaderView
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar)

from phenoapp.core.project import state
from phenoapp.core import load_grid, plot_region, VNIRCube
from phenoapp.core.spectral_indices import (
    IndexCatalogue, compute_plot_indices, write_index_rasters, compute_index_window,
    RECOMMENDED_BIOMASS, RECOMMENDED_WATER, DEFAULT_EXCLUDED_NM, DOMAIN_LABELS, parse_excluded,
    summarise_indices, composite_bands, guess_scale, read_band)
from phenoapp.core.paths import vnir_sidecar, write_provenance
from PyQt5.QtGui import QColor


def _excl_text(ranges):
    return ", ".join(f"{lo:g}-{hi:g}" for lo, hi in ranges)


# ======================================================================
# workers
# ======================================================================
class _IndexWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, vnir_path, grid_path, work_crs, region_mode, band_width, acronyms, mode,
                 excluded, out_csv, rasters_dir, spectra_npz):
        super().__init__()
        self._a = (vnir_path, grid_path, work_crs, region_mode, band_width, acronyms, mode,
                   excluded, out_csv, rasters_dir, spectra_npz)

    def run(self):
        try:
            (vnir_path, grid_path, work_crs, region_mode, band_width, acronyms, mode,
             excluded, out_csv, rasters_dir, spectra_npz) = self._a
            self.progress.emit(1, "Opening VNIR cube...")
            cube = VNIRCube(vnir_path)
            plots = load_grid(grid_path, target_crs=work_crs)
            regions = [plot_region(g, mode=region_mode, band_width=band_width) for g in plots.geometry]
            cat = IndexCatalogue()
            spectra = None
            if mode == "mean" and spectra_npz and os.path.exists(spectra_npz):
                try:
                    z = np.load(spectra_npz, allow_pickle=False)
                    if z["spectra"].shape == (len(plots), len(cube.wavelengths)) and \
                            list(z["plot_ids"]) == list(plots["Plot_ID"]):
                        spectra = z["spectra"]
                        self.progress.emit(5, "Re-using the per-plot mean spectra from the Biomass tab")
                except Exception:
                    spectra = None

            def cb(p, m):
                self.progress.emit(int(p * (0.6 if rasters_dir else 0.95)), m)
            df, qc, meta = compute_plot_indices(cube, plots, regions, acronyms, mode=mode, excluded=excluded,
                                                catalogue=cat, progress_cb=cb, spectra=spectra)
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            # carry plot naming columns for convenience
            for c in ("Plot", "B/R", "Bank", "Row", "Range"):
                if c in plots.columns and c not in df.columns:
                    df.insert(1, c, list(plots[c]))
            df.to_csv(out_csv, index=False)
            base = os.path.splitext(out_csv)[0]
            qc.to_csv(base + "_qc.csv", index=False)
            meta.update(region_mode=region_mode, band_width=band_width, grid=grid_path,
                        catalogue="Awesome Spectral Indices (Montero et al. 2023) + PhenoApp extras")
            with open(base + "_provenance.json", "w") as f:
                json.dump(meta, f, indent=1)
            msg = (f"Spectral indices ({mode} mode, {len(acronyms)} indices, {meta['n_cube_bands_read']} cube bands "
                   f"read) -> {out_csv}\nQC (sd + valid fraction) -> {base}_qc.csv\n"
                   f"Provenance (formulas, bands, wavelengths, exclusions) -> {base}_provenance.json\n\n"
                   + summarise_indices(df, qc if mode == "pixel" else None))
            if mode == "pixel":
                low = [c for c in df.columns if f"{c}_valid" in qc.columns and qc[f"{c}_valid"].mean() < 0.6]
                if low:
                    msg += (f"\n\nWARNING: {', '.join(low)} are valid on <60% of pixels - the bands they use are "
                            "clipped to 0 (negative reflectance) in the canopy; treat them as unreliable on this cube.")
            if rasters_dir:
                def cb2(p, m):
                    self.progress.emit(60 + int(p * 0.38), m)
                paths = write_index_rasters(cube, acronyms, rasters_dir, bounds=tuple(plots.total_bounds),
                                            excluded=excluded, catalogue=cat, progress_cb=cb2,
                                            base_name=os.path.splitext(os.path.basename(vnir_path))[0][:40])
                msg += f"\n\nIndex rasters ({len(paths)} GeoTIFFs, trial extent) -> {rasters_dir}"
            cube.close()
            self.progress.emit(100, "done")
            self.done_ok.emit(msg)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _CalibWorker(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self, kw): super().__init__(); self._kw = kw
    def run(self):
        try:
            from phenoapp.core.vnir_calibration import calibrate_cube
            kw = self._kw; grid = None
            if kw.get("grid_path") and os.path.exists(kw["grid_path"]):
                grid = load_grid(kw["grid_path"], target_crs=kw.get("work_crs"))
            res = calibrate_cube(kw["cube"], kw["mode"], reference=kw.get("reference"), grid=grid, targets=kw.get("targets"), spectra_csv=kw.get("spectra_csv"),
                                 soil_ndvi_max=kw.get("soil_ndvi_max", 0.2), progress_cb=lambda p, m: self.progress.emit(int(p), m))
            self.done_ok.emit(res)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _V3Worker(QThread):
    progress = pyqtSignal(int, str)
    done_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, vnir_path, grid_path, work_crs, region_mode, band_width, out_csv, excluded, cubes_dir=None, indices=None):
        super().__init__()
        self._a = (vnir_path, grid_path, work_crs, region_mode, band_width, out_csv, excluded, cubes_dir, indices)

    def run(self):
        try:
            from phenoapp.core.vnir_features import vnir_features_all, DEFAULT_V3_INDICES
            vnir_path, grid_path, work_crs, region_mode, band_width, out_csv, excluded, cubes_dir, indices = self._a
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            cube = VNIRCube(vnir_path); plots = load_grid(grid_path, target_crs=work_crs)
            regions = [plot_region(g, mode=region_mode, band_width=band_width) for g in plots.geometry]
            span = 0.6 if cubes_dir else 1.0
            df, sp, vf = vnir_features_all(cube, plots, regions, indices=indices, excluded=excluded, progress_cb=lambda p, m: self.progress.emit(int(p * span), m))
            n_cubes = 0
            if cubes_dir:
                idx = cube.write_plot_cubes(plots, cubes_dir, progress_cb=lambda p, m: self.progress.emit(60 + int(p * 0.4), m))
                n_cubes = len(idx)
            cube.close()
            df.to_csv(out_csv, index=False); sp.to_csv(vnir_sidecar(out_csv, "spectra"), index=False)
            vf.to_csv(vnir_sidecar(out_csv, "valid_fraction"), index=False)   # why a band is NaN: its valid-pixel fraction per plot
            # 0/1 band QC table (1 = usable, 0 = the matching spectra cell is NaN), same rule as the NaN cells
            import pandas as pd
            from phenoapp.core.spectral_indices import band_usable
            _wcols = [c for c in vf.columns if c != "Plot_ID"]
            bu = pd.DataFrame(band_usable(cube.wavelengths, vf[_wcols].values, bool(int(df.vnir_reflectance_ok.iloc[0])), df.frac_in_cube.values, excluded=excluded), columns=_wcols)
            bu.insert(0, "Plot_ID", vf["Plot_ID"].values); bu.insert(1, "n_usable_bands", bu[_wcols].sum(axis=1).values)
            bu.to_csv(vnir_sidecar(out_csv, "usable"), index=False)
            write_provenance(out_csv, excluded_nm=[list(e) for e in excluded], cube=vnir_path, grid=grid_path, region_mode=region_mode, band_width=band_width,
                             indices=list(indices or DEFAULT_V3_INDICES), n_plots=int(len(df)), per_plot_cubes=cubes_dir or "")
            t = df.veg_ndvi_threshold.iloc[0]
            msg = (f"VNIR features: {df.shape[1]} columns for {len(df)} plots -> {out_csv}\n"
                   + (f"Per-plot VNIR cubes: {n_cubes} GeoTIFFs (all bands, exact copies) -> {cubes_dir}\n" if cubes_dir else "")
                   + "Vegetation mask: " + (f"NDVI >= {t:.2f}" if np.isfinite(t) else "disabled (no soil/crop NDVI contrast - senesced canopy)") +
                   f"; vegetation fraction median {df.fcover_vnir.median():.2f}; plots with a usable red band {int(df.red_ok.sum())}/{len(df)}; vnir_qc_ok {int(df.vnir_qc_ok.sum())}/{len(df)}\n"
                   + ("" if int(df.vnir_reflectance_ok.iloc[0]) == 1 else
                      "WARNING: this cube does not look like reflectance (NDVI ~ 0 in every plot and/or float values far above 1 with no "
                      "reflectance scale factor). All plots are flagged vnir_qc_ok = 0; reprocess the orthomosaic to reflectance before using VNIR features.\n")
                   + (f"Red-clipped plots (red_ok = 0): {int((df.red_ok == 0).sum())}; their red-based indices are set to NaN.\n" if int((df.red_ok == 0).sum()) else "")
                   + f"Mean vegetation spectra -> {vnir_sidecar(out_csv, 'spectra')}\nBand QC tables -> {vnir_sidecar(out_csv, 'valid_fraction')}, {vnir_sidecar(out_csv, 'usable')}\n"
                   + "Band policy (excluded nm): " + (", ".join("%g-%g" % (lo, hi) for lo, hi in excluded) or "none") + " -> " + vnir_sidecar(out_csv, "provenance"))
            self.done_ok.emit(msg)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ======================================================================
# Indices sub-tab
# ======================================================================
class IndicesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cat = IndexCatalogue()
        self._wl = None
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            "Pick indices by category, then compute them per plot over the Traits-tab sampling region. "
            "Broad bands (N, R, G, RE1...) are synthesised from the cube's wavelengths using the catalogue's "
            "band ranges; narrow bands (R705...) use the closest cube band. Excluded wavelength ranges are "
            "never used, and a pixel counts only when every band feeding the index is > 0."))

        top = QGroupBox("Inputs")
        f = QFormLayout(top)
        row = QHBoxLayout()
        self.ed_vnir = QLineEdit()
        b = QPushButton("Browse..."); b.clicked.connect(self._pick_vnir)
        b2 = QPushButton("Read wavelengths"); b2.clicked.connect(self.refresh_catalogue)
        row.addWidget(self.ed_vnir); row.addWidget(b); row.addWidget(b2)
        f.addRow("VNIR cube (.bin):", row)
        self.ed_excl = QLineEdit(_excl_text(DEFAULT_EXCLUDED_NM))
        self.ed_excl.setToolTip(
            "Wavelength ranges (nm) never used to build a band, e.g. '0-415, 755-770, 928-962'.\n"
            "Defaults for GRYFN-processed cubes: the first bands (399-406 nm) carry a calibration "
            "spike, 410-500 nm is mostly clipped to 0 in canopy, 761 nm is the O2-A absorption residual "
            "and 930-960 nm the water-vapour feature. Press 'Read wavelengths' after editing.")
        f.addRow("Excluded ranges (nm):", self.ed_excl)
        pol = QHBoxLayout()
        self.cb_ex_edge = QCheckBox("sensor edge < 415 nm"); self.cb_ex_o2a = QCheckBox("O2-A 755-770 nm"); self.cb_ex_wv = QCheckBox("water vapour 928-962 nm")
        for cb in (self.cb_ex_edge, self.cb_ex_o2a, self.cb_ex_wv):
            cb.setChecked(True); cb.toggled.connect(self._policy_changed); pol.addWidget(cb)
        pol.addStretch()
        self.cb_ex_o2a.setToolTip("Exclude the oxygen A-band (760 nm): an ATMOSPHERIC absorption whose residual after reflectance conversion depends on "
                                  "flight height and path length, not on the plant. Untick to let the models and the indices use it (e.g. the DWI_O2A diagnostic).")
        self.cb_ex_wv.setToolTip("Exclude the 928-962 nm atmospheric water-vapour band. It overlaps the leaf-water absorption (the 970 nm feature behind WBI "
                                 "and the NWI indices stays available either way); untick to keep these bands for the band-based models and the WVD_940 diagnostic. "
                                 "Validate on a held-out flight: the atmospheric part of the signal changes with humidity and flight height.")
        f.addRow("Band policy:", pol)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Per pixel, then plot mean (recommended; reports sd + valid fraction)",
                               "On the plot mean spectrum (fast; same numbers as the Biomass tab)"])
        f.addRow("Computation:", self.cb_mode)
        self.lbl_region = QLabel("")
        f.addRow("Sampling region:", self.lbl_region)
        self.cb_rasters = QCheckBox("Also write index rasters (GeoTIFF, trial extent, for QGIS) for the ticked indices")
        f.addRow("", self.cb_rasters)
        v.addWidget(top)

        # ---- reflectance check / empirical calibration (4.0.4) ----
        cal = QGroupBox("Reflectance check and empirical calibration (for a cube delivered as radiance)"); fc = QFormLayout(cal)
        rowc = QHBoxLayout(); self.btn_check = QPushButton("Check cube"); self.btn_check.clicked.connect(self._check_cube); self.lbl_check = QLabel("not checked")
        self.lbl_check.setWordWrap(True); rowc.addWidget(self.btn_check); rowc.addWidget(self.lbl_check, stretch=1); fc.addRow("Calibration:", rowc)
        self.btn_check.setToolTip("Samples three windows of the cube and looks at the median spectrum: a reflectance cube has no atmospheric absorption lines, "
                                  "a radiance cube still shows the oxygen-A dip at 760 nm and the water-vapour dip at 940 nm, and NIR about equal to red.")
        self.cb_cal_mode = QComboBox()
        self.cb_cal_mode.addItems(["Reference targets in this flight (panels / tarps with known reflectance) - best",
                                   "Calibrated cube of the same trial (another flight of the day, or another date) - bare soil as invariant target"])
        self.cb_cal_mode.setCurrentIndex(1); self.cb_cal_mode.currentIndexChanged.connect(self._cal_mode); fc.addRow("Reference:", self.cb_cal_mode)
        self.ed_cal_ref = QLineEdit(); bref = QPushButton("Browse..."); bref.clicked.connect(lambda: self._pick_into(self.ed_cal_ref, "Calibrated reference cube (.bin)", "ENVI cubes (*.bin *.img *.dat);;All files (*)"))
        r1 = QHBoxLayout(); r1.addWidget(self.ed_cal_ref); r1.addWidget(bref); fc.addRow("Reference cube:", r1)
        self.ed_cal_targets = QLineEdit(); btg = QPushButton("Browse..."); btg.clicked.connect(lambda: self._pick_into(self.ed_cal_targets, "Reference target polygons", "Vector (*.shp *.gpkg *.geojson);;All files (*)"))
        r2 = QHBoxLayout(); r2.addWidget(self.ed_cal_targets); r2.addWidget(btg); fc.addRow("Target polygons:", r2)
        self.ed_cal_spectra = QLineEdit(); bsp = QPushButton("Browse..."); bsp.clicked.connect(lambda: self._pick_into(self.ed_cal_spectra, "Target reflectance spectra CSV", "CSV (*.csv);;All files (*)"))
        r3 = QHBoxLayout(); r3.addWidget(self.ed_cal_spectra); r3.addWidget(bsp); fc.addRow("Target spectra CSV:", r3)
        self.ed_cal_targets.setToolTip("Polygons of the reference panels / tarps in THIS flight. Either a column 'refl' with the constant reflectance (0-1) of each polygon, "
                                       "or a column 'name' matching a column of the spectra CSV (wavelength_nm + one reflectance column per target).")
        self.dsb_soil_ndvi = QDoubleSpinBox(); self.dsb_soil_ndvi.setRange(0.05, 0.5); self.dsb_soil_ndvi.setSingleStep(0.05); self.dsb_soil_ndvi.setValue(0.20)
        self.dsb_soil_ndvi.setToolTip("Bare soil = positions where the REFERENCE cube's NDVI is below this value (tracks, alleys). The gain is the median reference reflectance "
                                      "over these positions divided by the median value of this cube at the same positions, band by band.")
        fc.addRow("Soil NDVI below:", self.dsb_soil_ndvi)
        self.btn_cal = QPushButton("Calibrate -> write <cube>_reflectance_empirical.bin and use it"); self.btn_cal.clicked.connect(self._calibrate)
        self.btn_cal.setToolTip("R = gain x value (+ offset with two or more targets), per band. Writes the calibrated ENVI cube (uint16 x 10000, same grid) next to the "
                                "original with a gain table, a README with the recipe and caveats, and a before / after figure; the cube field then points to it.\n"
                                "Across DATES the bare-soil reference assumes the soil did not change: ratio indices stay reliable, absolute reflectance is uncertain by ~10-20 %. "
                                "A panel in the same flight, or a calibrated flight of the same day, avoids that.")
        fc.addRow("", self.btn_cal)
        v.addWidget(cal); self._cal_mode()

        # ---- catalogue tree ----
        mid = QGroupBox("Indices (tick to compute; hover for formula and reference)")
        mv = QVBoxLayout(mid)
        bar = QHBoxLayout()
        self.ed_search = QLineEdit(); self.ed_search.setPlaceholderText("filter by acronym / name / band...")
        self.ed_search.textChanged.connect(self._filter)
        bar.addWidget(self.ed_search, stretch=1)
        b_rec = QPushButton("Recommended (biomass)"); b_rec.clicked.connect(self._select_recommended)
        b_rec.setToolTip("Red-edge family + water and structure indices that carried biomass information on "
                         "pasture and wheat trials (NDRE variants, REP, LCI, S2REP, MTCI, CIRE, OSAVI, kNDVI, NIRv, WBI...).")
        b_wat = QPushButton("Water (plant water)"); b_wat.clicked.connect(self._select_water)
        b_wat.setToolTip("Every water index computable on a 400-1000 nm cube: WBI, NDWI970, WI/NDVI, the NWI 1-4 of Babar / Prasad, the ASI water "
                         "indices without SWIR bands (NDWI, DDWI, FWEI, NDCI, NDTI, OSI, PI, RNDVI, NDWIns). The ASI water indices that need SWIR "
                         "(S1 1550-1750 nm, S2 2080-2350 nm) are listed in the Water group greyed out - they cannot be built from a VNIR cube.")
        b_clr = QPushButton("Clear"); b_clr.clicked.connect(lambda: self._set_all(False))
        bar.addWidget(b_rec); bar.addWidget(b_wat); bar.addWidget(b_clr)
        mv.addLayout(bar)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Index", "Name", "Formula", "Bands"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Interactive)
        self.tree.setColumnWidth(2, 260)
        self.tree.itemChanged.connect(self._on_item_changed)
        mv.addWidget(self.tree, stretch=1)
        self.lbl_count = QLabel("")
        mv.addWidget(self.lbl_count)
        v.addWidget(mid, stretch=1)

        row3 = QHBoxLayout()
        self.btn_run = QPushButton("Compute selected indices per plot")
        self.btn_run.clicked.connect(self._run)
        self.btn_v3 = QPushButton("VNIR features per plot (soil mask, veg-only indices incl. water, spectral shape, PCA) -> vnir_features.csv")
        self.cb_v3_cubes = QCheckBox("Also write per-plot VNIR cubes (GeoTIFF, all bands, exact copies) into plots_vnir/ - needed by the Dataset / Deep Models tabs")
        self.cb_v3_cubes.setChecked(True)
        self.cb_v3_sel = QCheckBox("Add the ticked catalogue indices to the feature table (on top of the standard set)")
        self.cb_v3_sel.setChecked(False)
        self.btn_v3.setToolTip(
            "Per plot: clipped (0) band values treated as missing, excluded ranges never used, adaptive NDVI soil mask "
            "(disabled automatically on a senesced canopy), 30 indices over all pixels and over vegetation pixels (mean, sd), "
            "vegetation fraction, broad-band reflectances, red-edge derivative features, Guyot REP, PCA scores of the "
            "vegetation spectra, QC flags (red_ok, vnir_qc_ok). Writes <metrics>_vnir_v3.csv and _vnir_v3_spectra.csv "
            "- the inputs of the Height Models and Biomass ML tabs.")
        self.btn_v3.clicked.connect(self._run_v3)
        row3.addWidget(self.btn_run); row3.addWidget(self.btn_v3); row3.addStretch()
        v.addLayout(row3)
        v.addWidget(self.cb_v3_cubes); v.addWidget(self.cb_v3_sel)
        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas")
        self.out.setMaximumHeight(220)
        v.addWidget(self.out)
        self.refresh_catalogue()

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        s = state()
        if not self.ed_vnir.text().strip() and s.vnir_path:
            self.ed_vnir.setText(s.vnir_path)
            self.refresh_catalogue()
        self.lbl_region.setText(
            ("central band %.2f m" % s.band_width if s.region_mode == "central" else "whole plot (10 cm inset)")
            + "  - set on the LiDAR Features tab")

    def _pick_vnir(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick VNIR ENVI cube (.bin)", "",
                                           "ENVI cubes (*.bin *.img *.dat);;All files (*)")
        if p:
            self.ed_vnir.setText(p); state().vnir_path = p
            self.refresh_catalogue(); self._check_cube(quiet=True)

    # ---- reflectance check / empirical calibration (4.0.4) ----
    def _pick_into(self, edit, title, filt):
        p, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if p: edit.setText(p)

    def _cal_mode(self):
        targets = self.cb_cal_mode.currentIndex() == 0
        for w in (self.ed_cal_targets, self.ed_cal_spectra): w.setEnabled(targets)
        for w in (self.ed_cal_ref, self.dsb_soil_ndvi): w.setEnabled(not targets)

    def _check_cube(self, quiet=False):
        cube = self.ed_vnir.text().strip() or state().vnir_path
        if not cube or not os.path.exists(cube):
            if not quiet: QMessageBox.warning(self, "No cube", "Pick a VNIR cube first.")
            return
        try:
            from phenoapp.core.vnir_calibration import calibration_check
            c = calibration_check(cube)
        except Exception as e:
            self.lbl_check.setText(f"check failed: {e}"); return
        txt = (f"looks like {c['looks_like'].upper()}  ({c['dtype']}, scale factor {c['scale_factor'] or 'none'}; O2-A 760 nm dip {c['o2a_ratio']:.2f}, "
               f"water-vapour 940 nm dip {c['h2o_ratio']:.2f}, NIR/red {c['nir_red']:.2f}, NDVI median {c['ndvi_p50']:.2f})")
        if c["reasons"]: txt += " - " + "; ".join(c["reasons"])
        self.lbl_check.setText(txt)
        self.lbl_check.setStyleSheet("color: #b00000; font-weight: bold" if c["looks_like"] == "radiance" else ("color: #a06000" if c["looks_like"] == "doubtful" else "color: #006000"))
        if not quiet and c["looks_like"] != "reflectance":
            QMessageBox.warning(self, "Cube calibration", txt + "\n\nEvery vegetation index on this cube would be meaningless. Reprocess it to reflectance, or use the empirical "
                                "calibration below (a reference panel in this flight, or a calibrated cube of the same trial).")

    def _calibrate(self):
        s = state(); cube = self.ed_vnir.text().strip() or s.vnir_path
        if not cube or not os.path.exists(cube):
            QMessageBox.warning(self, "No cube", "Pick the VNIR cube to calibrate first."); return
        mode = "targets" if self.cb_cal_mode.currentIndex() == 0 else "cube"
        kw = dict(cube=cube, mode=mode, soil_ndvi_max=self.dsb_soil_ndvi.value(), work_crs=s.work_crs,
                  grid_path=(s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path))
        if mode == "cube":
            ref = self.ed_cal_ref.text().strip()
            if not ref or not os.path.exists(ref): QMessageBox.warning(self, "No reference cube", "Pick the calibrated reference cube of the same trial."); return
            if os.path.abspath(ref) == os.path.abspath(cube): QMessageBox.warning(self, "Reference", "The reference must be a different, calibrated cube."); return
            kw["reference"] = ref
        else:
            tg = self.ed_cal_targets.text().strip()
            if not tg or not os.path.exists(tg): QMessageBox.warning(self, "No targets", "Pick the polygons of the reference targets in this flight."); return
            kw["targets"] = tg; kw["spectra_csv"] = self.ed_cal_spectra.text().strip() or None
        self.btn_cal.setEnabled(False); self.progress.setValue(0); self.out.setPlainText("Empirical calibration running...")
        self._wc = _CalibWorker(kw)
        self._wc.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._wc.done_ok.connect(self._on_calibrated)
        self._wc.error.connect(lambda e: (self.btn_cal.setEnabled(True), self._on_err(e)))
        self._wc.start()

    def _on_calibrated(self, res):
        self.btn_cal.setEnabled(True); r = res["result"]; b = res["check_before"]; a = res["check_after"]
        self.ed_vnir.setText(res["out_cube"]); state().vnir_path = res["out_cube"]; self.refresh_catalogue(); self._check_cube(quiet=True)
        msg = (f"Calibrated cube -> {res['out_cube']}\nMethod: {r['method']}\nReference: {r['reference']}\n"
               f"Samples: {r['n_positions']} positions, {r['n_soil']} bare-soil; gain repeatability between two halves {r['repeatability_pct']:.2f} %\n"
               f"Before: {b['looks_like']} (O2-A {b['o2a_ratio']:.2f}, H2O {b['h2o_ratio']:.2f}, NIR/red {b['nir_red']:.2f}, NDVI {b['ndvi_p50']:.2f})\n"
               f"After : {a['looks_like']} (O2-A {a['o2a_ratio']:.2f}, H2O {a['h2o_ratio']:.2f}, NIR/red {a['nir_red']:.2f}, NDVI {a['ndvi_p50']:.2f})\n"
               f"Gain table, README (recipe + caveats) and QA figure next to the cube: {res['qa_png']}\n\n"
               "The cube field now points to the calibrated cube: run the VNIR features on it. Ratio indices are reliable; treat absolute reflectance from a cross-date "
               "soil reference as approximate (~10-20 %). A reprocessing to reflectance by the sensor software supersedes this file.")
        self.out.setPlainText(msg); QMessageBox.information(self, "Empirical calibration done", msg)

    def _policy_changed(self, *_):
        std = [((0.0, 415.0), self.cb_ex_edge), ((755.0, 770.0), self.cb_ex_o2a), ((928.0, 962.0), self.cb_ex_wv)]
        try:
            cur = parse_excluded(self.ed_excl.text())
        except ValueError:
            cur = list(DEFAULT_EXCLUDED_NM)
        custom = [r for r in cur if all(abs(r[0] - s[0]) > 0.5 or abs(r[1] - s[1]) > 0.5 for s, _ in std)]
        ranges = [s for s, cb in std if cb.isChecked()] + custom
        self.ed_excl.setText(_excl_text(ranges))
        self.refresh_catalogue()

    def _excluded(self):
        try:
            return parse_excluded(self.ed_excl.text())
        except ValueError as e:
            QMessageBox.warning(self, "Excluded ranges", str(e)); return DEFAULT_EXCLUDED_NM

    def refresh_catalogue(self):
        """Rebuild the tree for the cube's wavelengths (or a generic 400-1000 nm)."""
        p = self.ed_vnir.text().strip()
        wl = None
        if p and os.path.exists(p):
            try:
                from phenoapp.core.vnir import parse_envi_wavelengths, _find_header
                wl = parse_envi_wavelengths(_find_header(p))
            except Exception as e:
                self.out.append(f"Could not read wavelengths: {e}")
        if wl is None:
            wl = np.linspace(400, 1000, 172)
        self._wl = wl
        checked = self.selected()
        excl = self._excluded()
        self.tree.blockSignals(True)
        self.tree.clear()
        groups = self.cat.by_domain(wl, excl, include_uncomputable=False)
        n = 0
        for dom, entries in groups.items():
            top = QTreeWidgetItem([DOMAIN_LABELS.get(dom, dom.title()), f"{len(entries)} indices", "", ""])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            top.setCheckState(0, Qt.Unchecked)
            for e in entries:
                it = QTreeWidgetItem([e.acronym, e.name, e.formula, ", ".join(e.bands)])
                it.setData(0, Qt.UserRole, e.acronym)
                why = self.cat.why_not(e.acronym, wl, excl)
                if why:
                    # listed (the whole water family is shown) but not computable from this cube: greyed, unticked
                    it.setFlags(Qt.ItemIsSelectable)
                    for c in range(4):
                        it.setForeground(c, QColor("#999999")); it.setToolTip(c, e.tooltip() + "\n\nNOT computable from this cube: " + why)
                    it.setText(1, e.name + "   [not computable: " + why.split(" - ")[0] + "]")
                else:
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(0, Qt.Checked if e.acronym in checked else Qt.Unchecked)
                    for c in range(4):
                        it.setToolTip(c, e.tooltip())
                    n += 1
                top.addChild(it)
            self.tree.addTopLevelItem(top)
            top.setExpanded(dom == "hyperspectral narrow-band")
        self.tree.blockSignals(False)
        self._filter(self.ed_search.text())
        self.lbl_count.setText(f"{n} indices computable from this cube ({wl[0]:.0f}-{wl[-1]:.0f} nm, "
                               f"{len(wl)} bands); {len(checked)} selected")

    def _iter_leaves(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                yield top.child(j)

    def selected(self):
        return [it.data(0, Qt.UserRole) for it in self._iter_leaves() if (it.flags() & Qt.ItemIsUserCheckable) and it.checkState(0) == Qt.Checked]

    def _set_all(self, on):
        self.tree.blockSignals(True)
        for it in self._iter_leaves():
            if it.flags() & Qt.ItemIsUserCheckable:
                it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()

    def _select_recommended(self):
        self.tree.blockSignals(True)
        for it in self._iter_leaves():
            if it.data(0, Qt.UserRole) in RECOMMENDED_BIOMASS and (it.flags() & Qt.ItemIsUserCheckable):
                it.setCheckState(0, Qt.Checked)
        self.tree.blockSignals(False)
        self._update_count()

    def _select_water(self):
        self.tree.blockSignals(True)
        for it in self._iter_leaves():
            if it.data(0, Qt.UserRole) in RECOMMENDED_WATER and (it.flags() & Qt.ItemIsUserCheckable):
                it.setCheckState(0, Qt.Checked)
        self.tree.blockSignals(False)
        self._update_count()

    def _on_item_changed(self, *_):
        self._update_count()

    def _update_count(self):
        txt = self.lbl_count.text().split(";")[0]
        self.lbl_count.setText(f"{txt}; {len(self.selected())} selected")

    def _filter(self, text):
        t = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i); any_vis = False
            for j in range(top.childCount()):
                it = top.child(j)
                vis = (not t) or any(t in it.text(c).lower() for c in range(4))
                it.setHidden(not vis); any_vis |= vis
            top.setHidden(not any_vis)
            if t and any_vis:
                top.setExpanded(True)

    # ------------------------------------------------------------------
    def _run(self):
        s = state()
        vnir = self.ed_vnir.text().strip() or s.vnir_path
        if not vnir or not os.path.exists(vnir):
            QMessageBox.warning(self, "No VNIR cube", "Pick the ENVI .bin file first."); return
        grid_path = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
        if not grid_path or not os.path.exists(grid_path):
            QMessageBox.warning(self, "No grid", "Load / align a plot grid first (Project or Edit tab)."); return
        acr = self.selected()
        if not acr:
            QMessageBox.warning(self, "Nothing selected", "Tick at least one index."); return
        excl = self._excluded()
        s.vnir_path = vnir; s.derive_default_paths()
        s.spectral_sel = ",".join(acr); s.spectral_excl = self.ed_excl.text()
        mode = "pixel" if self.cb_mode.currentIndex() == 0 else "mean"
        rasters_dir = None
        if self.cb_rasters.isChecked():
            rasters_dir = os.path.join(os.path.dirname(s.spectral_csv), "index_rasters")
            if len(acr) > 12 and QMessageBox.question(
                    self, "Many rasters", f"Write {len(acr)} trial-extent GeoTIFFs?",
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return
        self.btn_run.setEnabled(False)
        self._w = _IndexWorker(vnir, grid_path, s.work_crs, s.region_mode, s.band_width, acr, mode, excl,
                               s.spectral_csv, rasters_dir, s.vnir_spectra)
        self._w.progress.connect(self._on_prog)
        self._w.done_ok.connect(self._on_done)
        self._w.error.connect(self._on_err)
        self._w.start()

    def _run_v3(self):
        s = state()
        vnir = self.ed_vnir.text().strip() or s.vnir_path
        if not vnir or not os.path.exists(vnir):
            QMessageBox.warning(self, "No VNIR cube", "Pick the ENVI .bin file first."); return
        grid_path = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
        if not grid_path or not os.path.exists(grid_path):
            QMessageBox.warning(self, "No grid", "Load / align a plot grid first."); return
        s.vnir_path = vnir; s.derive_default_paths()
        out_csv = s.vnir_csv if s.vnir_csv else os.path.join(os.path.dirname(vnir), "plots_vnir", "vnir_features.csv")
        cubes_dir = (s.plots_vnir_dir or os.path.dirname(out_csv)) if self.cb_v3_cubes.isChecked() else None
        indices = None
        if self.cb_v3_sel.isChecked():
            from phenoapp.core.vnir_features import DEFAULT_V3_INDICES
            indices = list(DEFAULT_V3_INDICES) + [a for a in self.selected() if a not in DEFAULT_V3_INDICES]
        s.spectral_excl = self.ed_excl.text()
        self.btn_v3.setEnabled(False)
        self._w3 = _V3Worker(vnir, grid_path, s.work_crs, s.region_mode, s.band_width, out_csv, self._excluded(), cubes_dir, indices)
        self._w3.progress.connect(self._on_prog)
        self._w3.done_ok.connect(lambda m: (self.btn_v3.setEnabled(True), self.out.setPlainText(m)))
        self._w3.error.connect(lambda e: (self.btn_v3.setEnabled(True), self._on_err(e)))
        self._w3.start()

    def _on_prog(self, p, m):
        self.progress.setValue(p); self.progress.setFormat(f"{p}% - {m}")

    def _on_done(self, msg):
        self.btn_run.setEnabled(True)
        self.out.setPlainText(msg)

    def _on_err(self, err):
        self.btn_run.setEnabled(True)
        self.out.append(f"\nERROR: {err}")
        QMessageBox.critical(self, "Failed", err.split("\n")[0])


# ======================================================================
# View sub-tab
# ======================================================================
class _MplCanvas(FigureCanvas):
    def __init__(self, parent=None, w=9, h=6):
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)


class ViewPanel(QWidget):
    """Whole-cube / selected-plot / per-plot-file viewer with click-spectrum."""

    def __init__(self, parent=None, indices_panel: IndicesPanel | None = None):
        super().__init__(parent)
        self._indices_panel = indices_panel
        self.cat = indices_panel.cat if indices_panel else IndexCatalogue()
        self._cube = None
        self._plots = None
        self._sources = []      # per drawn axes: dict(kind, path/cube, window, transform, decim, label)
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        left = QWidget(); lv = QVBoxLayout(left); left.setMaximumWidth(420)

        g1 = QGroupBox("Source"); f1 = QFormLayout(g1)
        self.cb_source = QComboBox()
        self.cb_source.addItems(["Whole orthomosaic (decimated)",
                                 "Selected plots from the cube (full resolution)",
                                 "Per-plot cube files (gallery)"])
        self.cb_source.currentIndexChanged.connect(self._source_changed)
        f1.addRow("Show:", self.cb_source)
        row = QHBoxLayout(); self.ed_dir = QLineEdit()
        b = QPushButton("..."); b.setMaximumWidth(32); b.clicked.connect(self._pick_dir)
        row.addWidget(self.ed_dir); row.addWidget(b)
        f1.addRow("Per-plot files:", row)
        self.lst = QListWidget(); self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst.setMaximumHeight(160)
        f1.addRow("Plots:", self.lst)
        rb = QHBoxLayout()
        b_all = QPushButton("All"); b_all.clicked.connect(self.lst.selectAll)
        b_none = QPushButton("None"); b_none.clicked.connect(self.lst.clearSelection)
        b_rl = QPushButton("Reload plots"); b_rl.clicked.connect(self.reload_plots)
        rb.addWidget(b_all); rb.addWidget(b_none); rb.addWidget(b_rl)
        f1.addRow("", rb)
        lv.addWidget(g1)

        g2 = QGroupBox("Display"); f2 = QFormLayout(g2)
        self.cb_disp = QComboBox()
        self.cb_disp.addItems(["Composite: NIR / red / green (800/670/550 nm)",
                               "Composite: true colour (640/550/470 nm)",
                               "Composite: custom wavelengths",
                               "Single band",
                               "Spectral index"])
        self.cb_disp.currentIndexChanged.connect(self._disp_changed)
        f2.addRow("Mode:", self.cb_disp)
        wrow = QHBoxLayout(); self.sp_nm = []
        for nm in (800, 670, 550):
            sp = QDoubleSpinBox(); sp.setRange(300, 2600); sp.setDecimals(0); sp.setValue(nm); sp.setSuffix(" nm")
            self.sp_nm.append(sp); wrow.addWidget(sp)
        f2.addRow("R / G / B or band:", wrow)
        self.cb_index = QComboBox()
        f2.addRow("Index:", self.cb_index)
        self.sp_maxpx = QSpinBox(); self.sp_maxpx.setRange(300, 6000); self.sp_maxpx.setValue(1500)
        self.sp_maxpx.setToolTip("Longest image side in pixels for display; larger windows are decimated. "
                                 "Zoomed plot views at 1.5 cm are full resolution up to this size.")
        f2.addRow("Max display px:", self.sp_maxpx)
        self.sp_stretch = QDoubleSpinBox(); self.sp_stretch.setRange(1, 100); self.sp_stretch.setValue(99)
        self.sp_stretch.setSuffix(" %"); self.sp_stretch.setToolTip("Upper percentile for the linear stretch")
        f2.addRow("Stretch:", self.sp_stretch)
        self.cb_grid = QCheckBox("Overlay plot polygons + IDs"); self.cb_grid.setChecked(True)
        f2.addRow("", self.cb_grid)
        lv.addWidget(g2)

        rr = QHBoxLayout()
        self.btn_render = QPushButton("Render"); self.btn_render.clicked.connect(self.render)
        self.btn_save = QPushButton("Save PNG..."); self.btn_save.clicked.connect(self._save_png)
        rr.addWidget(self.btn_render); rr.addWidget(self.btn_save)
        lv.addLayout(rr)
        self.lbl = QLabel("Click a pixel in the image to plot its spectrum."); self.lbl.setWordWrap(True)
        lv.addWidget(self.lbl)
        lv.addStretch()
        root.addWidget(left)

        split = QSplitter(Qt.Vertical)
        self.canvas = _MplCanvas(self, 9, 6)
        w = QWidget(); wv = QVBoxLayout(w); wv.setContentsMargins(0, 0, 0, 0)
        wv.addWidget(NavToolbar(self.canvas, self)); wv.addWidget(self.canvas)
        split.addWidget(w)
        self.spec = _MplCanvas(self, 9, 2.6)
        split.addWidget(self.spec)
        split.setStretchFactor(0, 4); split.setStretchFactor(1, 1)
        root.addWidget(split, stretch=1)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._disp_changed(0); self._source_changed(0)

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        if self._plots is None:
            self.reload_plots()
        s = state()
        if not self.ed_dir.text().strip():
            d = s.plots_vnir_dir or (os.path.join(os.path.dirname(s.out_csv), "plots_vnir") if s.out_csv else "")
            if d:
                self.ed_dir.setText(d)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with per-plot VNIR cubes (plot_<id>_*.tif)")
        if d:
            self.ed_dir.setText(d); state().plots_vnir_dir = d

    def _source_changed(self, i):
        self.ed_dir.setEnabled(i == 2)
        self.lst.setEnabled(i != 0)

    def _disp_changed(self, i):
        for k, sp in enumerate(self.sp_nm):
            sp.setEnabled(i == 2 or (i == 3 and k == 0))
        if i in (0, 1):
            for sp, nm in zip(self.sp_nm, (800, 670, 550) if i == 0 else (640, 550, 470)):
                sp.setValue(nm)
        self.cb_index.setEnabled(i == 4)
        if i == 4 and self.cb_index.count() == 0:
            self._fill_indices()

    def _fill_indices(self):
        wl = self._wavelengths()
        excl = self._excluded()
        self.cb_index.clear()
        pref = [a for a in RECOMMENDED_BIOMASS if self.cat.computable(a, wl, excl)]
        rest = sorted(a for a in self.cat.entries if a not in pref and self.cat.computable(a, wl, excl))
        for a in pref + rest:
            self.cb_index.addItem(f"{a} - {self.cat.entries[a].name}", a)

    def _excluded(self):
        try:
            return parse_excluded(self._indices_panel.ed_excl.text()) if self._indices_panel else DEFAULT_EXCLUDED_NM
        except Exception:
            return DEFAULT_EXCLUDED_NM

    def _wavelengths(self):
        c = self._open_cube()
        if c is not None:
            return np.asarray(c.wavelengths, float)
        return np.linspace(400, 1000, 172)

    def _open_cube(self):
        s = state()
        p = (self._indices_panel.ed_vnir.text().strip() if self._indices_panel else "") or s.vnir_path
        if not p or not os.path.exists(p):
            return None
        if self._cube is None or self._cube.path != p:
            try:
                if self._cube is not None:
                    self._cube.close()
                self._cube = VNIRCube(p)
            except Exception as e:
                self.lbl.setText(f"Cannot open cube: {e}"); self._cube = None
        return self._cube

    def reload_plots(self):
        s = state()
        self.lst.clear(); self._plots = None
        for cand in (s.saved_grid, s.grid_path):
            if cand and os.path.exists(cand):
                try:
                    self._plots = load_grid(cand, target_crs=s.work_crs)
                except Exception as e:
                    self.lbl.setText(f"Grid load failed: {e}"); return
                break
        if self._plots is None:
            self.lbl.setText("No grid loaded (Project tab) - only the whole orthomosaic can be shown."); return
        for pid, nm in zip(self._plots["Plot_ID"], self._plots["B/R"] if "B/R" in self._plots else self._plots["Plot_ID"]):
            self.lst.addItem(f"{pid}  ({nm})")
        self.lbl.setText(f"{len(self._plots)} plots loaded.")

    def _selected_plots(self):
        if self._plots is None:
            return None
        rows = sorted(i.row() for i in self.lst.selectedIndexes())
        return self._plots.iloc[rows] if rows else None

    # ------------------------------------------------------------------
    def _stretch(self, a, pct):
        v = a[np.isfinite(a) & (a > 0)]
        if v.size == 0:
            return np.zeros_like(a)
        lo, hi = np.percentile(v, [1, pct])
        return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)

    def _read_display(self, src, wl, window, out_shape, scale):
        """Return (image array for imshow, colour-bar label or None)."""
        mode = self.cb_disp.currentIndex()
        pct = self.sp_stretch.value()
        if mode in (0, 1, 2):
            nm = [sp.value() for sp in self.sp_nm]
            bands = [int(np.argmin(np.abs(wl - x))) + 1 for x in nm]
            chans = []
            anyvalid = None
            for b in bands:
                a = read_band(src, b, window, out_shape).astype(np.float32)
                anyvalid = (a > 0) if anyvalid is None else (anyvalid | (a > 0))
                chans.append(self._stretch(a, pct))
            img = np.dstack(chans)
            # outside the polygon / cube (0 in every channel) shown
            # white; a channel clipped to 0 inside the canopy just renders dark
            img[~anyvalid] = 1.0
            return img, None
        if mode == 3:
            b = int(np.argmin(np.abs(wl - self.sp_nm[0].value()))) + 1
            a = read_band(src, b, window, out_shape).astype(np.float32) / scale
            a[a <= 0] = np.nan
            return a, f"reflectance @ {wl[b-1]:.0f} nm"
        acr = self.cb_index.currentData()
        cube = self._open_cube()
        if cube is None or cube._src is not src:
            # per-plot file: wrap it in a light-weight cube-like object
            class _C:  # noqa
                pass
            cube = _C(); cube._src = src; cube.wavelengths = wl; cube.hdr = ""
        a = compute_index_window(cube, acr, window, excluded=self._excluded(), catalogue=self.cat, out_shape=out_shape)
        return a, acr

    def _draw(self, ax, img, label, extent, title):
        if img.ndim == 3:
            ax.imshow(img, extent=extent, interpolation="nearest")
        else:
            v = img[np.isfinite(img)]
            if v.size:
                lo, hi = np.percentile(v, [2, 98])
            else:
                lo, hi = 0, 1
            im = ax.imshow(img, extent=extent, cmap="RdYlGn" if label and label not in ("reflectance",) and not label.startswith("reflectance") else "gray",
                           vmin=lo, vmax=hi, interpolation="nearest")
            self.canvas.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    def _overlay(self, ax, plots, extent):
        if plots is None or not self.cb_grid.isChecked():
            return
        x0, x1, y0, y1 = extent
        for _, r in plots.iterrows():
            g = r.geometry
            if g.bounds[2] < x0 or g.bounds[0] > x1 or g.bounds[3] < y0 or g.bounds[1] > y1:
                continue
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="cyan", lw=0.8)
            c = g.centroid
            ax.annotate(str(r["Plot_ID"]), (c.x, c.y), color="black", fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))

    def render(self):
        import rasterio
        from rasterio.windows import from_bounds, Window
        self.canvas.fig.clear(); self._sources = []
        src_mode = self.cb_source.currentIndex()
        maxpx = self.sp_maxpx.value()
        try:
            if src_mode in (0, 1):
                cube = self._open_cube()
                if cube is None:
                    QMessageBox.warning(self, "No cube", "Pick the VNIR cube on the Indices sub-tab (or Biomass tab)."); return
                src, wl, scale = cube._src, np.asarray(cube.wavelengths, float), guess_scale(cube)
                if src_mode == 0:
                    w = Window(0, 0, src.width, src.height); sel = self._plots
                else:
                    sel = self._selected_plots()
                    if sel is None:
                        QMessageBox.warning(self, "No plots", "Select one or more plots in the list."); return
                    minx, miny, maxx, maxy = sel.total_bounds
                    m = 0.5
                    w = from_bounds(minx - m, miny - m, maxx + m, maxy + m, src.transform).round_offsets().round_lengths()
                    w = w.intersection(Window(0, 0, src.width, src.height))
                d = max(1, int(np.ceil(max(w.width, w.height) / maxpx)))
                out_shape = (int(w.height // d), int(w.width // d)) if d > 1 else None
                img, label = self._read_display(src, wl, w, out_shape, scale)
                tr = rasterio.windows.transform(w, src.transform)
                extent = (tr.c, tr.c + tr.a * w.width, tr.f + tr.e * w.height, tr.f)
                ax = self.canvas.fig.add_subplot(111)
                title = os.path.basename(cube.path) + (f"  (decimated x{d})" if d > 1 else "  (full resolution)")
                self._draw(ax, img, label, extent, title)
                self._overlay(ax, sel if src_mode == 1 else self._plots, extent)
                self._sources.append(dict(ax=ax, src=src, wl=wl, scale=scale, transform=src.transform, label="cube"))
            else:
                d = self.ed_dir.text().strip()
                if not d or not os.path.isdir(d):
                    QMessageBox.warning(self, "No folder", "Point 'Per-plot files' at a folder of plot_<id>_*.tif cubes."); return
                files = {}
                for fn in os.listdir(d):
                    m = re.match(r"^plot_(\d+)[_.]", fn)
                    if m and fn.lower().endswith(".tif"):
                        files[int(m.group(1))] = os.path.join(d, fn)
                sel = self._selected_plots()
                ids = [int(p) for p in sel["Plot_ID"]] if sel is not None else sorted(files)[:12]
                ids = [i for i in ids if i in files]
                if not ids:
                    QMessageBox.warning(self, "No files", "None of the selected plots has a plot_<id>_*.tif in that folder."); return
                if len(ids) > 30:
                    QMessageBox.information(self, "Gallery", f"Showing the first 30 of {len(ids)} plots."); ids = ids[:30]
                n = len(ids); nc = int(np.ceil(np.sqrt(n * 1.6))); nr = int(np.ceil(n / nc))
                for k, pid in enumerate(ids):
                    src = rasterio.open(files[pid])
                    wl = np.array([float(x.split()[0]) for x in src.descriptions]) if src.descriptions[0] else \
                        np.linspace(400, 1000, src.count)
                    dd = max(1, int(np.ceil(max(src.width, src.height) / max(200, maxpx // nc))))
                    out_shape = (src.height // dd, src.width // dd) if dd > 1 else None
                    img, label = self._read_display(src, wl, None, out_shape, 10000.0)
                    b = src.bounds; extent = (b.left, b.right, b.bottom, b.top)
                    ax = self.canvas.fig.add_subplot(nr, nc, k + 1)
                    self._draw(ax, img, label, extent, f"plot {pid}")
                    ax.set_xticks([]); ax.set_yticks([])
                    if self._plots is not None:
                        self._overlay(ax, self._plots[self._plots["Plot_ID"] == pid], extent)
                    self._sources.append(dict(ax=ax, src=src, wl=wl, scale=10000.0, transform=src.transform, label=f"plot {pid}"))
            self.canvas.draw()
            self.lbl.setText("Rendered. Click a pixel to plot its spectrum (all bands).")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Render failed", f"{e}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        for s in self._sources:
            if s["ax"] is ev.inaxes:
                break
        else:
            return
        try:
            from rasterio.windows import Window
            src = s["src"]; tr = s["transform"]
            col, row = ~tr * (ev.xdata, ev.ydata)
            col, row = int(col), int(row)
            if not (0 <= col < src.width and 0 <= row < src.height):
                return
            spec = src.read(window=Window(col, row, 1, 1)).ravel().astype(float)
            wl = s["wl"]; scale = s["scale"]
            refl = np.where(spec > 0, spec / scale, np.nan)
            self.spec.fig.clear(); ax = self.spec.fig.add_subplot(111)
            ax.plot(wl, refl, lw=1.2, color="darkgreen")
            for lo, hi in self._excluded():
                ax.axvspan(max(lo, wl[0]), min(hi, wl[-1]), color="grey", alpha=0.18, lw=0)
            n0 = int(np.sum(spec == 0))
            ax.set_title(f"{s['label']}  E {ev.xdata:.2f}  N {ev.ydata:.2f}   ({n0} of {len(spec)} bands = 0 / clipped; grey = excluded ranges)", fontsize=8)
            ax.set_xlabel("wavelength (nm)", fontsize=8); ax.set_ylabel("reflectance", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
            self.spec.draw()
        except Exception as e:
            self.lbl.setText(f"spectrum failed: {e}")

    def _save_png(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save figure", "", "PNG (*.png)")
        if p:
            self.canvas.fig.savefig(p, dpi=200, bbox_inches="tight")


# ======================================================================
class SpectralTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>VNIR Spectral Features: per-plot features, spectral indices (incl. water) and band viewer</h2>"))
        self.tabs = QTabWidget()
        self.indices = IndicesPanel()
        self.view = ViewPanel(indices_panel=self.indices)
        self.tabs.addTab(self.indices, "Indices (Awesome Spectral Indices catalogue)")
        self.tabs.addTab(self.view, "View bands / plots / orthomosaic")
        v.addWidget(self.tabs, stretch=1)
