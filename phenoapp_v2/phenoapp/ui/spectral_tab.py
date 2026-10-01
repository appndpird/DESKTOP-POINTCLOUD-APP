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
    RECOMMENDED_BIOMASS, DEFAULT_EXCLUDED_NM, DOMAIN_LABELS, parse_excluded,
    summarise_indices, composite_bands, guess_scale, read_band)


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
            "Defaults come from the GOBI/GRYFN cubes: the first bands (399-406 nm) carry a calibration "
            "spike, 410-500 nm is mostly clipped to 0 in canopy, 761 nm is the O2-A absorption residual "
            "and 930-960 nm the water-vapour feature. Press 'Read wavelengths' after editing.")
        f.addRow("Excluded ranges (nm):", self.ed_excl)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Per pixel, then plot mean (recommended; reports sd + valid fraction)",
                               "On the plot mean spectrum (fast; same numbers as the Biomass tab)"])
        f.addRow("Computation:", self.cb_mode)
        self.lbl_region = QLabel("")
        f.addRow("Sampling region:", self.lbl_region)
        self.cb_rasters = QCheckBox("Also write index rasters (GeoTIFF, trial extent, for QGIS) for the ticked indices")
        f.addRow("", self.cb_rasters)
        v.addWidget(top)

        # ---- catalogue tree ----
        mid = QGroupBox("Indices (tick to compute; hover for formula and reference)")
        mv = QVBoxLayout(mid)
        bar = QHBoxLayout()
        self.ed_search = QLineEdit(); self.ed_search.setPlaceholderText("filter by acronym / name / band...")
        self.ed_search.textChanged.connect(self._filter)
        bar.addWidget(self.ed_search, stretch=1)
        b_rec = QPushButton("Recommended (biomass)"); b_rec.clicked.connect(self._select_recommended)
        b_rec.setToolTip("Red-edge family + water and structure indices that carried biomass information on the "
                         "2025 DPIRD trials (NDRE variants, REP, LCI, S2REP, MTCI, CIRE, OSAVI, kNDVI, NIRv, WBI...).")
        b_clr = QPushButton("Clear"); b_clr.clicked.connect(lambda: self._set_all(False))
        bar.addWidget(b_rec); bar.addWidget(b_clr)
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
        row3.addWidget(self.btn_run); row3.addStretch()
        v.addLayout(row3)
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
            + "  - set on the Traits tab")

    def _pick_vnir(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick VNIR ENVI cube (.bin)", "",
                                           "ENVI cubes (*.bin *.img *.dat);;All files (*)")
        if p:
            self.ed_vnir.setText(p); state().vnir_path = p
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
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if e.acronym in checked else Qt.Unchecked)
                it.setData(0, Qt.UserRole, e.acronym)
                for c in range(4):
                    it.setToolTip(c, e.tooltip())
                top.addChild(it); n += 1
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
        return [it.data(0, Qt.UserRole) for it in self._iter_leaves() if it.checkState(0) == Qt.Checked]

    def _set_all(self, on):
        self.tree.blockSignals(True)
        for it in self._iter_leaves():
            it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_count()

    def _select_recommended(self):
        self.tree.blockSignals(True)
        for it in self._iter_leaves():
            if it.data(0, Qt.UserRole) in RECOMMENDED_BIOMASS:
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
            # nodata (0 in every channel: outside the polygon / cube) shown
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
        v.addWidget(QLabel("<h2>VNIR spectral indices &amp; band viewer</h2>"))
        self.tabs = QTabWidget()
        self.indices = IndicesPanel()
        self.view = ViewPanel(indices_panel=self.indices)
        self.tabs.addTab(self.indices, "Indices (Awesome Spectral Indices catalogue)")
        self.tabs.addTab(self.view, "View bands / plots / orthomosaic")
        v.addWidget(self.tabs, stretch=1)
