"""
Visualize tab.

Modes:
  - 2D: top-down canopy raster + saved grid overlay (always available)
  - 3D LiDAR viewer (PyVista): the whole trial cloud (decimated for display)
    or one / several per-plot LAS files written by the Traits tab, coloured
    by elevation, height above the plot ground, intensity or plot ID.
"""

from __future__ import annotations
import os
import re
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QToolBar,
    QMessageBox, QSizePolicy, QGraphicsPolygonItem, QGroupBox, QFormLayout,
    QComboBox, QLineEdit, QListWidget, QAbstractItemView, QSpinBox,
    QDoubleSpinBox, QFileDialog, QCheckBox, QSplitter
)

from phenoapp.core.project import state
from phenoapp.core import load_grid
from phenoapp.ui.canopy_view import CanopyView, shp_to_qpoly


class VisualizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._3d_window = None
        self._plots = None
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        tb = QToolBar()
        tb.addAction("Reload",        self.reload)
        tb.addAction("Fit View",      self.fit_view)
        tb.addSeparator()
        tb.addAction("🧊 Open 3D View (whole trial)",  self.open_3d)
        v.addWidget(tb)

        split = QSplitter(Qt.Horizontal)
        self.view = CanopyView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.coords_changed.connect(self._coords)
        split.addWidget(self.view)

        # ---- LiDAR plot viewer panel ----
        panel = QGroupBox("LiDAR plot viewer (3D)")
        panel.setMaximumWidth(400)
        f = QFormLayout(panel)
        self.cb_src = QComboBox()
        self.cb_src.addItems(["Whole trial point cloud (decimated)",
                              "Per-plot LAS files (Traits tab output)"])
        self.cb_src.currentIndexChanged.connect(
            lambda i: (self.ed_dir.setEnabled(i == 1), self.lst.setEnabled(i == 1)))
        f.addRow("Source:", self.cb_src)
        row = QHBoxLayout(); self.ed_dir = QLineEdit()
        b = QPushButton("..."); b.setMaximumWidth(32); b.clicked.connect(self._pick_dir)
        row.addWidget(self.ed_dir); row.addWidget(b)
        f.addRow("Per-plot folder:", row)
        self.lst = QListWidget(); self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst.setMaximumHeight(180)
        f.addRow("Plots:", self.lst)
        rb = QHBoxLayout()
        b_all = QPushButton("All"); b_all.clicked.connect(self.lst.selectAll)
        b_none = QPushButton("None"); b_none.clicked.connect(self.lst.clearSelection)
        rb.addWidget(b_all); rb.addWidget(b_none)
        f.addRow("", rb)
        self.cb_color = QComboBox()
        self.cb_color.addItems(["Elevation (z)", "Height above plot ground (z - plot p1)", "Intensity", "Plot ID"])
        f.addRow("Colour by:", self.cb_color)
        self.sp_pt = QDoubleSpinBox(); self.sp_pt.setRange(0.5, 10); self.sp_pt.setValue(2.0)
        self.sp_pt.setSingleStep(0.5)
        f.addRow("Point size:", self.sp_pt)
        self.sp_max = QSpinBox(); self.sp_max.setRange(100_000, 50_000_000); self.sp_max.setSingleStep(500_000)
        self.sp_max.setValue(5_000_000)
        self.sp_max.setToolTip("Display decimation only - analysis always uses every point.")
        f.addRow("Max points:", self.sp_max)
        self.cb_poly = QCheckBox("Draw plot outlines + IDs"); self.cb_poly.setChecked(True)
        f.addRow("", self.cb_poly)
        self.btn_3d = QPushButton("Open 3D view")
        self.btn_3d.clicked.connect(self._open_3d_selected)
        f.addRow("", self.btn_3d)
        self.ed_dir.setEnabled(False); self.lst.setEnabled(False)
        split.addWidget(panel)
        split.setStretchFactor(0, 4); split.setStretchFactor(1, 1)
        v.addWidget(split, stretch=1)

        self.lbl = QLabel("Click 'Reload' to load the canopy raster and saved grid.")
        v.addWidget(self.lbl)

    # -----
    def showEvent(self, ev):
        super().showEvent(ev)
        s = state()
        if not self.ed_dir.text().strip() and s.out_dir:
            self.ed_dir.setText(s.out_dir)
        if self._plots is None:
            self._load_plots()

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with per-plot LAS files (plot_<id>_*.las)")
        if d:
            self.ed_dir.setText(d)

    def _load_plots(self):
        s = state()
        self.lst.clear(); self._plots = None
        for cand in (s.saved_grid, s.grid_path):
            if cand and os.path.exists(cand):
                try:
                    self._plots = load_grid(cand, target_crs=s.work_crs)
                except Exception:
                    self._plots = None
                break
        if self._plots is not None:
            names = self._plots["B/R"] if "B/R" in self._plots else self._plots["Plot_ID"]
            for pid, nm in zip(self._plots["Plot_ID"], names):
                self.lst.addItem(f"{pid}  ({nm})")

    def reload(self):
        s = state()
        if s.canopy_tif and os.path.exists(s.canopy_tif):
            self.view.load_canopy_raster(s.canopy_tif)
        # Prefer saved (aligned) grid over input grid
        for cand in (s.saved_grid, s.grid_path):
            if cand and os.path.exists(cand):
                gdf = load_grid(cand, target_crs=s.work_crs)
                self._draw_grid(gdf)
                self.lbl.setText(f"Showing {len(gdf)} plots from {os.path.basename(cand)}")
                break
        self.view.fit_view()
        self._load_plots()

    def fit_view(self):
        self.view.fit_view()

    def _draw_grid(self, gdf):
        # Clear existing overlay polygons
        for it in list(self.view.scene().items()):
            if isinstance(it, QGraphicsPolygonItem):
                self.view.scene().removeItem(it)
        for poly in gdf.geometry:
            it = QGraphicsPolygonItem(shp_to_qpoly(poly))
            it.setPen(QPen(QColor(80, 220, 80), 0.10))
            it.setBrush(QBrush(QColor(80, 220, 80, 40)))
            self.view.scene().addItem(it)

    def _coords(self, wx, wy):
        pass

    # ---- 3D ----
    def _pv(self):
        try:
            import pyvista as pv
            return pv
        except ImportError:
            QMessageBox.warning(self, "PyVista not installed",
                "3D viewer requires PyVista. Install with:\n\n"
                "    pip install pyvista pyvistaqt\n\n"
                "Then restart the app.")
            return None

    @staticmethod
    def _decimate(arrays, n_max):
        n = len(arrays[0])
        if n > n_max:
            step = n // n_max + 1
            return [a[::step] for a in arrays]
        return arrays

    def _add_polygons(self, plotter, gdf, x, y, z):
        """Draw each polygon just above the local canopy top, with its ID."""
        z_fallback = float(np.quantile(z, 0.99)) + 0.10 if len(z) else 0.0
        labels, pts = [], []
        for _, r in gdf.iterrows():
            poly = r.geometry
            ring = list(poly.exterior.coords)
            if len(ring) < 2:
                continue
            minx, miny, maxx, maxy = poly.bounds
            m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
            z_top = float(np.quantile(z[m], 0.99)) + 0.10 if m.any() else z_fallback
            segs = []
            for i in range(len(ring) - 1):
                segs.append((ring[i][0], ring[i][1], z_top))
                segs.append((ring[i + 1][0], ring[i + 1][1], z_top))
            plotter.add_lines(np.asarray(segs), color="red", width=2)
            c = poly.centroid
            labels.append(str(r["Plot_ID"])); pts.append((c.x, c.y, z_top + 0.15))
        if pts:
            try:
                plotter.add_point_labels(np.asarray(pts), labels, font_size=12, text_color="yellow",
                                         point_size=1, shape=None, always_visible=True)
            except Exception:
                pass

    def _show_cloud(self, x, y, z, intensity, plot_id, gdf, title):
        pv = self._pv()
        if pv is None:
            return
        mode = self.cb_color.currentIndex()
        cloud = pv.PolyData(np.column_stack([x, y, z]))
        if mode == 0:
            scal, name, cmap = z, "elevation (m)", "viridis"
        elif mode == 1:
            scal, name, cmap = plot_id[1], "height above plot ground (m)", "viridis"
        elif mode == 2 and intensity is not None:
            scal, name, cmap = intensity.astype(np.float32), "intensity", "gray"
        else:
            scal, name, cmap = plot_id[0].astype(np.float32), "Plot_ID", "tab20"
        cloud[name] = scal
        plotter = pv.Plotter(title=title, window_size=(1200, 900))
        plotter.set_background("black")
        plotter.add_points(cloud, scalars=name, cmap=cmap, point_size=float(self.sp_pt.value()),
                           render_points_as_spheres=False, show_scalar_bar=True)
        if gdf is not None and self.cb_poly.isChecked():
            self._add_polygons(plotter, gdf, x, y, z)
        plotter.add_axes()
        plotter.show()

    def open_3d(self):
        """Whole trial cloud (project LAS), decimated for display."""
        if self._pv() is None:
            return
        s = state()
        if not s.las_path or not os.path.exists(s.las_path):
            QMessageBox.warning(self, "No LAS", "Load a project on the Project tab first.")
            return
        try:
            import laspy
            las = laspy.read(s.las_path)
            x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
            inten = np.asarray(las.intensity) if "intensity" in las.point_format.dimension_names else None
            arrays = [x, y, z] + ([inten] if inten is not None else [])
            arrays = self._decimate(arrays, self.sp_max.value())
            x, y, z = arrays[:3]; inten = arrays[3] if inten is not None else None
            gdf = self._plots
            pid = np.zeros(len(x), np.int32); hag = z - np.quantile(z, 0.01)
            if gdf is not None and self.cb_color.currentIndex() in (1, 3):
                from shapely import contains_xy
                for _, r in gdf.iterrows():
                    minx, miny, maxx, maxy = r.geometry.bounds
                    m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
                    if m.any():
                        idx = np.where(m)[0][contains_xy(r.geometry, x[m], y[m])]
                        pid[idx] = int(r["Plot_ID"])
                        if len(idx):
                            hag[idx] = z[idx] - np.quantile(z[idx], 0.01)
            self._show_cloud(x, y, z, inten, (pid, hag), gdf, "Point Cloud Viewer - whole trial")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "3D failed", f"{e}\n\n{traceback.format_exc()}")

    def load_selected_plot_points(self):
        """Read the selected per-plot LAS files (decimated). Returns
        (x, y, z, intensity, plot_id, height_above_plot_ground, ids) or None."""
        d = self.ed_dir.text().strip()
        if not d or not os.path.isdir(d):
            return None
        files = {}
        for fn in os.listdir(d):
            m = re.match(r"^plot_(\d+)[_.]", fn)
            if m and fn.lower().endswith(".las"):
                files[int(m.group(1))] = os.path.join(d, fn)
        rows = sorted(i.row() for i in self.lst.selectedIndexes())
        if self._plots is not None and rows:
            ids = [int(p) for p in self._plots.iloc[rows]["Plot_ID"]]
        else:
            ids = sorted(files)[:1]
        ids = [i for i in ids if i in files]
        if not ids:
            return None
        import laspy
        X, Y, Z, I, P, H = [], [], [], [], [], []
        per_plot_max = max(50_000, self.sp_max.value() // len(ids))
        for pid in ids:
            las = laspy.read(files[pid])
            x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
            inten = (np.asarray(las.intensity) if "intensity" in las.point_format.dimension_names
                     else np.zeros(len(x)))
            x, y, z, inten = self._decimate([x, y, z, inten], per_plot_max)
            X.append(x); Y.append(y); Z.append(z); I.append(inten)
            P.append(np.full(len(x), pid, np.int32))
            H.append(z - (np.quantile(z, 0.01) if len(z) else 0.0))
        return (np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(I),
                np.concatenate(P), np.concatenate(H), ids)

    def _open_3d_selected(self):
        if self.cb_src.currentIndex() == 0:
            self.open_3d(); return
        if self._pv() is None:
            return
        try:
            res = self.load_selected_plot_points()
            if res is None:
                QMessageBox.warning(self, "No files",
                                    "Point 'Per-plot folder' at the Traits-tab plots_las output and select plots "
                                    "that have a plot_<id>_*.las there.")
                return
            x, y, z, inten, pid, hag, ids = res
            gdf = (self._plots[self._plots["Plot_ID"].astype(int).isin(ids)]
                   if self._plots is not None else None)
            self._show_cloud(x, y, z, inten, (pid, hag), gdf, f"Point Cloud Viewer - {len(ids)} plot(s)")
            self.lbl.setText(f"3D: {len(ids)} plot(s), {len(x):,} points shown.")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "3D failed", f"{e}\n\n{traceback.format_exc()}")
