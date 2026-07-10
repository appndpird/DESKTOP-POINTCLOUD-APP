"""
Visualize tab.

Two modes:
  - 2D: top-down canopy raster + saved grid overlay (always available)
  - 3D: full point cloud rendering (PyVista, optional)
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QToolBar,
    QStackedWidget, QMessageBox, QSizePolicy, QGraphicsPolygonItem
)

from phenoapp.core.project import state
from phenoapp.core import load_grid
from phenoapp.ui.canopy_view import CanopyView, shp_to_qpoly


class VisualizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._3d_window = None
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        tb = QToolBar()
        tb.addAction("Reload",        self.reload)
        tb.addAction("Fit View",      self.fit_view)
        tb.addSeparator()
        tb.addAction("🧊 Open 3D View",  self.open_3d)
        v.addWidget(tb)

        self.view = CanopyView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.coords_changed.connect(self._coords)
        v.addWidget(self.view, stretch=1)

        self.lbl = QLabel("Click 'Reload' to load the canopy raster and saved grid.")
        v.addWidget(self.lbl)

    # -----
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
    def open_3d(self):
        try:
            import pyvista as pv
        except ImportError:
            QMessageBox.warning(self, "PyVista not installed",
                "3D viewer requires PyVista. Install with:\n\n"
                "    pip install pyvista pyvistaqt\n\n"
                "Then restart the app.")
            return
        s = state()
        if not s.las_path or not os.path.exists(s.las_path):
            QMessageBox.warning(self, "No LAS",
                "Load a project on the Project tab first.")
            return

        try:
            import laspy, numpy as np
            las = laspy.read(s.las_path)
            x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
            # Decimate for display only (does not affect any analysis)
            n = len(x)
            if n > 5_000_000:
                step = n // 5_000_000 + 1
                x = x[::step]; y = y[::step]; z = z[::step]
            pts = np.column_stack([x, y, z])
            cloud = pv.PolyData(pts)
            cloud["elevation"] = z

            plotter = pv.Plotter(title="Point Cloud Viewer", window_size=(1200, 900))
            plotter.set_background("black")
            plotter.add_points(cloud, scalars="elevation", cmap="viridis",
                               point_size=2, render_points_as_spheres=False,
                               show_scalar_bar=True)
            # Overlay polygons. pv.add_lines expects pairs of points (each
            # consecutive pair defines one segment), so expand the polygon
            # ring [p0, p1, p2, ..., p0] into [p0,p1, p1,p2, p2,p3, ...].
            # Each polygon is drawn just above the LOCAL canopy in that plot
            # (per-plot p99 + 10 cm), so the grid hugs the surface even when
            # the field has tall outliers (vehicles, sensor spikes) elsewhere.
            for cand in (s.saved_grid, s.grid_path):
                if cand and os.path.exists(cand):
                    gdf = load_grid(cand, target_crs=s.work_crs)
                    # Fallback height used when a polygon has no points in
                    # the (decimated) cloud - sit at field-wide p99 + 10 cm.
                    z_fallback = float(np.quantile(z, 0.99)) + 0.10
                    for poly in gdf.geometry:
                        ring = list(poly.exterior.coords)
                        if len(ring) < 2:
                            continue
                        # bbox prefilter, then per-plot height from canopy top
                        minx, miny, maxx, maxy = poly.bounds
                        m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
                        if m.any():
                            z_top = float(np.quantile(z[m], 0.99)) + 0.10
                        else:
                            z_top = z_fallback
                        segs = []
                        for i in range(len(ring) - 1):
                            segs.append((ring[i][0],   ring[i][1],   z_top))
                            segs.append((ring[i+1][0], ring[i+1][1], z_top))
                        plotter.add_lines(np.asarray(segs), color="red", width=2)
                    break
            plotter.show()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "3D failed",
                f"{e}\n\n{traceback.format_exc()}")
