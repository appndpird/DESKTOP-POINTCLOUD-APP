"""
Edit tab: align plot polygons over the canopy raster.

Click/drag to move; rubber-band for multi-select; toolbar/keyboard for
rotate, scale, undo, save.
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QThread
from PyQt5.QtGui import QPen, QBrush, QColor, QKeySequence, QPolygonF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolBar,
    QShortcut, QMessageBox, QAction, QRubberBand, QGraphicsItem,
    QGraphicsPolygonItem, QSizePolicy, QComboBox, QFileDialog
)
from shapely.affinity import (
    rotate as sh_rotate, translate as sh_translate, scale as sh_scale
)
from shapely.geometry import Polygon as ShpPolygon
from shapely.ops import unary_union
import geopandas as gpd
import math


def _resize_plot_rect(poly, dw: float, dl: float):
    """Adjust a plot polygon's width (short side) and length (long side) by
    dw and dl metres while preserving its centroid and orientation. Uses the
    minimum rotated rectangle to recover the plot's intrinsic axes."""
    mbr = poly.minimum_rotated_rectangle
    coords = list(mbr.exterior.coords)[:-1]   # 4 corners
    if len(coords) < 4:
        return poly
    cx = sum(p[0] for p in coords) / 4.0
    cy = sum(p[1] for p in coords) / 4.0
    # two adjacent edges from corner 0
    a = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    b = (coords[3][0] - coords[0][0], coords[3][1] - coords[0][1])
    la = math.hypot(*a); lb = math.hypot(*b)
    if la == 0 or lb == 0:
        return poly
    # width = shorter side, length = longer side
    if la <= lb:
        w, l = la, lb
        uw = (a[0]/la, a[1]/la);  ul = (b[0]/lb, b[1]/lb)
    else:
        w, l = lb, la
        uw = (b[0]/lb, b[1]/lb);  ul = (a[0]/la, a[1]/la)
    w_new = max(0.05, w + dw)
    l_new = max(0.05, l + dl)
    hw, hl = w_new / 2.0, l_new / 2.0
    corners = [
        (cx - uw[0]*hw - ul[0]*hl, cy - uw[1]*hw - ul[1]*hl),
        (cx + uw[0]*hw - ul[0]*hl, cy + uw[1]*hw - ul[1]*hl),
        (cx + uw[0]*hw + ul[0]*hl, cy + uw[1]*hw + ul[1]*hl),
        (cx - uw[0]*hw + ul[0]*hl, cy - uw[1]*hw + ul[1]*hl),
    ]
    return ShpPolygon(corners)

from phenoapp.core.project import state
from phenoapp.core import load_grid, save_grid, auto_align_grid, GRID_FILE_FILTER
from phenoapp.ui.canopy_view import CanopyView, shp_to_qpoly, qpoly_to_shp


class PlotItem(QGraphicsPolygonItem):
    def __init__(self, plot_idx, shp_poly):
        super().__init__(shp_to_qpoly(shp_poly))
        self.plot_idx = plot_idx
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._refresh_style()

    def _refresh_style(self):
        if self.isSelected():
            self.setPen(QPen(QColor(255, 220, 0), 0.10))
            self.setBrush(QBrush(QColor(255, 220, 0, 70)))
        else:
            self.setPen(QPen(QColor(255, 60, 60), 0.07))
            self.setBrush(QBrush(QColor(255, 0, 0, 0)))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._refresh_style()
        return super().itemChange(change, value)

    def shapely(self):
        return qpoly_to_shp(self.polygon())


class EditView(CanopyView):
    """Subclass that adds rubber-band selection, click-select, and drag-move."""
    def __init__(self, owner_tab):
        super().__init__()
        self._owner = owner_tab
        self._move_anchor = None
        self._move_origins = None
        self._rubber = None
        self._rubber_start_view = None
        self._move_active = False

    def mousePressEvent(self, event):
        # Pan via middle-click or Ctrl+left -> base class
        if event.button() == Qt.MiddleButton or \
           (event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier)):
            return super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            mods = event.modifiers()
            item = self._plot_at(scene_pos)
            if item is not None:
                shift = bool(mods & Qt.ShiftModifier)
                if shift:
                    item.setSelected(not item.isSelected())
                else:
                    if not item.isSelected():
                        self.scene().clearSelection()
                        item.setSelected(True)
                self._move_anchor = scene_pos
                self._move_origins = {it: it.polygon()
                                      for it in self.scene().selectedItems()
                                      if isinstance(it, PlotItem)}
                self._move_active = True
                event.accept()
                return
            else:
                if not (mods & Qt.ShiftModifier):
                    self.scene().clearSelection()
                self._rubber_start_view = event.pos()
                self._rubber = QRubberBand(QRubberBand.Rectangle, self)
                self._rubber.setGeometry(0, 0, 0, 0)
                self._rubber.show()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        wx, wy = scene_pos.x(), -scene_pos.y()
        self.coords_changed.emit(wx, wy)
        if self._move_active and self._move_anchor is not None:
            dx = scene_pos.x() - self._move_anchor.x()
            dy = scene_pos.y() - self._move_anchor.y()
            for item, orig in self._move_origins.items():
                item.setPolygon(QPolygonF(
                    [QPointF(p.x() + dx, p.y() + dy) for p in orig]))
            event.accept()
            return
        if self._rubber is not None:
            r = QRectF(self._rubber_start_view, event.pos()).normalized().toRect()
            self._rubber.setGeometry(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._move_active:
            scene_pos = self.mapToScene(event.pos())
            moved = (abs(scene_pos.x() - self._move_anchor.x()) > 1e-3 or
                     abs(scene_pos.y() - self._move_anchor.y()) > 1e-3)
            self._move_anchor = None
            self._move_origins = None
            self._move_active = False
            if moved:
                self._owner.push_undo()
            event.accept()
            return
        if self._rubber is not None:
            r = self._rubber.geometry()
            scene_rect = self.mapToScene(r).boundingRect()
            for it in self.scene().items(scene_rect):
                if isinstance(it, PlotItem):
                    it.setSelected(True)
            self._rubber.hide()
            self._rubber.deleteLater()
            self._rubber = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _plot_at(self, scene_pos):
        for it in self.scene().items(scene_pos):
            if isinstance(it, PlotItem):
                return it
        return None


class _RefineWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok  = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, kw):
        super().__init__(); self._kw = kw

    def run(self):
        try:
            kw = self._kw
            from phenoapp.core.grid_refine import refine_grid_to_canopy, render_refine_qa
            chm = kw["chm"]
            if not os.path.exists(chm):
                from phenoapp.core import LASManager
                from phenoapp.core.surface_models import build_surface_models
                mgr = LASManager(kw["las_path"], kw["work_crs"], kw["use_smrf"])
                self.progress.emit(2, "Loading LAS to build the CHM...")
                mgr.load(progress_cb=lambda p, m: self.progress.emit(int(2 + p * 0.3), m))
                base = os.path.splitext(os.path.basename(kw["las_path"]))[0]
                res = build_surface_models(mgr.x, mgr.y, mgr.z, kw["gdf"], os.path.dirname(chm), base, kw["work_crs"],
                                           hag=mgr.hag, dsm_res=0.10, dtm_mode="exterior",
                                           progress_cb=lambda p, m: self.progress.emit(int(35 + p * 0.3), m))
                chm = res["paths"]["chm"]
            gdf, rep, diag = refine_grid_to_canopy(chm, kw["gdf"], progress_cb=lambda p, m: self.progress.emit(int(65 + p * 0.3), m))
            save_grid(gdf, kw["out_shp"])
            stem = os.path.splitext(kw["out_shp"])[0]
            rep.round(3).to_csv(stem + "_report.csv", index=False)
            qa = render_refine_qa(chm, kw["gdf"], gdf, rep, stem + "_qa.png")
            self.progress.emit(100, "done")
            self.done_ok.emit(dict(gdf=gdf, diag=diag, out_shp=kw["out_shp"], report_csv=stem + "_report.csv", qa_png=qa))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class EditTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._undo_stack = []
        self._original_geoms = []
        self._grid_attrs = None
        self._grid_crs = None
        self._bg_mode = "canopy"   # "canopy" or "ortho"
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        tb = QToolBar()
        tb.addWidget(QLabel(" Base: "))
        self.cb_base = QComboBox()
        self.cb_base.addItems(["Canopy raster (LiDAR)", "Orthomosaic (RGB)"])
        self.cb_base.currentIndexChanged.connect(self._on_base_changed)
        tb.addWidget(self.cb_base)
        tb.addAction("Load Orthomosaic…", self.load_ortho)
        tb.addAction("Load Grid file…",   self.load_grid_file)
        tb.addSeparator()
        tb.addAction("Reload from Project", self.reload_grid_and_canopy)
        tb.addSeparator()
        tb.addAction("Select All",  self.select_all)
        tb.addAction("Deselect",    self.deselect_all)
        tb.addSeparator()
        tb.addAction("Rot -1°",   lambda: self.rotate_sel(-1))
        tb.addAction("Rot +1°",   lambda: self.rotate_sel(+1))
        tb.addAction("Rot -0.1°", lambda: self.rotate_sel(-0.1))
        tb.addAction("Rot +0.1°", lambda: self.rotate_sel(+0.1))
        tb.addAction("Scale +1%", lambda: self.scale_sel(1.01))
        tb.addAction("Scale -1%", lambda: self.scale_sel(0.99))
        tb.addSeparator()
        tb.addAction("W +5cm",  lambda: self.resize_sel(+0.05, 0))
        tb.addAction("W -5cm",  lambda: self.resize_sel(-0.05, 0))
        tb.addAction("L +5cm",  lambda: self.resize_sel(0, +0.05))
        tb.addAction("L -5cm",  lambda: self.resize_sel(0, -0.05))
        tb.addAction("W +25cm", lambda: self.resize_sel(+0.25, 0))
        tb.addAction("W -25cm", lambda: self.resize_sel(-0.25, 0))
        tb.addAction("L +25cm", lambda: self.resize_sel(0, +0.25))
        tb.addAction("L -25cm", lambda: self.resize_sel(0, -0.25))
        tb.addSeparator()
        tb.addAction("⟲ Auto-align to canopy", self.auto_align)
        tb.addAction("✚ Refine plots to crop (CHM)", self.refine_to_crop)
        tb.addAction("Undo (Z)",  self.undo)
        tb.addAction("Reset",     self.reset)
        tb.addSeparator()
        tb.addAction("Fit View",  self.fit_view)
        tb.addAction("💾 Save Grid", self.save_grid)
        v.addWidget(tb)

        self.view = EditView(self)
        self.view.coords_changed.connect(self._on_coords)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.view, stretch=1)

        bar = QHBoxLayout()
        self.lbl_coords = QLabel("E --, N --")
        self.lbl_sel    = QLabel("Selected: 0")
        self.lbl_status = QLabel("Load project then click Reload from Project.")
        bar.addWidget(self.lbl_coords)
        bar.addWidget(self.lbl_sel)
        bar.addStretch()
        bar.addWidget(self.lbl_status)
        v.addLayout(bar)

        # keyboard shortcuts
        for k, fn in [
            ("A", self.select_all), ("D", self.deselect_all),
            ("Z", self.undo), ("S", self.save_grid),
            ("Q",       lambda: self.rotate_sel(-1)),
            ("E",       lambda: self.rotate_sel(+1)),
            ("Shift+Q", lambda: self.rotate_sel(-5)),
            ("Shift+E", lambda: self.rotate_sel(+5)),
            ("Left",        lambda: self.translate_sel(-0.1, 0)),
            ("Right",       lambda: self.translate_sel(+0.1, 0)),
            ("Up",          lambda: self.translate_sel(0, +0.1)),
            ("Down",        lambda: self.translate_sel(0, -0.1)),
            ("Shift+Left",  lambda: self.translate_sel(-1.0, 0)),
            ("Shift+Right", lambda: self.translate_sel(+1.0, 0)),
            ("Shift+Up",    lambda: self.translate_sel(0, +1.0)),
            ("Shift+Down",  lambda: self.translate_sel(0, -1.0)),
            ("+", lambda: self.scale_sel(1.01)),
            ("=", lambda: self.scale_sel(1.01)),
            ("-", lambda: self.scale_sel(0.99)),
        ]:
            QShortcut(QKeySequence(k), self, fn)

        self.view.scene().selectionChanged.connect(self._on_sel_changed)

    # ---------- public ----------
    def _target_crs(self):
        return state().work_crs or "EPSG:7850"

    def reload_grid_and_canopy(self):
        s = state()
        # Background: whichever base is currently selected, falling back sensibly.
        if self._bg_mode == "ortho" and s.ortho_tif and os.path.exists(s.ortho_tif):
            self.view.load_ortho_raster(s.ortho_tif, dst_crs=self._target_crs())
        elif s.canopy_tif and os.path.exists(s.canopy_tif):
            self._bg_mode = "canopy"
            self.cb_base.blockSignals(True); self.cb_base.setCurrentIndex(0)
            self.cb_base.blockSignals(False)
            self.view.load_canopy_raster(s.canopy_tif)
        # Grid: prefer an aligned grid if it exists, else the project grid.
        if s.saved_grid and os.path.exists(s.saved_grid):
            self._load_grid(s.saved_grid)
        elif s.grid_path and os.path.exists(s.grid_path):
            self._load_grid(s.grid_path)
        self.view.fit_view()

    # ---------- base layer ----------
    def _on_base_changed(self, idx):
        if idx == 0:
            if not self._show_canopy():
                QMessageBox.information(self, "No canopy raster",
                    "No canopy raster in the project yet. Load a point cloud on "
                    "the Project tab, or use the orthomosaic as the base.")
        else:
            if not self._show_ortho():
                self.load_ortho()
                if self._bg_mode != "ortho":
                    self.cb_base.blockSignals(True); self.cb_base.setCurrentIndex(0)
                    self.cb_base.blockSignals(False)

    def _show_canopy(self):
        s = state()
        if s.canopy_tif and os.path.exists(s.canopy_tif):
            self._bg_mode = "canopy"
            self.view.load_canopy_raster(s.canopy_tif)
            self.view.fit_view()
            return True
        return False

    def _show_ortho(self):
        s = state()
        if s.ortho_tif and os.path.exists(s.ortho_tif):
            self._bg_mode = "ortho"
            self.view.load_ortho_raster(s.ortho_tif, dst_crs=self._target_crs())
            self.view.fit_view()
            return True
        return False

    def load_ortho(self):
        s = state()
        start = os.path.dirname(s.ortho_tif or s.las_path or "") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select orthomosaic (georeferenced GeoTIFF)",
            start, "GeoTIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            self.view.load_ortho_raster(path, dst_crs=self._target_crs())
        except Exception as e:
            QMessageBox.critical(self, "Could not load orthomosaic",
                f"{e}\n\nIt must be a georeferenced GeoTIFF; a different CRS is "
                "reprojected to the project CRS automatically.")
            return
        s.ortho_tif = path
        self._bg_mode = "ortho"
        self.cb_base.blockSignals(True); self.cb_base.setCurrentIndex(1)
        self.cb_base.blockSignals(False)
        self.view.fit_view()
        self.lbl_status.setText(
            f"Orthomosaic base: {os.path.basename(path)}. "
            "Drag / rotate / resize plots to fit, then Save Grid.")

    def load_grid_file(self):
        s = state()
        start = os.path.dirname(s.grid_path or s.saved_grid or s.ortho_tif
                                or s.las_path or "") or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load plot grid (Shapefile / GeoPackage / GeoJSON)",
            start, GRID_FILE_FILTER)
        if not path:
            return
        s.grid_path = path
        try:
            self._load_grid(path)
        except Exception as e:
            QMessageBox.critical(self, "Could not load grid", str(e))
            return
        self.view.fit_view()

    # ---------- grid loading ----------
    def _load_grid(self, path):
        gdf = load_grid(path, target_crs=state().work_crs)
        self._grid_attrs = gdf.drop(columns="geometry").reset_index(drop=True)
        self._grid_crs = gdf.crs
        self._original_geoms = list(gdf.geometry)
        # remove existing PlotItems
        for it in list(self.view.scene().items()):
            if isinstance(it, PlotItem):
                self.view.scene().removeItem(it)
        for i, g in enumerate(gdf.geometry):
            self.view.scene().addItem(PlotItem(i, g))
        self.lbl_status.setText(f"Loaded {len(gdf)} plots from {os.path.basename(path)}")

    # ---------- helpers ----------
    def _all_plots(self):
        return [it for it in self.view.scene().items() if isinstance(it, PlotItem)]
    def _sel(self):
        return [it for it in self.view.scene().selectedItems() if isinstance(it, PlotItem)]

    def push_undo(self):
        snap = [(it.plot_idx, it.shapely()) for it in self._all_plots()]
        self._undo_stack.append(snap)
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    # ---------- actions ----------
    def select_all(self):
        for it in self._all_plots(): it.setSelected(True)
    def deselect_all(self):
        self.view.scene().clearSelection()

    def undo(self):
        if not self._undo_stack: return
        snap = self._undo_stack.pop()
        by = {it.plot_idx: it for it in self._all_plots()}
        for idx, geom in snap:
            it = by.get(idx)
            if it: it.setPolygon(shp_to_qpoly(geom))

    def reset(self):
        if not self._original_geoms: return
        self.push_undo()
        by = {it.plot_idx: it for it in self._all_plots()}
        for i, g in enumerate(self._original_geoms):
            it = by.get(i)
            if it: it.setPolygon(shp_to_qpoly(g))

    def fit_view(self):
        self.view.fit_view()

    def translate_sel(self, dx, dy):
        sel = self._sel()
        if not sel: return
        self.push_undo()
        for it in sel:
            it.setPolygon(shp_to_qpoly(sh_translate(it.shapely(), dx, dy)))

    def rotate_sel(self, deg):
        sel = self._sel()
        if not sel: return
        self.push_undo()
        u = unary_union([it.shapely() for it in sel])
        cx, cy = u.centroid.x, u.centroid.y
        for it in sel:
            it.setPolygon(shp_to_qpoly(sh_rotate(it.shapely(), deg, origin=(cx, cy))))

    def scale_sel(self, f):
        sel = self._sel()
        if not sel: return
        self.push_undo()
        u = unary_union([it.shapely() for it in sel])
        cx, cy = u.centroid.x, u.centroid.y
        for it in sel:
            it.setPolygon(shp_to_qpoly(
                sh_scale(it.shapely(), xfact=f, yfact=f, origin=(cx, cy))))

    def resize_sel(self, dw, dl):
        """Resize each selected plot by (dw, dl) metres in its own local
        width/length axes. If nothing is selected, resize ALL plots
        (the 'whole grid' case). Each plot keeps its centroid + rotation."""
        sel = self._sel() or self._all_plots()
        if not sel: return
        self.push_undo()
        skipped = 0
        for it in sel:
            new_poly = _resize_plot_rect(it.shapely(), dw, dl)
            if new_poly is None or new_poly.is_empty:
                skipped += 1; continue
            it.setPolygon(shp_to_qpoly(new_poly))
        if skipped:
            self.lbl_status.setText(f"Resized {len(sel)-skipped}/{len(sel)} plots "
                                    f"(dw={dw:+.2f}m, dl={dl:+.2f}m).")
        else:
            scope = "selected" if self._sel() else "all"
            self.lbl_status.setText(f"Resized {len(sel)} {scope} plots "
                                    f"(dw={dw:+.2f}m, dl={dl:+.2f}m).")

    def auto_align(self):
        """FFT cross-correlate the grid mask with the canopy raster, then
        translate every polygon by the best-fit shift."""
        s = state()
        if not s.canopy_tif or not os.path.exists(s.canopy_tif):
            QMessageBox.warning(self, "No canopy raster",
                "Load a project first so a canopy raster exists.")
            return
        plots = self._all_plots()
        if not plots:
            QMessageBox.warning(self, "No grid",
                "Click 'Reload from Project' first to load plots.")
            return
        self.push_undo()
        polys = [it.shapely() for it in sorted(plots, key=lambda x: x.plot_idx)]
        try:
            shifted, dx, dy = auto_align_grid(s.canopy_tif, polys)
        except Exception as e:
            QMessageBox.critical(self, "Auto-align failed", str(e))
            return
        by_idx = {it.plot_idx: it for it in plots}
        for new_geom, it in zip(shifted, sorted(plots, key=lambda x: x.plot_idx)):
            it.setPolygon(shp_to_qpoly(new_geom))
        self.lbl_status.setText(
            f"Auto-align shift: dx={dx:+.3f} m  dy={dy:+.3f} m "
            f"(use Undo to revert)")

    def refine_to_crop(self):
        """Per-plot correction from the canopy height model: re-centre each
        plot along the sowing direction on the crop actually present, set its
        length from the crop edges, and apply a per-range across-row shift when
        the furrows are visible. Builds the CHM first if none exists."""
        s = state()
        if not s.las_path or not os.path.exists(s.las_path):
            QMessageBox.warning(self, "No LAS", "Load a project first (the CHM is built from the point cloud).")
            return
        plots = self._all_plots()
        if not plots or self._grid_attrs is None:
            QMessageBox.warning(self, "No grid", "Click 'Reload from Project' first to load plots.")
            return
        s.derive_default_paths()
        base = os.path.splitext(os.path.basename(s.las_path))[0]
        chm = os.path.join(os.path.dirname(s.out_csv), "surface_models", f"{base}_CHM.tif")
        polys = [it.shapely() for it in sorted(plots, key=lambda x: x.plot_idx)]
        gdf = gpd.GeoDataFrame(self._grid_attrs.copy(), geometry=polys, crs=self._grid_crs)
        out_shp = (s.saved_grid or s.grid_path or os.path.join(os.path.dirname(s.las_path), "aligned_grid.shp"))
        out_shp = os.path.splitext(out_shp)[0].replace("_refit", "") + "_refit.shp"
        self.lbl_status.setText("Refining plots to the crop... (building the CHM first if needed)")
        self._refine_worker = _RefineWorker(dict(las_path=s.las_path, work_crs=s.work_crs, use_smrf=s.use_smrf,
                                                 chm=chm, gdf=gdf, out_shp=out_shp))
        self._refine_worker.done_ok.connect(self._on_refined)
        self._refine_worker.error.connect(lambda e: QMessageBox.critical(self, "Refine failed", e))
        self._refine_worker.progress.connect(lambda p, m: self.lbl_status.setText(f"[{p}%] {m}"))
        self._refine_worker.start()

    def _on_refined(self, res):
        s = state()
        new = res["gdf"]; d = res["diag"]
        self.push_undo()
        plots = sorted(self._all_plots(), key=lambda x: x.plot_idx)
        for it, geom in zip(plots, new.geometry):
            it.setPolygon(shp_to_qpoly(geom))
        for col in ("plot_len", "along_off", "across_off", "crop_len", "refine_ok"):
            if col in new.columns:
                self._grid_attrs[col] = list(new[col].values)
        s.saved_grid = res["out_shp"]
        self.lbl_status.setText(f"Refined grid saved → {res['out_shp']} (Undo reverts the view; the file stays)")
        across = ", ".join(f"{k}: {v:+.2f} m" for k, v in d["across_off_by_group"].items())
        along = ", ".join(f"{k}: {v:+.2f} m" for k, v in d["along_off_median_by_group"].items())
        QMessageBox.information(self, "Plots refined to the crop",
            f"{d['n_along_ok']} of {d['n']} plots measured reliably (others use their range median).\n\n"
            f"Crop length median {d['crop_len_median']:.2f} m vs polygon {d['poly_len']:.2f} m -> new length median {d['new_len_median']:.2f} m\n"
            f"Along-plot offset applied, median by range: {along}\n"
            f"Across-row shift applied by range (0 = furrows not visible): {across}\n"
            f"Plot ends sticking into the alley: {d['ends_outside_crop_before']} before -> {d['ends_outside_crop_after']} after\n"
            f"Mean CHM inside plots: {d['mean_chm_inside_before']:.3f} -> {d['mean_chm_inside_after']:.3f} m\n\n"
            f"Saved: {res['out_shp']}\nReport: {res['report_csv']}\nQA figure: {res['qa_png']}\n\n"
            "The Traits tab now uses the refined grid.")

    def save_grid(self):
        if self._grid_attrs is None or not self._all_plots():
            QMessageBox.warning(self, "Nothing to save", "Load a grid first.")
            return
        s = state()
        base_dir = os.path.dirname(s.grid_path or s.ortho_tif or s.las_path or ".") or "."
        default = s.saved_grid or os.path.join(base_dir, "aligned_grid.shp")
        if default.lower().endswith(".gpkg"):
            default = default[:-5] + ".shp"    # prefer Shapefile by default
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save aligned grid", default,
            "Shapefile (*.shp);;GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)")
        if not out_path:
            return
        plots = sorted(self._all_plots(), key=lambda it: it.plot_idx)
        out = self._grid_attrs.copy()
        out["geometry"] = [it.shapely() for it in plots]
        gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=self._grid_crs)
        try:
            save_grid(gdf, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        s.saved_grid = out_path
        self.lbl_status.setText(f"Saved → {out_path}")
        QMessageBox.information(self, "Saved",
            f"Aligned grid saved to:\n{out_path}\n\n"
            "The Traits tab will use this aligned grid for extraction.")

    # ---------- callbacks ----------
    def _on_coords(self, wx, wy):
        self.lbl_coords.setText(f"E {wx:.2f}  N {wy:.2f}")
    def _on_sel_changed(self):
        n = sum(1 for it in self.view.scene().selectedItems() if isinstance(it, PlotItem))
        self.lbl_sel.setText(f"Selected: {n}")
