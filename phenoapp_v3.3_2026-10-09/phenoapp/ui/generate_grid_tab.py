"""
Generate Grid tab: click 4 corners on the canopy, set parameters, save.
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFileDialog, QMessageBox,
    QToolBar, QSizePolicy, QComboBox, QAction, QGraphicsEllipseItem,
    QGraphicsItem
)

from phenoapp.core.project import state
from phenoapp.core import generate_grid_from_corners, save_grid
from phenoapp.core.canopy_raster import COLORMAPS
from phenoapp.ui.canopy_view import CanopyView, world_to_scene


# Each entry: (label shown to user, click order as a tuple). The generator
# always wants (BL, BR, TR, TL); we permute the clicked list to that order.
CORNER_ORDERS = {
    "BL → BR → TR → TL  (counter-clockwise from bottom-left)":
        ("BL", "BR", "TR", "TL"),
    "TL → TR → BR → BL  (clockwise from top-left)":
        ("TL", "TR", "BR", "BL"),
}


class GenerateGridTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._corners = []         # list of (wx, wy) — picked points
        self._corner_dots = []     # graphics items for visualisation
        self._preview_items = []   # preview polygons
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        # ---- top: corner-order selector + instructions ----
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Click order:"))
        self.cb_order = QComboBox()
        self.cb_order.addItems(list(CORNER_ORDERS.keys()))
        self.cb_order.currentIndexChanged.connect(self._on_order_changed)
        order_row.addWidget(self.cb_order, stretch=1)
        v.addLayout(order_row)

        self.lbl_instructions = QLabel()
        self.lbl_instructions.setWordWrap(True)
        v.addWidget(self.lbl_instructions)
        self._on_order_changed()   # populate instruction text once

        # Which background is currently shown: "canopy" (LiDAR density raster)
        # or "ortho" (RGB orthomosaic). Corner picking works identically on both
        # because they share the same world-coordinate georeferencing.
        self._bg_mode = "canopy"

        # ---- toolbar (base-layer + display options + actions) ----
        tb = QToolBar()
        tb.addWidget(QLabel(" Base: "))
        self.cb_base = QComboBox()
        self.cb_base.addItems(["Canopy raster (LiDAR)", "Orthomosaic (RGB)"])
        self.cb_base.currentIndexChanged.connect(self._on_base_changed)
        tb.addWidget(self.cb_base)
        self.act_load_ortho = QAction("Load Orthomosaic…", self)
        self.act_load_ortho.triggered.connect(self.load_ortho)
        tb.addAction(self.act_load_ortho)
        tb.addSeparator()
        tb.addWidget(QLabel(" Mouse: "))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["📍 Place corners", "✋ Pan / move"])
        self.cb_mode.setToolTip(
            "Place corners: left-click drops the 4 trial corners.\n"
            "Pan / move: left-drag moves the scene freely.\n"
            "(Middle-drag or Ctrl+left-drag always pans; wheel zooms.)")
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self.cb_mode)
        tb.addSeparator()
        tb.addAction("Reload Canopy",   self.reload_canopy)
        tb.addAction("Clear Corners",   self.clear_corners)
        tb.addAction("Fit View",        self._fit_view)
        tb.addSeparator()
        tb.addWidget(QLabel(" Colour: "))
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(list(COLORMAPS))
        self.cb_cmap.setCurrentText("viridis")
        self.cb_cmap.currentTextChanged.connect(self._on_cmap_changed)
        tb.addWidget(self.cb_cmap)
        tb.addWidget(QLabel(" BG: "))
        self.cb_bg = QComboBox()
        self.cb_bg.addItems(["dark", "black", "white", "light-grey"])
        self.cb_bg.currentTextChanged.connect(self._on_bg_changed)
        tb.addWidget(self.cb_bg)
        tb.addSeparator()
        tb.addWidget(QLabel(" CRS: "))
        self.cb_crs = QComboBox()
        self.cb_crs.setEditable(True)
        self.cb_crs.setMinimumWidth(240)
        self.cb_crs.addItems([
            "EPSG:7850 — GDA2020 / MGA zone 50 (DPIRD WA)",
            "EPSG:28350 — GDA94 / MGA zone 50",
            "EPSG:32750 — WGS84 / UTM zone 50S",
            "EPSG:7845 — GDA2020 / MGA zone 51",
            "EPSG:32751 — WGS84 / UTM zone 51S",
            "EPSG:3577 — GDA94 / Australian Albers",
        ])
        self.cb_crs.setToolTip(
            "Target CRS for the orthomosaic reprojection AND the saved grid. "
            "Use a projected (metre) CRS so plot sizes in metres are correct.\n"
            "Type any 'EPSG:xxxx' or a bare code. Default: your project CRS.")
        # initialise to the project's working CRS
        self._set_crs_text(state().work_crs or "EPSG:7850")
        self.cb_crs.currentIndexChanged.connect(self._on_crs_changed)
        self.cb_crs.lineEdit().editingFinished.connect(self._on_crs_changed)
        tb.addWidget(self.cb_crs)
        v.addWidget(tb)

        # ---- canvas ----
        self.view = CanopyView()
        self.view.coords_changed.connect(self._on_coords)
        self.view.left_press.connect(self._on_left_click)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.view, stretch=1)

        # ---- parameters ----
        params = QGroupBox("Trial layout")
        f = QFormLayout(params)

        self.sb_banks = QSpinBox(); self.sb_banks.setRange(1, 999); self.sb_banks.setValue(4)
        self.sb_rows  = QSpinBox(); self.sb_rows.setRange(1, 9999); self.sb_rows.setValue(32)
        self.dsb_pw   = QDoubleSpinBox(); self.dsb_pw.setRange(0.05, 50); self.dsb_pw.setValue(1.5); self.dsb_pw.setSuffix(" m")
        self.dsb_pl   = QDoubleSpinBox(); self.dsb_pl.setRange(0.05, 50); self.dsb_pl.setValue(6.0); self.dsb_pl.setSuffix(" m")
        self.dsb_gr   = QDoubleSpinBox(); self.dsb_gr.setRange(0, 50); self.dsb_gr.setValue(0.5); self.dsb_gr.setSuffix(" m")
        self.dsb_gb   = QDoubleSpinBox(); self.dsb_gb.setRange(0, 50); self.dsb_gb.setValue(2.0); self.dsb_gb.setSuffix(" m")
        f.addRow("Rows (along 1st→2nd clicked corner edge):", self.sb_rows)
        f.addRow("Ranges (along 1st→4th clicked corner edge):", self.sb_banks)
        f.addRow("Plot width (across rows):",  self.dsb_pw)
        f.addRow("Plot length (along range):", self.dsb_pl)
        f.addRow("Gap between rows:",  self.dsb_gr)
        f.addRow("Gap between ranges:",self.dsb_gb)

        from PyQt5.QtWidgets import QCheckBox
        self.cb_fit = QCheckBox("Stretch grid to fit the 4 clicked corners "
                                "(fills the trial exactly; plot size/gaps used "
                                "as ratios)")
        self.cb_fit.setChecked(True)
        self.cb_fit.setToolTip(
            "On: the whole grid is stretched to fill the quadrilateral you "
            "clicked — best for rotated or slightly irregular trials. The plot "
            "width/length and gaps set only the plot-to-alley proportions.\n"
            "Off: plots use the exact metre dimensions and the grid is placed "
            "from the bottom-left corner (may not reach the other corners).")
        f.addRow("", self.cb_fit)
        v.addWidget(params)

        # ---- buttons ----
        h = QHBoxLayout()
        self.btn_preview = QPushButton("👁️ Preview Grid")
        self.btn_preview.clicked.connect(self.preview)
        self.btn_save    = QPushButton("💾 Save Grid")
        self.btn_save.clicked.connect(self.save)
        self.lbl_status  = QLabel("Click 0/4 corners.")
        h.addWidget(self.btn_preview)
        h.addWidget(self.btn_save)
        h.addStretch()
        h.addWidget(self.lbl_status)
        v.addLayout(h)

    # ---- background layers ----
    def reload_canopy(self):
        s = state()
        if s.canopy_tif and os.path.exists(s.canopy_tif):
            self._bg_mode = "canopy"
            self.cb_base.blockSignals(True)
            self.cb_base.setCurrentIndex(0)
            self.cb_base.blockSignals(False)
            self.view.load_canopy_raster(s.canopy_tif)

    def reload_ortho(self):
        """Show the orthomosaic already recorded in the project state, if any."""
        s = state()
        if s.ortho_tif and os.path.exists(s.ortho_tif):
            self._bg_mode = "ortho"
            self.view.load_ortho_raster(s.ortho_tif, dst_crs=self._target_crs())
            self.lbl_status.setText(
                f"Orthomosaic loaded: {os.path.basename(s.ortho_tif)}. "
                "Click the 4 trial corners.")
            return True
        return False

    def load_ortho(self):
        """Pick an RGB orthomosaic GeoTIFF and show it as the drawing base."""
        s = state()
        start = (os.path.dirname(s.ortho_tif) if s.ortho_tif else
                 (os.path.dirname(s.las_path) if s.las_path else
                  os.path.expanduser("~")))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select orthomosaic (georeferenced GeoTIFF)",
            start, "GeoTIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            self.view.load_ortho_raster(path, dst_crs=self._target_crs())
        except Exception as e:
            QMessageBox.critical(self, "Could not load orthomosaic",
                f"{e}\n\nMake sure it is a georeferenced GeoTIFF. If it is in a "
                "different CRS it will be reprojected to the target CRS "
                "selected in the toolbar.")
            return
        s.ortho_tif = path
        self._bg_mode = "ortho"
        self.cb_base.blockSignals(True)
        self.cb_base.setCurrentIndex(1)
        self.cb_base.blockSignals(False)
        self.lbl_status.setText(
            f"Orthomosaic loaded: {os.path.basename(path)}. "
            "Click the 4 trial corners.")

    def _on_base_changed(self, idx):
        if idx == 0:
            self.reload_canopy()
        else:
            if not self.reload_ortho():
                # nothing stored yet — prompt for a file
                self.load_ortho()
                if self._bg_mode != "ortho":
                    # user cancelled — revert selector to canopy
                    self.cb_base.blockSignals(True)
                    self.cb_base.setCurrentIndex(0)
                    self.cb_base.blockSignals(False)

    def _on_mode_changed(self, idx):
        self.view.set_left_mode("emit" if idx == 0 else "pan")

    # ---- CRS chooser ----
    def _target_crs(self) -> str:
        """The CRS chosen in the toolbar: used to reproject the ortho for
        display AND to tag the saved grid. Accepts the descriptive dropdown
        entries ('EPSG:7850 — ...'), a bare 'EPSG:xxxx', or just a number."""
        raw = self.cb_crs.currentText().strip() if hasattr(self, "cb_crs") else ""
        if not raw:
            return state().work_crs or "EPSG:7850"
        token = raw.split()[0]              # 'EPSG:7850 — ...' -> 'EPSG:7850'
        if token.isdigit():
            return f"EPSG:{token}"
        return token

    def _set_crs_text(self, crs: str):
        """Select the matching dropdown row for `crs`, else set the edit text."""
        for i in range(self.cb_crs.count()):
            if self.cb_crs.itemText(i).split()[0].upper() == str(crs).upper():
                self.cb_crs.setCurrentIndex(i)
                return
        self.cb_crs.setEditText(crs)

    def _on_crs_changed(self, *_):
        # Coordinates change meaning when the CRS changes, so drop any picked
        # corners and re-render the current background in the new CRS.
        self.clear_corners()
        if self._bg_mode == "ortho":
            self.reload_ortho()
        self.lbl_status.setText(f"Target CRS set to {self._target_crs()}. "
                                "Corners cleared — click them again.")

    def _fit_view(self):
        self.view.fit_view()

    # ---- corner-order helpers ----
    def _current_order(self):
        return CORNER_ORDERS[self.cb_order.currentText()]

    def _on_order_changed(self, *_):
        order = self._current_order()
        self.lbl_instructions.setText(
            f"<b>Click the 4 corners of the trial</b> in this order: "
            f"<b>{' → '.join(order)}</b> on the base layer (canopy raster or "
            f"orthomosaic). Then set the trial layout below "
            f"and click <b>Preview Grid</b>.")
        # If the user already started clicking, refresh labels on the dots
        # to match the new order.
        if self._corners:
            self._redraw_corner_labels()

    def _redraw_corner_labels(self):
        # Remove existing dots/labels and re-add with the new label order.
        coords = list(self._corners)
        self.clear_corners()
        for c in coords:
            self._add_corner_at(QPointF(c[0], -c[1]))

    def _on_cmap_changed(self, name):
        self.view.set_colormap(name)

    def _on_bg_changed(self, name):
        bg = {"dark": (20, 20, 20), "black": (0, 0, 0),
              "white": (245, 245, 245), "light-grey": (200, 200, 200)}[name]
        self.view.set_background_rgb(bg)

    # ---- corner picking ----
    def _on_left_click(self, scene_pos):
        if len(self._corners) >= 4:
            return
        self._add_corner_at(scene_pos)

    def _add_corner_at(self, scene_pos):
        wx, wy = scene_pos.x(), -scene_pos.y()
        self._corners.append((wx, wy))
        # Draw the marker at a FIXED on-screen pixel size (ItemIgnores
        # Transformations) so it looks the same at any zoom and in any CRS.
        # A hard-coded world-unit radius would be invisibly small in metres or
        # kilometres-wide in geographic degrees (which painted the whole scene).
        R = 6  # pixel radius
        dot = QGraphicsEllipseItem(-R, -R, 2 * R, 2 * R)
        pen = QPen(QColor("red")); pen.setCosmetic(True); pen.setWidth(2)
        dot.setPen(pen)
        dot.setBrush(QBrush(QColor(255, 60, 60)))
        dot.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        dot.setPos(scene_pos)
        dot.setZValue(1000)
        self.view.scene().addItem(dot)

        labels = self._current_order()
        txt = self.view.scene().addText(labels[len(self._corners) - 1])
        txt.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        txt.setPos(scene_pos)
        txt.setDefaultTextColor(QColor("red"))
        txt.setZValue(1001)
        self._corner_dots += [dot, txt]
        self.lbl_status.setText(f"Click {len(self._corners)}/4 corners.")

    def clear_corners(self):
        self._corners = []
        for it in self._corner_dots:
            self.view.scene().removeItem(it)
        self._corner_dots = []
        self._clear_preview()
        self.lbl_status.setText("Click 0/4 corners.")

    def _clear_preview(self):
        for it in self._preview_items:
            self.view.scene().removeItem(it)
        self._preview_items = []

    # ---- preview / save ----
    def _build_grid(self):
        if len(self._corners) != 4:
            raise RuntimeError("Need 4 corners")
        # generate_grid_from_corners always expects (BL, BR, TR, TL); permute
        # the user's clicks (in their chosen order) to that canonical order.
        order = self._current_order()
        as_dict = dict(zip(order, self._corners))
        canonical = [as_dict["BL"], as_dict["BR"], as_dict["TR"], as_dict["TL"]]
        return generate_grid_from_corners(
            canonical,
            self.sb_banks.value(),
            self.sb_rows.value(),
            self.dsb_pw.value(),
            self.dsb_pl.value(),
            self.dsb_gr.value(),
            self.dsb_gb.value(),
            crs=self._target_crs(),   # CRS chosen in the toolbar
            fit_to_corners=self.cb_fit.isChecked(),
        )

    def preview(self):
        if len(self._corners) != 4:
            QMessageBox.warning(self, "Need 4 corners",
                "Click all 4 corners (BL, BR, TR, TL) on the canopy first.")
            return
        try:
            gdf = self._build_grid()
        except Exception as e:
            QMessageBox.critical(self, "Build failed", str(e))
            return
        self._clear_preview()
        for poly in gdf.geometry:
            from phenoapp.ui.canopy_view import shp_to_qpoly
            from PyQt5.QtWidgets import QGraphicsPolygonItem
            it = QGraphicsPolygonItem(shp_to_qpoly(poly))
            it.setPen(QPen(QColor(80, 200, 255), 0.07))
            it.setBrush(QBrush(QColor(80, 200, 255, 40)))
            self.view.scene().addItem(it)
            self._preview_items.append(it)
        self.lbl_status.setText(f"Preview: {len(gdf)} plots.")

    def save(self):
        if len(self._corners) != 4:
            QMessageBox.warning(self, "Need 4 corners", "Click all 4 corners first.")
            return
        try:
            gdf = self._build_grid()
        except Exception as e:
            QMessageBox.critical(self, "Build failed", str(e))
            return
        s = state()
        base_dir = os.path.dirname(s.las_path or s.ortho_tif or ".") or "."
        path, _ = QFileDialog.getSaveFileName(
            self, "Save grid (Shapefile for later use)",
            os.path.join(base_dir, "plot_grid.shp"),
            "Shapefile (*.shp);;GeoPackage (*.gpkg);;GeoJSON (*.geojson);;All files (*)")
        if not path:
            return
        try:
            save_grid(gdf, path)
            state().grid_path = path
            self.lbl_status.setText(f"Saved → {path}")
            QMessageBox.information(self, "Saved",
                f"Grid saved to:\n{path}\n\nGo to the Project tab and reload, "
                "or switch to the Edit tab and click 'Reload from Project' to align it.")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_coords(self, wx, wy):
        self.lbl_status.setText(
            f"Cursor: E {wx:.2f}  N {wy:.2f}   |   Corners clicked: {len(self._corners)}/4")
