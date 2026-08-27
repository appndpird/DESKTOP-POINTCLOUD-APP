"""
Shared canvas widget for displaying the canopy raster + interactive overlays.

Used as the base for:
  - Edit tab: draggable/rotatable plot polygons
  - Generate Grid tab: click 4 corners + preview
  - Visualize tab: overlay of saved grid

Provides:
  - Mouse-wheel zoom centred on cursor
  - Middle-click or Ctrl+left drag for pan
  - Coordinate readout (world coordinates)
  - Background canopy raster loaded once

Subclasses override:
  - on_left_press / on_left_drag / on_left_release for tool-specific behaviour
"""

from __future__ import annotations
import os
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QImage, QPixmap,
                         QTransform, QPolygonF)
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGraphicsPolygonItem, QRubberBand)


# Y is flipped between Qt (down=+Y) and world (up=+Y).
def shp_to_qpoly(poly):
    return QPolygonF([QPointF(x, -y) for x, y in list(poly.exterior.coords)[:-1]])


def qpoly_to_shp(qpoly):
    from shapely.geometry import Polygon
    return Polygon([(p.x(), -p.y()) for p in qpoly])


def world_to_scene(x, y):
    return QPointF(x, -y)


def scene_to_world(p):
    return p.x(), -p.y()


# ---------------------------------------------------------------
class CanopyView(QGraphicsView):
    """Shared zoomable/pannable view with canopy raster + overlay."""

    coords_changed = pyqtSignal(float, float)
    left_press     = pyqtSignal(object)   # QPointF in scene coords
    left_drag      = pyqtSignal(object)
    left_release   = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(20, 20, 20)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._panning = False
        self._pan_start = None
        self._dragging_left = False
        self._left_drag_started = False
        self._raster_item = None
        self._tif_path  = None
        self._colormap  = "viridis"
        self._bg_rgb    = (20, 20, 20)
        # "emit" -> left-click places corners (emits signals);
        # "pan"  -> left-drag pans the scene freely. Middle-click and
        # Ctrl+left always pan regardless of mode.
        self._left_mode = "emit"

    def set_left_mode(self, mode: str):
        """Switch left-mouse behaviour: 'emit' (place points) or 'pan'."""
        self._left_mode = "pan" if mode == "pan" else "emit"
        self.viewport().setCursor(
            Qt.OpenHandCursor if self._left_mode == "pan" else Qt.CrossCursor)

    # ---------- Background ----------
    def set_colormap(self, name: str):
        """Switch the canopy colormap and re-render the current raster."""
        self._colormap = name
        if self._tif_path:
            self.load_canopy_raster(self._tif_path)

    def set_background_rgb(self, rgb_tuple):
        self._bg_rgb = tuple(int(c) for c in rgb_tuple)
        self.setBackgroundBrush(QBrush(QColor(*self._bg_rgb)))
        if self._tif_path:
            self.load_canopy_raster(self._tif_path)

    def load_canopy_raster(self, tif_path: str):
        """Load a canopy density GeoTIFF as the scene background."""
        from phenoapp.core.canopy_raster import raster_display_array
        if not tif_path or not os.path.exists(tif_path):
            return
        img, bounds, _crs = raster_display_array(
            tif_path, colormap=self._colormap, background_rgb=self._bg_rgb)
        self._tif_path = tif_path
        self._place_raster(img, bounds)

    def load_ortho_raster(self, tif_path: str, dst_crs=None):
        """Load an RGB orthomosaic GeoTIFF as the scene background.

        Reprojects to `dst_crs` (the working CRS) if the ortho is in a different
        CRS, so corners clicked on the ortho land in the same world coordinates
        (metres) as if clicked on the canopy raster."""
        from phenoapp.core.canopy_raster import ortho_display_array
        if not tif_path or not os.path.exists(tif_path):
            return
        img, bounds, _crs = ortho_display_array(tif_path, dst_crs=dst_crs)
        # An ortho is not colormapped; remember its path so colormap/bg toggles
        # don't try to re-render it through the canopy path.
        self._tif_path = None
        self._place_raster(img, bounds)

    def _place_raster(self, img, bounds):
        """Position an HxWx4 RGBA uint8 image at its georeferenced bounds."""
        if self._raster_item is not None:
            self.scene().removeItem(self._raster_item)
            self._raster_item = None
        h, w = img.shape[:2]
        qimg = QImage(img.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(qimg)
        item = self.scene().addPixmap(pix)
        # Position pixel (0,0) -> (left, -top); pixel (w,h) -> (right, -bottom)
        left, bottom, right, top = bounds
        t = QTransform()
        t.translate(left, -top)
        t.scale((right - left) / w, (top - bottom) / h)
        item.setTransform(t)
        item.setZValue(-100)
        self._raster_item = item
        self.fit_view()

    def fit_view(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)

    # ---------- Zoom & pan ----------
    def wheelEvent(self, event):
        factor = 1.20 if event.angleDelta().y() > 0 else 1 / 1.20
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        mods = event.modifiers()
        want_pan = (
            event.button() == Qt.MiddleButton or
            (event.button() == Qt.LeftButton and (mods & Qt.ControlModifier)) or
            (event.button() == Qt.LeftButton and self._left_mode == "pan"))
        if want_pan:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._left_mode == "emit":
            self._dragging_left = True
            self._left_drag_started = False
            self.left_press.emit(scene_pos)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        wx, wy = scene_to_world(scene_pos)
        self.coords_changed.emit(wx, wy)

        if self._panning and self._pan_start is not None:
            d = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            event.accept()
            return

        if self._dragging_left:
            self._left_drag_started = True
            self.left_drag.emit(scene_pos)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor if self._left_mode == "pan"
                           else Qt.ArrowCursor)
            event.accept()
            return
        if self._dragging_left and event.button() == Qt.LeftButton:
            self._dragging_left = False
            self.left_release.emit(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseReleaseEvent(event)
