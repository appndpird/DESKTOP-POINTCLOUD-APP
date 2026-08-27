"""About tab — credits + DPIRD/APPN logos."""

import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)


def _resource_path(rel: str) -> str:
    """Resolve a path inside the package, also handling PyInstaller frozen mode."""
    base_candidates = []
    if hasattr(sys, "_MEIPASS"):
        base_candidates.append(sys._MEIPASS)
    here = os.path.dirname(os.path.abspath(__file__))
    base_candidates.append(os.path.dirname(here))   # phenoapp/
    base_candidates.append(here)
    for base in base_candidates:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
        p2 = os.path.join(base, "phenoapp", rel)
        if os.path.exists(p2):
            return p2
    return ""


class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignTop)

        title = QLabel("Plant Phenotyping from Point Clouds")
        f = QFont(); f.setPointSize(22); f.setBold(True)
        title.setFont(f); title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)

        sub = QLabel("Plot extraction and physical-trait computation toolkit")
        f2 = QFont(); f2.setPointSize(12); f2.setItalic(True)
        sub.setFont(f2); sub.setAlignment(Qt.AlignCenter)
        v.addWidget(sub)
        v.addSpacing(28)

        credits = QLabel(
            "<div style='font-size:13pt; line-height:170%;'>"
            "<b>Developed by</b><br>"
            "Dr. Muhammad Ibrahim<br>"
            "Research Scientist, DPIRD<br><br>"
            "<b>Under the supervision of</b><br>"
            "Dr. Hammad Khan<br>"
            "Senior Research Scientist, DPIRD<br>"
            "APPN Director for DPIRD node"
            "</div>")
        credits.setAlignment(Qt.AlignCenter)
        v.addWidget(credits)
        v.addStretch(1)

        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        v.addWidget(line)

        h = QHBoxLayout()
        h.setAlignment(Qt.AlignCenter)
        h.setSpacing(60)
        h.addWidget(self._logo(_resource_path("assets/dpird_logo.png"), "DPIRD"))
        h.addWidget(self._logo(_resource_path("assets/appn_logo.png"),  "APPN"))
        v.addLayout(h)

        v.addSpacing(8)
        ver = QLabel("v1.0 — All phases (Project · Generate · Edit · Visualize · Traits · Statistics · Guidance)")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("color: #888; padding: 6px;")
        v.addWidget(ver)

    def _logo(self, path: str, fallback: str):
        if path and os.path.exists(path):
            lbl = QLabel()
            pix = QPixmap(path)
            if not pix.isNull():
                if pix.height() > 90:
                    pix = pix.scaledToHeight(90, Qt.SmoothTransformation)
                lbl.setPixmap(pix); lbl.setAlignment(Qt.AlignCenter)
                return lbl
        ph = QLabel(f"[{fallback} logo]")
        ph.setStyleSheet(
            "border: 2px dashed #aaa; padding: 18px 32px; "
            "color: #666; font-size: 14pt;")
        ph.setAlignment(Qt.AlignCenter)
        return ph
