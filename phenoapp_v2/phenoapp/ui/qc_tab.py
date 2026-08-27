"""
Point Cloud Quality Check tab.

Assesses drone LiDAR quality before any trait extraction:
  * header / metadata conformance (CRS, GPS time encoding, attribute
    population, return + classification content)
  * geometry (density, z-outliers, per-point noise)
  * survey-grade vertical consistency against surveyed GCPs
  * optionally runs the genuine FREE LAStools reporters (lasinfo,
    lasvalidate). Paid LAStools tools are never used, so no license is
    required and nothing is watermarked or perturbed.
"""

from __future__ import annotations
import os
import json

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QProgressBar, QFileDialog, QMessageBox, QLineEdit, QTextEdit, QFormLayout
)

from phenoapp.core.project import state
from phenoapp.core import run_qc, format_report


class _QCWorker(QThread):
    progress = pyqtSignal(int, str)
    done_ok  = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, las_path, gcp_csv, lastools_dir):
        super().__init__()
        self._a = (las_path, gcp_csv, lastools_dir)

    def run(self):
        try:
            las_path, gcp_csv, lastools_dir = self._a
            def cb(p, m): self.progress.emit(p, m)
            rep = run_qc(las_path, gcp_csv or None, lastools_dir or None,
                         progress_cb=cb)
            self.done_ok.emit(rep)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class QCTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rep = None
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Point cloud quality check</h2>"))
        v.addWidget(QLabel(
            "Run this before extracting traits. Geometry problems (tilt, "
            "noise, missing CRS) invalidate everything downstream; metadata "
            "problems (single-return, no classification) change which ground "
            "model you should use on the Traits tab."))

        box = QGroupBox("Inputs")
        f = QFormLayout(box)

        row1 = QHBoxLayout()
        self.ed_las = QLineEdit()
        b1 = QPushButton("Browse..."); b1.clicked.connect(self._pick_las)
        row1.addWidget(self.ed_las); row1.addWidget(b1)
        f.addRow("LAS/LAZ file:", row1)

        row2 = QHBoxLayout()
        self.ed_gcp = QLineEdit()
        self.ed_gcp.setPlaceholderText(
            "optional - CSV with easting, northing, height columns")
        b2 = QPushButton("Browse..."); b2.clicked.connect(self._pick_gcp)
        row2.addWidget(self.ed_gcp); row2.addWidget(b2)
        f.addRow("Surveyed GCPs:", row2)

        row3 = QHBoxLayout()
        self.ed_lt = QLineEdit()
        self.ed_lt.setPlaceholderText(
            "optional - LAStools bin folder (only the FREE lasinfo/"
            "lasvalidate are used)")
        b3 = QPushButton("Browse..."); b3.clicked.connect(self._pick_lt)
        row3.addWidget(self.ed_lt); row3.addWidget(b3)
        f.addRow("LAStools folder:", row3)
        v.addWidget(box)

        h = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run quality check")
        self.btn_run.clicked.connect(self._run)
        self.btn_save = QPushButton("Save report...")
        self.btn_save.clicked.connect(self._save)
        self.btn_save.setEnabled(False)
        h.addWidget(self.btn_run); h.addWidget(self.btn_save); h.addStretch()
        v.addLayout(h)

        self.progress = QProgressBar(); self.progress.setTextVisible(True)
        v.addWidget(self.progress)

        self.out = QTextEdit(); self.out.setReadOnly(True)
        self.out.setFontFamily("Consolas")
        v.addWidget(self.out, stretch=1)

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        s = state()
        if not self.ed_las.text() and s.las_path:
            self.ed_las.setText(s.las_path)
        if not self.ed_gcp.text() and s.gcp_csv:
            self.ed_gcp.setText(s.gcp_csv)
        if not self.ed_lt.text() and s.lastools_dir:
            self.ed_lt.setText(s.lastools_dir)

    def _pick_las(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick LAS/LAZ", "",
                                           "LAS files (*.las *.laz)")
        if p: self.ed_las.setText(p)

    def _pick_gcp(self):
        p, _ = QFileDialog.getOpenFileName(self, "Pick GCP CSV", "",
                                           "CSV files (*.csv)")
        if p:
            self.ed_gcp.setText(p)
            state().gcp_csv = p

    def _pick_lt(self):
        p = QFileDialog.getExistingDirectory(self, "Pick LAStools bin folder")
        if p:
            self.ed_lt.setText(p)
            state().lastools_dir = p

    # ------------------------------------------------------------------
    def _run(self):
        las = self.ed_las.text().strip()
        if not las or not os.path.exists(las):
            QMessageBox.warning(self, "No LAS", "Pick a LAS/LAZ file first.")
            return
        self.btn_run.setEnabled(False)
        self.out.clear()
        self._w = _QCWorker(las, self.ed_gcp.text().strip(),
                            self.ed_lt.text().strip())
        self._w.progress.connect(self._on_prog)
        self._w.done_ok.connect(self._on_done)
        self._w.error.connect(self._on_err)
        self._w.start()

    def _on_prog(self, p, m):
        self.progress.setValue(p)
        self.progress.setFormat(f"{p}% — {m}")

    def _on_done(self, rep):
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self._rep = rep
        self.out.setPlainText(format_report(rep))

    def _on_err(self, err):
        self.btn_run.setEnabled(True)
        self.out.append(f"ERROR: {err}")
        QMessageBox.critical(self, "QC failed", err.split("\n")[0])

    def _save(self):
        if not self._rep:
            return
        base = os.path.splitext(self.ed_las.text().strip())[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save QC report", base + "_qc_report.txt",
            "Text (*.txt);;JSON (*.json)")
        if not path:
            return
        if path.lower().endswith(".json"):
            with open(path, "w") as f:
                json.dump(self._rep, f, indent=2, default=str)
        else:
            with open(path, "w") as f:
                f.write(format_report(self._rep))
        QMessageBox.information(self, "Saved", path)
