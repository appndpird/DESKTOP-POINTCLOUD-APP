"""
Dataset tab (PhenoFuse 4.0): build and inspect a multi-flight dataset folder.

    <dataset>/lidar|vnir|rgb/<flight>/plots/ + <modality>_features.csv + ground_truth.csv, dataset.json manifest,
    ground_truth/<flight>.csv, Models/Height models|Biomass models/<run>/

"Add this flight" copies the current flight's per-plot files and feature tables (the ticked modalities) plus its
ground-truth CSV into the dataset; the modelling tabs then train on any selection of flights and modalities.
"""
from __future__ import annotations
import os
import re

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QMessageBox, QLineEdit, QTextEdit,
                             QCheckBox, QFormLayout, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView)

from phenoapp.core.project import state
from phenoapp.ui.common import file_row


class _AddWorker(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)

    def __init__(self, dataset_dir, flight, trial, stage, date, items, gt_csv, copy_plots):
        super().__init__(); self._a = (dataset_dir, flight, trial, stage, date, items, gt_csv, copy_plots)

    def run(self):
        try:
            from phenoapp.core.dataset import add_flight
            dataset_dir, flight, trial, stage, date, items, gt_csv, copy_plots = self._a
            lines = []
            for i, (mod, csv, pdir) in enumerate(items):
                self.progress.emit(int(100 * i / len(items)), f"{mod}: copying")
                fdir = add_flight(dataset_dir, flight, mod, csv, plots_dir=pdir, gt_csv=gt_csv, trial=trial, stage=stage, date=date, copy_plots=copy_plots,
                                  progress_cb=lambda p, m: self.progress.emit(int(100 * (i + p / 100) / len(items)), m))
                lines.append(f"{mod}: {csv} -> {fdir}" + (f" (+ plot files from {pdir})" if (copy_plots and pdir) else ""))
            self.progress.emit(100, "done"); self.done_ok.emit("\n".join(lines))
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class DatasetTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Dataset: several flights, three modalities, one folder</h2>"))
        v.addWidget(QLabel("A dataset folder holds, per modality and per flight, the per-plot files (LAS / VNIR cubes / RGB crops), the feature table "
                           "and the ground truth: <dataset>/lidar|vnir|rgb/<flight>/plots/ + lidar_features.csv (vnir_features.csv, rgb_features.csv) + "
                           "ground_truth.csv. The Biomass, Height and Deep Models tabs train on any selection of flights and modalities of a dataset, "
                           "and their models land in <dataset>/Models/Height models/ and Biomass models/."))
        g = QGroupBox("Dataset folder"); f = QFormLayout(g)
        self.ed_dataset = QLineEdit(); f.addRow("Folder:", file_row(self, self.ed_dataset, "Dataset folder (existing or new)", directory=True, on_change=lambda p: self.rescan()))
        row = QHBoxLayout(); b = QPushButton("Rescan"); b.clicked.connect(self.rescan); row.addWidget(b); row.addStretch(); f.addRow("", row)
        v.addWidget(g)
        self.table = QTableWidget(0, 10); self.table.setHorizontalHeaderLabels(["flight", "trial", "stage", "date", "lidar plots", "lidar files", "vnir plots", "vnir files", "rgb plots", "ground truth"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); self.table.setMinimumHeight(140); v.addWidget(self.table, stretch=1)

        g2 = QGroupBox("Add this flight to the dataset"); f2 = QFormLayout(g2)
        self.ed_flight = QLineEdit(); self.ed_trial = QLineEdit(); self.cb_stage = QComboBox(); self.cb_stage.setEditable(True); self.cb_stage.addItems(["anthesis", "maturity", "tillering", "stem elongation", "booting", "grain filling", "other"])
        self.ed_date = QLineEdit(); self.ed_date.setPlaceholderText("YYYY-MM-DD (optional; guessed from the LAS name)")
        f2.addRow("Flight name:", self.ed_flight); f2.addRow("Trial:", self.ed_trial); f2.addRow("Stage:", self.cb_stage); f2.addRow("Date:", self.ed_date)
        self.ed_flight.setToolTip("Folder name of this flight inside the dataset, e.g. 20250930_NUE_I_DPIRD_Muresk_F_Gobi.")
        self.ed_trial.setToolTip("Flights of the same trial share plot IDs: the same plot on several dates is kept in one cross-validation fold.")
        self.cb_lidar = QCheckBox("LiDAR: plots_lidar/ + lidar_features.csv"); self.cb_vnir = QCheckBox("VNIR: plots_vnir/ + vnir_features.csv (+ spectra, band QC)"); self.cb_rgb = QCheckBox("RGB: plots_rgb/ + rgb_features.csv")
        for cb in (self.cb_lidar, self.cb_vnir, self.cb_rgb):
            cb.setChecked(True); f2.addRow("", cb)
        self.cb_copy = QCheckBox("Copy the per-plot files too (LAS / cubes / crops; needed by the Deep Models tab, large)"); self.cb_copy.setChecked(True); f2.addRow("", self.cb_copy)
        self.ed_gt = QLineEdit(); self.ed_gt.setPlaceholderText("Plot_ID + biomass_kg_ha / height_cm / any target column (optional for inference-only flights)")
        f2.addRow("Ground-truth CSV:", file_row(self, self.ed_gt, "Ground-truth table"))
        row = QHBoxLayout(); self.btn_add = QPushButton("Add this flight to the dataset"); self.btn_add.clicked.connect(self._add); row.addWidget(self.btn_add); row.addStretch(); f2.addRow("", row)
        v.addWidget(g2)
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); self.out.setMaximumHeight(160); v.addWidget(self.out)

    def showEvent(self, ev):
        super().showEvent(ev); s = state()
        if s.las_path:
            s.derive_default_paths()
        if not self.ed_dataset.text().strip() and s.dataset_dir:
            self.ed_dataset.setText(s.dataset_dir); self.rescan()
        if not self.ed_flight.text().strip() and s.flight_name:
            self.ed_flight.setText(s.flight_name)
        if not self.ed_trial.text().strip():
            self.ed_trial.setText(s.trial_name or re.sub(r"^\d{8}_", "", s.flight_name or ""))
        if not self.ed_date.text().strip():
            m = re.match(r"^(\d{4})(\d{2})(\d{2})", s.flight_name or "")
            if m: self.ed_date.setText(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        if not self.ed_gt.text().strip() and s.gt_csv:
            self.ed_gt.setText(s.gt_csv)
        self.cb_stage.setCurrentText(s.stage or "anthesis")

    def rescan(self):
        from phenoapp.core.dataset import scan
        d = self.ed_dataset.text().strip(); self.table.setRowCount(0)
        if not d or not os.path.isdir(d):
            return
        state().dataset_dir = d
        try:
            t = scan(d)
        except Exception as e:
            self.out.append(f"scan failed: {e}"); return
        self.table.setRowCount(len(t))
        for i, r in t.iterrows():
            vals = [r.flight, r.trial, r.stage, r.date, r.lidar, r.lidar_files, r.vnir, r.vnir_files, r.rgb, f"{r.gt_rows} rows: {r.gt_columns}"]
            for j, val in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.out.append(f"{len(t)} flight(s) in {d}")

    def _add(self):
        s = state(); d = self.ed_dataset.text().strip()
        if not d:
            QMessageBox.warning(self, "No folder", "Pick (or type) the dataset folder."); return
        flight = self.ed_flight.text().strip()
        if not flight:
            QMessageBox.warning(self, "No name", "Give the flight a name."); return
        items = []
        if self.cb_lidar.isChecked():
            p = s.lidar_features_csv()
            if p and os.path.exists(p): items.append(("lidar", p, os.path.dirname(p) if os.path.basename(p) == "lidar_features.csv" else s.out_dir))
        if self.cb_vnir.isChecked():
            p = s.vnir_features_csv()
            if p and os.path.exists(p): items.append(("vnir", p, s.plots_vnir_dir if (s.plots_vnir_dir and os.path.isdir(s.plots_vnir_dir)) else os.path.dirname(p)))
        if self.cb_rgb.isChecked():
            p = s.rgb_features_csv()
            if p and os.path.exists(p): items.append(("rgb", p, os.path.dirname(p)))
        if not items:
            QMessageBox.warning(self, "Nothing to add", "No feature table of the ticked modalities exists for this flight - run the LiDAR / VNIR / RGB Features tabs first."); return
        gt = self.ed_gt.text().strip() or None
        s.dataset_dir = d; s.flight_name = flight; s.trial_name = self.ed_trial.text().strip(); s.stage = self.cb_stage.currentText().strip()
        if gt: s.gt_csv = gt
        self.btn_add.setEnabled(False); self.progress.setValue(0)
        self._w = _AddWorker(d, flight, s.trial_name, s.stage, self.ed_date.text().strip(), items, gt, self.cb_copy.isChecked())
        self._w.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._w.done_ok.connect(lambda t: (self.btn_add.setEnabled(True), self.out.append(t), self.rescan()))
        self._w.error.connect(lambda e: (self.btn_add.setEnabled(True), self.out.append("ERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._w.start()
