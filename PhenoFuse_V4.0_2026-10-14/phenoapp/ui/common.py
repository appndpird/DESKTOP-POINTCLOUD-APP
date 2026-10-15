"""Widgets shared by the modelling tabs (PhenoFuse 4.0): training-data source, modality choice, file rows."""
from __future__ import annotations
import os
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QGroupBox, QFormLayout,
                             QCheckBox, QComboBox, QRadioButton, QListWidget, QAbstractItemView, QLabel)

from phenoapp.core.project import state


def file_row(parent, line_edit: QLineEdit, title="Pick file", filt="CSV (*.csv)", directory=False, on_change=None):
    row = QHBoxLayout(); b = QPushButton("Browse...")

    def pick():
        if directory:
            p = QFileDialog.getExistingDirectory(parent, title, line_edit.text().strip() or os.path.expanduser("~"))
        else:
            p, _ = QFileDialog.getOpenFileName(parent, title, os.path.dirname(line_edit.text().strip()) if line_edit.text().strip() else "", filt)
        if p:
            line_edit.setText(p)
            if on_change:
                on_change(p)
    b.clicked.connect(pick)
    row.addWidget(line_edit); row.addWidget(b)
    return row


class ModalityBox(QWidget):
    """LiDAR / VNIR / RGB tick boxes + the VNIR input mode."""
    def __init__(self, parent=None, with_vnir_mode=True, on_change=None):
        super().__init__(parent)
        from phenoapp.core.biomass_ml import VNIR_MODES
        from phenoapp.core.pca_families import BAND_TRANSFORMS
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        self.cb_lidar = QCheckBox("LiDAR"); self.cb_vnir = QCheckBox("VNIR"); self.cb_rgb = QCheckBox("RGB")
        self.cb_lidar.setChecked(True); self.cb_vnir.setChecked(True)
        for cb in (self.cb_lidar, self.cb_vnir, self.cb_rgb):
            h.addWidget(cb)
            if on_change:
                cb.toggled.connect(lambda *_: on_change())
        self.cb_lidar.setToolTip("Features of the LiDAR Features tab (lidar_features.csv).")
        self.cb_vnir.setToolTip("Features of the VNIR Spectral Features tab (vnir_features.csv + vnir_spectra.csv).")
        self.cb_rgb.setToolTip("Features of the RGB Features tab (rgb_features.csv).")
        self.cb_mode = None; self.cb_band = None
        if with_vnir_mode:
            h.addSpacing(12); h.addWidget(QLabel("VNIR input:"))
            self.cb_mode = QComboBox()
            for k, v in VNIR_MODES.items():
                self.cb_mode.addItem(f"{k}: {v}", k)
            self.cb_mode.setToolTip("What the VNIR modality feeds the models with: the spectral indices and shape features, "
                                    "the valid reflectance bands of the mean vegetation spectrum (log10, SNV; excluded ranges and "
                                    "bands clipped on > 5 % of the plots left out), or both.")
            h.addWidget(self.cb_mode, stretch=1)
            if on_change:
                self.cb_mode.currentIndexChanged.connect(lambda *_: on_change())
            h.addSpacing(8); h.addWidget(QLabel("Band values:"))
            self.cb_band = QComboBox()
            for k, v in BAND_TRANSFORMS.items():
                self.cb_band.addItem(f"{k}: {v}", k)
            self.cb_band.setToolTip("How the reflectance bands of vnir_spectra.csv enter the models (bands / both modes and the *_bands_PCA "
                                    "families). snv: log10 reflectance then per-plot standard normal variate - removes the overall brightness "
                                    "of a plot, so it transfers better between flights and illumination conditions. raw: the mean vegetation "
                                    "reflectance (0-1) as measured - keeps the absolute level, which can carry biomass / cover signal within one "
                                    "flight but shifts with sun angle and calibration between flights. Try both with the same validation; a "
                                    "saved model remembers its choice and predictions use it automatically.")
            h.addWidget(self.cb_band, stretch=1)
            if on_change:
                self.cb_band.currentIndexChanged.connect(lambda *_: on_change())
        h.addStretch()

    def modalities(self):
        return [m for m, cb in (("lidar", self.cb_lidar), ("vnir", self.cb_vnir), ("rgb", self.cb_rgb)) if cb.isChecked()]

    def vnir_mode(self):
        return self.cb_mode.currentData() if self.cb_mode is not None else "indices"

    def band_transform(self):
        return self.cb_band.currentData() if self.cb_band is not None else "snv"


class TrainingSource(QGroupBox):
    """Where the training rows come from: this flight (feature tables of tabs 6-8 + a ground-truth CSV) or a dataset
    folder (several flights, ground truth inside)."""
    def __init__(self, parent=None, title="Training data", gt_hint="Plot_ID + target column"):
        super().__init__(title, parent)
        f = QFormLayout(self)
        self.rb_flight = QRadioButton("This flight (feature tables of the LiDAR / VNIR / RGB tabs + ground-truth CSV)")
        self.rb_dataset = QRadioButton("Dataset folder (several flights; features and ground truth inside, see the Dataset tab)")
        self.rb_flight.setChecked(True)
        f.addRow(self.rb_flight); f.addRow(self.rb_dataset)
        self.ed_lidar = QLineEdit(); self.ed_vnir = QLineEdit(); self.ed_rgb = QLineEdit(); self.ed_gt = QLineEdit()
        self.ed_gt.setPlaceholderText(gt_hint)
        f.addRow("LiDAR features CSV:", file_row(self, self.ed_lidar, "LiDAR features table"))
        f.addRow("VNIR features CSV:", file_row(self, self.ed_vnir, "VNIR features table"))
        f.addRow("RGB features CSV:", file_row(self, self.ed_rgb, "RGB features table"))
        f.addRow("Ground-truth CSV:", file_row(self, self.ed_gt, "Ground-truth table"))
        self.ed_dataset = QLineEdit()
        f.addRow("Dataset folder:", file_row(self, self.ed_dataset, "Dataset folder", directory=True, on_change=lambda p: self.refresh_flights()))
        self.lst_flights = QListWidget(); self.lst_flights.setSelectionMode(QAbstractItemView.ExtendedSelection); self.lst_flights.setMaximumHeight(110)
        row = QHBoxLayout(); row.addWidget(self.lst_flights, stretch=1)
        col = QVBoxLayout(); b1 = QPushButton("All"); b1.clicked.connect(self.lst_flights.selectAll); b2 = QPushButton("Rescan"); b2.clicked.connect(self.refresh_flights)
        col.addWidget(b1); col.addWidget(b2); col.addStretch(); row.addLayout(col)
        f.addRow("Flights (select):", row)
        self.rb_flight.toggled.connect(self._mode)
        self._mode()

    def _mode(self):
        fl = self.rb_flight.isChecked()
        for w in (self.ed_lidar, self.ed_vnir, self.ed_rgb, self.ed_gt):
            w.setEnabled(fl)
        self.ed_dataset.setEnabled(not fl); self.lst_flights.setEnabled(not fl)

    def sync_from_state(self):
        s = state()
        if not self.ed_lidar.text().strip():
            p = s.lidar_features_csv()
            if p and os.path.exists(p): self.ed_lidar.setText(p)
        if not self.ed_vnir.text().strip():
            p = s.vnir_features_csv()
            if p and os.path.exists(p): self.ed_vnir.setText(p)
        if not self.ed_rgb.text().strip():
            p = s.rgb_features_csv()
            if p and os.path.exists(p): self.ed_rgb.setText(p)
        if not self.ed_gt.text().strip() and s.gt_csv and os.path.exists(s.gt_csv):
            self.ed_gt.setText(s.gt_csv)
        if not self.ed_dataset.text().strip() and s.dataset_dir:
            self.ed_dataset.setText(s.dataset_dir); self.refresh_flights()

    def refresh_flights(self):
        from phenoapp.core.dataset import scan
        self.lst_flights.clear(); d = self.ed_dataset.text().strip()
        if not d or not os.path.isdir(d):
            return
        state().dataset_dir = d
        try:
            t = scan(d)
        except Exception:
            return
        for _, r in t.iterrows():
            mods = "+".join(m for m in ("lidar", "vnir", "rgb") if r.get(m, 0) > 0)
            self.lst_flights.addItem(f"{r.flight}   [{mods}; GT: {r.gt_columns or 'none'}]")
        self.lst_flights.selectAll()

    def selected_flights(self):
        return [self.lst_flights.item(i).text().split("   [")[0] for i in range(self.lst_flights.count()) if self.lst_flights.item(i).isSelected()]

    def is_dataset(self):
        return self.rb_dataset.isChecked()

    def load(self, modalities, gt_required=True):
        """-> dict (see core.dataset.load_training_table) and the root folder for the Models directory."""
        from phenoapp.core.dataset import load_training_table, single_flight_table
        s = state()
        if self.is_dataset():
            d = self.ed_dataset.text().strip()
            if not d or not os.path.isdir(d):
                raise RuntimeError("pick a dataset folder")
            fl = self.selected_flights()
            if not fl:
                raise RuntimeError("select at least one flight")
            t = load_training_table(d, modalities, flights=fl)
            if gt_required and (t["gt"] is None or len(t["gt"]) == 0):
                raise RuntimeError("none of the selected flights has a ground_truth.csv")
            return t, d
        lidar = self.ed_lidar.text().strip() if "lidar" in modalities else ""
        vnir = self.ed_vnir.text().strip() if "vnir" in modalities else ""
        rgb = self.ed_rgb.text().strip() if "rgb" in modalities else ""
        gt = self.ed_gt.text().strip()
        for m, p in (("lidar", lidar), ("vnir", vnir), ("rgb", rgb)):
            if m in modalities and not (p and os.path.exists(p)):
                raise RuntimeError(f"{m.upper()} is ticked but its features CSV is missing - run the {m.upper()} tab or untick it")
        if gt_required and not (gt and os.path.exists(gt)):
            raise RuntimeError("pick the ground-truth CSV")
        if gt:
            s.gt_csv = gt
        t = single_flight_table(lidar, vnir, rgb, gt, flight=s.flight_name or "flight")
        root = os.path.dirname(os.path.dirname(lidar or vnir or rgb)) if (lidar or vnir or rgb) else (s.proj_dir or os.path.expanduser("~"))
        return t, root


class InferenceSource(QGroupBox):
    """Feature tables of the flight to predict (this flight, or one flight of a dataset)."""
    def __init__(self, parent=None, title="Data to predict"):
        super().__init__(title, parent)
        f = QFormLayout(self)
        self.rb_flight = QRadioButton("This flight (feature tables of the LiDAR / VNIR / RGB tabs)")
        self.rb_dataset = QRadioButton("A flight inside a dataset folder")
        self.rb_flight.setChecked(True); f.addRow(self.rb_flight); f.addRow(self.rb_dataset)
        self.ed_lidar = QLineEdit(); self.ed_vnir = QLineEdit(); self.ed_rgb = QLineEdit()
        f.addRow("LiDAR features CSV:", file_row(self, self.ed_lidar, "LiDAR features table"))
        f.addRow("VNIR features CSV:", file_row(self, self.ed_vnir, "VNIR features table"))
        f.addRow("RGB features CSV:", file_row(self, self.ed_rgb, "RGB features table"))
        self.ed_dataset = QLineEdit(); self.cb_flight = QComboBox()
        f.addRow("Dataset folder:", file_row(self, self.ed_dataset, "Dataset folder", directory=True, on_change=lambda p: self.refresh_flights()))
        f.addRow("Flight:", self.cb_flight)
        self.rb_flight.toggled.connect(self._mode); self._mode()

    def _mode(self):
        fl = self.rb_flight.isChecked()
        for w in (self.ed_lidar, self.ed_vnir, self.ed_rgb):
            w.setEnabled(fl)
        self.ed_dataset.setEnabled(not fl); self.cb_flight.setEnabled(not fl)

    def sync_from_state(self):
        s = state()
        for ed, p in ((self.ed_lidar, s.lidar_features_csv()), (self.ed_vnir, s.vnir_features_csv()), (self.ed_rgb, s.rgb_features_csv())):
            if not ed.text().strip() and p and os.path.exists(p):
                ed.setText(p)
        if not self.ed_dataset.text().strip() and s.dataset_dir:
            self.ed_dataset.setText(s.dataset_dir); self.refresh_flights()

    def refresh_flights(self):
        from phenoapp.core.dataset import scan
        self.cb_flight.clear(); d = self.ed_dataset.text().strip()
        if d and os.path.isdir(d):
            try:
                for fl in scan(d).flight:
                    self.cb_flight.addItem(fl)
            except Exception:
                pass

    def load(self):
        """-> (table dict, output folder, label)"""
        from phenoapp.core.dataset import load_training_table, single_flight_table
        s = state()
        if self.rb_dataset.isChecked():
            d = self.ed_dataset.text().strip(); fl = self.cb_flight.currentText()
            if not d or not fl:
                raise RuntimeError("pick a dataset folder and a flight")
            have = [m for m in ("lidar", "vnir", "rgb") if os.path.exists(os.path.join(d, m, fl, f"{m}_features.csv"))]
            t = load_training_table(d, have, flights=[fl])
            return t, os.path.join(d, "predictions"), fl
        lidar, vnir, rgb = (e.text().strip() for e in (self.ed_lidar, self.ed_vnir, self.ed_rgb))
        t = single_flight_table(lidar if os.path.exists(lidar) else "", vnir if os.path.exists(vnir) else "", rgb if os.path.exists(rgb) else "", None,
                                flight=s.flight_name or "flight")
        root = os.path.dirname(os.path.dirname(lidar or vnir or rgb))
        return t, os.path.join(root, "Models", "predictions"), s.flight_name or "flight"
