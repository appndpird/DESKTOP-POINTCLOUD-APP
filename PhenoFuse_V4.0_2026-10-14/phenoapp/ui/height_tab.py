"""
Height Models tab (PhenoFuse 4.0).

Two sections:
  Use a pretrained model   a saved height model / top-three ensemble applied to the feature tables of a flight (+ optional
                           per-flight offset and a check against a measured CSV).
  Train from scratch       LiDAR (+ VNIR, + RGB) features of this flight or of a dataset of several flights against measured
                           plant height; linear calibration on cth_p95 / H95, structure models, nested-selection and PCA
                           models; models saved under Models/Height models/<run>/.
"""
from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget

from phenoapp.ui.model_panels import TrainPanel, PretrainedPanel


class HeightTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Plant-height models</h2>"))
        v.addWidget(QLabel("Uses the per-plot feature tables (LiDAR Features tab with the canopy-top cth_* heights and 'LiDAR v3' ticked; VNIR and RGB "
                           "tables optional). Within a trial the linear calibration on cth_p95 is the reference; the PCA and random-forest models are "
                           "the ones to carry to another date with a per-flight offset. Validate only against a height measured on the day of the flight."))
        self.tabs = QTabWidget()
        self.pretrained = PretrainedPanel("height"); self.train = TrainPanel("height")
        self.tabs.addTab(self.pretrained, "Use a pretrained model (new trial / flight)")
        self.tabs.addTab(self.train, "Train, validate and test from scratch")
        v.addWidget(self.tabs, stretch=1)
