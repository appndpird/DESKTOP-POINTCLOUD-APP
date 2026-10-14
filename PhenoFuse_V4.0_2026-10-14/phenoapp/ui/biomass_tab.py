"""
Biomass Models tab (PhenoFuse 4.0) - the v3 "Biomass & Fusion" and "Biomass ML" tabs merged.

Two sections:
  Use a pretrained model   pick a saved model / ensemble (this flight's Models folder, a dataset's Models folder, the bundled
                           ones or any .joblib), point it at the feature tables of a flight, predict every plot.
  Train from scratch       LiDAR / VNIR / RGB in any combination, this flight or a dataset of several flights, simple
                           linear calibrations + core / PCA / fused families x learners, any ground-truth unit, nested
                           cross-validation; models saved under Models/Biomass models/<run>/.
No index is recomputed here: the inputs are the feature tables of tabs 6, 7 and 8.
"""
from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget

from phenoapp.ui.model_panels import TrainPanel, PretrainedPanel


class BiomassTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Biomass models: LiDAR, VNIR, RGB and fusion</h2>"))
        v.addWidget(QLabel("Inputs are the per-plot feature tables of the LiDAR, VNIR Spectral and RGB Features tabs (nothing is recomputed from the bands). "
                           "Use a pretrained model on a new flight, or train from scratch on this flight or on a dataset of several flights "
                           "(any modality alone or fused; VNIR as indices, valid reflectance bands or both; ground truth in any biomass unit)."))
        self.tabs = QTabWidget()
        self.pretrained = PretrainedPanel("biomass"); self.train = TrainPanel("biomass")
        self.tabs.addTab(self.pretrained, "Use a pretrained model (new trial / flight)")
        self.tabs.addTab(self.train, "Train, validate and test from scratch")
        v.addWidget(self.tabs, stretch=1)
