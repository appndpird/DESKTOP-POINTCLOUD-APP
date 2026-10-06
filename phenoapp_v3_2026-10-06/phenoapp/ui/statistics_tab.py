"""
Statistics tab: load the per-plot metrics CSV and produce graphs.

Provides:
  - Distribution plots (histograms) per trait
  - Field heatmap (Bank × Row coloured by trait)
  - Bank-level box plots
  - Correlation matrix
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QSizePolicy, QGroupBox, QFormLayout,
    QTabWidget
)

# Use Qt5Agg for matplotlib embedding
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar
)

from phenoapp.core.project import state


class _MplCanvas(FigureCanvas):
    def __init__(self, parent=None, w=8, h=5):
        self.fig = Figure(figsize=(w, h), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)


class StatisticsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._numeric_cols = []
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        # ---- top bar ----
        bar = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load metrics CSV")
        self.btn_load.clicked.connect(self.load_csv)
        bar.addWidget(self.btn_load)

        self.btn_reload = QPushButton("🔄 Reload from project")
        self.btn_reload.clicked.connect(self.reload_from_state)
        bar.addWidget(self.btn_reload)

        self.lbl_status = QLabel("No metrics loaded.")
        bar.addWidget(self.lbl_status); bar.addStretch()
        v.addLayout(bar)

        # ---- choose trait ----
        f = QFormLayout()
        self.cmb_trait = QComboBox()
        self.cmb_trait.currentTextChanged.connect(self._refresh_plots)
        f.addRow("Active trait:", self.cmb_trait)
        v.addLayout(f)

        # ---- plot tabs ----
        self.plots = QTabWidget()
        self.cv_hist = _MplCanvas(); self.cv_heat = _MplCanvas()
        self.cv_box  = _MplCanvas(); self.cv_corr = _MplCanvas()
        self.plots.addTab(self._wrap(self.cv_hist), "Distribution")
        self.plots.addTab(self._wrap(self.cv_heat), "Field heatmap")
        self.plots.addTab(self._wrap(self.cv_box),  "By bank")
        self.plots.addTab(self._wrap(self.cv_corr), "Correlations")
        v.addWidget(self.plots, stretch=1)

    def _wrap(self, canvas):
        w = QWidget(); l = QVBoxLayout(w)
        l.addWidget(NavToolbar(canvas, w))
        l.addWidget(canvas)
        return w

    # ---- loading ----
    def reload_from_state(self):
        path = state().out_csv
        if path and os.path.exists(path):
            self._load_path(path)
        else:
            QMessageBox.information(self, "No metrics CSV",
                "Run the Traits tab first to produce metrics.")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open metrics CSV", state().out_csv or "",
            "CSV (*.csv);;All files (*)")
        if path: self._load_path(path)

    def _load_path(self, path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._df = df
        # numeric columns
        num = df.select_dtypes(include="number").columns.tolist()
        # exclude IDs
        for skip in ("Plot_ID", "Bank", "Row"):
            if skip in num: num.remove(skip)
        self._numeric_cols = num
        self.cmb_trait.clear()
        self.cmb_trait.addItems(num)
        self.lbl_status.setText(f"{len(df)} plots, {len(num)} numeric traits.  Source: {os.path.basename(path)}")
        self._refresh_plots()

    # ---- plotting ----
    def _refresh_plots(self):
        if self._df is None: return
        trait = self.cmb_trait.currentText()
        if not trait or trait not in self._df.columns: return
        df = self._df[self._df[trait].notna()]
        s = df[trait].astype(float)

        # 1. histogram
        self.cv_hist.fig.clear()
        ax = self.cv_hist.fig.add_subplot(111)
        ax.hist(s, bins=30, color="#3a7", edgecolor="black", alpha=0.85)
        ax.set_xlabel(trait); ax.set_ylabel("count")
        ax.set_title(f"Distribution of {trait}  (n={len(s)})")
        ax.axvline(s.mean(), color="red", ls="--", label=f"mean={s.mean():.2f}")
        ax.axvline(s.median(), color="orange", ls="--", label=f"median={s.median():.2f}")
        ax.legend()
        self.cv_hist.draw()

        # 2. field heatmap (Bank × Row)
        self.cv_heat.fig.clear()
        ax = self.cv_heat.fig.add_subplot(111)
        if "Bank" in df.columns and "Row" in df.columns:
            piv = df.pivot_table(index="Bank", columns="Row", values=trait, aggfunc="mean")
            im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
            ax.set_xlabel("Row"); ax.set_ylabel("Bank"); ax.set_title(f"{trait} — field heatmap")
            ax.set_xticks(range(0, piv.shape[1], max(1, piv.shape[1] // 20)))
            ax.set_xticklabels(piv.columns[::max(1, piv.shape[1] // 20)])
            ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
            self.cv_heat.fig.colorbar(im, ax=ax, label=trait)
        else:
            ax.text(0.5, 0.5, "No Bank/Row columns in CSV", ha="center", transform=ax.transAxes)
        self.cv_heat.draw()

        # 3. boxplot by bank
        self.cv_box.fig.clear()
        ax = self.cv_box.fig.add_subplot(111)
        if "Bank" in df.columns:
            banks = sorted(df["Bank"].dropna().unique())
            data = [df[df["Bank"] == b][trait].dropna().values for b in banks]
            ax.boxplot(data, labels=[f"Bank {int(b)}" for b in banks])
            ax.set_ylabel(trait); ax.set_title(f"{trait} by bank")
        else:
            ax.text(0.5, 0.5, "No Bank column", ha="center", transform=ax.transAxes)
        self.cv_box.draw()

        # 4. correlation matrix
        self.cv_corr.fig.clear()
        ax = self.cv_corr.fig.add_subplot(111)
        if len(self._numeric_cols) >= 2:
            corr = self._df[self._numeric_cols].corr()
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index, fontsize=8)
            ax.set_title("Pearson correlation between traits")
            self.cv_corr.fig.colorbar(im, ax=ax)
        self.cv_corr.draw()
