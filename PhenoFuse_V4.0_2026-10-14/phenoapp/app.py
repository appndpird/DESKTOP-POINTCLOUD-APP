"""
PhenoFuse - multi-sensor plot phenotyping (LiDAR, VNIR hyperspectral, RGB) - desktop (PyQt5) entry point.

Launch with:
    python -m phenoapp.app
or from the project root:
    python run.py

Tabs
  1 Project            2 Point Cloud QC     3 Generate Grid     4 Edit             5 Visualize
  6 LiDAR Features     7 VNIR Spectral Features   8 RGB Features
  9 Biomass Models     10 Height Models     11 Deep Models      12 Dataset
  13 Statistics        14 Guidance          About
Every modelling tab has two sections: use a pretrained model on new data, or train / validate / test from scratch
(this flight, or a dataset folder with several flights).
"""

from __future__ import annotations
import sys
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QScrollArea, QWidget, QSizePolicy
)

# Make sure we can import the package whether run as module or script
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from phenoapp import __version__, APP_NAME, APP_DATE, APP_TAGLINE
from phenoapp.ui.project_tab        import ProjectTab
from phenoapp.ui.generate_grid_tab  import GenerateGridTab
from phenoapp.ui.edit_tab           import EditTab
from phenoapp.ui.visualize_tab      import VisualizeTab
from phenoapp.ui.traits_tab         import TraitsTab
from phenoapp.ui.spectral_tab       import SpectralTab
from phenoapp.ui.rgb_tab            import RGBTab
from phenoapp.ui.biomass_tab        import BiomassTab
from phenoapp.ui.height_tab         import HeightTab
from phenoapp.ui.deep_tab           import DeepTab
from phenoapp.ui.dataset_tab        import DatasetTab
from phenoapp.ui.qc_tab             import QCTab
from phenoapp.ui.statistics_tab     import StatisticsTab
from phenoapp.ui.guidance_tab       import GuidanceTab
from phenoapp.ui.about_tab          import AboutTab, _resource_path


def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap a tab in a scroll area so the main window can be made smaller than the tab's natural size
    (every edge of the window then resizes freely; the tab scrolls instead of blocking the resize)."""
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.NoFrame)
    sa.setWidget(widget)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    sa.setMinimumSize(200, 150)
    return sa


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__} ({APP_DATE}) - {APP_TAGLINE}")

        icon_path = _resource_path("assets/appn_logo.png")
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.tabs = QTabWidget()
        self.tabs.setMovable(False)
        self.tabs.setUsesScrollButtons(True)
        self.setCentralWidget(self.tabs)
        self.setMinimumSize(480, 360)

        # Build all tabs
        self.tab_project   = ProjectTab()
        self.tab_qc        = QCTab()
        self.tab_grid_gen  = GenerateGridTab()
        self.tab_edit      = EditTab()
        self.tab_visualize = VisualizeTab()
        self.tab_traits    = TraitsTab()
        self.tab_spectral  = SpectralTab()
        self.tab_rgb       = RGBTab()
        self.tab_biomass   = BiomassTab()
        self.tab_height    = HeightTab()
        self.tab_deep      = DeepTab()
        self.tab_dataset   = DatasetTab()
        self.tab_stats     = StatisticsTab()
        self.tab_guide     = GuidanceTab()
        self.tab_about     = AboutTab()

        for w, name in ((self.tab_project,   "1. Project"),
                        (self.tab_qc,        "2. Point Cloud QC"),
                        (self.tab_grid_gen,  "3. Generate Grid"),
                        (self.tab_edit,      "4. Edit"),
                        (self.tab_visualize, "5. Visualize"),
                        (self.tab_traits,    "6. LiDAR Features"),
                        (self.tab_spectral,  "7. VNIR Spectral Features"),
                        (self.tab_rgb,       "8. RGB Features"),
                        (self.tab_biomass,   "9. Biomass Models"),
                        (self.tab_height,    "10. Height Models"),
                        (self.tab_deep,      "11. Deep Models"),
                        (self.tab_dataset,   "12. Dataset"),
                        (self.tab_stats,     "13. Statistics"),
                        (self.tab_guide,     "14. Guidance"),
                        (self.tab_about,     "About")):
            self.tabs.addTab(_scrollable(w), name)

        self.tab_project.project_ready.connect(self._on_project_ready)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Welcome. Start on the Project tab.", 5000)

        # open at 90 % of the available screen, never larger than the screen; the window stays resizable from
        # every edge because each tab lives in a scroll area
        try:
            geo = QApplication.primaryScreen().availableGeometry()
            w, h = int(geo.width() * 0.9), int(geo.height() * 0.9)
            self.resize(min(1500, w), min(950, h))
            self.move(geo.x() + (geo.width() - self.width()) // 2, geo.y() + (geo.height() - self.height()) // 2)
        except Exception:
            self.resize(1400, 900)

    def _on_project_ready(self):
        for fn in (self.tab_grid_gen.reload_canopy, self.tab_edit.reload_grid_and_canopy, self.tab_visualize.reload):
            try:
                fn()
            except Exception:
                pass
        self.statusBar().showMessage(
            "Project loaded. Generate Grid (if needed), Edit to align, then LiDAR / VNIR / RGB Features.", 8000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
