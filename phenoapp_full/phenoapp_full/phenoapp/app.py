"""
Plant Phenotyping App — Desktop (PyQt5) entry point
====================================================

Launch with:
    python -m phenoapp.app
or from the project root:
    python run.py
"""

from __future__ import annotations
import sys
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QLabel
)

# Make sure we can import the package whether run as module or script
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from phenoapp.ui.project_tab        import ProjectTab
from phenoapp.ui.generate_grid_tab  import GenerateGridTab
from phenoapp.ui.edit_tab           import EditTab
from phenoapp.ui.visualize_tab      import VisualizeTab
from phenoapp.ui.traits_tab         import TraitsTab
from phenoapp.ui.statistics_tab     import StatisticsTab
from phenoapp.ui.guidance_tab       import GuidanceTab
from phenoapp.ui.about_tab          import AboutTab, _resource_path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Phenotyping from Point Clouds")
        self.resize(1500, 950)

        # Window icon — use the APPN logo if present
        icon_path = _resource_path("assets/appn_logo.png")
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.tabs = QTabWidget()
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)

        # Build all tabs
        self.tab_project   = ProjectTab()
        self.tab_grid_gen  = GenerateGridTab()
        self.tab_edit      = EditTab()
        self.tab_visualize = VisualizeTab()
        self.tab_traits    = TraitsTab()
        self.tab_stats     = StatisticsTab()
        self.tab_guide     = GuidanceTab()
        self.tab_about     = AboutTab()

        self.tabs.addTab(self.tab_project,   "1. Project")
        self.tabs.addTab(self.tab_grid_gen,  "2. Generate Grid")
        self.tabs.addTab(self.tab_edit,      "3. Edit")
        self.tabs.addTab(self.tab_visualize, "4. Visualize")
        self.tabs.addTab(self.tab_traits,    "5. Traits")
        self.tabs.addTab(self.tab_stats,     "6. Statistics")
        self.tabs.addTab(self.tab_guide,     "7. Guidance")
        self.tabs.addTab(self.tab_about,     "About")

        # When project loads, refresh the dependent tabs
        self.tab_project.project_ready.connect(self._on_project_ready)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Welcome. Start on the Project tab.", 5000)

    def _on_project_ready(self):
        # Auto-refresh all tabs that depend on the canopy/grid
        try: self.tab_grid_gen.reload_canopy()
        except Exception: pass
        try: self.tab_edit.reload_grid_and_canopy()
        except Exception: pass
        try: self.tab_visualize.reload()
        except Exception: pass
        self.statusBar().showMessage(
            "Project loaded. Use Generate Grid (if needed), Edit to align, "
            "Traits to compute.", 8000)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
