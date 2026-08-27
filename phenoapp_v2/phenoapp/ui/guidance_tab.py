"""
Guidance tab: embedded markdown documentation.
"""

from __future__ import annotations
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextBrowser


class GuidanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        v.addWidget(self.text)
        self._load_markdown()

    def _load_markdown(self):
        # Look for guidance.md in package docs/ — tolerate frozen (PyInstaller) layouts
        candidates = []
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "..", "docs", "guidance.md"))
        candidates.append(os.path.join(os.path.dirname(here), "docs", "guidance.md"))
        # PyInstaller _MEIPASS layout
        try:
            import sys
            mp = getattr(sys, "_MEIPASS", None)
            if mp:
                candidates.append(os.path.join(mp, "phenoapp", "docs", "guidance.md"))
                candidates.append(os.path.join(mp, "docs", "guidance.md"))
        except Exception:
            pass

        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        md = f.read()
                    # Qt 5 QTextBrowser supports markdown via setMarkdown
                    if hasattr(self.text, "setMarkdown"):
                        self.text.setMarkdown(md)
                    else:
                        self.text.setPlainText(md)
                    return
                except Exception:
                    continue
        self.text.setPlainText("Guidance file not found.")
