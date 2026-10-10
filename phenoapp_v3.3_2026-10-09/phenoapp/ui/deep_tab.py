"""
Deep Models tab (v3.1): two-stream network (sparse 3D CNN on the plot cloud + spectral transformer on VNIR
pixels) for plot biomass and plant height.

  1. Compute environment: PhenoApp finds a Python environment with torch (and spconv); GPU is used
     automatically when CUDA is available, otherwise CPU.
  2. Prepare plot tensors from the project LAS + VNIR cube + grid (LiDAR v3 preprocessing, cleaned VNIR).
  3. Predict with the bundled pretrained fold ensemble (wheat, anthesis and maturity flights), or
  4. Train / cross-validate the network on this trial's ground truth.
"""

from __future__ import annotations
import os
import json

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QFileDialog,
                             QMessageBox, QLineEdit, QTextEdit, QFormLayout, QComboBox, QSpinBox)

from phenoapp.core.project import state
from phenoapp.core import load_grid, LASManager, VNIRCube
from phenoapp.core.deep_models import (auto_select_env, probe_env, prepare_plot_tensors, run_two_stream, pretrained_weights_dir, find_torch_pythons)


class _Probe(QThread):
    done = pyqtSignal(object)
    def __init__(self, exe=None): super().__init__(); self.exe = exe
    def run(self):
        try: self.done.emit(probe_env(self.exe) if self.exe else auto_select_env())
        except Exception as e: self.done.emit({"error": str(e)})


class _Prep(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)
    def __init__(self, las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width):
        super().__init__(); self._a = (las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width)
    def run(self):
        try:
            import pandas as pd
            las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width = self._a
            self.progress.emit(1, "loading point cloud..."); mgr = LASManager(las, crs, use_smrf=False); mgr.load(progress_cb=lambda p, m: None)
            cube = VNIRCube(vnir); plots = load_grid(grid, target_crs=crs); gt = pd.read_csv(gt_csv) if gt_csv and os.path.exists(gt_csv) else None
            idx = prepare_plot_tensors(mgr, cube, plots, out_dir, name, stage, gt=gt, progress_cb=lambda p, m: self.progress.emit(p, m), region_mode=region_mode, band_width=band_width)
            cube.close(); del mgr
            self.done_ok.emit(f"{len(idx[idx.dataset == name])} plot tensors written to {os.path.join(out_dir, name)}\nindex -> {os.path.join(out_dir, 'index.csv')}")
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _Run(QThread):
    line = pyqtSignal(str); done_ok = pyqtSignal(int)
    def __init__(self, *a, **k): super().__init__(); self.a = a; self.k = k
    def run(self):
        try: rc = run_two_stream(*self.a, log_cb=lambda l: self.line.emit(l), **self.k)
        except Exception as e: self.line.emit(f"ERROR: {e}"); rc = -1
        self.done_ok.emit(rc)


class DeepTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._env = None; self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Deep models: two-stream network (sparse 3D CNN + spectral transformer)</h2>"))
        v.addWidget(QLabel("Pretrained fold ensemble (wheat plots, anthesis and maturity flights). Within a trial the feature models of the Height Models / Biomass ML tabs "
                           "were as good or better; use this tab to score a new flight with the ensemble "
                           "or to retrain once several hundred labelled plots are available. The GPU is used automatically when CUDA is available."))
        g = QGroupBox("1. Compute environment"); f = QFormLayout(g)
        row = QHBoxLayout(); self.cb_env = QComboBox(); self.cb_env.setEditable(True); self.cb_env.addItems(find_torch_pythons())
        b = QPushButton("Detect automatically"); b.clicked.connect(self._auto); b2 = QPushButton("Probe selected"); b2.clicked.connect(self._probe)
        row.addWidget(self.cb_env, stretch=3); row.addWidget(b); row.addWidget(b2); f.addRow("Python with torch:", row)
        self.cb_device = QComboBox(); self.cb_device.addItems(["auto (GPU if available, else CPU)", "cuda (GPU)", "cpu"]); f.addRow("Device:", self.cb_device)
        self.lbl_env = QLabel("not probed yet"); f.addRow("Status:", self.lbl_env)
        v.addWidget(g)
        g2 = QGroupBox("2. Plot tensors (LiDAR v3 preprocessing + cleaned VNIR pixels per plot)"); f2 = QFormLayout(g2)
        self.ed_name = QLineEdit("my_trial_anthesis"); self.cb_stage = QComboBox(); self.cb_stage.addItems(["anthesis", "maturity"])
        row = QHBoxLayout(); row.addWidget(self.ed_name, stretch=2); row.addWidget(QLabel("stage:")); row.addWidget(self.cb_stage); f2.addRow("Dataset name:", row)
        row = QHBoxLayout(); self.ed_gt = QLineEdit(); self.ed_gt.setPlaceholderText("optional: Plot_ID, biomass_kg_ha and/or height_cm (needed for training / scoring)")
        b = QPushButton("Browse..."); b.clicked.connect(lambda: self._pick(self.ed_gt)); row.addWidget(self.ed_gt); row.addWidget(b); f2.addRow("Ground truth CSV:", row)
        self.btn_prep = QPushButton("Prepare tensors from the project LAS + VNIR cube + grid"); self.btn_prep.clicked.connect(self._prep); f2.addRow("", self.btn_prep)
        v.addWidget(g2)
        g3 = QGroupBox("3. Run"); f3 = QFormLayout(g3)
        self.cb_target = QComboBox(); self.cb_target.addItems(["biomass", "height"]); self.cb_variant = QComboBox(); self.cb_variant.addItems(["two_stream", "lidar_only", "vnir_only"])
        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(5, 500); self.sp_epochs.setValue(100); self.sp_folds = QSpinBox(); self.sp_folds.setRange(2, 20); self.sp_folds.setValue(10)
        row = QHBoxLayout(); row.addWidget(QLabel("target")); row.addWidget(self.cb_target); row.addWidget(QLabel("variant")); row.addWidget(self.cb_variant); row.addWidget(QLabel("epochs")); row.addWidget(self.sp_epochs); row.addWidget(QLabel("folds")); row.addWidget(self.sp_folds); row.addStretch(); f3.addRow("Options:", row)
        row = QHBoxLayout(); self.ed_weights = QLineEdit(pretrained_weights_dir()); b = QPushButton("Browse..."); b.clicked.connect(lambda: self._pick_dir(self.ed_weights)); row.addWidget(self.ed_weights); row.addWidget(b); f3.addRow("Weights folder:", row)
        self.cb_finetune = QCheckBox("Fine-tune: warm start the training from the weights folder (pretrained ensemble) instead of random weights")
        self.cb_finetune.setChecked(True)
        self.cb_finetune.setToolTip("Training only. The encoders and the head of the matching fold model are loaded, the trial/stage embedding is re-initialised for the "
                                    "new dataset(s), and the peak learning rate is lowered to 3e-4. Use it when the new trial has fewer than a few hundred labelled plots; "
                                    "untick to train from scratch. Band policy: the common 111-band list shipped with the ensemble.")
        f3.addRow("", self.cb_finetune)
        row = QHBoxLayout(); self.btn_pred = QPushButton("Predict with pretrained ensemble"); self.btn_pred.clicked.connect(lambda: self._run("predict"))
        self.btn_train = QPushButton("Train / cross-validate on this trial"); self.btn_train.clicked.connect(lambda: self._run("train")); row.addWidget(self.btn_pred); row.addWidget(self.btn_train); row.addStretch(); f3.addRow("", row)
        v.addWidget(g3)
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); v.addWidget(self.out, stretch=1)

    def _data_dir(self):
        s = state(); base = os.path.dirname(s.out_csv) if s.out_csv else (os.path.dirname(s.las_path) if s.las_path else "")
        return os.path.join(base, "dl_data")

    def _pick(self, ed):
        p, _ = QFileDialog.getOpenFileName(self, "CSV", "", "CSV (*.csv)")
        if p: ed.setText(p)

    def _pick_dir(self, ed):
        d = QFileDialog.getExistingDirectory(self, "Folder")
        if d: ed.setText(d)

    def _auto(self):
        self.lbl_env.setText("probing environments..."); self._pr = _Probe(None); self._pr.done.connect(self._on_probe); self._pr.start()

    def _probe(self):
        exe = self.cb_env.currentText().strip(); self.lbl_env.setText("probing..."); self._pr = _Probe(exe); self._pr.done.connect(self._on_probe); self._pr.start()

    def _on_probe(self, d):
        if not d or not d.get("torch"):
            self.lbl_env.setText("no torch found: " + str((d or {}).get("error", ""))[:120]); self._env = None; return
        self._env = d
        if self.cb_env.findText(d["python"]) < 0: self.cb_env.addItem(d["python"])
        self.cb_env.setCurrentText(d["python"])
        self.lbl_env.setText(f"torch {d['torch']} | " + (f"GPU: {d['gpu']}" if d.get("cuda") else "no CUDA -> CPU") + f" | spconv {'yes' if d.get('spconv') else 'NO (LiDAR stream unavailable, VNIR-only fallback)'}")

    def _prep(self):
        s = state()
        for nm, p in (("LAS", s.las_path), ("VNIR cube", s.vnir_path)):
            if not p or not os.path.exists(p): QMessageBox.warning(self, f"No {nm}", f"Set the {nm} on the Project / VNIR Spectral tab first."); return
        grid = s.saved_grid if (s.saved_grid and os.path.exists(s.saved_grid)) else s.grid_path
        if not grid or not os.path.exists(grid): QMessageBox.warning(self, "No grid", "Load / align a plot grid first."); return
        self.btn_prep.setEnabled(False); self.progress.setValue(0)
        self._pp = _Prep(s.las_path, s.vnir_path, grid, s.work_crs, self._data_dir(), self.ed_name.text().strip() or "trial", self.cb_stage.currentText(), self.ed_gt.text().strip(), s.region_mode, s.band_width)
        self._pp.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._pp.done_ok.connect(lambda t: (self.btn_prep.setEnabled(True), self.out.append(t)))
        self._pp.error.connect(lambda e: (self.btn_prep.setEnabled(True), self.out.append("ERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._pp.start()

    def _run(self, mode):
        exe = self.cb_env.currentText().strip()
        if not exe or not os.path.exists(exe): QMessageBox.warning(self, "No environment", "Pick or detect a Python environment with torch."); return
        data = self._data_dir()
        if not os.path.exists(os.path.join(data, "index.csv")): QMessageBox.warning(self, "No tensors", "Prepare the plot tensors first (step 2)."); return
        device = ("auto", "cuda", "cpu")[self.cb_device.currentIndex()]; target = self.cb_target.currentText(); variant = self.cb_variant.currentText()
        s = state(); base = os.path.splitext(s.out_csv)[0] if s.out_csv else os.path.join(data, "dl")
        out = base + f"_dl_{target}_{variant}_pretrained_predictions.csv" if mode == "predict" else base + f"_dl_{target}_training"
        self.btn_pred.setEnabled(False); self.btn_train.setEnabled(False); self.out.append(f"\n=== {mode}: target {target}, variant {variant}, device {device}")
        wdir = self.ed_weights.text().strip() or None
        self._rn = _Run(exe, mode, data, out, target=target, device=device, variant=variant, weights=wdir, epochs=self.sp_epochs.value(), folds=self.sp_folds.value(),
                        init=(wdir or pretrained_weights_dir()) if (mode == "train" and self.cb_finetune.isChecked()) else None)
        self._rn.line.connect(self.out.append)
        self._rn.done_ok.connect(lambda rc: (self.btn_pred.setEnabled(True), self.btn_train.setEnabled(True), self.out.append(f"finished (exit code {rc}) -> {out}")))
        self._rn.start()
