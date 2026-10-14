"""
Deep Models tab (PhenoFuse 4.0): two-stream network (sparse 3D CNN on the plot cloud + spectral transformer on VNIR
pixels) for plot biomass and plant height, in an external Python environment with torch (+ spconv).

Two sections:
  Use a pretrained ensemble   weights folder (bundled or any training run) + the plot tensors of ONE flight (prepared here from
                              this flight's LAS + VNIR cube + grid, or from a flight of a dataset) -> predictions CSV.
  Train from scratch          plot tensors of SEVERAL flights of a dataset (per-plot LAS files and VNIR cubes of the dataset folder
                              + ground_truth.csv), or of this flight; k-fold training, optional warm start from a weights folder;
                              outputs under Models/<Biomass models | Height models>/<run>_two_stream/.
"""
from __future__ import annotations
import os
import time

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel, QProgressBar, QFileDialog, QMessageBox, QLineEdit,
                             QTextEdit, QFormLayout, QComboBox, QSpinBox, QCheckBox, QTabWidget, QRadioButton, QListWidget, QAbstractItemView)

from phenoapp.core.project import state
from phenoapp.core import load_grid, LASManager, VNIRCube
from phenoapp.core.deep_models import (auto_select_env, probe_env, prepare_plot_tensors, prepare_tensors_from_dataset, run_two_stream,
                                       pretrained_weights_dir, find_torch_pythons)
from phenoapp.core.model_runs import models_root, KIND_DIR
from phenoapp.ui.common import file_row


class _Probe(QThread):
    done = pyqtSignal(object)
    def __init__(self, exe=None): super().__init__(); self.exe = exe
    def run(self):
        try: self.done.emit(probe_env(self.exe) if self.exe else auto_select_env())
        except Exception as e: self.done.emit({"error": str(e)})


class _PrepFlight(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)
    def __init__(self, las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width):
        super().__init__(); self._a = (las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width)
    def run(self):
        try:
            import pandas as pd
            las, vnir, grid, crs, out_dir, name, stage, gt_csv, region_mode, band_width = self._a
            self.progress.emit(1, "loading point cloud..."); mgr = LASManager(las, crs, use_smrf=False); mgr.load(progress_cb=lambda p, m: None)
            from phenoapp.core.dataset import read_csv
            cube = VNIRCube(vnir); plots = load_grid(grid, target_crs=crs); gt = read_csv(gt_csv) if gt_csv and os.path.exists(gt_csv) else None
            idx = prepare_plot_tensors(mgr, cube, plots, out_dir, name, stage, gt=gt, progress_cb=lambda p, m: self.progress.emit(p, m), region_mode=region_mode, band_width=band_width)
            cube.close(); del mgr
            self.done_ok.emit(f"{len(idx[idx.dataset == name])} plot tensors written to {os.path.join(out_dir, name)}\nindex -> {os.path.join(out_dir, 'index.csv')}")
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _PrepDataset(QThread):
    progress = pyqtSignal(int, str); done_ok = pyqtSignal(str); error = pyqtSignal(str)
    def __init__(self, dataset_dir, flights, out_dir, variant):
        super().__init__(); self._a = (dataset_dir, flights, out_dir, variant)
    def run(self):
        try:
            dataset_dir, flights, out_dir, variant = self._a
            idx = prepare_tensors_from_dataset(dataset_dir, flights, out_dir, progress_cb=lambda p, m: self.progress.emit(p, m), variant=variant)
            n = idx[idx.dataset.isin(flights)]
            txt = f"{len(n)} plot tensors for {len(flights)} flight(s) -> {out_dir}\nindex -> {os.path.join(out_dir, 'index.csv')}\n" + \
                  "\n".join(f"  {fl}: {int((n.dataset == fl).sum())} plots, {int(n[n.dataset == fl].biomass_kg_ha.notna().sum())} with biomass, {int(n[n.dataset == fl].height_cm.notna().sum())} with height" for fl in flights)
            if os.path.exists(os.path.join(out_dir, "tensor_errors.csv")):
                txt += f"\nsome plots failed - see {os.path.join(out_dir, 'tensor_errors.csv')}"
            self.done_ok.emit(txt)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class _Run(QThread):
    line = pyqtSignal(str); done_ok = pyqtSignal(int)
    def __init__(self, *a, **k): super().__init__(); self.a = a; self.k = k
    def run(self):
        try: rc = run_two_stream(*self.a, log_cb=lambda l: self.line.emit(l), **self.k)
        except Exception as e: self.line.emit(f"ERROR: {e}"); rc = -1
        self.done_ok.emit(rc)


class _TensorSource(QGroupBox):
    """Plot tensors: from this flight (LAS + cube + grid) or from flights of a dataset folder (per-plot files)."""
    def __init__(self, parent, multi):
        super().__init__("Plot tensors (LiDAR v3 preprocessing + cleaned VNIR pixels per plot)", parent)
        f = QFormLayout(self); self.multi = multi
        self.rb_flight = QRadioButton("This flight: project LAS + VNIR cube + grid"); self.rb_dataset = QRadioButton("Dataset folder: per-plot LAS files + VNIR cubes of the selected flight(s)")
        self.rb_flight.setChecked(True); f.addRow(self.rb_flight); f.addRow(self.rb_dataset)
        self.ed_name = QLineEdit(); self.cb_stage = QComboBox(); self.cb_stage.setEditable(True); self.cb_stage.addItems(["anthesis", "maturity"])
        row = QHBoxLayout(); row.addWidget(self.ed_name, stretch=2); row.addWidget(QLabel("stage:")); row.addWidget(self.cb_stage); f.addRow("Flight name:", row)
        self.ed_gt = QLineEdit(); self.ed_gt.setPlaceholderText("optional: Plot_ID, biomass_kg_ha and/or height_cm (needed for training / scoring)")
        f.addRow("Ground truth CSV:", file_row(self, self.ed_gt, "Ground truth"))
        self.ed_dataset = QLineEdit(); f.addRow("Dataset folder:", file_row(self, self.ed_dataset, "Dataset folder", directory=True, on_change=lambda p: self.refresh()))
        self.lst = QListWidget(); self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection if multi else QAbstractItemView.SingleSelection); self.lst.setMaximumHeight(100)
        row = QHBoxLayout(); row.addWidget(self.lst, stretch=1); b = QPushButton("All" if multi else "Rescan"); b.clicked.connect(self.lst.selectAll if multi else self.refresh); row.addWidget(b)
        f.addRow("Flights:" if multi else "Flight:", row)
        self.ed_out = QLineEdit(); f.addRow("Tensor folder (dl_data):", file_row(self, self.ed_out, "Tensor folder", directory=True))
        self.btn = QPushButton("Prepare plot tensors"); f.addRow("", self.btn)
        self.rb_flight.toggled.connect(self._mode); self._mode()

    def _mode(self):
        fl = self.rb_flight.isChecked()
        for w in (self.ed_name, self.cb_stage, self.ed_gt): w.setEnabled(fl)
        for w in (self.ed_dataset, self.lst): w.setEnabled(not fl)

    def sync(self):
        s = state()
        if not self.ed_name.text().strip() and s.flight_name: self.ed_name.setText(s.flight_name)
        if not self.ed_gt.text().strip() and s.gt_csv: self.ed_gt.setText(s.gt_csv)
        if not self.ed_dataset.text().strip() and s.dataset_dir: self.ed_dataset.setText(s.dataset_dir); self.refresh()
        if not self.ed_out.text().strip():
            base = self.ed_dataset.text().strip() if self.rb_dataset.isChecked() else s.proj_dir
            if base: self.ed_out.setText(os.path.join(base, "dl_data"))
        self.cb_stage.setCurrentText(s.stage or "anthesis")

    def refresh(self):
        from phenoapp.core.dataset import scan
        self.lst.clear(); d = self.ed_dataset.text().strip()
        if d and os.path.isdir(d):
            state().dataset_dir = d
            try:
                for _, r in scan(d).iterrows():
                    self.lst.addItem(f"{r.flight}   [lidar files {r.lidar_files}, vnir files {r.vnir_files}; GT: {r.gt_columns or 'none'}]")
            except Exception:
                pass
            if self.multi: self.lst.selectAll()
            elif self.lst.count(): self.lst.setCurrentRow(0)
            if not self.ed_out.text().strip() or self.ed_out.text().strip().endswith("dl_data") and not self.ed_out.text().strip().startswith(d):
                self.ed_out.setText(os.path.join(d, "dl_data"))

    def flights(self):
        return [self.lst.item(i).text().split("   [")[0] for i in range(self.lst.count()) if self.lst.item(i).isSelected()]

    def data_dir(self):
        s = state(); d = self.ed_out.text().strip()
        if d: return d
        base = self.ed_dataset.text().strip() if self.rb_dataset.isChecked() else s.proj_dir
        return os.path.join(base or os.path.expanduser("~"), "dl_data")


class DeepTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._env = None; self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.addWidget(QLabel("<h2>Deep models: two-stream network (sparse 3D CNN + spectral transformer)</h2>"))
        v.addWidget(QLabel("The network runs in an external Python environment with torch (and spconv); the GPU is used automatically when CUDA is available. "
                           "Use the bundled pretrained ensembles (wheat, anthesis and maturity flights) on a new flight, or train from scratch on the plots of "
                           "several flights of a dataset (fine-tuning from the bundled weights is the default for a few hundred plots)."))
        g = QGroupBox("Compute environment"); f = QFormLayout(g)
        row = QHBoxLayout(); self.cb_env = QComboBox(); self.cb_env.setEditable(True); self.cb_env.addItems(find_torch_pythons())
        b = QPushButton("Detect automatically"); b.clicked.connect(self._auto); b2 = QPushButton("Probe selected"); b2.clicked.connect(self._probe)
        row.addWidget(self.cb_env, stretch=3); row.addWidget(b); row.addWidget(b2); f.addRow("Python with torch:", row)
        self.cb_device = QComboBox(); self.cb_device.addItems(["auto (GPU if available, else CPU)", "cuda (GPU)", "cpu"]); f.addRow("Device:", self.cb_device)
        self.lbl_env = QLabel("not probed yet"); f.addRow("Status:", self.lbl_env)
        v.addWidget(g)
        self.tabs = QTabWidget(); v.addWidget(self.tabs, stretch=1)
        self.tabs.addTab(self._build_pretrained(), "Use a pretrained ensemble (new trial / flight)")
        self.tabs.addTab(self._build_train(), "Train from scratch (several flights)")
        self.progress = QProgressBar(); self.progress.setTextVisible(True); v.addWidget(self.progress)
        self.out = QTextEdit(); self.out.setReadOnly(True); self.out.setFontFamily("Consolas"); self.out.setMinimumHeight(180); v.addWidget(self.out, stretch=1)

    # ---------------------------------------------------------------- pretrained
    def _build_pretrained(self):
        w = QWidget(); v = QVBoxLayout(w)
        g = QGroupBox("Pretrained ensemble"); f = QFormLayout(g)
        self.ed_weights = QLineEdit(pretrained_weights_dir()); f.addRow("Weights folder:", file_row(w, self.ed_weights, "Weights folder", directory=True))
        self.ed_weights.setToolTip("Folder with <variant>_<target>_fold<k>.pt files (+ fold_scores.csv, common_bands.csv): the bundled assets/dl_models, "
                                   "a Models/<kind>/<run>_two_stream folder of a training run, or an unzipped weights release.")
        self.cb_target_p = QComboBox(); self.cb_target_p.addItems(["biomass", "height"]); self.cb_variant_p = QComboBox(); self.cb_variant_p.addItems(["two_stream", "lidar_only", "vnir_only"])
        self.cb_fsel = QComboBox(); self.cb_fsel.addItems(["all fold models (average of every fold model in the weights folder)", "best three folds (lowest test RMSE in fold_scores.csv) - optimistic scores"])
        row = QHBoxLayout(); row.addWidget(QLabel("target")); row.addWidget(self.cb_target_p); row.addWidget(QLabel("variant")); row.addWidget(self.cb_variant_p); row.addStretch(); f.addRow("Options:", row)
        f.addRow("Fold models:", self.cb_fsel)
        v.addWidget(g)
        self.src_p = _TensorSource(w, multi=False); self.src_p.btn.clicked.connect(lambda: self._prep(self.src_p)); v.addWidget(self.src_p)
        row = QHBoxLayout(); self.btn_pred = QPushButton("Predict with the pretrained ensemble"); self.btn_pred.clicked.connect(self._predict); row.addWidget(self.btn_pred); row.addStretch(); v.addLayout(row)
        v.addStretch()
        return w

    # ---------------------------------------------------------------- train
    def _build_train(self):
        w = QWidget(); v = QVBoxLayout(w)
        self.src_t = _TensorSource(w, multi=True); self.src_t.rb_dataset.setChecked(True); self.src_t.btn.clicked.connect(lambda: self._prep(self.src_t)); v.addWidget(self.src_t)
        g = QGroupBox("Training"); f = QFormLayout(g)
        self.cb_target_t = QComboBox(); self.cb_target_t.addItems(["biomass", "height"]); self.cb_variant_t = QComboBox(); self.cb_variant_t.addItems(["two_stream", "lidar_only", "vnir_only"])
        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(5, 500); self.sp_epochs.setValue(100); self.sp_folds = QSpinBox(); self.sp_folds.setRange(2, 20); self.sp_folds.setValue(10)
        row = QHBoxLayout()
        for lab, wd in (("target", self.cb_target_t), ("variant", self.cb_variant_t), ("epochs", self.sp_epochs), ("folds", self.sp_folds)):
            row.addWidget(QLabel(lab)); row.addWidget(wd)
        row.addStretch(); f.addRow("Options:", row)
        self.cb_finetune = QCheckBox("Fine-tune: warm start from a weights folder (bundled ensemble) instead of random weights"); self.cb_finetune.setChecked(True)
        self.ed_init = QLineEdit(pretrained_weights_dir()); f.addRow("", self.cb_finetune); f.addRow("Warm-start weights:", file_row(w, self.ed_init, "Weights folder", directory=True))
        self.cb_bands = QComboBox(); self.cb_bands.addItems(["common band list of the warm-start weights (common_bands.csv)", "per plot: every usable band + availability mask"])
        f.addRow("Band policy:", self.cb_bands)
        self.ed_run = QLineEdit(); self.ed_run.setPlaceholderText("optional run name"); f.addRow("Run name:", self.ed_run)
        self.lbl_out = QLabel(""); f.addRow("Models go to:", self.lbl_out)
        v.addWidget(g)
        row = QHBoxLayout(); self.btn_train = QPushButton("Train / cross-validate"); self.btn_train.clicked.connect(self._train); row.addWidget(self.btn_train); row.addStretch(); v.addLayout(row)
        v.addStretch()
        return w

    def showEvent(self, ev):
        super().showEvent(ev); self.src_p.sync(); self.src_t.sync(); self._update_out()

    def _update_out(self):
        s = state(); base = self.src_t.ed_dataset.text().strip() if self.src_t.rb_dataset.isChecked() else s.proj_dir
        self.lbl_out.setText(os.path.join(models_root(base), "...") if base else "(pick a dataset / load a project)")

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

    def _prep(self, src):
        s = state(); out = src.data_dir(); src.btn.setEnabled(False); self.progress.setValue(0)
        if src.rb_flight.isChecked():
            for nm, p in (("LAS", s.las_path), ("VNIR cube", s.vnir_path)):
                if not p or not os.path.exists(p): QMessageBox.warning(self, f"No {nm}", f"Set the {nm} on the Project / VNIR Spectral Features tab first."); src.btn.setEnabled(True); return
            grid = s.grid_file()
            if not grid or not os.path.exists(grid): QMessageBox.warning(self, "No grid", "Load / align a plot grid first."); src.btn.setEnabled(True); return
            name = src.ed_name.text().strip() or s.flight_name or "flight"; s.flight_name = name; s.stage = src.cb_stage.currentText()
            self._pp = _PrepFlight(s.las_path, s.vnir_path, grid, s.work_crs, out, name, src.cb_stage.currentText(), src.ed_gt.text().strip(), s.region_mode, s.band_width)
        else:
            d = src.ed_dataset.text().strip(); fl = src.flights()
            if not d or not fl: QMessageBox.warning(self, "No flights", "Pick a dataset folder and select the flight(s)."); src.btn.setEnabled(True); return
            variant = (self.cb_variant_t if src is self.src_t else self.cb_variant_p).currentText()
            self._pp = _PrepDataset(d, fl, out, variant)
        self._pp.progress.connect(lambda p, m: (self.progress.setValue(p), self.progress.setFormat(f"{p}% - {m}")))
        self._pp.done_ok.connect(lambda t: (src.btn.setEnabled(True), self.out.append(t)))
        self._pp.error.connect(lambda e: (src.btn.setEnabled(True), self.out.append("ERROR: " + e), QMessageBox.critical(self, "Failed", e.split("\n")[0])))
        self._pp.start()

    def _exe(self):
        exe = self.cb_env.currentText().strip()
        if not exe or not os.path.exists(exe):
            QMessageBox.warning(self, "No environment", "Pick or detect a Python environment with torch."); return None
        return exe

    def _predict(self):
        exe = self._exe()
        if not exe: return
        data = self.src_p.data_dir()
        if not os.path.exists(os.path.join(data, "index.csv")): QMessageBox.warning(self, "No tensors", "Prepare the plot tensors first."); return
        s = state(); device = ("auto", "cuda", "cpu")[self.cb_device.currentIndex()]; target = self.cb_target_p.currentText(); variant = self.cb_variant_p.currentText()
        wdir = self.ed_weights.text().strip() or pretrained_weights_dir()
        fl = (self.src_p.flights() or [self.src_p.ed_name.text().strip() or s.flight_name or "flight"])[0]
        out_dir = os.path.join(os.path.dirname(data), "Models", "predictions"); os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{fl}_{target}_{variant}_twostream_{os.path.basename(wdir.rstrip(os.sep)) or 'weights'}.csv")
        self.btn_pred.setEnabled(False); self.out.append(f"\n=== predict: target {target}, variant {variant}, device {device}, weights {wdir}")
        self._rn = _Run(exe, "predict", data, out, target=target, device=device, variant=variant, weights=wdir, folds_select=("all", "best3")[self.cb_fsel.currentIndex()])
        self._rn.line.connect(self.out.append)
        self._rn.done_ok.connect(lambda rc: (self.btn_pred.setEnabled(True), self.out.append(f"finished (exit code {rc}) -> {out}")))
        self._rn.start()

    def _train(self):
        exe = self._exe()
        if not exe: return
        data = self.src_t.data_dir()
        if not os.path.exists(os.path.join(data, "index.csv")): QMessageBox.warning(self, "No tensors", "Prepare the plot tensors of the training flights first."); return
        s = state(); device = ("auto", "cuda", "cpu")[self.cb_device.currentIndex()]; target = self.cb_target_t.currentText(); variant = self.cb_variant_t.currentText()
        base = self.src_t.ed_dataset.text().strip() if self.src_t.rb_dataset.isChecked() else (s.proj_dir or os.path.dirname(data))
        run = time.strftime("%Y-%m-%d_%H%M") + "_" + (self.ed_run.text().strip() or target) + f"_two_stream_{variant}"
        out = os.path.join(models_root(base), KIND_DIR[target], run); os.makedirs(out, exist_ok=True)
        init = (self.ed_init.text().strip() or pretrained_weights_dir()) if self.cb_finetune.isChecked() else None
        self.btn_train.setEnabled(False); self.out.append(f"\n=== train: target {target}, variant {variant}, device {device}, data {data}" + (f", warm start {init}" if init else ""))
        self._rn = _Run(exe, "train", data, out, target=target, device=device, variant=variant, epochs=self.sp_epochs.value(), folds=self.sp_folds.value(), init=init,
                        bands=("common", "per_plot")[self.cb_bands.currentIndex()])
        self._rn.line.connect(self.out.append)
        self._rn.done_ok.connect(lambda rc: (self.btn_train.setEnabled(True), self.out.append(f"finished (exit code {rc}) -> {out}")))
        self._rn.start()
