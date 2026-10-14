"""
Multi-flight dataset layout (PhenoFuse 4.0).

    <dataset>/
      dataset.json                      manifest: flights (trial, stage, date), modalities present
      lidar/<flight>/plots/             per-plot LAS files of the flight (classified when LiDAR v3 was on)
      lidar/<flight>/lidar_features.csv
      lidar/<flight>/ground_truth.csv   (optional) Plot_ID + biomass / height columns
      vnir/<flight>/plots/              per-plot VNIR cubes (GeoTIFF, all bands)
      vnir/<flight>/vnir_features.csv + vnir_spectra.csv + vnir_band_valid_fraction.csv + vnir_band_usable.csv + vnir_provenance.json
      vnir/<flight>/ground_truth.csv
      rgb/<flight>/plots/               per-plot RGB crops
      rgb/<flight>/rgb_features.csv
      rgb/<flight>/ground_truth.csv
      ground_truth/<flight>.csv         (optional) one ground-truth table per flight shared by every modality
      Models/Height models/, Models/Biomass models/   training runs on this dataset

A user adds the current flight (any subset of the modalities) with add_flight(); the modelling tabs load a
training table with load_training_table(): one row per (flight, plot), the feature columns of the selected
modalities merged on Plot_ID, a ground-truth column joined from the flight's ground_truth.csv, and a group id
"<trial>:<Plot_ID>" so that the same plot measured on several dates stays in one cross-validation fold.
"""
from __future__ import annotations
import os
import re
import json
import shutil
import numpy as np

def _isnum(series):
    from pandas.api.types import is_numeric_dtype
    return bool(is_numeric_dtype(series))


MODALITIES = ("lidar", "vnir", "rgb")
FEATURE_FILE = {"lidar": "lidar_features.csv", "vnir": "vnir_features.csv", "rgb": "rgb_features.csv"}
SIDE_FILES = {"vnir": ["vnir_spectra.csv", "vnir_band_valid_fraction.csv", "vnir_band_usable.csv", "vnir_provenance.json", "vnir_indices.csv"],
              "lidar": ["lidar_features_ground_model.json", "lidar_features_targets.csv", "lidar_features_target_calibration.json"], "rgb": ["plots_rgb_index.csv"]}
GT_FILE = "ground_truth.csv"
MANIFEST = "dataset.json"
MODEL_DIRS = {"height": os.path.join("Models", "Height models"), "biomass": os.path.join("Models", "Biomass models")}


def read_csv(path, **kw):
    """pandas.read_csv that tolerates an Excel UTF-8 BOM and stray spaces in the header (Plot_ID stays Plot_ID)."""
    import pandas as pd
    t = pd.read_csv(path, encoding="utf-8-sig", **kw)
    t.columns = [str(c).strip().lstrip("\ufeff") for c in t.columns]
    return t


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "flight"


def read_manifest(dataset_dir: str) -> dict:
    p = os.path.join(dataset_dir, MANIFEST)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"flights": {}}


def write_manifest(dataset_dir: str, man: dict):
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, MANIFEST), "w") as f:
        json.dump(man, f, indent=1, default=str)


def _copytree_plots(src_dir, dst_dir, pattern=r"^plot_", progress_cb=None, label=""):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(src_dir)) if re.match(pattern, f)] if (src_dir and os.path.isdir(src_dir)) else []
    for i, f in enumerate(files):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        if progress_cb and (i % 10 == 0 or i == len(files) - 1):
            progress_cb(int(100 * (i + 1) / max(len(files), 1)), f"{label}: copying plot files {i + 1}/{len(files)}")
    return len(files)


def add_flight(dataset_dir, flight, modality, features_csv, plots_dir=None, gt_csv=None, trial="", stage="", date="",
               copy_plots=True, progress_cb=None, extra_meta=None):
    """Copy one modality of one flight into the dataset. Returns the flight folder."""
    import pandas as pd
    flight = safe_name(flight); modality = modality.lower()
    if modality not in MODALITIES:
        raise ValueError(modality)
    if not features_csv or not os.path.exists(features_csv):
        raise FileNotFoundError(f"{modality} features table not found: {features_csv}")
    fdir = os.path.join(dataset_dir, modality, flight); os.makedirs(fdir, exist_ok=True)
    shutil.copy2(features_csv, os.path.join(fdir, FEATURE_FILE[modality]))
    src_dir = os.path.dirname(features_csv)
    for side in SIDE_FILES.get(modality, []):
        for cand in (os.path.join(src_dir, side), os.path.join(src_dir, side.replace("vnir_", "plot_metrics_vnir_v3_").replace("lidar_features", "plot_metrics"))):
            if os.path.exists(cand):
                shutil.copy2(cand, os.path.join(fdir, side)); break
    n_plots_files = 0
    if copy_plots and plots_dir and os.path.isdir(plots_dir):
        n_plots_files = _copytree_plots(plots_dir, os.path.join(fdir, "plots"), progress_cb=progress_cb, label=f"{modality}/{flight}")
    n_gt = 0
    if gt_csv and os.path.exists(gt_csv):
        gt = read_csv(gt_csv)
        if "Plot_ID" not in gt.columns:
            raise ValueError("ground-truth CSV needs a Plot_ID column")
        gt.to_csv(os.path.join(fdir, GT_FILE), index=False); n_gt = int(len(gt))
        gdir = os.path.join(dataset_dir, "ground_truth"); os.makedirs(gdir, exist_ok=True)
        gt.to_csv(os.path.join(gdir, f"{flight}.csv"), index=False)
    n_feat = int(len(read_csv(features_csv)))
    man = read_manifest(dataset_dir); fl = man.setdefault("flights", {}).setdefault(flight, {})
    fl.update({k: v for k, v in dict(trial=trial, stage=stage, date=date).items() if v})
    fl.setdefault("modalities", {})[modality] = dict(features=FEATURE_FILE[modality], n_plots=n_feat, plot_files=n_plots_files,
                                                     source=os.path.abspath(features_csv), ground_truth_rows=n_gt)
    if extra_meta:
        fl.update(extra_meta)
    write_manifest(dataset_dir, man)
    return fdir


def gt_path(dataset_dir, flight):
    """First ground-truth table found for a flight (shared folder, then any modality folder)."""
    cands = [os.path.join(dataset_dir, "ground_truth", f"{flight}.csv")] + [os.path.join(dataset_dir, m, flight, GT_FILE) for m in MODALITIES]
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def scan(dataset_dir):
    """DataFrame: flight, trial, stage, date, lidar, vnir, rgb (plot counts or 0), gt_rows, gt_columns."""
    import pandas as pd
    man = read_manifest(dataset_dir); flights = set(man.get("flights", {}))
    for m in MODALITIES:
        d = os.path.join(dataset_dir, m)
        if os.path.isdir(d):
            flights |= {f for f in os.listdir(d) if os.path.exists(os.path.join(d, f, FEATURE_FILE[m]))}
    rows = []
    for fl in sorted(flights):
        meta = man.get("flights", {}).get(fl, {})
        row = dict(flight=fl, trial=meta.get("trial", ""), stage=meta.get("stage", ""), date=meta.get("date", ""))
        for m in MODALITIES:
            p = os.path.join(dataset_dir, m, fl, FEATURE_FILE[m])
            n = 0
            if os.path.exists(p):
                try:
                    n = int(len(read_csv(p, usecols=["Plot_ID"])))
                except Exception:
                    n = -1
            row[m] = n
            pdir = os.path.join(dataset_dir, m, fl, "plots")
            row[f"{m}_files"] = len(os.listdir(pdir)) if os.path.isdir(pdir) else 0
        g = gt_path(dataset_dir, fl)
        if g:
            try:
                gt = read_csv(g); row["gt_rows"] = int(len(gt)); row["gt_columns"] = ", ".join(c for c in gt.columns if c != "Plot_ID")
            except Exception:
                row["gt_rows"] = -1; row["gt_columns"] = "unreadable"
        else:
            row["gt_rows"] = 0; row["gt_columns"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def uid(flight, pid):
    return f"{flight}:{pid}"


def load_training_table(dataset_dir, modalities, flights=None, require_all=True):
    """Pooled table over flights.

    Returns dict(df, gt, vnir_cols, rgb_cols, spectra, valid_fraction, groups, flights, excluded)
      df       : Plot_ID = "<flight>:<plot>" (string), Plot_ID_orig, flight, trial, stage + feature columns
      gt       : Plot_ID (uid) + every ground-truth column found
      spectra  : vnir_spectra tables concatenated (Plot_ID = uid) or None
      groups   : "<trial>:<plot>" per row of df (same plot on several dates -> one CV group)
    """
    import pandas as pd
    from .paths import read_excluded
    modalities = [m.lower() for m in modalities]
    man = read_manifest(dataset_dir); allf = scan(dataset_dir)
    use = [f for f in allf.flight if (flights is None or f in set(flights))]
    dfs, gts, specs, vfs = [], [], [], []
    vnir_cols, rgb_cols, excluded = [], [], None
    for fl in use:
        meta = man.get("flights", {}).get(fl, {}); trial = meta.get("trial") or fl; stage = meta.get("stage", "")
        merged = None; ok = True
        for m in modalities:
            p = os.path.join(dataset_dir, m, fl, FEATURE_FILE[m])
            if not os.path.exists(p):
                ok = False
                if require_all:
                    break
                continue
            t = read_csv(p)
            if m == "vnir":
                vnir_cols = sorted(set(vnir_cols) | {c for c in t.columns if c != "Plot_ID" and _isnum(t[c])})
                if excluded is None:
                    excluded = read_excluded(p)
                sp = os.path.join(dataset_dir, m, fl, "vnir_spectra.csv")
                if os.path.exists(sp):
                    s = read_csv(sp); s["Plot_ID"] = [uid(fl, v) for v in s["Plot_ID"]]; specs.append(s)
                vf = os.path.join(dataset_dir, m, fl, "vnir_band_valid_fraction.csv")
                if os.path.exists(vf):
                    v = read_csv(vf); v["Plot_ID"] = [uid(fl, x) for x in v["Plot_ID"]]; vfs.append(v)
            if m == "rgb":
                rgb_cols = sorted(set(rgb_cols) | {c for c in t.columns if c.startswith("rgb_")})
            if merged is None:
                merged = t
            else:
                t = t[[c for c in t.columns if c == "Plot_ID" or c not in merged.columns]]
                merged = merged.merge(t, on="Plot_ID", how="inner")
        if merged is None or (require_all and not ok):
            continue
        merged = merged.copy()
        merged.insert(0, "flight", fl); merged.insert(1, "trial", trial); merged.insert(2, "stage", stage)
        merged.insert(3, "Plot_ID_orig", merged["Plot_ID"].values)
        merged["Plot_ID"] = [uid(fl, v) for v in merged["Plot_ID_orig"]]
        dfs.append(merged)
        g = gt_path(dataset_dir, fl)
        if g:
            gt = read_csv(g); gt["Plot_ID"] = [uid(fl, v) for v in gt["Plot_ID"]]; gt.insert(1, "flight", fl); gts.append(gt)
    if not dfs:
        raise RuntimeError("no flight in the dataset has the features of every selected modality (" + ", ".join(modalities) + ")")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    gt = pd.concat(gts, ignore_index=True, sort=False) if gts else pd.DataFrame({"Plot_ID": [], "flight": []})
    spectra = pd.concat(specs, ignore_index=True, sort=False) if specs else None
    valid_fraction = pd.concat(vfs, ignore_index=True, sort=False) if vfs else None
    groups = np.array([f"{t}:{p}" for t, p in zip(df["trial"], df["Plot_ID_orig"])])
    return dict(df=df, gt=gt, vnir_cols=vnir_cols, rgb_cols=rgb_cols, spectra=spectra, valid_fraction=valid_fraction,
                groups=groups, flights=[f for f in use if f in set(df.flight)], excluded=excluded)


def single_flight_table(lidar_csv=None, vnir_csv=None, rgb_csv=None, gt_csv=None, flight="flight"):
    """The same structure as load_training_table for the current project's flight (no copying)."""
    import pandas as pd
    from .paths import vnir_sidecar, existing, read_excluded
    merged = None; vnir_cols, rgb_cols = [], []; spectra = valid_fraction = None; excluded = None
    for m, p in (("lidar", lidar_csv), ("vnir", vnir_csv), ("rgb", rgb_csv)):
        if not p or not os.path.exists(p):
            continue
        t = read_csv(p)
        if m == "vnir":
            vnir_cols = [c for c in t.columns if c != "Plot_ID" and _isnum(t[c])]
            sp = existing(vnir_sidecar(p, "spectra")); spectra = read_csv(sp) if sp else None
            vf = existing(vnir_sidecar(p, "valid_fraction")); valid_fraction = read_csv(vf) if vf else None
            excluded = read_excluded(p)
        if m == "rgb":
            rgb_cols = [c for c in t.columns if c.startswith("rgb_")]
        if merged is None:
            merged = t
        else:
            t = t[[c for c in t.columns if c == "Plot_ID" or c not in merged.columns]]
            merged = merged.merge(t, on="Plot_ID", how="inner")
    if merged is None:
        raise RuntimeError("no feature table found for this flight - run the LiDAR / VNIR / RGB Features tabs first")
    gt = read_csv(gt_csv) if (gt_csv and os.path.exists(gt_csv)) else None
    groups = merged["Plot_ID"].astype(str).to_numpy()
    return dict(df=merged, gt=gt, vnir_cols=vnir_cols, rgb_cols=rgb_cols, spectra=spectra, valid_fraction=valid_fraction,
                groups=groups, flights=[flight], excluded=excluded)
