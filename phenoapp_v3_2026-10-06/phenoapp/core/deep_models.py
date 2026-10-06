"""
Deep models (two-stream network) from inside PhenoApp.

PhenoApp's own environment has no torch. The network runs in an external Python environment that has torch
(and spconv for the LiDAR stream); this module finds such an environment, probes it (CUDA / GPU / spconv),
prepares the per-plot tensors with PhenoApp's own LiDAR v3 preprocessing and VNIR cleaning, and launches
phenoapp/dl/two_stream.py as a subprocess for prediction with the bundled pretrained fold ensembles or for
training on a new trial. Device selection is automatic: CUDA when available, otherwise CPU (spconv's native
algorithm on CPU, same weights, slower).

Public API
----------
find_torch_pythons()                      candidate interpreters that import torch
probe_env(python_exe)                     dict(torch, cuda, gpu, spconv) or error
prepare_plot_tensors(mgr, cube, plots, out_dir, dataset_name, stage, gt=None, progress_cb=None) -> index DataFrame
run_two_stream(python_exe, mode, data_dir, out, target, device="auto", variant="two_stream", weights=None,
               epochs=60, folds=10, log_cb=None) -> return code
pretrained_weights_dir()                  bundled assets/dl_models
"""

from __future__ import annotations
import os
import sys
import json
import glob
import subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
NPIX, MAXPTS = 512, 24000


def _pkg_root():
    for base in ([getattr(sys, "_MEIPASS", None)] if hasattr(sys, "_MEIPASS") else []) + [os.path.dirname(_HERE)]:
        if base and os.path.isdir(os.path.join(base, "dl" if base.endswith("phenoapp") else "phenoapp")):
            return base if base.endswith("phenoapp") else os.path.join(base, "phenoapp")
    return os.path.dirname(_HERE)


def script_path() -> str:
    return os.path.join(_pkg_root(), "dl", "two_stream.py")


def pretrained_weights_dir() -> str:
    return os.path.join(_pkg_root(), "assets", "dl_models")


def find_torch_pythons():
    """Interpreters likely to have torch: the running one, conda envs, common install locations."""
    cands = [sys.executable]
    home = os.path.expanduser("~")
    for pat in (os.path.join(home, ".conda", "envs", "*", "python.exe"), os.path.join(home, "anaconda3", "envs", "*", "python.exe"), os.path.join(home, "miniconda3", "envs", "*", "python.exe"),
                r"C:\ProgramData\anaconda3\envs\*\python.exe", os.path.join(home, ".conda", "envs", "*", "bin", "python"), "/opt/conda/envs/*/bin/python"):
        cands += sorted(glob.glob(pat))
    seen, out = set(), []
    for c in cands:
        if c and os.path.exists(c) and c.lower() not in seen:
            seen.add(c.lower()); out.append(c)
    return out


def probe_env(python_exe: str, timeout=120) -> dict:
    try:
        r = subprocess.run([python_exe, script_path(), "probe"], capture_output=True, text=True, timeout=timeout)
        line = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
        d = json.loads(line[-1]) if line else {"error": (r.stderr or r.stdout)[-400:]}
    except Exception as e:
        d = {"error": str(e)}
    d["python"] = python_exe
    return d


def auto_select_env(progress_cb=None):
    """First interpreter with torch; prefers CUDA + spconv."""
    best = None
    for p in find_torch_pythons():
        if progress_cb: progress_cb(0, f"probing {p}")
        d = probe_env(p)
        if d.get("torch"):
            score = (2 if d.get("cuda") else 0) + (1 if d.get("spconv") else 0)
            if best is None or score > best[0]:
                best = (score, d)
            if score == 3: break
    return best[1] if best else None


# ---------------------------------------------------------------- tensors
def _otsu(v):
    v = v[np.isfinite(v)]
    if len(v) < 100: return 0.3
    hist, edges = np.histogram(v, bins=200, range=(-0.2, 1.0)); c = edges[:-1] + np.diff(edges) / 2
    w1 = np.cumsum(hist); w2 = w1[-1] - w1; m1 = np.cumsum(hist * c) / np.maximum(w1, 1); m2 = (np.cumsum((hist * c)[::-1])[::-1] / np.maximum(w2, 1))
    var = w1[:-1] * w2[:-1] * (m1[:-1] - m2[1:]) ** 2; return float(c[np.argmax(var)])


def prepare_plot_tensors(mgr, cube, plots, out_dir, dataset_name, stage, gt=None, progress_cb=None, region_mode="whole", band_width=0.5):
    """Write dl_data/<dataset_name>/plot_<id>.npz + dl_data/index.csv for every plot.

    mgr   : loaded LASManager (x, y, z, points with intensity / return_number)
    cube  : VNIRCube
    gt    : optional DataFrame Plot_ID [+ biomass_kg_ha] [+ height_cm]
    """
    import pandas as pd
    from shapely import contains_xy
    from shapely.ops import unary_union
    from .regions import plot_region
    from .lidar_features import preprocess_plot
    os.makedirs(os.path.join(out_dir, dataset_name), exist_ok=True)
    x, y, z = mgr.x, mgr.y, mgr.z
    I = mgr.intensity if getattr(mgr, "intensity", None) is not None else np.zeros(len(x))
    RN = mgr.return_number if getattr(mgr, "return_number", None) is not None else np.ones(len(x))
    plots = plots.reset_index(drop=True); all_buf = unary_union(list(plots.geometry.buffer(0.25)))
    src = cube._src; wl = np.asarray(cube.wavelengths, float)
    regions = [plot_region(g, mode=region_mode, band_width=band_width) for g in plots.geometry]
    lab = cube._labels(regions).ravel(); sel = np.flatnonzero(lab); lab_sel = lab[sel]
    S = np.zeros((len(sel), src.count), np.uint16)
    for b in range(1, src.count + 1):
        S[:, b - 1] = src.read(b).ravel()[sel]
        if progress_cb and b % 20 == 0: progress_cb(int(30 * b / src.count), f"tensors: VNIR band {b}/{src.count}")
    b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800)))
    r670 = S[:, b670].astype(np.float32); r800 = S[:, b800].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_all = np.where((r670 > 0) & (r800 > 0), (r800 - r670) / (r800 + r670), np.nan)
    t_raw = _otsu(ndvi_all); veg_t = float(np.clip(t_raw, 0.15, 0.50))
    if t_raw < 0.12 or float(np.nanmean(ndvi_all >= veg_t)) < 0.10: veg_t = -9.0
    rng = np.random.default_rng(0); rows = []; n = len(plots)
    gtm = gt.set_index("Plot_ID") if gt is not None else None
    for i, r in plots.iterrows():
        pid = int(r["Plot_ID"]); poly = r.geometry; ring = poly.buffer(1.0).difference(all_buf); bx = ring.bounds
        m = (x >= bx[0]) & (x <= bx[2]) & (y >= bx[1]) & (y <= bx[3]); xs, ys, zs, is_, rn = x[m], y[m], z[m], I[m], RN[m]
        ip = contains_xy(poly, xs, ys); ir = contains_xy(ring, xs, ys)
        if ip.sum() < 100: continue
        pre = preprocess_plot(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, is_[ir])
        keep = pre["keep"]; px, py, h = xs[ip][keep], ys[ip][keep], pre["h"]; pi = is_[ip][keep] / pre["inorm"]; prn = rn[ip][keep]
        mrr = poly.minimum_rotated_rectangle; co = np.array(mrr.exterior.coords)[:-1]; e = np.diff(np.vstack([co, co[:1]]), axis=0); L = np.hypot(e[:, 0], e[:, 1]); ax_ = e[np.argmax(L)] / L.max()
        cxp, cyp = poly.centroid.x, poly.centroid.y; u = (px - cxp) * ax_[0] + (py - cyp) * ax_[1]; v = -(px - cxp) * ax_[1] + (py - cyp) * ax_[0]
        if len(h) > MAXPTS:
            k = rng.choice(len(h), MAXPTS, replace=False); u, v, h, pi, prn = u[k], v[k], h[k], pi[k], prn[k]
        pts = np.column_stack([u, v, h, pi, prn]).astype(np.float32)
        px_idx = np.flatnonzero(lab_sel == i + 1)
        if len(px_idx) < 50: continue
        spec = S[px_idx].astype(np.float32); rr = spec[:, b670]; nn_ = spec[:, b800]
        with np.errstate(invalid="ignore", divide="ignore"): ndvi = np.where((rr > 0) & (nn_ > 0), (nn_ - rr) / (nn_ + rr), -1)
        veg = ndvi >= veg_t; full = (spec[:, wl > 500] > 0).all(1)
        pref = np.flatnonzero(veg & full); pool = pref if len(pref) >= NPIX else (np.flatnonzero(veg) if veg.sum() >= NPIX else np.arange(len(spec)))
        kk = rng.choice(pool, NPIX, replace=len(pool) < NPIX); pix = S[px_idx][kk]
        vspec = spec[veg] if veg.sum() > 10 else spec
        mean_spec = np.where((vspec > 0).sum(0) > 0, np.nansum(np.where(vspec > 0, vspec, np.nan), 0) / np.maximum((vspec > 0).sum(0), 1), 0).astype(np.float32)
        bio = float(gtm.loc[pid, "biomass_kg_ha"]) if (gtm is not None and pid in gtm.index and "biomass_kg_ha" in gtm.columns) else np.nan
        hgt = float(gtm.loc[pid, "height_cm"]) if (gtm is not None and pid in gtm.index and "height_cm" in gtm.columns) else np.nan
        np.savez_compressed(os.path.join(out_dir, dataset_name, f"plot_{pid}.npz"), points=pts, pixels=pix, mean_spec=mean_spec, wavelengths=wl.astype(np.float32),
                            biomass_kg_ha=bio, height_cm=hgt, fcover_vnir=float(veg.mean()))
        rows.append(dict(dataset=dataset_name, stage=stage, Plot_ID=pid, Plot=r.get("Plot", pid), Variety=r.get("Variety", ""), biomass_kg_ha=bio, height_cm=hgt, n_points=len(pts),
                         n_pixels_region=len(px_idx), fcover_vnir=float(veg.mean()), file=f"{dataset_name}/plot_{pid}.npz"))
        if progress_cb and (i % 8 == 0 or i == n - 1): progress_cb(30 + int(70 * (i + 1) / n), f"tensors: plot {i + 1}/{n}")
    idx = pd.DataFrame(rows)
    ip_ = os.path.join(out_dir, "index.csv")
    if os.path.exists(ip_):
        old = pd.read_csv(ip_); old = old[old.dataset != dataset_name]; idx = pd.concat([old, idx], ignore_index=True)
    idx.to_csv(ip_, index=False)
    return idx


def run_two_stream(python_exe, mode, data_dir, out, target="biomass", device="auto", variant="two_stream", weights=None, epochs=60, folds=10, log_cb=None):
    cmd = [python_exe, "-u", script_path(), mode, "--data", data_dir, "--target", target, "--device", device, "--variant", variant, "--out", out]
    if mode == "predict":
        cmd += ["--weights", weights or pretrained_weights_dir()]
    else:
        cmd += ["--epochs", str(epochs), "--folds", str(folds)]
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    for line in p.stdout:
        if log_cb and line.strip() and "is_fx_tracing" not in line:
            log_cb(line.rstrip())
    return p.wait()
