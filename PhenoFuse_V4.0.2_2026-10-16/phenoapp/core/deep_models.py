"""
Deep models (two-stream network) from inside PhenoFuse.

PhenoFuse's own environment has no torch. The network runs in an external Python environment that has torch
(and spconv for the LiDAR stream); this module finds such an environment, probes it (CUDA / GPU / spconv),
prepares the per-plot tensors and launches phenoapp/dl/two_stream.py as a subprocess for prediction with a
pretrained fold ensemble or for training.

Tensors can be prepared
  * from the current flight (LAS + VNIR cube + grid; prepare_plot_tensors) - the v3 route, one flight, or
  * from a DATASET folder (prepare_tensors_from_dataset): every selected flight's per-plot classified LAS files
    (lidar/<flight>/plots/) and per-plot VNIR cubes (vnir/<flight>/plots/) plus the flight's ground_truth.csv.
    Several flights go into one dl_data folder (index.csv, dataset column = flight name), which is what training
    on more than one flight needs.

Public API
----------
find_torch_pythons()                      candidate interpreters that import torch
probe_env(python_exe)                     dict(torch, cuda, gpu, spconv) or error
prepare_plot_tensors(mgr, cube, plots, out_dir, dataset_name, stage, gt=None, progress_cb=None) -> index DataFrame
prepare_tensors_from_dataset(dataset_dir, flights, out_dir, progress_cb=None, variant="two_stream") -> index DataFrame
run_two_stream(python_exe, mode, data_dir, out, target, device="auto", variant="two_stream", weights=None,
               epochs=60, folds=10, log_cb=None) -> return code
pretrained_weights_dir()                  bundled assets/dl_models
"""

from __future__ import annotations
import os
import re
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


def _site_packages(python_exe):
    d = os.path.dirname(python_exe)
    return [sp for sp in [os.path.join(d, "Lib", "site-packages"), os.path.join(d, "..", "lib", "site-packages")] +
            glob.glob(os.path.join(d, "..", "lib", "python3*", "site-packages")) if os.path.isdir(sp)]


def _has_pkg(python_exe, pkg):
    """Cheap check (no subprocess): the package folder exists in the interpreter's site-packages."""
    return any(os.path.isdir(os.path.join(sp, pkg)) for sp in _site_packages(python_exe))


def find_torch_pythons(only_with_torch=True):
    """Interpreters with torch (conda envs, common install locations, the running one when it is a real Python); the ones that
    also have spconv come first. The frozen PhenoFuse.exe is never offered. only_with_torch=False lists every interpreter found."""
    cands = [] if getattr(sys, "frozen", False) else [sys.executable]
    home = os.path.expanduser("~")
    for pat in (os.path.join(home, ".conda", "envs", "*", "python.exe"), os.path.join(home, "anaconda3", "envs", "*", "python.exe"), os.path.join(home, "miniconda3", "envs", "*", "python.exe"),
                r"C:\ProgramData\anaconda3\envs\*\python.exe", os.path.join(home, "anaconda3", "python.exe"), os.path.join(home, "miniconda3", "python.exe"),
                os.path.join(home, ".conda", "envs", "*", "bin", "python"), "/opt/conda/envs/*/bin/python"):
        cands += sorted(glob.glob(pat))
    seen, out = set(), []
    for c in cands:
        if c and os.path.exists(c) and c.lower() not in seen:
            seen.add(c.lower())
            if not only_with_torch or _has_pkg(c, "torch"):
                out.append(c)
    return sorted(out, key=lambda c: 0 if _has_pkg(c, "spconv") else 1)


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


def _plot_frame(px, py, poly=None):
    """(u, v) coordinates of the points along the plot's long axis, centred on the plot."""
    if poly is None:
        from shapely.geometry import MultiPoint
        k = np.random.default_rng(0).choice(len(px), min(len(px), 4000), replace=False)
        poly = MultiPoint(np.column_stack([px[k], py[k]])).convex_hull
    mrr = poly.minimum_rotated_rectangle; co = np.array(mrr.exterior.coords)[:-1]; e = np.diff(np.vstack([co, co[:1]]), axis=0)
    L = np.hypot(e[:, 0], e[:, 1]); ax_ = e[np.argmax(L)] / L.max()
    cxp, cyp = poly.centroid.x, poly.centroid.y
    u = (px - cxp) * ax_[0] + (py - cyp) * ax_[1]; v = -(px - cxp) * ax_[1] + (py - cyp) * ax_[0]
    return u, v


def _pack_points(u, v, h, pi, prn, rng):
    if len(h) > MAXPTS:
        k = rng.choice(len(h), MAXPTS, replace=False); u, v, h, pi, prn = u[k], v[k], h[k], pi[k], prn[k]
    return np.column_stack([u, v, h, pi, prn]).astype(np.float32)


def _pack_pixels(spec, wl, veg_t, rng, refl_ok):
    """spec: (n_px, n_bands) raw cube values of the plot's pixels -> (pixels, mean_spec, band_mask, veg fraction)."""
    b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800)))
    rr = spec[:, b670].astype(np.float32); nn_ = spec[:, b800].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"): ndvi = np.where((rr > 0) & (nn_ > 0), (nn_ - rr) / (nn_ + rr), -1)
    veg = ndvi >= veg_t; full = (spec[:, wl > 500] > 0).all(1)
    pref = np.flatnonzero(veg & full); pool = pref if len(pref) >= NPIX else (np.flatnonzero(veg) if veg.sum() >= NPIX else np.arange(len(spec)))
    kk = rng.choice(pool, NPIX, replace=len(pool) < NPIX); pix = spec[kk]
    vspec = spec[veg] if veg.sum() > 10 else spec
    mean_spec = np.where((vspec > 0).sum(0) > 0, np.nansum(np.where(vspec > 0, vspec, np.nan), 0) / np.maximum((vspec > 0).sum(0), 1), 0).astype(np.float32)
    band_mask = (((spec > 0).mean(0) >= 0.5) & refl_ok).astype(np.uint8)
    return pix, mean_spec, band_mask, float(veg.mean())


def _write_index(out_dir, idx, dataset_names):
    import pandas as pd
    ip_ = os.path.join(out_dir, "index.csv")
    if os.path.exists(ip_):
        old = pd.read_csv(ip_); old = old[~old.dataset.isin(dataset_names)]; idx = pd.concat([old, idx], ignore_index=True)
    idx.to_csv(ip_, index=False)
    return idx


def prepare_plot_tensors(mgr, cube, plots, out_dir, dataset_name, stage, gt=None, progress_cb=None, region_mode="whole", band_width=0.5):
    """Write dl_data/<dataset_name>/plot_<id>.npz + dl_data/index.csv for every plot of the CURRENT flight.

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
    S = np.zeros((len(sel), src.count), np.dtype(src.dtypes[0]))   # cube's own dtype (float cubes must not be truncated to uint16)
    for b in range(1, src.count + 1):
        S[:, b - 1] = src.read(b).ravel()[sel]
        if progress_cb and b % 20 == 0: progress_cb(int(30 * b / src.count), f"tensors: VNIR band {b}/{src.count}")
    b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800)))
    r670 = S[:, b670].astype(np.float32); r800 = S[:, b800].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_all = np.where((r670 > 0) & (r800 > 0), (r800 - r670) / (r800 + r670), np.nan)
    t_raw = _otsu(ndvi_all); veg_t = float(np.clip(t_raw, 0.15, 0.50))
    if t_raw < 0.12 or float(np.nanmean(ndvi_all >= veg_t)) < 0.10: veg_t = -9.0
    refl_ok = bool(np.isfinite(ndvi_all).any() and float(np.nanmedian(ndvi_all)) >= 0.05)
    rng = np.random.default_rng(0); rows = []; n = len(plots)
    gtm = gt.set_index("Plot_ID") if gt is not None else None
    for i, r in plots.iterrows():
        pid = int(r["Plot_ID"]); poly = r.geometry; ring = poly.buffer(1.0).difference(all_buf); bx = ring.bounds
        m = (x >= bx[0]) & (x <= bx[2]) & (y >= bx[1]) & (y <= bx[3]); xs, ys, zs, is_, rn = x[m], y[m], z[m], I[m], RN[m]
        ip = contains_xy(poly, xs, ys); ir = contains_xy(ring, xs, ys)
        if ip.sum() < 100: continue
        pre = preprocess_plot(xs[ip], ys[ip], zs[ip], xs[ir], ys[ir], zs[ir], poly, is_[ir])
        keep = pre["keep"]; px, py, h = xs[ip][keep], ys[ip][keep], pre["h"]; pi = is_[ip][keep] / pre["inorm"]; prn = rn[ip][keep]
        u, v = _plot_frame(px, py, poly)
        pts = _pack_points(u, v, h, pi, prn, rng)
        px_idx = np.flatnonzero(lab_sel == i + 1)
        if len(px_idx) < 50: continue
        pix, mean_spec, band_mask, fveg = _pack_pixels(S[px_idx].astype(np.float32), wl, veg_t, rng, refl_ok)
        bio = float(gtm.loc[pid, "biomass_kg_ha"]) if (gtm is not None and pid in gtm.index and "biomass_kg_ha" in gtm.columns) else np.nan
        hgt = float(gtm.loc[pid, "height_cm"]) if (gtm is not None and pid in gtm.index and "height_cm" in gtm.columns) else np.nan
        np.savez_compressed(os.path.join(out_dir, dataset_name, f"plot_{pid}.npz"), points=pts, pixels=pix, mean_spec=mean_spec, wavelengths=wl.astype(np.float32),
                            band_mask=band_mask, biomass_kg_ha=bio, height_cm=hgt, fcover_vnir=fveg, vnir_reflectance_ok=int(refl_ok), n_usable_bands=int(band_mask.sum()))
        rows.append(dict(dataset=dataset_name, stage=stage, Plot_ID=pid, Plot=r.get("Plot", pid), Variety=r.get("Variety", ""), biomass_kg_ha=bio, height_cm=hgt, n_points=len(pts),
                         n_pixels_region=len(px_idx), fcover_vnir=fveg, vnir_reflectance_ok=int(refl_ok), file=f"{dataset_name}/plot_{pid}.npz"))
        if progress_cb and (i % 8 == 0 or i == n - 1): progress_cb(30 + int(70 * (i + 1) / n), f"tensors: plot {i + 1}/{n}")
    return _write_index(out_dir, pd.DataFrame(rows), [dataset_name])


def _read_plot_las(path):
    """(x, y, h, intensity_norm, return_number, ground_rms) from a per-plot LAS; HeightAboveGround when present
    (classified files of the LiDAR Features tab), else a plot-local ground plane from the lowest returns."""
    import laspy
    las = laspy.read(path); dims = set(las.point_format.dimension_names)
    x = np.asarray(las.x, float); y = np.asarray(las.y, float); z = np.asarray(las.z, float)
    I = np.asarray(las.intensity, float) if "intensity" in dims else np.zeros(len(x))
    RN = np.asarray(las.return_number, float) if "return_number" in dims else np.ones(len(x))
    cls = np.asarray(las.classification) if "classification" in dims else None
    keep = np.ones(len(x), bool)
    if cls is not None and (cls == 7).any():
        keep = cls != 7
    if "HeightAboveGround" in dims:
        h = np.asarray(las["HeightAboveGround"], float)
    else:
        from .lidar_features import preprocess_plot
        pre = preprocess_plot(x, y, z, None, None, None, None, None); h = pre["h_all"]; keep &= pre["keep"]
    ground = (cls == 2) if cls is not None else (np.abs(h) <= 0.08)
    inorm = float(np.median(I[ground & keep])) if (ground & keep).sum() > 50 and np.median(I[ground & keep]) > 0 else (float(np.median(I[keep])) if np.median(I[keep]) > 0 else 1.0)
    return x[keep], y[keep], h[keep], I[keep] / inorm, RN[keep]


def _read_plot_cube(path):
    """(pixels inside the mask (n, bands) raw values, wavelengths) of a per-plot VNIR GeoTIFF."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read()                                  # (bands, h, w)
        mask = src.read_masks(1) > 0
        desc = src.descriptions
        try:
            wl = np.array([float(str(d).split()[0]) for d in desc])
        except Exception:
            tag = src.tags().get("wavelengths_nm")
            wl = np.array([float(v) for v in tag.split(",")]) if tag else np.linspace(400, 1000, src.count)
    inside = mask & ~(arr == 0).all(0)
    return arr[:, inside].T, wl


def per_plot_files(folder, ext):
    """{plot id: path} of the per-plot files 'plot_<id>_*.<ext>' in a folder; a '_classified' LAS (LiDAR Features tab output,
    HeightAboveGround + classes) wins over a raw cut of the same plot."""
    out = {}
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        m = re.match(r"^plot_(\d+)[_.]", f)
        if m and f.lower().endswith(ext):
            pid = int(m.group(1))
            if pid not in out or "_classified" in f.lower():
                out[pid] = os.path.join(folder, f)
    return out


def flight_folder_files(folder):
    """(las_files, tif_files) of a flight folder written by the LiDAR / VNIR Spectral Features tabs:
    <flight>/plots_lidar/plot_<id>_*.las and <flight>/plots_vnir/plot_<id>_*.tif (the folder itself is accepted too)."""
    folder = folder or ""
    las = per_plot_files(os.path.join(folder, "plots_lidar"), ".las") or per_plot_files(folder, ".las")
    tif = per_plot_files(os.path.join(folder, "plots_vnir"), ".tif") or per_plot_files(folder, ".tif")
    return las, tif


def prepare_tensors_from_dataset(dataset_dir, flights, out_dir, progress_cb=None, variant="two_stream"):
    """Write dl_data tensors for every selected flight of a dataset folder (lidar/<flight>/plots/*.las and
    vnir/<flight>/plots/plot_<id>_*.tif + ground_truth.csv). variant 'lidar_only' / 'vnir_only' needs only that modality."""
    from .dataset import read_manifest, gt_path, read_csv
    man = read_manifest(dataset_dir); specs = []
    for fl in flights:
        meta = man.get("flights", {}).get(fl, {})
        ldir = os.path.join(dataset_dir, "lidar", fl, "plots"); vdir = os.path.join(dataset_dir, "vnir", fl, "plots")
        g = gt_path(dataset_dir, fl)
        specs.append(dict(name=fl, stage=meta.get("stage", ""), las_files=per_plot_files(ldir, ".las"), tif_files=per_plot_files(vdir, ".tif"),
                          gt=read_csv(g) if g else None, lidar_dir=ldir, vnir_dir=vdir))
    return prepare_tensors_from_files(specs, out_dir, progress_cb=progress_cb, variant=variant)


def prepare_tensors_from_flight_folder(folder, name, stage, out_dir, gt_csv=None, progress_cb=None, variant="two_stream"):
    """Write dl_data tensors of ONE flight from the per-plot files already extracted into its flight folder
    (plots_lidar/*.las of the LiDAR Features tab, plots_vnir/*.tif of the VNIR Spectral Features tab); no project LAS / cube needed."""
    from .dataset import read_csv
    las, tif = flight_folder_files(folder)
    gt = read_csv(gt_csv) if gt_csv and os.path.exists(gt_csv) else None
    spec = dict(name=name, stage=stage, las_files=las, tif_files=tif, gt=gt, lidar_dir=os.path.join(folder, "plots_lidar"), vnir_dir=os.path.join(folder, "plots_vnir"))
    return prepare_tensors_from_files([spec], out_dir, progress_cb=progress_cb, variant=variant)


def prepare_tensors_from_files(specs, out_dir, progress_cb=None, variant="two_stream"):
    """Shared tensor writer: specs = [{name, stage, las_files {pid: path}, tif_files {pid: path}, gt (DataFrame with Plot_ID or None)}].
    variant 'lidar_only' / 'vnir_only' needs only that modality."""
    import pandas as pd
    rng = np.random.default_rng(0); rows = []; flights = [sp["name"] for sp in specs]
    need_lidar = variant != "vnir_only"; need_vnir = variant != "lidar_only"
    for fi, sp in enumerate(specs):
        fl = sp["name"]; stage = sp.get("stage", "") or ""; las_files = sp["las_files"]; tif_files = sp["tif_files"]
        if need_lidar and not las_files:
            raise RuntimeError(f"{fl}: no per-plot LAS files (plot_<id>_*.las) in {sp.get('lidar_dir', '?')}")
        if need_vnir and not tif_files:
            raise RuntimeError(f"{fl}: no per-plot VNIR cubes (plot_<id>_*.tif) in {sp.get('vnir_dir', '?')}")
        pids = sorted((set(las_files) if need_lidar else set(tif_files)) & (set(tif_files) if need_vnir else set(las_files)) if (need_lidar and need_vnir)
                      else (set(las_files) if need_lidar else set(tif_files)))
        gt = sp.get("gt")
        gtm = None
        if gt is not None and "Plot_ID" in gt.columns:
            gtm = gt.copy(); gtm["Plot_ID"] = pd.to_numeric(gtm["Plot_ID"], errors="coerce"); gtm = gtm.dropna(subset=["Plot_ID"])
            gtm["Plot_ID"] = gtm["Plot_ID"].astype(int); gtm = gtm.drop_duplicates("Plot_ID").set_index("Plot_ID")
        os.makedirs(os.path.join(out_dir, fl), exist_ok=True)
        # flight-level vegetation threshold from a sample of plots
        veg_t = 0.3; refl_ok = True; wl = None
        if need_vnir:
            nd = []
            for pid in pids[:: max(1, len(pids) // 12)][:12]:
                sp, wl = _read_plot_cube(tif_files[pid]); b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800)))
                rr = sp[:, b670].astype(np.float32); nn_ = sp[:, b800].astype(np.float32)
                with np.errstate(invalid="ignore", divide="ignore"): nd.append(np.where((rr > 0) & (nn_ > 0), (nn_ - rr) / (nn_ + rr), np.nan))
            nd = np.concatenate(nd) if nd else np.array([np.nan])
            t_raw = _otsu(nd); veg_t = float(np.clip(t_raw, 0.15, 0.50))
            if t_raw < 0.12 or float(np.nanmean(nd >= veg_t)) < 0.10: veg_t = -9.0
            refl_ok = bool(np.isfinite(nd).any() and float(np.nanmedian(nd)) >= 0.05)
        for i, pid in enumerate(pids):
            try:
                if need_lidar:
                    px, py, h, pi, prn = _read_plot_las(las_files[pid])
                    if len(h) < 100: continue
                    u, v = _plot_frame(px, py); pts = _pack_points(u, v, h, pi, prn, rng)
                else:
                    pts = np.zeros((0, 5), np.float32)
                if need_vnir:
                    sp, wl = _read_plot_cube(tif_files[pid])
                    if len(sp) < 50: continue
                    pix, mean_spec, band_mask, fveg = _pack_pixels(sp.astype(np.float32), wl, veg_t, rng, refl_ok)
                else:
                    wl = np.linspace(400, 1000, 172) if wl is None else wl
                    pix = np.zeros((NPIX, len(wl)), np.float32); mean_spec = np.zeros(len(wl), np.float32); band_mask = np.zeros(len(wl), np.uint8); fveg = np.nan
                bio = float(gtm.loc[pid, "biomass_kg_ha"]) if (gtm is not None and pid in gtm.index and "biomass_kg_ha" in gtm.columns) else np.nan
                hgt = float(gtm.loc[pid, "height_cm"]) if (gtm is not None and pid in gtm.index and "height_cm" in gtm.columns) else np.nan
                np.savez_compressed(os.path.join(out_dir, fl, f"plot_{pid}.npz"), points=pts, pixels=pix, mean_spec=mean_spec, wavelengths=np.asarray(wl, np.float32),
                                    band_mask=band_mask, biomass_kg_ha=bio, height_cm=hgt, fcover_vnir=fveg, vnir_reflectance_ok=int(refl_ok), n_usable_bands=int(band_mask.sum()))
                rows.append(dict(dataset=fl, stage=stage, Plot_ID=pid, Plot=pid, Variety="", biomass_kg_ha=bio, height_cm=hgt, n_points=len(pts),
                                 n_pixels_region=len(sp) if need_vnir else 0, fcover_vnir=fveg, vnir_reflectance_ok=int(refl_ok), file=f"{fl}/plot_{pid}.npz"))
            except Exception as e:
                rows.append(dict(dataset=fl, stage=stage, Plot_ID=pid, error=str(e)[:120]))
            if progress_cb and (i % 5 == 0 or i == len(pids) - 1):
                progress_cb(int(100 * (fi + (i + 1) / max(len(pids), 1)) / len(flights)), f"tensors {fl}: plot {i + 1}/{len(pids)}")
    idx = pd.DataFrame(rows)
    if "error" in idx.columns:
        bad = idx[idx.error.notna()]; idx = idx[idx.error.isna()].drop(columns=["error"])
        if len(bad):
            bad.to_csv(os.path.join(out_dir, "tensor_errors.csv"), index=False)
    return _write_index(out_dir, idx, list(flights))


def run_two_stream(python_exe, mode, data_dir, out, target="biomass", device="auto", variant="two_stream", weights=None, epochs=100, folds=10, log_cb=None,
                   init=None, bands="common", folds_select="all", excluded=None):
    """init: folder of fold models to warm-start from (fine-tuning; e.g. pretrained_weights_dir()); bands: 'common' | 'per_plot' for training;
    folds_select (predict): 'all' | 'best3' | comma list of fold indices; excluded: list of (lo, hi) nm ranges for the per-plot band policy."""
    cmd = [python_exe, "-u", script_path(), mode, "--data", data_dir, "--target", target, "--device", device, "--variant", variant, "--out", out]
    if mode == "predict":
        cmd += ["--weights", weights or pretrained_weights_dir(), "--folds-select", folds_select or "all"]
    else:
        cmd += ["--epochs", str(epochs), "--folds", str(folds), "--bands", bands]
        if init:
            cmd += ["--init", init]
    if excluded is not None:
        cmd += ["--excluded", ",".join(f"{lo:g}-{hi:g}" for lo, hi in excluded)]
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    for line in p.stdout:
        if log_cb and line.strip() and "is_fx_tracing" not in line:
            log_cb(line.rstrip())
    return p.wait()
