"""
Add the four oxygen-A bands (758-768 nm) to the Muresk per-plot mean vegetation spectra.
The tool writes NaN for those bands (excluded range); here they are recomputed from the per-plot cubes with the same
vegetation mask (NDVI >= the plot's veg_ndvi_threshold from features_vnir.csv, pixels inside the polygon), so that a
feature model can use every band except the first bands (0-415 nm) and the water-vapour tail (928-962 nm).
Output: dataset\muresk\<flight>\vnir_spectra_o2a.csv (copy of vnir_spectra.csv with the four columns filled).
"""
import os, glob, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vnir_envi import read_plot_tif
D = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\dataset\muresk"
for fl in ("2025-09-30", "2025-11-21"):
    S = pd.read_csv(os.path.join(D, fl, "vnir_spectra.csv")); V = pd.read_csv(os.path.join(D, fl, "features_vnir.csv")).set_index("Plot_ID")
    wcols = [c for c in S.columns if c != "Plot_ID"]; wl = np.array([float(c) for c in wcols]); o2a = [c for c, w in zip(wcols, wl) if 755 <= w <= 770]
    b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800))); io2 = [wcols.index(c) for c in o2a]
    files = {int(os.path.basename(p).split("_")[1]): p for p in glob.glob(os.path.join(D, fl, "vnir", "plot_*.tif"))}
    S = S.set_index("Plot_ID"); n = 0
    for pid in S.index:
        if pid not in files: continue
        a, _, _ = read_plot_tif(files[pid]); a = a.astype(np.float32)
        inside = a[b800] > 0
        with np.errstate(invalid="ignore", divide="ignore"): ndvi = np.where(inside & (a[b670] > 0), (a[b800] - a[b670]) / (a[b800] + a[b670]), -1)
        thr = float(V.loc[pid, "veg_ndvi_threshold"]) if pid in V.index and np.isfinite(V.loc[pid, "veg_ndvi_threshold"]) else 0.3
        veg = inside & (ndvi >= thr)
        if veg.sum() < 10: veg = inside
        for c, i in zip(o2a, io2):
            v = a[i][veg]; v = v[v > 0]
            S.loc[pid, c] = float(v.mean()) if len(v) else np.nan
        n += 1
    S.reset_index().to_csv(os.path.join(D, fl, "vnir_spectra_o2a.csv"), index=False)
    print(fl, "plots updated:", n, "| O2A bands:", o2a, "| NaN share now:", S[o2a].isna().mean().round(3).tolist(), "| mean values:", S[o2a].mean().round(1).tolist())
