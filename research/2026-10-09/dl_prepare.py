"""
Per-plot tensors for the two-stream network, built from Dataset_2026-10-09 only (no re-read of the whole clouds):
  LiDAR : classified per-plot LAS (class 7 noise excluded), HeightAboveGround, intensity / alley-ring median (sidecar
          JSON), return number; centred on the plot centroid and rotated so the plot's long axis is x; <= 24,000 points
  VNIR  : 512 pixel spectra (172 bands, uint16 reflectance x 10000) from the regenerated per-plot cube inside the
          sampling region (polygon inset 10 cm); vegetation pixels preferred (NDVI >= the flight threshold used by the
          features), pixels with every band > 500 nm valid preferred. For a cube that is not reflectance (AGT
          2025-11-13) the pixels are all zero, so the spectral stream carries no information for those plots.
  Target: biomass_kg_ha (+ height_cm for Muresk)
Output: <Models_2026-10-09>/two_stream/dl_data/<dataset>/plot_<Plot_ID>.npz and dl_data/index.csv
"""
import os, sys, re, json, glob, time, numpy as np, pandas as pd, laspy
from shapely import contains_xy
TOOL = r"C:\Users\appn\Downloads\DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06"; sys.path.insert(0, TOOL)
from phenoapp.core import plot_region
from grid_pyshp import load_grid                       # fiona / pyogrio are blocked on this machine: pyshp-based grid reader
from vnir_envi import read_plot_tif, _parse_hdr, _hdr  # rasterio is blocked too: tifffile reader for the per-plot cubes
B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; AGT = os.path.join(B, "2025_NUE_AGT_I_DPIRD")
ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); DS = os.path.join(ROOT, "dataset"); OUT = os.path.join(ROOT, "two_stream", "dl_data"); os.makedirs(OUT, exist_ok=True)
CRS = "EPSG:7850"; MAXPTS, NPIX = 24000, 512
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
CUBES = {"2025-09-30": os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin"), "2025-11-21": os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin"),
         "2025-09-22": os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin"), "2025-11-13_f1": os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin"),
         "2025-11-13_f2": os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin")}
GRIDS = {"2025-09-30": os.path.join(B, "2025-09-30", "aligned_grid_refit_20250930.shp"), "2025-11-21": os.path.join(B, "Grid_Refit_Comparison", "aligned_grid_refit.shp"),
         "2025-09-22": os.path.join(AGT, "2025-09-22", "grid_qa", "refit_auto3.shp"), "2025-11-13_f1": os.path.join(AGT, "2025-11-13", "flight1", "aligned_grid_refit_flight1.shp"),
         "2025-11-13_f2": os.path.join(AGT, "2025-11-13", "flight2", "aligned_grid_refit_flight2.shp")}
JOBS = {"muresk_anthesis": ("muresk", "2025-09-30", "anthesis"), "muresk_maturity": ("muresk", "2025-11-21", "maturity"),
        "agt_anthesis": ("agt", "2025-09-22", "anthesis"), "agt_maturity": ("agt", "2025-11-13_merged", "maturity")}
rng = np.random.default_rng(0); index = []
for name, (trial, key, stage) in JOBS.items():
    A = pd.read_csv(os.path.join(DS, trial, key, "features_all.csv"))
    A = A[(A.has_ground_truth == 1) & (A.valid_lidar == 1)].reset_index(drop=True)
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    polys = {}; vegT = {}; reflok = {}; busable = {}; wl_hdr = {}
    srcs = sorted(A.source_flight.unique()) if "source_flight" in A.columns else [key]
    for sk in srcs:
        g = load_grid(GRIDS[sk], target_crs=CRS); polys[sk] = dict(zip(g.Plot_ID.astype(int), g.geometry))
        wl_hdr[sk] = _parse_hdr(_hdr(CUBES[sk]))["wavelengths"]
        V = pd.read_csv(os.path.join(DS, trial, sk, "features_vnir.csv"))
        vegT[sk] = float(V.veg_ndvi_threshold.iloc[0]) if np.isfinite(V.veg_ndvi_threshold.iloc[0]) else -9.0
        reflok[sk] = bool(int(V.vnir_reflectance_ok.iloc[0]))
        # per plot, per band: 1 = usable (valid on >= 50 % of the pixels, not excluded, reflectance cube); the network must not see band 0
        BU = pd.read_csv(os.path.join(DS, trial, sk, "vnir_band_usable.csv")).set_index("Plot_ID")
        busable[sk] = BU[[c for c in BU.columns if c not in ("cube_reflectance_ok", "n_usable_bands")]].astype(np.int8)
    log(f"{name}: {len(A)} GT plots with valid LiDAR; source flights {srcs}; veg thresholds {vegT}; reflectance ok {reflok}")
    n_ok = 0
    for i, r in A.iterrows():
        pid = int(r.Plot_ID); sk = r.source_flight if "source_flight" in A.columns else key
        fdir = os.path.join(DS, trial, sk); poly = polys[sk][pid]
        lf = glob.glob(os.path.join(fdir, "lidar_classified", f"plot_{pid}_*_classified.las"))
        if not lf: continue
        las = laspy.read(lf[0]); meta = json.load(open(lf[0][:-4] + ".json"))
        cls = np.asarray(las.classification); keep = cls != 7
        px, py = np.asarray(las.x, float)[keep], np.asarray(las.y, float)[keep]; h = np.asarray(las.HeightAboveGround, np.float32)[keep]
        dims = set(las.point_format.dimension_names)
        pi = (np.asarray(las.intensity, np.float32)[keep] if "intensity" in dims else np.zeros(len(h), np.float32)) / max(float(meta.get("intensity_ring_median") or 1.0), 1.0)
        prn = np.asarray(las.return_number, np.float32)[keep] if "return_number" in dims else np.ones(len(h), np.float32)
        if len(h) < 100: continue
        mrr = poly.minimum_rotated_rectangle; co = np.array(mrr.exterior.coords)[:-1]; e = np.diff(np.vstack([co, co[:1]]), axis=0); L = np.hypot(e[:, 0], e[:, 1]); ax_ = e[np.argmax(L)] / L.max()
        cxp, cyp = poly.centroid.x, poly.centroid.y; u = (px - cxp) * ax_[0] + (py - cyp) * ax_[1]; v = -(px - cxp) * ax_[1] + (py - cyp) * ax_[0]
        if len(h) > MAXPTS:
            k = rng.choice(len(h), MAXPTS, replace=False); u, v, h, pi, prn = u[k], v[k], h[k], pi[k], prn[k]
        pts = np.column_stack([u, v, h, pi, prn]).astype(np.float32)
        # ---- VNIR pixels from the regenerated per-plot cube
        tf = glob.glob(os.path.join(fdir, "vnir", f"plot_{pid}_*.tif")); pix = np.zeros((NPIX, 172), np.uint16); fcover = np.nan; npx = 0; wl = None; vavail = 0
        if tf and reflok[sk]:
            cube, (tx0, ty0, tps), wl = read_plot_tif(tf[0])                               # (bands, H, W)
            if wl is None: wl = wl_hdr[sk]
            region = plot_region(poly, mode="whole", band_width=0.5)
            H_, W_ = cube.shape[1], cube.shape[2]
            XX, YY = np.meshgrid(tx0 + (np.arange(W_) + 0.5) * tps, ty0 - (np.arange(H_) + 0.5) * tps)
            inside = contains_xy(region, XX.ravel(), YY.ravel()).reshape(H_, W_)          # pixel-centre rule, as rasterize(all_touched=False)
            spec = cube[:, inside].T.astype(np.float32); npx = len(spec)                    # (npx, bands)
            if npx >= 50:
                b670 = int(np.argmin(np.abs(wl - 670))); b800 = int(np.argmin(np.abs(wl - 800)))
                r670 = spec[:, b670]; r800 = spec[:, b800]
                with np.errstate(invalid="ignore", divide="ignore"): ndvi = np.where((r670 > 0) & (r800 > 0), (r800 - r670) / (r800 + r670), -1)
                veg = ndvi >= vegT[sk]; full = (spec[:, wl > 500] > 0).all(1)
                pref = np.flatnonzero(veg & full); pool = pref if len(pref) >= NPIX else (np.flatnonzero(veg) if veg.sum() >= NPIX else np.arange(len(spec)))
                kk = rng.choice(pool, NPIX, replace=len(pool) < NPIX)
                pix = np.clip(np.rint(spec[kk]), 0, 65535).astype(np.uint16); fcover = float(veg.mean()); vavail = 1
        if wl is None: wl = np.linspace(399.14, 1000.12, 172)
        band_mask = busable[sk].loc[pid].to_numpy().astype(np.int8) if pid in busable[sk].index else np.zeros(172, np.int8)
        if vavail: pix = np.where(band_mask[None, :] == 1, pix, 0).astype(np.uint16)      # unusable bands are zeroed = masked in the network
        np.savez_compressed(os.path.join(OUT, name, f"plot_{pid}.npz"), points=pts, pixels=pix, band_mask=band_mask, wavelengths=wl.astype(np.float32),
                            biomass_kg_ha=float(r.biomass_kg_ha), height_cm=float(r.height_cm) if np.isfinite(r.height_cm) else np.nan,
                            fcover_vnir=fcover, vnir_available=vavail, vnir_reflectance_ok=int(reflok[sk]), n_usable_bands=int(band_mask.sum()))
        index.append(dict(dataset=name, trial=trial, stage=stage, Plot_ID=pid, Plot=r.Plot, Variety=r.Variety, biomass_kg_ha=float(r.biomass_kg_ha),
                          height_cm=float(r.height_cm) if np.isfinite(r.height_cm) else np.nan, n_points=len(pts), n_pixels_region=npx, fcover_vnir=fcover,
                          vnir_available=vavail, vnir_reflectance_ok=int(reflok[sk]), n_usable_bands=int(band_mask.sum()), source_flight=sk, file=f"{name}/plot_{pid}.npz"))
        n_ok += 1
        if i % 64 == 0: log(f"  {name} plot {i+1}/{len(A)}")
    log(f"{name}: {n_ok} samples written")
pd.DataFrame(index).to_csv(os.path.join(OUT, "index.csv"), index=False)
log(f"index: {len(index)} samples -> {os.path.join(OUT, 'index.csv')}")
