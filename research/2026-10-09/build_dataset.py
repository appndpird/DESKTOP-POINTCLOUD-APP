"""
Build the clean, combined per-plot dataset for both modalities (Dataset_2026-10-09), one folder per trial / flight:

  ground_truth.csv              design factors + biomass (kg/ha dry) [+ ruler height cm, Muresk] per plot
  features_lidar.csv            PhenoApp v3.2 LiDAR pipeline (gap noise rule, plot-local ground, canopy classification,
                                canopy-only metrics, interior gap metrics, canopy-top heights, intensity) - all plots
  lidar_classified/plot_*.las   per-plot LAS with classification 2 ground / 3 low veg / 5 canopy / 7 noise (withheld
                                bit), HeightAboveGround extra dimension, + sidecar JSON (ground plane, counts)
  vnir/plot_*.tif               per-plot VNIR cubes regenerated with the v3.0.1 writer (every band, exact copy of the
                                orthomosaic inside the polygon, internal mask outside) + plots_vnir_index.csv
  features_vnir.csv             PhenoApp v3.0.1 cleaned VNIR features: NaN where a band/index is unusable (clipped on
                                >= 50 % of the pixels, excluded range, or non-reflectance cube), reflectance flag, red_ok
  vnir_spectra.csv              vegetation mean spectrum per plot (NaN = band unusable)
  vnir_band_valid_fraction.csv  fraction of region pixels valid per band
  vnir_band_usable.csv          1 = band usable, 0 = NaN in the spectra
  features_all.csv              ground truth + LiDAR + VNIR + QC flags + valid_* flags (what the models read)

Run in the phenoapp venv:  python build_dataset.py [flight_key ...]
"""
import os, sys, json, time, gc, traceback, shutil, numpy as np, pandas as pd, laspy
TOOL = r"C:\Users\appn\Downloads\DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06"
sys.path.insert(0, TOOL)
from phenoapp.core import plot_region
from grid_pyshp import load_grid                       # fiona / pyogrio are blocked on this machine: pyshp-based grid reader
from vnir_envi import ENVICube as VNIRCube             # rasterio / GDAL DLLs are blocked too: numpy memmap reader for the ENVI cubes
from phenoapp.core.lidar_features import lidar_features_all, LIDAR_V3_CORE, NOISE_RULES
from phenoapp.core.vnir_features import vnir_features_all, VNIR_V3_CORE
from phenoapp.core.spectral_indices import band_usable, MIN_VALID_FRAC, DEFAULT_EXCLUDED_NM

B = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment"; AGT = os.path.join(B, "2025_NUE_AGT_I_DPIRD"); DS = os.path.join(B, "Biomass Dataset")
ROOT = os.path.join(B, "Biomass_Height_2026-10-09"); OUT = os.path.join(ROOT, "dataset"); CRS = "EPSG:7850"; NOISE_RULE = "gap"
VERIFIED_CUBES = {"2025-09-30": os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-09-30", "vnir"), "2025-11-21": os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-11-21", "vnir"),
                  "2025-09-22": os.path.join(AGT, "2025-09-22", "dataset", "vnir"), "2025-11-13_f1": os.path.join(AGT, "2025-11-13", "flight1", "dataset", "vnir"),
                  "2025-11-13_f2": os.path.join(AGT, "2025-11-13", "flight2", "dataset", "vnir")}
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "build_dataset.log"), "a")
t0 = time.time()
def log(*a):
    s = f"[{time.time()-t0:6.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

FLIGHTS = [
    dict(trial="muresk", key="2025-09-30", stage="anthesis",
         las=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_LiDAR_CombinedPointCloud.las"),
         cube=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin"),
         grid=os.path.join(B, "2025-09-30", "aligned_grid_refit_20250930.shp")),
    dict(trial="muresk", key="2025-11-21", stage="maturity",
         las=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_LiDAR_CombinedPointCloud.las"),
         cube=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin"),
         grid=os.path.join(B, "Grid_Refit_Comparison", "aligned_grid_refit.shp")),
    dict(trial="agt", key="2025-09-22", stage="anthesis",
         las=os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_LiDAR_CombinedPointCloud.las"),
         cube=os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-09-22", "grid_qa", "refit_auto3.shp")),
    dict(trial="agt", key="2025-11-13_f1", stage="maturity",
         las=os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_LiDAR_CombinedPointCloud.las"),
         cube=os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-11-13", "flight1", "aligned_grid_refit_flight1.shp")),
    dict(trial="agt", key="2025-11-13_f2", stage="maturity",
         las=os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_LiDAR_CombinedPointCloud.las"),
         cube=os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-11-13", "flight2", "aligned_grid_refit_flight2.shp")),
]

# ---------------------------------------------------------------- ground truth
def ground_truth(fl):
    if fl["trial"] == "muresk":
        g = pd.read_excel(os.path.join(B, "Biomass_25NO43_AGT&DPIRD.xlsx"), sheet_name="DPIRD_NUE_2025")
        g.columns = ["Plot", "Range", "Row", "Block", "Rep", "N_treatment", "Variety", "N_available_kg_ha", "biomass_anthesis_kg_ha", "biomass_maturity_kg_ha", "height_cm"]
        g.insert(0, "Plot_ID", np.arange(1, 129))                   # sheet order = Plot_ID 1..128 (verified on the maturity data)
        g["biomass_kg_ha"] = g["biomass_anthesis_kg_ha"] if fl["stage"] == "anthesis" else g["biomass_maturity_kg_ha"]
        g["height_source"] = "ruler at maturity (one measurement per plot, used for both flights)"
        return g
    date = fl["key"].split("_")[0]
    g = pd.read_csv(os.path.join(DS, "2025_NUE_AGT_I_DPIRD", date, "ground_truth.csv"))
    g = g[[c for c in g.columns if not c.startswith(("lidar_", "vnir_"))]].copy()
    bcol = "biomass_anthesis_kg_ha" if fl["stage"] == "anthesis" else "biomass_maturity_kg_ha"
    g = g.rename(columns={"sheet_plot": "Plot"}); g["biomass_kg_ha"] = g[bcol]; g["height_cm"] = np.nan; g["height_source"] = ""
    return g

# ---------------------------------------------------------------- one flight
def run_flight(fl):
    fdir = os.path.join(OUT, fl["trial"], fl["key"]); os.makedirs(fdir, exist_ok=True)
    log(f"=== {fl['trial']} {fl['key']} ({fl['stage']}) -> {fdir}")
    plots = load_grid(fl["grid"], target_crs=CRS).reset_index(drop=True); plots["Plot_ID"] = plots["Plot_ID"].astype(int)
    gt = ground_truth(fl); gt["Plot_ID"] = gt["Plot_ID"].astype(int)
    gt_out = gt.copy(); gt_out["has_ground_truth"] = gt_out["biomass_kg_ha"].notna().astype(int)
    gt_out.to_csv(os.path.join(fdir, "ground_truth.csv"), index=False)
    log(f"  grid {len(plots)} plots; ground truth rows {len(gt)} ({int(gt.biomass_kg_ha.notna().sum())} with biomass)")

    # -------- LiDAR (v3.2 pipeline on the whole cloud) --------
    lidar_csv = os.path.join(fdir, "features_lidar.csv")
    if os.path.exists(lidar_csv) and "--redo-lidar" not in sys.argv:
        L = pd.read_csv(lidar_csv); log(f"  LiDAR features reused ({L.shape})")
    else:
        las = laspy.read(fl["las"])
        x = np.asarray(las.x, dtype=np.float64); y = np.asarray(las.y, dtype=np.float64); z = np.asarray(las.z, dtype=np.float64)
        dims = set(las.point_format.dimension_names)
        inten = np.asarray(las.intensity, dtype=np.float32) if "intensity" in dims else None
        rn = np.asarray(las.return_number) if "return_number" in dims else None
        nr = np.asarray(las.number_of_returns) if "number_of_returns" in dims else None
        log(f"  cloud {len(x)/1e6:.0f} M points, format {las.header.point_format.id}, intensity {inten is not None}")
        cdir = os.path.join(fdir, "lidar_classified")
        L = lidar_features_all(x, y, z, inten, rn, nr, plots, write_classified_dir=cdir, las_header=las.header, points=las.points,
                               noise_rule=NOISE_RULE, drop_noise=False,
                               progress_cb=lambda p, m: (log(f"    lidar {p}% {m}") if p % 20 == 0 else None))
        del las, x, y, z, inten, rn, nr; gc.collect()
        L["Plot_ID"] = L["Plot_ID"].astype(int)
        L.to_csv(lidar_csv, index=False)
        log(f"  LiDAR features {L.shape}; qc_ok {int(L.qc_ok.sum())}/{len(L)}; classified LAS files {len([f for f in os.listdir(cdir) if f.endswith('.las')])}")

    # -------- VNIR (v3.0.1: per-plot cubes regenerated + cleaned features with the NaN rule) --------
    cube = VNIRCube(fl["cube"]); wl = np.asarray(cube.wavelengths, float)
    regions = [plot_region(g, mode="whole", band_width=0.5) for g in plots.geometry]
    vfe = os.path.join(fdir, "features_vnir.csv")
    if os.path.exists(vfe) and "--redo-vnir-features" not in sys.argv:
        V = pd.read_csv(vfe); SP = pd.read_csv(os.path.join(fdir, "vnir_spectra.csv")); VF = pd.read_csv(os.path.join(fdir, "vnir_band_valid_fraction.csv")); BU = pd.read_csv(os.path.join(fdir, "vnir_band_usable.csv"))
        wcols = [c for c in VF.columns if c != "Plot_ID"]; refl_ok = bool(int(V.vnir_reflectance_ok.iloc[0])); log(f"  VNIR features reused ({V.shape})")
    else:
        V, SP, VF = vnir_features_all(cube, plots, regions, progress_cb=lambda p, m: (log(f"    vnir {p}% {m}") if p in (50, 100) else None))
        V["Plot_ID"] = V["Plot_ID"].astype(int); SP["Plot_ID"] = SP["Plot_ID"].astype(int); VF["Plot_ID"] = VF["Plot_ID"].astype(int)
        wcols = [c for c in VF.columns if c != "Plot_ID"]
        refl_ok = bool(int(V.vnir_reflectance_ok.iloc[0]))
        BU = pd.DataFrame(band_usable(wl, VF[wcols].values, refl_ok, V.frac_in_cube.values), columns=wcols)
        BU.insert(0, "Plot_ID", V["Plot_ID"].values); BU.insert(1, "cube_reflectance_ok", int(refl_ok)); BU.insert(2, "n_usable_bands", BU[wcols].sum(axis=1).values)
        V.to_csv(vfe, index=False); SP.to_csv(os.path.join(fdir, "vnir_spectra.csv"), index=False)
        VF.to_csv(os.path.join(fdir, "vnir_band_valid_fraction.csv"), index=False); BU.to_csv(os.path.join(fdir, "vnir_band_usable.csv"), index=False)
        log(f"  VNIR features {V.shape}; reflectance_ok {refl_ok}; vnir_qc_ok {int(V.vnir_qc_ok.sum())}/{len(V)}; red_ok {int(V.red_ok.sum())}/{len(V)}; "
            f"usable band cells {100*BU[wcols].to_numpy().mean():.1f}%")
    # per-plot VNIR cubes: the GeoTIFF writer needs GDAL (blocked here), so the per-plot files verified against the
    # orthomosaic with ENVI on 8-9 Oct 2026 (pixel-exact in all 172 bands) are copied from the existing datasets
    vdir = os.path.join(fdir, "vnir"); os.makedirs(vdir, exist_ok=True); src_dir = VERIFIED_CUBES[fl["key"]]
    if os.path.exists(os.path.join(vdir, "plots_vnir_index.csv")) and "--redo-vnir" not in sys.argv:
        idx = pd.read_csv(os.path.join(vdir, "plots_vnir_index.csv")); log(f"  per-plot VNIR cubes reused ({len(idx)})")
    else:
        idx = pd.read_csv(os.path.join(src_dir, "plots_vnir_index.csv")); keep = []; n_copied = 0
        best = VF.set_index("Plot_ID")[wcols].max(axis=1)
        for _, r in idx.iterrows():
            base = os.path.basename(str(r.path)); src = os.path.join(src_dir, base)
            if not os.path.exists(src): continue
            for ext in ("", ".aux.xml"):
                if os.path.exists(src + ext) and not os.path.exists(os.path.join(vdir, base + ext)): shutil.copy2(src + ext, os.path.join(vdir, base + ext)); n_copied += 1
            q = src[:-4] + ".qml"
            if os.path.exists(q) and not os.path.exists(os.path.join(vdir, base[:-4] + ".qml")): shutil.copy2(q, os.path.join(vdir, base[:-4] + ".qml"))
            keep.append(dict(Plot_ID=int(r.Plot_ID), name=r.get("name", ""), path=base, width=r.get("width"), height=r.get("height"), px_inside=r.get("px_inside"),
                             vnir_status=("ok" if best.get(int(r.Plot_ID), 0) >= 0.5 else ("partial" if best.get(int(r.Plot_ID), 0) >= 0.05 else "low")),
                             source=src_dir, verified="pixel-exact vs orthomosaic (ENVI, 2026-10-08/09)" + (" except 4 edge pixels" if (fl["key"], int(r.Plot_ID)) == ("2025-11-13_f2", 1001) else (" except 1 edge pixel" if (fl["key"], int(r.Plot_ID)) == ("2025-11-13_f2", 1024) else ""))))
        idx = pd.DataFrame(keep); idx.to_csv(os.path.join(vdir, "plots_vnir_index.csv"), index=False)
        log(f"  per-plot VNIR cubes copied: {len(idx)} files ({n_copied} new) from {src_dir}")
    cube.close()

    # -------- join --------
    A = gt_out.merge(L, on="Plot_ID", how="right").merge(V, on="Plot_ID", how="left")
    A["has_ground_truth"] = A["has_ground_truth"].fillna(0).astype(int)
    A["trial"] = fl["trial"]; A["flight"] = fl["key"]; A["stage"] = fl["stage"]
    A["valid_lidar"] = (A["qc_ok"] == 1).astype(int)
    A["valid_vnir"] = ((A["vnir_qc_ok"] == 1) & (A["red_ok"] == 1)).fillna(False).astype(int)     # strict: usable red band too
    A["valid_vnir_lenient"] = (A["vnir_qc_ok"] == 1).fillna(False).astype(int)                     # red-based indices are NaN anyway
    A["valid_fused"] = ((A["valid_lidar"] == 1) & (A["valid_vnir"] == 1)).astype(int)
    front = ["Plot_ID", "trial", "flight", "stage", "has_ground_truth", "valid_lidar", "valid_vnir", "valid_vnir_lenient", "valid_fused"]
    A = A[front + [c for c in A.columns if c not in front]]
    A.to_csv(os.path.join(fdir, "features_all.csv"), index=False)
    log(f"  features_all {A.shape}: GT plots {int(A.has_ground_truth.sum())}, of which valid_lidar {int((A.has_ground_truth.eq(1) & A.valid_lidar.eq(1)).sum())}, "
        f"valid_vnir {int((A.has_ground_truth.eq(1) & A.valid_vnir.eq(1)).sum())}, valid_fused {int((A.has_ground_truth.eq(1) & A.valid_fused.eq(1)).sum())}")
    return dict(trial=fl["trial"], flight=fl["key"], stage=fl["stage"], plots=len(plots), gt_plots=int(A.has_ground_truth.sum()),
                lidar_qc_ok=int(A.valid_lidar.sum()), vnir_reflectance_ok=int(refl_ok), vnir_qc_ok=int(A.valid_vnir_lenient.sum()), red_ok=int((A.red_ok == 1).sum()),
                gt_valid_lidar=int((A.has_ground_truth.eq(1) & A.valid_lidar.eq(1)).sum()), gt_valid_vnir=int((A.has_ground_truth.eq(1) & A.valid_vnir.eq(1)).sum()),
                gt_valid_fused=int((A.has_ground_truth.eq(1) & A.valid_fused.eq(1)).sum()), usable_band_cells_pct=round(100 * BU[wcols].to_numpy().mean(), 1),
                las=os.path.basename(fl["las"]), cube=os.path.basename(fl["cube"]), grid=os.path.basename(fl["grid"]))

# ---------------------------------------------------------------- AGT maturity: flight 1 preferred, flight 2 where flight 1 has no usable LiDAR
def merge_agt_maturity():
    d1 = os.path.join(OUT, "agt", "2025-11-13_f1", "features_all.csv"); d2 = os.path.join(OUT, "agt", "2025-11-13_f2", "features_all.csv")
    if not (os.path.exists(d1) and os.path.exists(d2)):
        return None
    A1 = pd.read_csv(d1); A2 = pd.read_csv(d2).set_index("Plot_ID")
    rows = []
    for _, r in A1.iterrows():
        use2 = (r.valid_lidar != 1) and (r.Plot_ID in A2.index) and (A2.loc[r.Plot_ID, "valid_lidar"] == 1)
        rr = A2.loc[r.Plot_ID].copy() if use2 else r.copy()
        if use2: rr["Plot_ID"] = r.Plot_ID
        rr["source_flight"] = "2025-11-13_f2" if use2 else "2025-11-13_f1"; rows.append(rr)
    M = pd.DataFrame(rows); M["flight"] = "2025-11-13"
    mdir = os.path.join(OUT, "agt", "2025-11-13_merged"); os.makedirs(mdir, exist_ok=True)
    M.to_csv(os.path.join(mdir, "features_all.csv"), index=False)
    shutil.copy(os.path.join(OUT, "agt", "2025-11-13_f1", "ground_truth.csv"), os.path.join(mdir, "ground_truth.csv"))
    log(f"  AGT maturity merged: {len(M)} plots, {int((M.source_flight == '2025-11-13_f2').sum())} taken from flight 2; GT plots valid_lidar "
        f"{int((M.has_ground_truth.eq(1) & M.valid_lidar.eq(1)).sum())}/{int(M.has_ground_truth.sum())}; VNIR invalid for every plot (radiance cube)")
    return M

if __name__ == "__main__":
    keys = [a for a in sys.argv[1:] if not a.startswith("--")]
    summary = []
    for fl in FLIGHTS:
        if keys and fl["key"] not in keys: continue
        try:
            summary.append(run_flight(fl))
        except Exception as e:
            log(f"  FAILED {fl['key']}: {e}"); log(traceback.format_exc())
            summary.append(dict(trial=fl["trial"], flight=fl["key"], stage=fl["stage"], error=str(e)))
        gc.collect()
    merge_agt_maturity()
    S = pd.DataFrame(summary)
    old = os.path.join(OUT, "dataset_summary.csv")
    if os.path.exists(old) and keys:
        S0 = pd.read_csv(old); S = pd.concat([S0[~S0.flight.isin(S.flight)], S], ignore_index=True)
    S.to_csv(old, index=False)
    manifest = dict(built=time.strftime("%Y-%m-%d %H:%M"), tool=TOOL, tool_version="PhenoApp v3.3 (v3.0.1 VNIR QC + v3.2 LiDAR), 2026-10-09",
                    noise_rule=NOISE_RULE, lidar_core_features=LIDAR_V3_CORE, vnir_core_features=VNIR_V3_CORE, min_valid_frac=MIN_VALID_FRAC,
                    excluded_nm=DEFAULT_EXCLUDED_NM, crs=CRS, flights=[{k: v for k, v in f.items()} for f in FLIGHTS],
                    notes=["per-plot VNIR cubes are the files verified pixel-exact against the orthomosaics with ENVI (2026-10-08/09); AGT 2025-11-13 flight 2 plots 1001 and 1024 lack 4 and 1 edge pixels (writer defect fixed in v3.0.1, files to be regenerated on a machine with GDAL)",
                           "AGT 2025-11-13 VNIR cubes are at-sensor radiance (not reflectance): every VNIR feature is NaN and vnir_qc_ok = 0 for both flights",
                           "AGT 2025-09-22: 670 nm clipped on ~half of the pixels; red-based indices are NaN where red_ok = 0 (valid_vnir = strict filter)",
                           "ground truth: Muresk biomass + ruler height from Biomass_25NO43_AGT&DPIRD.xlsx (DPIRD_NUE_2025); AGT biomass from ground_truth.csv per date"])
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1, default=str)
    log("summary:\n" + S.to_string(index=False)); log("BUILD DONE")
