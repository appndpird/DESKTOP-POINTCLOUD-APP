"""
Re-extract per-plot LiDAR traits and VNIR spectral indices for both trials with the
corrected methods (2026-10-01) and save one Excel workbook per trial.

LiDAR  : PhenoApp extract_all_plots on the refit grid of each flight - every catalogue
         trait incl. the canopy-top heights (cth_*: noise filter, plot-local alley ground,
         2 cm voxel thinning, 5 cm top-surface raster, P95 of cell maxima) with QC flags,
         exterior ground for the legacy h_* traits, whole-plot region (10 cm inset).
VNIR   : every Awesome-Spectral-Indices + PhenoApp index computable from the cube,
         per pixel then plot mean, excluded ranges 0-415 / 755-770 / 928-962 nm,
         sd + valid-pixel fraction per index (QC), plus the plot mean spectrum.
Output : <Biomass Dataset>/<trial>/<flight>/plot_lidar_traits.csv, plot_vnir_indices.csv,
         plot_vnir_indices_qc.csv, plot_vnir_spectra.csv, provenance JSON;
         <Biomass Dataset>/<trial>/<trial>_plot_traits_indices.xlsx (one sheet set per flight).
Usage  : python build_plot_traits_indices.py [muresk] [agt]
"""
import os, sys, time, json, traceback, numpy as np, pandas as pd
# 2026-10-09: needs PhenoApp v3.0.1 or later (NaN rule for unusable bands, band_usable, reflectance check)
sys.path.insert(0, r"E:\Software\DESKTOP-POINTCLOUD-APP\phenoapp_v3_2026-10-06")
from phenoapp.core import LASManager, load_grid, extract_all_plots, TRAIT_KEYS, plot_region, VNIRCube
from phenoapp.core.spectral_indices import IndexCatalogue, compute_plot_indices, DEFAULT_EXCLUDED_NM, RECOMMENDED_BIOMASS, band_usable, MIN_VALID_FRAC

B = r"F:\Ibrahim's Workspace2\Biomass Experiment"; DS = os.path.join(B, "Biomass Dataset"); CRS = "EPSG:7850"
AGT = os.path.join(B, "2025_NUE_AGT_I_DPIRD")
TRIALS = {
    "muresk": dict(folder="NUE_I_DPIRD_Muresk_F_Gobi", flights=[
        dict(key="2025-09-30", las=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_LiDAR_CombinedPointCloud.las"),
             vnir=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin"),
             grid=os.path.join(B, "2025-09-30", "aligned_grid_refit_20250930.shp"), out="2025-09-30", stage="anthesis"),
        dict(key="2025-11-21", las=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_LiDAR_CombinedPointCloud.las"),
             vnir=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin"),
             grid=os.path.join(B, "Grid_Refit_Comparison", "aligned_grid_refit.shp"), out="2025-11-21", stage="maturity")]),
    "agt": dict(folder="2025_NUE_AGT_I_DPIRD", flights=[
        dict(key="2025-09-22", las=os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_LiDAR_CombinedPointCloud.las"),
             vnir=os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin"),
             grid=os.path.join(AGT, "2025-09-22", "grid_qa", "refit_auto3.shp"), out="2025-09-22", stage="anthesis"),
        dict(key="2025-11-13_f1", las=os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_LiDAR_CombinedPointCloud.las"),
             vnir=os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin"),
             grid=os.path.join(AGT, "2025-11-13", "flight1", "aligned_grid_refit_flight1.shp"), out="2025-11-13", stage="maturity flight1"),
        dict(key="2025-11-13_f2", las=os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_LiDAR_CombinedPointCloud.las"),
             vnir=os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin"),
             grid=os.path.join(AGT, "2025-11-13", "flight2", "aligned_grid_refit_flight2.shp"), out="2025-11-13", stage="maturity flight2 (south part only)")]),
}
t0 = time.time()
LOG = open(os.path.join(DS, "build_plot_traits_indices.log"), "a")
def log(*a):
    s = f"[{time.time()-t0:6.0f}s] " + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def ground_truth(trial, fl):
    """Design + ground-truth columns keyed by Plot_ID."""
    if trial == "muresk":
        g = pd.read_excel(os.path.join(B, "Biomass_25NO43_AGT&DPIRD.xlsx"), sheet_name="DPIRD_NUE_2025")
        g.columns = ["Plot", "Range", "Row", "Block", "Rep", "N_treatment", "Variety", "N_available_kg_ha", "biomass_anthesis_kg_ha", "biomass_maturity_kg_ha", "plant_height_maturity_cm"]
        g["Plot_ID"] = np.arange(1, 129)
        return g[["Plot_ID", "Plot", "Range", "Row", "Block", "Rep", "N_treatment", "Variety", "N_available_kg_ha", "biomass_anthesis_kg_ha", "biomass_maturity_kg_ha", "plant_height_maturity_cm"]]
    p = os.path.join(DS, "2025_NUE_AGT_I_DPIRD", fl["out"], "ground_truth.csv")
    g = pd.read_csv(p)
    keep = [c for c in g.columns if not c.startswith(("lidar_", "vnir_"))]
    return g[keep]

cat = IndexCatalogue()
only = [a for a in sys.argv[1:] if a in TRIALS] or list(TRIALS)
FORCE = [a.split(":", 1)[1] for a in sys.argv[1:] if ":" in a]      # e.g. agt:2025-11-13_f2 -> recompute only that flight, reuse the rest
for trial in only:
    T = TRIALS[trial]; tdir = os.path.join(DS, T["folder"]); sheets = {}; readme = []
    for fl in T["flights"]:
        fdir = os.path.join(tdir, fl["out"]); os.makedirs(fdir, exist_ok=True); sfx = "" if fl["key"] == fl["out"] else "_" + fl["key"].split("_")[-1]
        log(f"=== {trial} {fl['key']}: {fl['stage']}")
        try:
            plots = load_grid(fl["grid"], target_crs=CRS).reset_index(drop=True)
            gt = ground_truth(trial, fl)
            done = all(os.path.exists(os.path.join(fdir, f"{n}{sfx}.csv")) for n in ("plot_lidar_traits", "plot_vnir_indices", "plot_vnir_indices_qc", "plot_vnir_spectra"))
            if FORCE and fl["key"] not in FORCE and done:
                # reuse the outputs of an earlier run
                ld = pd.read_csv(os.path.join(fdir, f"plot_lidar_traits{sfx}.csv")); vi = pd.read_csv(os.path.join(fdir, f"plot_vnir_indices{sfx}.csv"))
                qc = pd.read_csv(os.path.join(fdir, f"plot_vnir_indices_qc{sfx}.csv")); sp = pd.read_csv(os.path.join(fdir, f"plot_vnir_spectra{sfx}.csv"))
                if "cth_ok" not in ld.columns:
                    raise RuntimeError("existing LiDAR CSV has no cth_* columns - rerun this flight")
                bu_p = os.path.join(fdir, f"plot_vnir_band_usable{sfx}.csv"); vf_p = os.path.join(fdir, f"plot_vnir_band_valid_fraction{sfx}.csv")
                if not (os.path.exists(bu_p) and os.path.exists(vf_p)):
                    raise RuntimeError("existing VNIR outputs have no band-QC tables (pre v3.0.1) - rerun this flight")
                bu = pd.read_csv(bu_p); bvf = pd.read_csv(vf_p)
                gm = json.load(open(os.path.join(fdir, f"plot_lidar_traits{sfx}_ground_model.json"))) if os.path.exists(os.path.join(fdir, f"plot_lidar_traits{sfx}_ground_model.json")) else {}
                acr = [c for c in vi.columns if f"{c}_valid" in qc.columns]
                low = [c for c in acr if qc[f"{c}_valid"].mean() < 0.6]
                lid = ld.merge(gt, on="Plot_ID", how="left"); vij = vi.drop(columns=[c for c in ("Plot", "B/R", "Range", "Row") if c in vi.columns]).merge(gt, on="Plot_ID", how="left")
                front = [c for c in gt.columns if c in lid.columns]
                lid = lid[front + [c for c in lid.columns if c not in front]]; vij = vij[front + [c for c in vij.columns if c not in front]]
                k = fl["key"]; sheets[f"{k}_LiDAR"] = lid; sheets[f"{k}_VNIR"] = vij; sheets[f"{k}_VNIR_QC"] = qc; sheets[f"{k}_Spectra"] = sp
                sheets[f"{k}_Bands_ValidFrac"] = bvf; sheets[f"{k}_Bands_QC"] = bu
                readme.append(dict(flight=k, stage=fl["stage"], plots=len(plots), plots_with_ground_truth=int(lid[front[-1]].notna().sum()) if front else 0,
                                   las=os.path.basename(fl["las"]), vnir=os.path.basename(fl["vnir"]), grid=os.path.basename(fl["grid"]),
                                   exterior_ground_rms_cm=round(gm.get("rms_resid_m", float("nan")) * 100, 1), cth_ok=int(ld.cth_ok.sum()),
                                   vnir_indices=len(acr), vnir_indices_low_valid=", ".join(low[:15]) + (" ..." if len(low) > 15 else ""),
                                   vnir_reflectance_ok=int(bu.get("cube_reflectance_ok", pd.Series([1])).iloc[0]), usable_band_cells_pct=round(100 * bu.filter(regex=r"^\d").to_numpy().mean(), 1)))
                log(f"  reused existing outputs ({len(ld)} plots)")
                continue
            # ---------------- LiDAR ----------------
            lidar_csv = os.path.join(fdir, f"plot_lidar_traits{sfx}.csv")
            mgr = LASManager(fl["las"], CRS, use_smrf=False); mgr.load(progress_cb=lambda p, m: None)
            log(f"  LAS {len(mgr.x)/1e6:.0f} M points, {len(plots)} plots, grid {os.path.basename(fl['grid'])}")
            traits = [k for k in TRAIT_KEYS if k != "vert_profile"]
            ld = extract_all_plots(mgr, plots, out_dir=os.path.join(fdir, "lidar"), out_csv=lidar_csv, selected_traits=traits, write_las=False,
                                   height_cut=0.15, voxel=0.05, biomass_k=1.0, ground_mode="exterior", region_mode="whole", band_width=0.5,
                                   cth_ground="local", progress_cb=lambda p, m: (log(f"    lidar {p}% {m}") if p in (3, 50, 99, 100) and "plot" not in m.lower() else None))
            gm = ld.attrs.get("ground_model", {})
            log(f"  LiDAR traits: {ld.shape}, exterior ground rms {gm.get('rms_resid_m', float('nan'))*100:.1f} cm; cth_ok {int(ld.cth_ok.sum())}/{len(ld)}, "
                f"local-vs-trial ground offset sd {ld.cth_ground_offset.std()*100:.1f} cm")
            del mgr
            # ---------------- VNIR ----------------
            cube = VNIRCube(fl["vnir"]); wl = cube.wavelengths
            regions = [plot_region(g, mode="whole", band_width=0.5) for g in plots.geometry]
            acr = [e.acronym for d in cat.by_domain(wl, DEFAULT_EXCLUDED_NM).values() for e in d]
            acr = [a for a in RECOMMENDED_BIOMASS if a in acr] + [a for a in acr if a not in RECOMMENDED_BIOMASS]
            vi, qc, meta = compute_plot_indices(cube, plots, regions, acr, mode="pixel", excluded=DEFAULT_EXCLUDED_NM, catalogue=cat,
                                                progress_cb=lambda p, m: (log(f"    vnir {p}% {m}") if p in (70, 100) else None))
            spectra, npx = cube.plot_spectra(plots, regions)      # v3.0.1: NaN where a band is unusable for the plot (see band_usable)
            band_vf = cube.last_band_valid_frac; refl_ok = bool(cube.last_reflectance_ok)
            cube.close()
            if not refl_ok:
                log("  WARNING: cube is not reflectance (NDVI ~ 0 in every plot) - every spectral value written as NaN, every band flagged 0")
            for c in ("Plot", "B/R", "Range", "Row"):
                if c in plots.columns and c not in vi.columns:
                    vi.insert(1, c, plots[c].values)
            vi["vnir_px_valid_800nm"] = npx
            region_px = np.array([r.area for r in regions]) / (cube.res[0] * cube.res[1])
            vi["vnir_coverage"] = np.clip(npx / np.maximum(region_px, 1), 0, 1)
            wcols = [f"{w:.1f}" for w in wl]
            sp = pd.DataFrame(spectra, columns=wcols); sp.insert(0, "Plot_ID", plots["Plot_ID"].values)
            # band QC: 1 = band usable for the plot (valid on >= MIN_VALID_FRAC of the region pixels, not excluded, cube is reflectance),
            # 0 = the matching Spectra cell is NaN; plus the valid-pixel fraction behind each decision
            bu = pd.DataFrame(band_usable(wl, band_vf, refl_ok), columns=wcols)
            bu.insert(0, "Plot_ID", plots["Plot_ID"].values); bu.insert(1, "cube_reflectance_ok", int(refl_ok)); bu.insert(2, "n_usable_bands", bu[wcols].sum(axis=1).values)
            bvf = pd.DataFrame(band_vf, columns=wcols); bvf.insert(0, "Plot_ID", plots["Plot_ID"].values)
            if not refl_ok:
                vi[[c for c in vi.columns if c not in ("Plot_ID", "Plot", "B/R", "Range", "Row")]] = np.nan
            vi.to_csv(os.path.join(fdir, f"plot_vnir_indices{sfx}.csv"), index=False); qc.to_csv(os.path.join(fdir, f"plot_vnir_indices_qc{sfx}.csv"), index=False)
            sp.to_csv(os.path.join(fdir, f"plot_vnir_spectra{sfx}.csv"), index=False)
            bu.to_csv(os.path.join(fdir, f"plot_vnir_band_usable{sfx}.csv"), index=False); bvf.to_csv(os.path.join(fdir, f"plot_vnir_band_valid_fraction{sfx}.csv"), index=False)
            meta.update(grid=fl["grid"], las=fl["las"], region="whole plot, 10 cm inset", lidar_ground="exterior surface (h_*), plot-local ring plane (cth_*)")
            with open(os.path.join(fdir, f"plot_provenance{sfx}.json"), "w") as f:
                json.dump(meta, f, indent=1, default=str)
            low = [c for c in vi.columns if f"{c}_valid" in qc.columns and qc[f"{c}_valid"].mean() < 0.6]
            log(f"  VNIR: {len(acr)} indices, {meta['n_cube_bands_read']} bands read; coverage median {np.median(vi.vnir_coverage):.2f}; indices valid on <60% px: {len(low)}")
            # ---------------- join + sheets ----------------
            lid = ld.merge(gt, on="Plot_ID", how="left"); vij = vi.drop(columns=[c for c in ("Plot", "B/R", "Range", "Row") if c in vi.columns]).merge(gt, on="Plot_ID", how="left")
            front = [c for c in gt.columns if c in lid.columns]
            lid = lid[front + [c for c in lid.columns if c not in front]]; vij = vij[front + [c for c in vij.columns if c not in front]]
            k = fl["key"]
            sheets[f"{k}_LiDAR"] = lid; sheets[f"{k}_VNIR"] = vij; sheets[f"{k}_VNIR_QC"] = qc; sheets[f"{k}_Spectra"] = sp
            sheets[f"{k}_Bands_ValidFrac"] = bvf; sheets[f"{k}_Bands_QC"] = bu
            readme.append(dict(flight=k, stage=fl["stage"], plots=len(plots), plots_with_ground_truth=int(lid[front[-1]].notna().sum()) if front else 0,
                               las=os.path.basename(fl["las"]), vnir=os.path.basename(fl["vnir"]), grid=os.path.basename(fl["grid"]),
                               exterior_ground_rms_cm=round(gm.get("rms_resid_m", float("nan")) * 100, 1), cth_ok=int(ld.cth_ok.sum()),
                               vnir_indices=len(acr), vnir_indices_low_valid=", ".join(low[:15]) + (" ..." if len(low) > 15 else ""),
                               vnir_reflectance_ok=int(refl_ok), usable_band_cells_pct=round(100 * bu[wcols].to_numpy().mean(), 1)))
        except Exception as e:
            log("  FAILED:", e); log(traceback.format_exc())
            readme.append(dict(flight=fl["key"], stage=fl["stage"], error=str(e)))
    # ---------------- workbook ----------------
    xlsx = os.path.join(tdir, f"{T['folder']}_plot_traits_indices.xlsx")
    notes = pd.DataFrame([
        ["Built", time.strftime("%Y-%m-%d %H:%M")],
        ["LiDAR sheets", "PhenoApp v2 traits on the refit grid; h_* on the trial-wide exterior ground; cth_* = canopy-top surface heights on the plot-local alley ground "
                         "(noise filter, 25 cm alley cells 5th pct robust plane, 2 cm voxel thinning, 5 cm cell-max raster, 15 cm edge trim); cth_p95 recommended; "
                         "cth_ok/cth_cover/cth_ground_cells/cth_ground_rms/cth_ground_offset are QC flags; heights in metres."],
        ["VNIR sheets", "Awesome Spectral Indices (Montero et al. 2023) + PhenoApp narrow-band indices, per pixel then plot mean over the whole-plot region (10 cm inset); "
                        "broad bands synthesised from the cube wavelengths, excluded 0-415 / 755-770 / 928-962 nm; a pixel counts only when every band feeding the index is > 0."],
        ["VNIR_QC sheets", "<index>_sd = within-plot sd of the per-pixel index; <index>_valid = fraction of region pixels valid. Indices valid on <60% of pixels use bands clipped to 0 "
                           "(negative reflectance in the GRYFN cube) and should not be used for calibration."],
        ["NaN rule (v3.0.1)", f"An index is NaN in the VNIR sheet (and its _sd in VNIR_QC) when <index>_valid < {MIN_VALID_FRAC}, i.e. computable on fewer than half of the "
                              "plot's pixels because an input band is clipped; every index is NaN for a cube that fails the reflectance check. NaN cells are to be left out, never filled."],
        ["Spectra sheets", "Plot mean reflectance x 10000 per band (zeros excluded per band); column = wavelength nm. NaN = band unusable for that plot "
                           f"(valid on < {int(MIN_VALID_FRAC*100)}% of the region pixels, excluded range 0-415 / 755-770 / 928-962 nm, or cube not reflectance); never fill these."],
        ["Bands_QC sheets", "One row per plot, one column per band: 1 = usable, 0 = do not use (the Spectra cell is NaN). cube_reflectance_ok = 0 marks a cube that is "
                            "still radiance (every band 0 until reprocessed); n_usable_bands counts usable bands per plot."],
        ["Bands_ValidFrac sheets", "Fraction of the plot's region pixels with a valid (non-zero) value in each band of the original orthomosaic: the number behind each Bands_QC decision."],
        ["Recommended biomass indices", ", ".join(RECOMMENDED_BIOMASS)],
        ["Caveats", "Ruler plant height has ~4 cm SD noise; GRYFN cubes clip negative reflectance to 0 (blue bands unusable in canopy); AGT flight2 covers the south part only."],
    ], columns=["item", "value"])
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            notes.to_excel(xw, sheet_name="README", index=False)
            pd.DataFrame(readme).to_excel(xw, sheet_name="Flights", index=False)
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
        log(f"workbook written: {xlsx} ({len(sheets)} data sheets)")
    except Exception as e:
        log("workbook FAILED:", e); log(traceback.format_exc())
log("ALL DONE")
