"""Regenerate every per-plot VNIR GeoTIFF with the mask-based writer (PhenoApp v2.2.1):
pixel values are exact copies of the orthomosaic, the plot polygon is carried by an internal GDAL
mask instead of a nodata tag, so zero-valued (clipped) pixels inside the plot are no longer shown
as holes in QGIS. File names and folders are unchanged. Also rewrites the README lines."""
import os, sys, re, glob, time
sys.path.insert(0, r"E:\Software\DESKTOP-POINTCLOUD-APP\phenoapp_v2")
from phenoapp.core import load_grid, VNIRCube
B = r"F:\Ibrahim's Workspace2\Biomass Experiment"; DS = os.path.join(B, "Biomass Dataset"); AGT = os.path.join(B, "2025_NUE_AGT_I_DPIRD")
JOBS = [
    dict(cube=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin"),
         grid=os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-09-30", "grid_used.shp"),
         out=os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-09-30", "vnir"), name_fmt="plot_{pid}_{name}", name_col="B/R", ids=None),
    dict(cube=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin"),
         grid=os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-11-21", "grid_used.shp"),
         out=os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-11-21", "vnir"), name_fmt="plot_{pid}_{name}", name_col="B/R", ids=None),
    dict(cube=os.path.join(AGT, "2025-09-22", "20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-09-22", "grid_qa", "refit_auto3.shp"),
         out=os.path.join(DS, "2025_NUE_AGT_I_DPIRD", "2025-09-22", "vnir"), name_fmt="plot_{pid}", name_col=None, ids="plot_(\\d+)\\.tif$"),
    dict(cube=os.path.join(AGT, "2025-11-13", "flight1", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-11-13", "flight1", "aligned_grid_refit_flight1.shp"),
         out=os.path.join(DS, "2025_NUE_AGT_I_DPIRD", "2025-11-13", "vnir"), name_fmt="plot_{pid}_flight1", name_col=None, ids="plot_(\\d+)_flight1\\.tif$"),
    dict(cube=os.path.join(AGT, "2025-11-13", "flight2", "20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin"),
         grid=os.path.join(AGT, "2025-11-13", "flight2", "aligned_grid_refit_flight2.shp"),
         out=os.path.join(DS, "2025_NUE_AGT_I_DPIRD", "2025-11-13", "vnir"), name_fmt="plot_{pid}_flight2", name_col=None, ids="plot_(\\d+)_flight2\\.tif$"),
    # the original build folders (duplicates of the Muresk sets) - keep them consistent
    dict(cube=os.path.join(B, "2025-09-30", "20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin"),
         grid=os.path.join(B, "2025-09-30", "aligned_grid_refit_20250930.shp"),
         out=os.path.join(B, "2025-09-30", "dataset", "vnir"), name_fmt="plot_{pid}_{name}", name_col="B/R", ids=None),
    dict(cube=os.path.join(B, "2025-11-21", "20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin"),
         grid=os.path.join(B, "Grid_Refit_Comparison", "aligned_grid_refit.shp"),
         out=os.path.join(B, "2025-11-21", "dataset", "vnir"), name_fmt="plot_{pid}_{name}", name_col="B/R", ids=None),
]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
for j in JOBS:
    plots = load_grid(j["grid"], target_crs="EPSG:7850")
    ids = None
    if j["ids"]:
        ids = sorted({int(re.search(j["ids"], os.path.basename(f)).group(1)) for f in glob.glob(os.path.join(j["out"], "*.tif")) if re.search(j["ids"], os.path.basename(f))})
    # remove stale index / aux files for this flight before rewriting
    for f in glob.glob(os.path.join(j["out"], "*.aux.xml")):
        os.remove(f)
    cube = VNIRCube(j["cube"])
    tmp = j["out"].rstrip("\\/") + "_regen"
    idx = cube.write_plot_cubes(plots, tmp, name_fmt=j["name_fmt"], name_col=j["name_col"] or "__none__", plot_ids=ids,
                                progress_cb=lambda p, m: (log("   " + m) if p % 48 == 0 else None))
    cube.close()
    # move into place; a file locked by another program (e.g. open in QGIS) stays in the _regen folder
    locked = []
    for f in glob.glob(os.path.join(tmp, "*")):
        if f.endswith("plots_vnir_index.csv"):
            continue
        dest = os.path.join(j["out"], os.path.basename(f))
        try:
            os.replace(f, dest)
        except PermissionError:
            locked.append(os.path.basename(f))
    if not locked:
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)
        os.rmdir(tmp)
    log(f"{os.path.basename(j['cube'])[:40]}: {len(idx)} files regenerated -> {j['out']}" +
        (f"; LOCKED (still in {tmp}): {locked}" if locked else ""))
# README notes
NOTE = ("\nVNIR per-plot files (regenerated 2026-10-01, PhenoApp v2.2.1): every band, pixel values are exact copies of the orthomosaic "
        "(verified bit-exact), DEFLATE lossless, same 1.5/2.5 cm grid, EPSG:7850. The plot polygon is stored as an internal GDAL mask "
        "(outside = masked/transparent); NO nodata tag, so a 0 inside the plot is a real orthomosaic value (GRYFN clips negative "
        "reflectance to 0 - mostly 400-500 nm, a few % at 670 nm) and is not a hole. The .qml sidecar makes QGIS open the file as a "
        "NIR/red/green composite (bands 115/78/44) instead of bands 1/2/3.\n")
for rd in (os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-09-30", "README.txt"), os.path.join(DS, "NUE_I_DPIRD_Muresk_F_Gobi", "2025-11-21", "README.txt"),
           os.path.join(DS, "2025_NUE_AGT_I_DPIRD", "README.txt")):
    if os.path.exists(rd) and "internal GDAL mask" not in open(rd, encoding="utf-8", errors="ignore").read():
        open(rd, "a", encoding="utf-8").write(NOTE)
log("ALL DONE")
