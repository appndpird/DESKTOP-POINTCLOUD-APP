"""Bundle the plot-grouped common-band two-stream ensembles into the tool (PhenoApp v3.3.3):
biomass = dl_biomass_commonbands_grouped_<variant>_fold0-4 (760 plots, 4 datasets),
height  = dl_height_commonbands_grouped_maturity_<variant>_fold0-4 (128 plots, maturity flight with the ruler).
The previously bundled (ungrouped) files are moved to <tool>\\_dl_models_v3.3.2_superseded (outside the bundled assets)."""
import os, glob, shutil, torch, pandas as pd
SRC = r"D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream\models"
TOOL = r"C:\Users\appn\Downloads\PhenoFuse-main\phenoapp_v3_2026-10-06"
DST = os.path.join(TOOL, "phenoapp", "assets", "dl_models"); OLD = os.path.join(TOOL, "_dl_models_v3.3.2_superseded"); os.makedirs(OLD, exist_ok=True)
for f in glob.glob(os.path.join(DST, "*.pt")): shutil.move(f, os.path.join(OLD, os.path.basename(f)))
n = 0
for target, tag in (("biomass", "commonbands_grouped"), ("height", "commonbands_grouped_maturity")):
    for variant in ("two_stream", "lidar_only"):
        for k in range(5):
            s = os.path.join(SRC, f"dl_{target}_{tag}_{variant}_fold{k}.pt")
            if not os.path.exists(s): raise SystemExit(f"missing {s}")
            sd = torch.load(s, map_location="cpu"); torch.save({kk: (v.half() if v.is_floating_point() else v) for kk, v in sd.items()}, os.path.join(DST, f"{variant}_{target}_fold{k}.pt")); n += 1
open(os.path.join(DST, "README.txt"), "w").write(
    "PhenoApp v3.3.3 pretrained two-stream ensembles (2026-10-12): {variant}_{target}_fold{k}.pt, k = 0..4, float16 state dicts.\n"
    "Trained with folds GROUPED BY PLOT (both flights of a plot always in the same fold) and the common band policy: common_bands.csv lists the 111 bands\n"
    "(common_95 = 1) used for every plot; the runner applies it automatically when this file is present.\n"
    "biomass: 760 wheat plots, two trials x anthesis/maturity; log(kg/ha) target, normalisation mean 9.239603 sd 0.227668; dataset embedding order (sorted names):\n"
    "  0 anthesis trial B, 1 maturity trial B, 2 anthesis trial A, 3 maturity trial A; a new dataset is scored with every training embedding of its stage, averaged.\n"
    "height: 128 plots of the maturity flight (the only flight with a ruler measurement); cm, mean 84.656250 sd 5.091641; one dataset embedding.\n"
    "Held-out 10-fold scores (grouped): research/2026-10-09/results_summary/two_stream (dl_*_commonbands_grouped*_comparison.csv).\n")
print("bundled", n, "fold files; folder MB:", round(sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)) / 1e6, 1))
