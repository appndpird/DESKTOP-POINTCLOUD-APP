# Height and biomass models, rebuild of 9 October 2026

Rebuilt from scratch on the combined dataset `Biomass Experiment\Dataset_2026-10-09` with PhenoApp v3.3
(= v3.2 LiDAR pipeline from the point-cloud session + v3.0.1 VNIR QC from the VNIR session), tool source in
`DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06`.

## Folders
- `scripts\`      the exact scripts that produced everything here, in run order:
  1. `build_dataset.py`   whole cloud -> v3.2 LiDAR features + classified per-plot LAS; cube -> v3.0.1 VNIR features,
                          spectra, band valid fractions, band usable table; verified per-plot VNIR cubes copied;
                          ground truth joined -> `features_all.csv` per flight (+ AGT maturity merged f1/f2)
  2. `dl_prepare.py`      per-plot tensors for the two-stream network from the dataset (classified LAS + per-plot cubes,
                          invalid bands masked with the per-plot band mask)
  3. `height_models.py`   Muresk plant height, LiDAR core / all / + VNIR, nested repeated 10-fold, LOOCV, flight transfer
  4. `biomass_models.py`  biomass per dataset (Muresk anthesis / maturity, AGT anthesis / maturity), LiDAR_core / VNIR_core /
                          Fused_core / Fused_all, pooled and cross-trial
  5. `dl_twostream.py`    two-stream network (sparse 3D CNN + spectral transformer), 10-fold, 100 epochs per fold,
                          on-the-fly augmentation - NOT RUN HERE (see below)
  6. `report_builder.py`  figures + `REPORT_height_biomass_2026-10-09.html`
  helpers: `grid_pyshp.py` (shapefile reader without fiona), `vnir_envi.py` (ENVI cube reader without rasterio)
- `height\`       results\ (model_comparison.csv, per_plot_predictions.csv, best_models.csv, feature sets, RF importances), models\ (joblib)
- `biomass\`      results\, models\, features\ (the exact feature tables fed to the models, one per dataset)
- `two_stream\`   dl_data\ (tensors + index.csv), results\, models\ (empty until trained on a torch/spconv machine)
- `figures\`, `REPORT_height_biomass_2026-10-09.html`

## What changed against the 6 October run
- LiDAR: v3.2 pipeline for all five flights (gap noise rule, 20 cm-interior gap metrics, 2 cm cover / ground visibility,
  density layers from the 10 cm canopy threshold); core set = `LIDAR_V3_CORE` (22) + gap layers / densities / PVI for biomass.
- VNIR: NaN rule (band or index unusable when valid on < 50 % of the plot's pixels, in an excluded range, or from a
  non-reflectance cube), flight-level reflectance check, red-clipping censoring; features computed on the same sampling
  region (polygon inset 10 cm) and verified identical to the previous tool output where the rule does not apply.
- QC applied to the models: plots need ground truth + LiDAR QC; VNIR families need vnir_qc_ok = 1; a feature is used only
  when finite on > 95 % of the plots, otherwise it is dropped (red-based indices at AGT anthesis), the rest median-filled.
- AGT 2025-11-13 (maturity): both VNIR cubes are at-sensor radiance -> VNIR invalid for every plot -> LiDAR-only models.
  Per-plot VNIR files for the two AGT flight-2 plots 1001 / 1024 still lack 4 / 1 edge pixels (writer defect fixed in
  v3.0.1; regenerate on a machine with GDAL).

## Two-stream network: results of the overnight runs (9-10 Oct 2026)

Two complete runs, each 10-fold CV stratified by dataset, 100 epochs per fold, three variants (two_stream = LiDAR + VNIR,
lidar_only, vnir_only), biomass on 760 plots and height on 256 plots, GPU 1 (TITAN Xp), the two runs sharing the GPU
from 18:00: **per-plot bands** (every band usable on the plot, availability mask given to the network) and **common
bands** (the fixed list of 111 bands usable on >= 95 % of the plots of every reflectance flight, `dataset\vnir_common_bands.csv`,
`--bands common`). Held-out scores per dataset (`two_stream\results\dl_*_comparison.csv`):

| Target / dataset | two_stream per-plot | two_stream common | lidar_only per-plot | lidar_only common | vnir_only (both) | best feature model |
|---|---|---|---|---|---|---|
| Biomass, Muresk anthesis (n 128) | R2 0.23 / 1,673 | **0.27 / 1,636** | 0.32 / 1,573 | 0.29 / 1,609 | ~0 | 0.37 / 1,513 (Fused_PCA_joint GPR) |
| Biomass, Muresk maturity (128) | 0.07 / 1,388 | **0.13 / 1,340** | 0.14 / 1,332 | 0.13 / 1,342 | ~0 | 0.27 / 1,237 (Fused_bands_PCA LightGBM) |
| Biomass, AGT anthesis (252) | 0.06 / 1,475 | **0.07 / 1,467** | 0.10 / 1,444 | 0.09 / 1,450 | ~0 | 0.19 / 1,374 (Fused_bands_PCA Ridge) |
| Biomass, AGT maturity (252, LiDAR only) | 0.03 / 1,867 | **0.05 / 1,848** | 0.02 / 1,877 | 0.03 / 1,870 | ~0 | 0.06 / 1,847 (LiDAR_PCA Ridge) |
| Height, Muresk anthesis (128) | 0.47 / 3.70 cm | 0.46 / 3.74 cm | 0.44 / 3.80 cm | 0.41 / 3.9 cm | ~0 | 0.59 / 3.26 cm (joint PCA Ridge) |
| Height, Muresk maturity (128) | 0.35 / 4.11 cm | 0.35 / 4.10 cm | 0.37 / 4.03 cm | 0.41 / 3.9 cm | ~0 | 0.55 / 3.42 cm (joint PCA Ridge) |

(R2 / RMSE in kg/ha or cm; the pooled "ALL" rows of the CSVs, R2 0.53-0.55 for biomass, are inflated by the between-dataset
differences in mean biomass and are not comparable with the per-dataset values.)

Reading: (1) the network is below the fused PCA feature models on every dataset and both targets at this sample size;
(2) the spectral stream alone carries no within-dataset signal (vnir_only R2 about 0 everywhere; its pooled R2 comes from
learning the dataset means), and the fused network is at best equal to LiDAR-only; (3) the common band list is equal or
better than the per-plot policy for the fused network (anthesis 0.27 vs 0.23, maturity 0.13 vs 0.07 at Muresk) and equal
for height, consistent with the per-plot availability pattern acting as a trial fingerprint rather than as information.
**Decision (agreed with the user on 9 Oct): the common 111-band list is the standard band policy**; `dl_twostream.py`
defaults to `--bands common`, the tool's runner applies the ensemble's `common_bands.csv` automatically, and the bundled
pretrained ensembles of PhenoApp v3.3.2 are the common-band fold models (`two_stream_<target>_fold0-4.pt`,
`lidar_only_<target>_fold0-4.pt`, float16).

Timing per fold on the shared TITAN Xp: biomass two_stream 36-46 min, lidar_only 32-39 min, vnir_only 15 min; height
two_stream 10-18 min, lidar_only 9-17 min, vnir_only 5 min; the per-plot run took 16:43 to 13:08 (biomass 15 h, height 5.3 h),
the common run 18:00 to about 14:30 the next day. The first fold of a run includes a few minutes of spconv kernel compilation.

## Environment notes (this machine, 9 Oct 2026)
- Feature pipeline: Python venv `C:\Users\appn\phenoapp-venv` (3.12; numpy, pandas, scipy, scikit-learn, xgboost, lightgbm,
  laspy, shapely, tifffile). An application-control policy blocked the DLLs of fiona, pyogrio and rasterio, so: shapefiles are
  read with pyshp, ENVI cubes with numpy memmap, per-plot GeoTIFFs with tifffile; per-plot cubes are copied (verified) rather
  than rewritten.
- Deep learning: conda environment `phenoapp_dl` (Miniforge at `C:\Users\appn\miniforge3`, created 9 Oct 2026 16:13-16:27 by
  `scripts\env\conda_setup.ps1`): Python 3.11.17, torch 2.6.0+cu124, spconv-cu124 2.3.8 (cumm-cu124 0.7.11), numpy 2.4.6,
  pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, laspy 2.7.0, lazrs 0.8.2, shapely 2.2.0, tifffile, pyshp, xgboost 3.2.0,
  lightgbm 4.7.0, matplotlib, openpyxl (full list: `scripts\env\phenoapp_dl_requirements.txt`). GPUs: 2 x NVIDIA TITAN Xp
  (12 GB, Pascal sm_61), driver 560.94; training runs on `cuda:1` (GPU 0 drives the desktop).
  The spconv wheel ships no prebuilt kernels for sm_61, so spconv compiles them at run time with NVRTC; that needs the CUDA
  12.4 headers and tools, installed into the env from conda-forge (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`,
  `cuda-nvrtc-dev`, `cuda-cuxxfilt`, all 12.4). cumm locates them through `nvcc` on PATH: `run_dl.ps1` prepends
  `<env>\Library\bin` to PATH and sets `CUDA_PATH=<env>\Library`. First use of each kernel shape compiles for a few minutes.
- Two-stream run (`scripts\env\run_dl.ps1`):
  `C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe dl_twostream.py --target biomass --epochs 100 --device cuda:1`
  then the same with `--target height`. 100 epochs per fold is the default; AdamW + OneCycle; training-time augmentation =
  random 180-degree rotation, mirror across the row axis, 0-30 % point dropout, 1 cm xy jitter, bootstrap resampling of the
  plot pixels and a 0.9-1.1 brightness factor; no augmentation at test time. If spconv is missing the script trains the
  vnir_only variant and skips the LiDAR variants; results of separate runs are merged per variant. Rerun `report_builder.py`
  afterwards to fill section 5 of the report.
- QC scripts of the 8 October VNIR audit (ENVI/IDL) and the workbook build are in `scripts\vnir_qc_idl\` and
  `scripts\workbooks\`; the dataset itself is described in `dataset\README.md`.
- Tool executable of v3.3: `Downloads\DESKTOP-POINTCLOUD-APP-main\phenoapp_v3_2026-10-06\dist_v3.3_2026-10-09\PhenoApp_v3\PhenoApp_v3.exe`
  (PyInstaller, built 9 Oct 2026 16:23 from the v3.3 source; GitHub: appndpird/DESKTOP-POINTCLOUD-APP, `phenoapp_v3.3_2026-10-09/`).
