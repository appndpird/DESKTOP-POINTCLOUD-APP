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

## Environment notes (this machine, 9 Oct 2026)
- Python venv `C:\Users\appn\phenoapp-venv` (numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, laspy, shapely, tifffile).
- An application-control policy blocks the DLLs of fiona, pyogrio, rasterio and torch, so: shapefiles are read with pyshp,
  ENVI cubes with numpy memmap, per-plot GeoTIFFs with tifffile; per-plot cubes are copied (verified) rather than rewritten;
  the two-stream network could not be trained (torch DLLs blocked, no spconv wheel) - run `dl_twostream.py` in the
  soilnet environment on the GPU machine with `two_stream\dl_data` as input:
  `python dl_twostream.py --target biomass --epochs 100` and `python dl_twostream.py --target height --epochs 100`
  (100 epochs per fold is the default; AdamW + OneCycle; training-time augmentation = random 180-degree rotation,
  mirror across the row axis, 0-30 % point dropout, 1 cm xy jitter, bootstrap resampling of the plot pixels and a
  0.9-1.1 brightness factor; no augmentation at test time). Then rerun `report_builder.py` to add section 5.
