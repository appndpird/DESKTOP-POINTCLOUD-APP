# PhenoApp user guide: training, testing and inference

Taken from the Guidance tab of PhenoApp v3.3.1 (2026-10-09); the in-app version also has the trait reference, the feature table of every model and the deep-model walkthrough.


This section is the end-to-end recipe for anyone who wants to fit a biomass or height model on their own trial, test it
on held-out plots or on another flight, and predict plots that have no ground truth. The same text is shipped as
`USER_GUIDE.md` in the repository.

### 1. What you need

| Item | Requirement |
|---|---|
| Point cloud | one LAS/LAZ per flight, projected CRS in metres, intensity and return number present (single-return clouds are fine) |
| Plot grid | shapefile / GeoPackage / GeoJSON with one polygon per plot and a `Plot_ID` column; align it on the Edit tab and run "Refine plots to crop" once |
| VNIR cube | the ENVI reflectance orthomosaic of the same flight (reflectance x 10,000 or 0-1; a radiance / DN cube is detected and refused for spectral features) |
| Ground truth | CSV with `Plot_ID` and `biomass_kg_ha` (dry matter per hectare) and / or `height_cm`; one row per measured plot; the IDs must match the grid |
| Settings kept identical across flights | noise rule, region mode and band width, excluded wavelength ranges, canopy threshold; a model only transfers to features extracted the same way |

### 2. Extract the features (identical for training and inference)

1. **Point Cloud QC** (tab 2): run it once per flight; fix tilt, CRS or noise problems before anything else.
2. **Project** (tab 1): load the LAS and the grid. **Edit** (tab 4): align, then "Refine plots to crop", save the grid.
3. **Traits** (tab 6): keep "LiDAR v3" ticked, the noise rule on "Gap only", tick the canopy-top heights; tick "Write classified per-plot LAS" if you want the labelled clouds. Compute. Outputs: `<metrics>.csv` and `<metrics>_lidar_v3.csv` with the per-plot QC column `qc_ok`.
4. **VNIR Spectral** (tab 11): load the cube, run "VNIR v3". Outputs next to the metrics CSV: `_vnir_v3.csv` (features, NaN where unusable, with `vnir_qc_ok`, `red_ok`, `vnir_reflectance_ok`), `_vnir_v3_spectra.csv`, `_vnir_v3_band_valid_fraction.csv`, `_vnir_v3_band_usable.csv`.
5. Look at the usable-band table before modelling: a band column that is 0 on most plots (clipped blue or red) means the indices built on it are NaN and will be left out; a flight whose every band is 0 is a radiance cube.

### 3. Train and validate

**Plant height, Height Models tab (tab 8).** Inputs: the metrics CSV, the ground-truth CSV, optionally the `_vnir_v3.csv` for the fused sets. Tick the models (defaults: linear calibration, 7-feature ridge and random forest, LiDAR core PLS, LiDAR + VNIR PLS, LiDAR PCA ridge, fused PCA ridge). Validation: leave-one-out for the headline number on fewer than about 150 plots, repeated 10-fold for a stable comparison of many models, 5-fold for speed. Press "Fit and validate". Outputs: `<metrics>_height_predictions.csv` (measured, full-fit `pred_<model>` and held-out `predcv_<model>` per plot), `<metrics>_height_models.json` (metrics, features, calibration line), one `<metrics>_height_model_<key>.joblib` per model.

**Biomass or any per-plot trait, Biomass ML tab (tab 9).** Inputs: metrics CSV, VNIR v3 CSV, ground-truth CSV, the spectra CSV is found automatically; set the target column. Families: start with LiDAR core, fused core, LiDAR PCA and fused PCA; add the spectral-band PCA families when the spectra are clean; learners: ridge, PLS and random forest first, the others for comparison. Validation: repeated 10-fold with nested selection (default). Outputs: `<metrics>_<target>_ml_predictions.csv`, `_ml_results.json`, one `_ml_model_<family>_<learner>.joblib` per pair, `_ml_top_features.csv`.

**Reading the table.** Report the held-out columns (cvR2, cvRMSE, rRMSE, accuracy), never the fit columns; a fit R² far above the cv R² is overfitting. Prefer the simplest model within about 0.02 R² of the best. The "within 5 cm" share (height) and rRMSE (biomass) are the numbers agronomists read. With fewer than about 60 labelled plots, use the linear calibration or ridge only.

### 4. Test on held-out plots or on another flight

- **Same flight:** the `predcv_` columns are already held-out predictions; plot them against the measured values (Statistics tab, or any spreadsheet) and read the residuals per block or treatment to see whether an error is systematic.
- **Another flight or trial:** extract the features there with the same settings, open "Apply a saved model" on the Height Models or Biomass ML tab, pick the `.joblib`, press "Predict every plot". Output: `<metrics>_height_applied_<key>.csv` or `<metrics>_<target>_ml_applied_<family>_<learner>.csv` with the prediction per plot and `model_ok` (0 when a required feature is NaN on that plot). Merge with the measured plots of that flight to get the test metrics; the snippet in section 6 does it in five lines.
- **Expect an offset.** Between dates and trials the structure-to-trait relation keeps its slope but not its intercept (different stage, variety set, ruler protocol, soil). Measure a handful of plots on the new flight (five is enough for height) and enter the mean difference as the offset on the apply panel; without it, use the predictions as a ranking, not as absolute values. Within one trial and date the models are at the noise floor of the ground truth; across trials expect R² well below the within-trial value for biomass.
- **Which model to carry:** random forest and the PCA families transfer between dates better than PLS; the linear cth_p95 calibration is the safest height model for a new trial once the offset is set.

### 5. Inference on a flight without ground truth

Extract features (section 2), apply the saved model (section 4) with the offset you established, then sanity-check: the predicted range should sit inside the training range (a saved model does not extrapolate reliably), `model_ok = 0` plots need their missing feature explained (outside the cube, red clipped, too few points), and the Statistics tab heatmap should show the field pattern, not the flight-line pattern. Keep the metrics CSV, the model file and the offset together with the predictions; the JSON next to the model records the features, the validation and the training inputs.

### 6. Command line and Python

The research scripts in the repository (`research/<date>/`) reproduce every table of the report from a dataset folder: `build_dataset.py` (per-plot extraction and QC for all flights), `height_models.py`, `biomass_models.py` (feature families, PCA families, nested validation, pooled and cross-trial schemes), `dl_prepare.py` and `dl_twostream.py` (network tensors and training; `--bands per_plot|common`, `--epochs`, `--folds`), `report_builder.py`. A saved feature model is applied in Python with:

```python
import joblib, pandas as pd
m = joblib.load("flight_biomass_kg_ha_ml_model_Fused_PCA_Ridge.joblib")      # dict: model, features, meta
X = pd.read_csv("new_flight_metrics_merged.csv")                             # metrics + VNIR v3 columns merged on Plot_ID
ok = X[m["features"]].notna().all(axis=1)
X.loc[ok, "predicted"] = m["model"].predict(X.loc[ok, m["features"]].to_numpy(float)) + offset
test = X.merge(gt, on="Plot_ID"); err = test.predicted - test.biomass_kg_ha      # RMSE = (err**2).mean()**0.5
```

PCA pipelines carry their scaler and PCA inside `model`, so the input is always the raw feature columns listed in `features`.

### 7. Metrics

R² = 1 - sum of squared errors / total variance of the measured values (can be negative on transfer); RMSE in the target's unit; rRMSE = RMSE / mean measured, in %; MAE; bias = mean(prediction - measured), positive means over-prediction; r = Pearson correlation; accuracy = 100 - mean absolute percentage error; "within 5 cm" = share of plots whose height error is below 5 cm. Held-out values come from plots the model never saw in that fold; fit values are on the training plots and only show how flexible the model is.

### 8. Checklist before trusting a number

- Point cloud QC passed; grid refined; same noise rule and region on every flight.
- VNIR cube is reflectance (`vnir_reflectance_ok = 1`); usable-band table inspected; red-based indices absent where `red_ok = 0`.
- Plots with `qc_ok = 0` or `vnir_qc_ok = 0` excluded from the families that need them (the tabs do this).
- Held-out metrics reported; model and its JSON kept with the predictions; offset documented when applied to another flight.
- Deep models: see "Using the pretrained deep models, step by step"; the feature models remain the reporting models until several hundred labelled plots per crop are available.
