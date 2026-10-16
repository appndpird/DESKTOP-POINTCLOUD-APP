# PhenoApp v3.3.4 (2026-10-12)

Windows executable: `PhenoApp_v3.3.4_win64.zip` (unzip anywhere and run `PhenoApp_v3\PhenoApp_v3.exe`). The Deep Models tab
runs the network in a separate Python environment with torch and spconv (Guidance tab). `PhenoApp_v3.3.4_twostream_weights.zip`:
all fold models (float16) of the final network runs, one folder per run, each usable as a weights folder of the Deep Models tab.

## New in 3.3.4
- **Valid-band plot selection** in the Biomass ML tab ("Plots"): train and validate only on plots whose spectra have no band
  problem - rule A (every band of the common band list unclipped on at least 50 % of the plot's pixels) or rule B (95 %, plot
  entirely inside the cube); both require VNIR QC and a reflectance cube.
- Everything from 3.3.3: top-three ensembles per category (LiDAR only, VNIR only, fused) applicable in the Height Models and
  Biomass ML tabs; network ensembles trained with folds grouped by plot; height error metrics only for a flight with its own
  ruler measurement.

## Results
Reported results follow one rule: height on the flight with the ruler measurement; biomass as one overall result pooled over
all trials and stages, on the test set of each of the 10 folds, for features and deep learning in each category; plus the same
on valid-band plots only (rules A and B). Tables and fold-by-fold results: `research/2026-10-09/REPORT_height_biomass_2026-10-09.html`
(sections 3 and 4) and `research/2026-10-09/results_summary/reporting/`.
