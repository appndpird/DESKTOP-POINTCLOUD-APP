# PhenoApp v3.3.3 (2026-10-12)

Windows executable: `PhenoApp_v3.3.3_win64.zip` (unzip anywhere and run `PhenoApp_v3\PhenoApp_v3.exe`). No Python is needed
for the LiDAR, VNIR, Height Models and Biomass ML tabs. The Deep Models tab runs the network in a separate Python
environment with torch and spconv (Guidance tab, "Using the pretrained deep models").
`PhenoApp_v3.3.3_twostream_weights.zip`: all fold models (float16) of the final two-stream runs; each folder works as a
weights folder of the Deep Models tab.

## New in 3.3.3
- **Top-three ensembles.** The "Apply a saved model" panels of the Height Models and Biomass ML tabs accept mean ensembles:
  the three best models of a category (LiDAR only, VNIR only, fused), chosen by nested cross-validation and averaged.
  Ready-made ensembles for height and biomass: `research/2026-10-09/models/<target>/*_top3_ensemble.joblib`.
- **Network ensembles retrained with folds grouped by plot** (both flights of a plot always in the same fold); the height
  ensemble uses only the flight with a ruler measurement.
- **Height reporting rule:** error metrics only for a flight with a height measured on that flight; other flights get a
  predicted height and the difference between dates, never an error score.
- Fixed: the Biomass ML apply panel; applying models with spectral-band inputs now merges the spectra table.

## Validation (nested top three per category; R2 / RMSE)
| Target, dataset | LiDAR only | VNIR only | Fused |
|---|---|---|---|
| Biomass, wheat trial A anthesis | 0.34 / 1,553 kg/ha | 0.25 / 1,656 | 0.37 / 1,516 |
| Biomass, trial A maturity | 0.18 / 1,311 | 0.09 / 1,378 | 0.21 / 1,283 |
| Biomass, trial B anthesis | 0.13 / 1,418 | 0.05 / 1,485 | 0.17 / 1,385 |
| Biomass, trial B maturity | 0.05 / 1,849 | - | - |
| Height, trial A maturity | 0.40 / 4.0 cm | 0.17 / 4.6 cm | 0.53 / 3.5 cm |

Two-stream network (grouped folds): biomass 0.24 / 0.12 / 0.05 / 0.03 R2 on the same four datasets; height 4.0 cm
(LiDAR-only 3.8 cm). Full report: `research/2026-10-09/REPORT_height_biomass_2026-10-09.html`.

## Source and models
Tool source: `phenoapp_v3.3_2026-10-09/` (build with `build_windows.bat` or `pyinstaller phenoapp.spec`).
Research scripts, results and saved models: `research/2026-10-09/`.
