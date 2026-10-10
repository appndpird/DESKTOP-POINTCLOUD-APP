# PhenoApp v3.3.2 (2026-10-10)

Windows executable: `PhenoApp_v3.3.2_win64.zip` (release asset; unzip anywhere and run `PhenoApp_v3\PhenoApp_v3.exe`).
No Python is needed for the LiDAR, VNIR, Height Models and Biomass ML tabs. The Deep Models tab runs the network in a
separate Python environment with torch and spconv (see the Guidance tab, "Using the pretrained deep models"), which is
why torch is not inside the executable.

## What is in this version
- **LiDAR v3.2 per-plot pipeline**: gap noise rule (default), plot-local ground plane, classification 2 / 3 / 5 / 7,
  HeightAboveGround, interior gap metrics, classified per-plot LAS with sidecar JSON.
- **VNIR v3.0.1 QC**: exact per-plot cubes, flight-level reflectance check, NaN rule for unusable bands and indices,
  red-clipping censoring, band usable / valid-fraction tables, spectra tables.
- **Feature models**: Height Models tab (calibration, 7-feature models, LiDAR core, LiDAR + VNIR, PCA sets) and
  Biomass ML tab (core families, Fused_all, PCA families incl. spectral-band PCA), nested repeated 10-fold validation,
  saved models applicable to another flight with an offset.
- **Deep Models tab**: two-stream network (sparse 3D CNN + spectral transformer), bundled common-band ensembles
  (two_stream and lidar_only, biomass and height, 5 folds each, 100 epochs), fine-tune (warm start) option, band policy
  handled by the runner, stage-averaged embeddings for new datasets.
- **Guidance tab**: user guide for training, testing and inference; the input list of every model; deep-model walkthrough.
- No trial or dataset names in the tool.

## Validation (held out, 9-10 October 2026 rebuild)
Height: joint PCA ridge R2 0.59 / 3.3 cm (anthesis), 0.55 / 3.4 cm (maturity); network 3.7 / 4.1 cm.
Biomass: fused PCA R2 0.37 / 1,513 kg/ha (anthesis), 0.27 / 1,237 (maturity) on the trial with ruler heights;
network 0.27 / 1,636 and 0.13 / 1,340. Details: research/2026-10-09/REPORT_height_biomass_2026-10-09.html.

## Source and models
Tool source: `phenoapp_v3.3_2026-10-09/` (build with `build_windows.bat` or `pyinstaller phenoapp.spec`).
Research scripts, saved feature models and all two-stream fold models: `research/2026-10-09/`.
