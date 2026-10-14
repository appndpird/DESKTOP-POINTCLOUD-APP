# PhenoFuse 4.0 (2026-10-14)

PhenoFuse is the new name of PhenoApp ("Plant Phenotyping from Point Clouds"): the tool now handles three modalities
per plot - LiDAR point clouds, VNIR hyperspectral cubes and RGB orthomosaics - and fuses them for biomass and plant-height
models. Windows executable: `PhenoFuse_v4.0_win64.zip` (unzip anywhere, run `PhenoFuse\PhenoFuse.exe`). The internal
Python package keeps the name `phenoapp` so that models saved by v3 still load.

## Tabs (new order)

| # | Tab | What changed |
|---|-----|--------------|
| 6 | **LiDAR Features** | was "Traits". ONE table per flight, `plots_lidar/lidar_features.csv` (legacy traits + canopy-top heights + LiDAR v3 canopy features; no separate `_lidar_v3.csv`), and the per-plot LAS files in the same folder - the classified ones (2 ground / 3 low veg / 5 canopy / 7 noise, HeightAboveGround) when LiDAR v3 is on. The ground / canopy class thresholds (8 / 10 cm) are now settings and are recorded in `lidar_features_provenance.json`. |
| 7 | **VNIR Spectral Features** | was tab 11 "VNIR Spectral". Writes `plots_vnir/vnir_features.csv` (+ `vnir_spectra.csv`, band QC tables, `vnir_provenance.json`), the per-plot VNIR cubes and the catalogue index table `vnir_indices.csv`. **All 34 water indices of the Awesome Spectral Indices catalogue are listed** in the Water group; the 10 that a 400-1000 nm cube can build (NDWI, DDWI, FWEI, NDCI, NDTI, NDWIns, OSI, PI, RNDVI and TWI's VNIR part) are computable, the SWIR-based ones are shown greyed with the missing band. Added hyperspectral plant-water indices: WI/NDVI (Penuelas 1997), NWI-1..4 (Babar 2006, Prasad 2007), WBI on the 960-990 nm mean, and two diagnostics on the atmospheric bands (DWI_O2A at 760 nm, WVD_940). "Water (plant water)" selects them all. **Band policy** tick boxes: exclude the sensor edge (< 415 nm), the O2-A band (755-770 nm) and the water-vapour band (928-962 nm) individually; the choice is written to the provenance file and honoured by the band-based models. Water indices (NDWI, WI/NDVI, NWI-1, NWI-3, DDWI, NDCI, FWEI) are in the standard feature set. |
| 8 | **RGB Features** (new) | Per-plot crops of the RGB orthomosaic with the same grid: cut on the orthomosaic's own pixel grid (no resampling, same dtype / CRS / transform, lossless DEFLATE, polygon as internal mask), so the pixel values are exact copies of the orthomosaic. `plots_rgb/rgb_features.csv`: ExG, ExR, ExGR, GLI, VARI, NGRDI, MGRVI, RGBVI, CIVE, TGI, VEG, COM, GCC over all pixels and over vegetation pixels, RGB cover (ExG Otsu threshold per flight), HSV, texture, QC. |
| 9 | **Biomass Models** | tabs 7 ("Biomass & Fusion") and 9 ("Biomass ML") merged. Inputs are the three feature tables - nothing is recomputed from the bands. Two sections: **Use a pretrained model** (bundled top-three ensembles, this flight's or a dataset's `Models/` runs, or any .joblib; optional offset; optional check against a measured CSV) and **Train from scratch** (this flight or a dataset of several flights; LiDAR / VNIR / RGB in any combination; VNIR as indices, valid reflectance bands or both; the simple one-predictor linear calibrations of the old tab 7, core / fused / PCA families, every learner; ground truth in any biomass unit with fresh / dry selection; valid-band plot rules). |
| 10 | **Height Models** | was tab 8. Same two sections; models for LiDAR, VNIR, RGB and their fusions (linear calibration on cth_p95 / H95, structure models, nested-selection PLS / Ridge / GPR, PCA sets incl. RGB and reflectance bands). A missing legacy column no longer stops the tab: models whose inputs are absent are skipped with a message. |
| 11 | **Deep Models** | two sections. **Use a pretrained ensemble**: weights folder + the tensors of one flight (from this flight's LAS + cube, or from a flight of a dataset). **Train from scratch**: tensors of SEVERAL flights of a dataset (built from the per-plot classified LAS files and per-plot VNIR cubes of the dataset folder + ground_truth.csv), fine-tuning from the bundled weights by default. All 10 fold models of the three variants (two_stream, lidar_only, vnir_only) for biomass and height are bundled. |
| 12 | **Dataset** (new) | Builds `<dataset>/lidar|vnir|rgb/<flight>/plots/ + <modality>_features.csv + ground_truth.csv` with one click per flight ("Add this flight"), keeps a manifest (trial, stage, date) and lists the dataset. Training on a dataset keeps the same plot of several dates in one cross-validation fold. |

Models of every training run go to `<flight folder or dataset>/Models/Height models/<run>/` or `Models/Biomass models/<run>/`
(predictions, results, one .joblib per model, `top3_ensemble.joblib`, `run.json`). The version ships a `Models/` folder with the
latest height and biomass models (top-three ensembles, fold ensembles, two-stream fold models) organised the same way.

## Fixes
- **Wrong output folder**: picking a different LAS now moves every derived path (feature tables, per-plot folders, Models) to the
  new flight's folder; v3 kept the first flight's paths, so a second flight was written next to the first one.
- **Window resize**: every tab lives in a scroll area, so the window resizes from any edge down to 480 x 360; it opens at 90 % of
  the screen at most.
- Ground-truth CSVs saved by Excel (UTF-8 BOM) are read correctly.
- The Project tab holds the VNIR cube, the RGB orthomosaic and the ground-truth CSV of the flight, plus flight / trial / stage names.

## Questions answered in the Guidance tab (section "PhenoFuse 4.0")
DSM / DTM / CHM usage, ground classification (alley-ring plane vs SMRF / CSF / PMF), the 8 / 10 cm class thresholds, and
the water-vapour / O2-A bands for dry biomass.
