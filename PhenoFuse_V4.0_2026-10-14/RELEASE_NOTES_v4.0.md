# PhenoFuse 4.0.2 (2026-10-16) - pretrained models on a new flight

- **LightGBM bundled.** The Muresk maturity biomass ensembles (fused and LiDAR) contain a LightGBM member; 4.0.1 shipped
  without the `lightgbm` module, so loading them failed with `No module named 'lightgbm'`. The build environment now has
  lightgbm 4.6.0 (the version the research models were trained with) and the executable bundles it.
- **Band models on a flight whose usable bands differ from training.** A pretrained model that uses reflectance bands
  (`sb_<nm>` features) is now given exactly the bands it was trained on, in the same per-plot SNV normalisation, instead of
  the flight's own usable-band set. A band that is unusable on every plot of the flight (the GRYFN cube clips the blue
  bands 417-466 nm to zero on some flights) is set to the SNV mean and the Biomass / Height Models tab prints a warning
  listing the range. Before, such a flight failed with `input table lacks sb_416.7, ...`.
- **Index features that are NaN on every plot.** Some indices need a clipped band (mND705 uses 445 nm) and the VNIR tab
  writes NaN for them; a pretrained model that needs such a feature predicted no plot at all. Every model now stores the
  medians of its training table (`feature_medians`): a feature NaN on the whole flight is filled with the training median
  and the tab says which ones were filled (and names the features it could not fill, instead of silently predicting
  nothing). Models trained from scratch in the app store their medians from now on; the bundled research models were
  stamped with the medians of their research training tables (`features_<dataset>.csv`) and given a `meta` block
  (target, modalities, dataset, flight), so the tab's model description no longer shows `?`.
- **Per-flight spectral PCA scores are no longer model inputs.** `spec_pc1..5` from the VNIR tab are PCA scores of that
  flight's spectra: their sign and scale differ between extraction runs (the same flight re-extracted gave r = -0.99),
  which made the research VNIR biomass ensemble predict ~22 t/ha on a 9.5 t/ha flight. They stay in the tables for
  within-flight analysis, the training families skip them, and a pretrained model that has them gets its training median
  with a note in the output.
- **Training-range check.** Every model also stores the 5-95 % range of each training feature (`feature_ranges`). When
  the flight's median of an input feature falls outside it, the tab says so with examples (e.g. the maturity VNIR
  biomass ensemble applied to the 30 Sep anthesis flight: CIRE_veg 2.87 vs training 0.33-0.61) and labels the
  predictions as extrapolations. That model predicts ~20 t/ha on any anthesis table, including the research team's
  own, because the canopy is green at anthesis; it is a growth-stage mismatch, not an extraction difference (218 of
  265 shared features are identical between the app's tables and the research tables of the same flight).
- **Raw reflectance bands as model input.** Next to "VNIR input" (indices / bands / both) the training panels now have
  "Band values": `snv` (log10 reflectance + per-plot SNV, the default, illumination-robust) or `raw` (the mean
  vegetation reflectance of `vnir_spectra.csv` as measured, keeps the absolute level). Every saved model records its
  choice (`band_transform`) and prediction rebuilds the bands the same way; the pretrained-model description shows it.
- **Use most of the features, not only the curated cores.** Biomass Models: new families `LiDAR_all`, `VNIR_all`, `RGB_all`,
  `Fused_all` (LiDAR+VNIR), `LiDAR+RGB_all`, `VNIR+RGB_all`, `Fused3_all` = every usable column of those modalities (on the
  Muresk flight: 110 LiDAR, 152 VNIR, 53 RGB columns, plus the bands per the VNIR mode). Height Models: `*_all_*` models
  (LiDAR / VNIR / RGB / fused, Ridge / PLS / RF / GPR) on the same full column sets. "Features kept" is now on both panels
  and accepts 0 = all (no selection), so the learner sees every feature; otherwise the random-forest top-k selection runs
  inside each fold as before. The PCA families remain the compressed way to use everything.
- The false "input feature(s) missing" warning for band features is gone (the bands come from `vnir_spectra.csv`,
  not from the feature table).

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
