## PhenoFuse 4.0 (2026-10-14): three modalities, datasets, pretrained / from-scratch modelling

PhenoFuse is PhenoApp renamed: one plot grid, three sensors (LiDAR point cloud, VNIR hyperspectral cube, RGB orthomosaic),
one feature table per modality per flight, and biomass / plant-height models on any of them or on their fusion.

### Per-flight outputs (next to the LAS)

| Folder | Files | Written by |
|---|---|---|
| `plots_lidar/` | `plot_<id>_<name>.las` (classified: 2 ground / 3 low veg / 5 canopy / 7 noise, HeightAboveGround) + **`lidar_features.csv`** (one table: legacy traits, canopy-top `cth_*`, LiDAR v3 canopy features) + `lidar_features_provenance.json` | 6. LiDAR Features |
| `plots_vnir/` | `plot_<id>_<name>.tif` (all bands, exact copies) + **`vnir_features.csv`** + `vnir_spectra.csv` + `vnir_band_valid_fraction.csv` + `vnir_band_usable.csv` + `vnir_provenance.json` (band policy) + `vnir_indices.csv` (catalogue indices) | 7. VNIR Spectral Features |
| `plots_rgb/` | `plot_<id>_<name>.tif` (exact pixel copies of the orthomosaic) + **`rgb_features.csv`** + `plots_rgb_index.csv` | 8. RGB Features |
| `Models/Height models/<run>/`, `Models/Biomass models/<run>/` | `predictions.csv`, `results.json`, `run.json`, one `.joblib` per model, `top3_ensemble.joblib` | 9 / 10 (train from scratch), 11 (two-stream runs) |

Picking another LAS on the Project tab moves all of these to the new flight's folder.

### Dataset folder (tab 12)

```
<dataset>/
  dataset.json                      flights: trial, stage, date; modalities present
  lidar/<flight>/plots/*.las        lidar_features.csv   ground_truth.csv
  vnir/<flight>/plots/*.tif         vnir_features.csv    vnir_spectra.csv  vnir_band_*.csv  vnir_provenance.json  ground_truth.csv
  rgb/<flight>/plots/*.tif          rgb_features.csv     ground_truth.csv
  ground_truth/<flight>.csv
  Models/Height models/…  Models/Biomass models/…
  dl_data/                          two-stream tensors of the flights (index.csv)
```
"Add this flight" on the Dataset tab copies the current flight's tables and per-plot files for the ticked modalities. The
modelling tabs then train on any selection of flights and modalities. The same plot measured on several dates gets the
group id `<trial>:<Plot_ID>`, so it stays in one cross-validation fold (otherwise its twin from another date leaks into
training and the held-out score is optimistic).

### Modelling tabs: two sections each

* **Use a pretrained model / ensemble** - pick a `.joblib` (bundled top-three ensembles, a run under `Models/`, or any file),
  point it at the feature tables of a flight (this flight or a flight of a dataset), optional offset, optional check against
  a measured CSV, predict every plot. Deep Models: a weights folder + the plot tensors of one flight.
* **Train, validate and test from scratch** - LiDAR / VNIR / RGB in any combination; VNIR input as spectral indices, valid
  reflectance bands (log10, SNV, bands clipped on > 5 % of the plots and the excluded ranges left out) or both; the simple
  one-predictor linear calibrations (H95, mean height, PVI, canopy volume, NDRE, WBI, RGB ExG, cover ...), the core
  families with nested feature selection, the fused families with CHM-weighted indices, the PCA families (one PCA per
  modality, 95 % of the variance, fitted inside each fold); every learner; ground truth in any biomass unit (kg/ha, t/ha,
  g/m2, kg/m2, kg or g per plot) with a fresh / dry selector; repeated 10-fold, 5-fold, LOOCV or fit-only; the valid-band
  plot rules. Height models: linear calibration on cth_p95 or H95, structure models, nested-selection PLS / Ridge / GPR / RF,
  PCA sets per modality and fusion. Every run saves its models and a `top3_ensemble.joblib` (mean of the three best by
  held-out RMSE, refitted on all plots) under `Models/`.
* **Deep Models** train on the tensors of several flights at once (built from the dataset's per-plot classified LAS files
  and per-plot VNIR cubes), warm-started from the bundled weights by default. All 10 fold models of the three variants for
  biomass and height are bundled (`assets/dl_models`, common band policy).

### Water indices (tab 7)
The Water group lists all 34 water indices of the Awesome Spectral Indices catalogue (Montero et al. 2023). A 400-1000 nm
VNIR cube can build NDWI (G, N), DDWI, FWEI, NDCI, NDTI, NDWIns, OSI, PI and RNDVI; ANDWI, AWEI, MNDWI, LSWI, MBWI, MuWIR,
NWI, S2WI, SWM, WI1/2, WI2015, WRI and the rest need SWIR bands (S1 1550-1750 nm, S2 2080-2350 nm) and are listed greyed
with the missing band. PhenoFuse adds the hyperspectral plant-water indices that do work on VNIR: WBI (R900/R970), NDWI970,
WI/NDVI (Penuelas et al. 1997), NWI-1..4 (Babar et al. 2006; Prasad et al. 2007), WBI on the 960-990 nm mean, and two
diagnostics of the atmospheric bands (DWI_O2A at 760 nm, WVD_940). "Water (plant water)" ticks all of them; NDWI, WI/NDVI,
NWI-1, NWI-3, DDWI, NDCI and FWEI are in the standard feature set of `vnir_features.csv`.

### Water-vapour and O2-A bands for dry biomass
The O2-A feature at 760 nm is an absorption by atmospheric oxygen; its residual after reflectance conversion depends on
flight height, sun angle and the calibration panel, not on the plant, so it is excluded by default. The 930-960 nm band is
atmospheric water vapour, which overlaps the leaf-water absorption around 970 nm; the plant signal is retained through the
970 nm indices (WBI, NWI) that sit just outside the excluded range. Dry-matter content is a ratio of dry to fresh weight;
canopy water content (fresh minus dry) does leave a spectral signature at 970 nm (and more strongly at 1200 / 1450 nm,
outside this sensor), so WBI / NWI are legitimate predictors of fresh biomass and of the water fraction - and through them
of dry biomass only indirectly. Both ranges can now be kept (Band policy tick boxes): untick them, recompute the VNIR
features, and compare the held-out scores on a different flight; a gain that does not transfer between flights is the
atmosphere, not the crop.

### DSM, DTM, CHM
Yes, the tool builds them (LiDAR Features tab, "Export DSM / DTM / CHM"): the DSM is the maximum z per cell, the DTM comes
from the exterior ground surface (alleys), from the ground inside each plot, from SMRF or from an external GeoTIFF, and the
CHM = DSM - DTM; zonal CHM statistics per plot are written as well. The per-plot features themselves do not go through a
raster: heights are computed per point against a plot-local ground plane (LiDAR v3) or the trial-wide exterior surface, which
avoids the raster-resolution and interpolation artefacts of a CHM; the canopy-top metrics (`cth_*`) use a 5 cm top-surface
grid of the plot's own points.

### Ground classification: alley-ring plane vs SMRF, CSF, PMF
The LiDAR v3 route fits a robust plane through the 5th-percentile cells of a 0.25-1.0 m ring around each plot (the alley,
neighbours excluded) and refines it with in-plot points within the ground band. Classical filters (PMF, Zhang et al. 2003;
SMRF, Pingel et al. 2013; CSF, Zhang et al. 2016) look for the lowest local surface inside a window: on a dense closed
canopy where the laser rarely reaches the soil inside the plot they ride along the canopy bottom and compress heights
(on a dense pasture trial SMRF flipped the LiDAR-biomass relationship; the exterior surface doubled the LOOCV R2). The
alley plane uses only returns that are genuinely ground, which is why the Muresk maturity validation reached a ruler
correlation of 0.70 (0.31 with the trial-wide surface) and a 3.4-3.8 cm RMSE at the ruler noise floor. The limits: it needs
visible alleys (fallback: in-plot 1st percentile), and it is a plane per plot, so within-plot micro-relief is not modelled;
SMRF / CSF remain the right choice for terrain with slope breaks inside plots, and the DTM export offers SMRF for that.
Literature on crops using the same idea (ground from bare strips / plot edges, canopy percentiles from normalised points):
Jimenez-Berni et al. 2018 (Front. Plant Sci., wheat, LiDAR Phenomobile), ten Harkel et al. 2020 (Remote Sens., UAV LiDAR
crop height), Walter et al. 2019 (Plant Methods).

### The 8 cm / 10 cm class thresholds
Ground = |h| <= 8 cm, low vegetation = 8-10 cm, canopy = h > 10 cm, with h the height above the plot-local plane. They are
empirical: the plane-fit RMS on the validation flights was 2-4 cm, so 8 cm (about two RMS of the worst case) keeps ground
noise and stubble in class 2; 10 cm is the vegetation cutoff used by most UAV-LiDAR crop studies (height-above-ground
thresholds of 0.1 m are common, e.g. ten Harkel et al. 2020), and on the ruler validation the canopy percentiles H95 / H99 and
`cth_p95` changed by < 1 cm when the cutoff was moved between 5 and 15 cm. They are not a published standard; the LiDAR
Features tab now exposes both ("Class thresholds") and records them in the provenance file. Raise them for noisy ground or
tall stubble, lower them for short crops at early stages, and keep them identical across the flights of one dataset.

### Window resize
Every tab sits in a scroll area, so the window can be dragged from any edge down to 480 x 360 pixels; it opens at 90 % of
the screen at most.

### Deep Models on an extracted flight (4.0.2)

The pretrained deep ensembles need plot tensors, and the Deep Models tab can build them from three sources: *This flight*
(project LAS + VNIR cube + grid, cut now), *Flight folder* (the per-plot files already written by the LiDAR Features and
VNIR Spectral Features tabs: `plots_lidar\plot_<id>_*_classified.las` and `plots_vnir\plot_<id>_*.tif`), or a *Dataset
folder*. For a flight that has been through tabs 6 and 7, pick *Flight folder*: the folder is guessed from the ground-truth
CSV or the project folder, the flight name from the folder name, and the tensors go to `<flight>\dl_data`. *This flight*
switches to the folder by itself when no project LAS is loaded. Variant `lidar_only` needs only the LAS files,
`vnir_only` only the cubes. The "Python with torch" list offers the interpreters on this machine that have torch
installed (those with spconv first); press "Probe selected" to see the torch version and GPU. Predictions land in
`<flight>\Models\predictions\<flight>_<target>_<variant>_twostream_<weights>.csv` with the per-fold spread and, when a
ground-truth CSV was given, the measured values and the R2 / RMSE in the log.

### Pretrained models on a new flight (4.0.2)

A bundled model was trained on one trial, flight and growth stage (its description names them). Before trusting its
numbers on a new flight, read the notes under the prediction:

- **Bands filled** - the model uses reflectance bands that are unusable on every plot of this flight (the GRYFN cube
  clips the blue bands 417-466 nm to zero on some flights). They are set to the SNV mean; predictions lean on the rest.
- **Features filled with the training median** - an index that needs a clipped band (mND705 uses 445 nm) is NaN on the
  whole flight; the model's stored training median is used instead.
- **Spectral PCA scores replaced** - `spec_pc1..5` are per-flight PCA scores whose sign and scale differ between
  extraction runs; they are never transferable and are not used when training in the app.
- **Out of training range** - the flight's median of several inputs lies outside the model's training 5-95 % range.
  That is the growth-stage signature: a maturity model applied to an anthesis flight sees a green canopy (CIRE, NDRE,
  OSAVI several times higher) and extrapolates (the Muresk maturity VNIR ensemble predicts ~20 t/ha at anthesis).
  Use a model trained at the same stage, or train from scratch on this flight / a dataset of similar flights.
- **Curated core, all features, or PCA?** Three ways to feed a modality: the *_core families (12-30 curated
  features, the research setting), the *_all families / *_all_* height models (every usable column: ~110 LiDAR, ~150
  VNIR, ~50 RGB, plus bands) with "features kept" = top-k per fold or 0 = all, and the PCA families (every column
  compressed to the components holding 95 % of the variance). With 128 plots, all-features Ridge / PLS usually match
  the cores and random forest likes the extra columns; compare on the same validation and keep the simplest model
  that wins.
- **Raw bands or SNV bands?** "Band values" on the training panels: `snv` removes each plot's overall brightness
  (log10 + standard normal variate), so a model transfers better between flights and sun angles; `raw` keeps the
  absolute reflectance, which can add cover / biomass signal within one flight but drifts between flights. Train both
  with the same validation and compare the held-out RMSE; a saved model remembers its choice.
- Keep the LiDAR preprocessing the same as the training run: the research models were trained with per-plot noise
  removal on (rule `sor`); with the noise rule set to `gap` nothing is removed and the density features shift a little.

