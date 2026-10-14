# Models shipped with PhenoFuse 4.0 (2026-10-14)

Height-wise and biomass-wise, the same layout the tool creates for its own training runs
(`<flight folder or dataset>/Models/Height models/<run>/`, `Models/Biomass models/<run>/`).

## Height models/
Feature models of the Muresk NUE trial, maturity flight 2025-11-21 (the only flight with a ruler measurement; cm):
- `height_muresk_maturity_{lidar,vnir,fused}_top3_ensemble.joblib` - the final top-three mean ensembles per category
  (members chosen by nested repeated 10-fold selection and refitted on all 128 plots); these are also bundled in the app
  (`phenoapp/assets/pretrained/Height models/`) and appear as "[bundled]" in the Height Models tab.
- `height_muresk_maturity_{lidar,vnir,fused}_fold_ensembles.joblib` - one top-three ensemble per cross-validation fold with its
  test score (use "all fold models" for an honest prediction; "best three folds" is optimistic).
- `height_<family>_<learner>_2025-11-21.joblib` - the best learner of every feature family; `best_models.csv` lists R2, RMSE,
  accuracy and the input features of each file. Reference: LiDAR+VNIR joint PCA Ridge R2 0.55, RMSE 3.4 cm (ruler noise
  floor about 3.8 cm).
- `two_stream/` - the two-stream network fold models for height (`two_stream_height_fold0-9.pt`, `lidar_only_...`,
  `vnir_only_...`), common band policy (`common_bands.csv`, 111 bands), `fold_scores.csv`. Use the folder as the weights
  folder of the Deep Models tab.

## Biomass models/
- `biomass_{muresk_anthesis,muresk_maturity,agt_anthesis,agt_maturity}_{lidar,vnir,fused}_top3_ensemble.joblib` - final
  top-three mean ensembles per trial x stage x category (kg/ha dry matter); `__ruleA` / `__ruleB` = trained on valid-band plots
  only. Bundled in the app under `phenoapp/assets/pretrained/Biomass models/`.
- `best_models.csv`, `summary_best_per_family.csv`, `pca_components.csv` - scores and inputs of every feature family.
- `two_stream/` - the network fold models for biomass (three variants x 10 folds), `common_bands.csv`, `fold_scores.csv`.
- The per-fold feature ensembles (`*_fold_ensembles.joblib`, 300 MB) and the single-family files are in
  `research/2026-10-09/models/biomass/` of the repository.

## How to use
Height / Biomass Models tab -> "Use a pretrained model" -> pick the file (the bundled ones are listed) -> point it at the
feature tables of the flight (this flight or a flight of a dataset) -> Predict. Models transfer between dates and trials only
with an offset calibrated on a few measured plots; validate against a measurement taken on the day of the flight.
Deep Models tab -> "Use a pretrained ensemble" -> weights folder = `Models/<kind>/two_stream` -> prepare the flight's tensors
-> Predict.
