# PhenoFuse 4.0 (2026-10-14) - multi-sensor plot phenotyping

**PhenoFuse** (formerly PhenoApp, "Plant Phenotyping from Point Clouds") extracts per-plot features from three UAV
modalities - **LiDAR** point clouds, **VNIR** hyperspectral cubes and **RGB** orthomosaics - over one plot grid, and fits
biomass and plant-height models on any modality or on their fusion, with pretrained models for new trials and
training from scratch on datasets of several flights. Desktop application (PyQt5), Windows executable
`PhenoFuse_v4.0_win64.zip` (unzip, run `PhenoFuse\PhenoFuse.exe`).

Developed by **Dr. Muhammad Ibrahim**, Research Scientist, DPIRD, under the supervision of **Dr. Hammad Khan**,
Senior Research Scientist, DPIRD, APPN Director for DPIRD node.

## Tabs

| # | Tab | Purpose |
|---|-----|---------|
| 1 | Project | LAS, grid, canopy raster, VNIR cube, RGB orthomosaic, ground-truth CSV, flight / trial / stage names. Picking another LAS moves every output to the new flight's folder. |
| 2 | Point Cloud QC | LAS checks (lasinfo / lasvalidate), GCP comparison |
| 3 | Generate Grid | plot grid from four corners |
| 4 | Edit | align / refine the grid on the canopy or the orthomosaic |
| 5 | Visualize | 2D / 3D plot viewer |
| 6 | LiDAR Features | per-plot LAS (classified) + `plots_lidar/lidar_features.csv`: legacy traits, canopy-top heights, LiDAR v3 canopy features; class thresholds exposed; DSM / DTM / CHM export |
| 7 | VNIR Spectral Features | `plots_vnir/vnir_features.csv` (+ spectra, band QC, provenance), per-plot cubes, the Awesome Spectral Indices catalogue incl. **all water indices**, band policy (sensor edge / O2-A / water vapour), band viewer |
| 8 | RGB Features | per-plot orthomosaic crops (exact pixel copies) + `plots_rgb/rgb_features.csv` (ExG, GLI, VARI, NGRDI, MGRVI, RGBVI, TGI, ... cover, HSV, texture) |
| 9 | Biomass Models | **Use a pretrained model** / **Train from scratch**: LiDAR, VNIR, RGB or fused; VNIR as indices, bands or both; simple linear calibrations, core, fused and PCA families, all learners; any ground-truth unit; nested CV |
| 10 | Height Models | same two sections for plant height (linear calibration on cth_p95 / H95, structure, nested-selection and PCA models) |
| 11 | Deep Models | two-stream network (sparse 3D CNN + spectral transformer) in an external torch environment: pretrained ensembles on a flight, or training on several flights of a dataset |
| 12 | Dataset | build / inspect `<dataset>/lidar|vnir|rgb/<flight>/` with per-plot files, feature tables and ground truth |
| 13 | Statistics | distributions, heatmaps, correlations of any per-plot CSV |
| 14 | Guidance | the embedded manual (methods, the 4.0 chapter, answers on DSM/DTM, ground classification, thresholds, water bands) |

`Models/` in this folder holds the latest height and biomass models (top-three ensembles, fold ensembles, two-stream fold
models) - see `Models/README.md`. The same models are bundled in the executable (`phenoapp/assets/pretrained`,
`phenoapp/assets/dl_models`).

## Run from source

```bash
conda create -n phenoapp python=3.11 -y
conda activate phenoapp
conda install -c conda-forge pdal python-pdal gdal geopandas shapely rasterio fiona laspy lazrs pyproj scikit-learn joblib -y
pip install PyQt5 pyvista pyvistaqt scipy matplotlib pandas numpy xgboost lightgbm
cd PhenoFuse_V4.0_2026-10-14
python run.py
```
The Deep Models tab needs a second environment with torch (+ spconv for the LiDAR stream); the tab finds it automatically.

## Build the executable

`build_windows.bat` (PyInstaller, `phenoapp.spec`; set `ENVROOT` to the conda env that has PyQt5 + pyinstaller). The Python
package keeps the name `phenoapp` so that models saved by v3 still load.

## Release notes
`RELEASE_NOTES_v4.0.md` (this version), `RELEASE_NOTES_v3.3.*.md` (previous). User recipe: `USER_GUIDE.md`.
