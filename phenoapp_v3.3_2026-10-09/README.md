# Plant Phenotyping from Point Clouds

**v3.3 (2026-10-09) - combined release:** the v3.2 LiDAR update (noise rule, interior gap metrics, labelled per-plot LAS)
and the v3.0.1 VNIR QC update (exact per-plot cubes, NaN rule for unusable bands and indices, flight-level reflectance
check, red-clipping censoring, band QC tables) in one tree, verified together on Muresk NUE 2025 (both flights) and
AGT NUE 2025 (three flights): every per-plot VNIR cube pixel-exact against its orthomosaic, LiDAR features identical to
the labelled-plot run. Research pipeline used for the 9 October rebuild of the height and biomass models (dataset build
without fiona/rasterio, PCA feature families, two-stream tensors with the per-plot band mask) is in `research/2026-10-09/`.

**v3.2 (2026-10-09) - LiDAR noise rule, interior gap metrics, labelled per-plot LAS (validated on Muresk NUE 2025-09-30):**
the Traits tab gains a **Noise rule** selector (`gap` = points > 30 cm above the canopy, now the default; `none`;
`legacy` upper-canopy SOR; `sor` k=10 at 3 or 5 sd) and a **Drop flagged noise points** box (default off: flagged
points stay in the classified LAS as class 7 with the LAS *withheld* bit, so every reader can skip or recover them).
Against 128 ruler heights the SOR variants removed sparse heads/awns and lowered the cth_p95 agreement (r 0.70 / LOOCV
3.70 cm unfiltered, 0.68 / 3.79 cm at 5 sd, 0.66 / 3.88 cm at 3 sd), while the gap rule flagged one point in the trial.
LiDAR v3 features add the 20 cm-interior gap metrics `Pgap_interior`, `LAI_proxy_interior`, `cover_2cm_interior`,
`ground_visible_2cm_interior` (+ `cover_2cm`, `ground_visible_2cm`, `ground_edge_frac`, `Pgap_interior_below_*`): refit
polygons include alley soil along the long sides (68 % of ground returns within 20 cm of the edge; Pgap 8.3 % whole vs
3.5 % interior), and at 30,000 pts/m2 `cover_5cm` saturates at 1.0, so `LIDAR_V3_CORE` now uses the interior / 2 cm
versions (22 features). Density layers `dens_1..10` now start at the 10 cm canopy threshold (`dens_1` was always 0).
The classified per-plot LAS gains a `HeightAboveGround` extra dimension and a sidecar JSON (ground plane, counts, rule).
Core API: `lidar_features.noise_mask`, `preprocess_plot(..., rule=, top_gap=)` returns `h_all`,
`lidar_features_all(..., noise_rule=, nsig=, drop_noise=)`, `extract_all_plots(..., noise_rule=, noise_nsig=, drop_noise=)`.
Originals of the changed files are in `_backup_before_v3.2_2026-10-09/`.

**v3.0.1 (2026-10-08) - QC fixes after the plot-level VNIR audit (ENVI, all plots of Muresk 2025 and AGT 2025):**
per-plot VNIR writer no longer zero-pads a plot whose window ends on the last row/column of the union read window
(5 polygon pixels were lost in 2 of 1,785 AGT plots; every other plot was bit-exact); VNIR v3 features and the
deep-model tensors read the cube in its own dtype instead of casting to uint16 (float32 cubes were truncated);
VNIR v3 adds a flight-level reflectance check (`vnir_reflectance_ok`; a radiance/DN cube such as AGT 2025-11-13
now fails `vnir_qc_ok` instead of passing), censors red-based indices to NaN on red-clipped plots (`red_ok = 0`),
and keeps the excluded O2-A bands out of the red-edge derivative; the Biomass ML tab now excludes `vnir_qc_ok = 0`
plots from every family that uses VNIR features, not only from VNIR_core.
**NaN rule for every per-plot band table (`MIN_VALID_FRAC = 0.5` in `spectral_indices.py`):** a band is written as NaN
for a plot when it is valid on fewer than half of the plot's pixels (clipped reflectance), when it lies in an excluded
wavelength range, or when the whole cube fails the reflectance check; indices built from such bands are NaN too. Applies
to the Biomass-tab spectra (`<spectra>.npz`, new `_spectra.csv` and `_band_valid_fraction.csv`), the VNIR index table,
the Spectral-tab index tables and the VNIR v3 features and spectra (new `_vnir_v3_band_valid_fraction.csv`), so a value
that should not be analysed never appears as a number. Deep-model `index.csv` gains `vnir_reflectance_ok`.
Each spectra table also gets a `*_band_usable.csv` companion (`band_usable()` in `spectral_indices.py`): one row per
plot, one column per band, 1 = usable / 0 = the spectra cell is NaN, plus `n_usable_bands`; the trial workbooks built by
`Biomass Dataset\build_plot_traits_indices.py` carry it as `<flight>_Bands_QC` and `<flight>_Bands_ValidFrac` sheets.

**v3.0 (2026-10-06)** - LiDAR v3 preprocessing + point classification (noise / ground / low veg / canopy) with
LiDAR360-style canopy-only features, cleaned VNIR v3 features (soil mask, veg-only indices, spectral shape, PCA),
Height Models v3 sets (LiDAR core, LiDAR+VNIR; PLS / ridge / GPR / RF with nested selection, repeated 10-fold),
new Biomass ML tab (fused features, 8 learners, nested CV, saved models), and (v3.1) a Deep Models tab running the
two-stream network (sparse 3D CNN + spectral transformer) in an external torch environment with automatic
GPU/CPU selection and bundled pretrained fold ensembles for biomass and height. Build name PhenoApp_v3. See
`phenoapp/docs/guidance.md`, section "PhenoApp v3".

Desktop application for extracting plot polygons and per-plot physical traits
from UAV LiDAR point clouds.

**VNIR Spectral tab (added 2026-10-01):** tab 8 computes any of ~200 spectral
indices from the bundled Awesome Spectral Indices catalogue
(`phenoapp/assets/spectral_indices/`, MIT) plus PhenoApp narrow-band extras,
grouped by category with a checkbox per index, per plot (per-pixel or on the
mean spectrum), with QC (sd, valid-pixel fraction) and a provenance JSON;
optional index GeoTIFFs. Its View sub-tab shows the cube, selected plots or
per-plot files as composites, single bands or indices with click-to-spectrum.
Per-plot VNIR cubes get a QGIS `.qml` sidecar. The Visualize tab gained a
per-plot LiDAR 3D viewer. **v2.2:** tab 8 "Height Models" trains and saves plant-height models
(cth_p95 calibration, ridge / random forest / XGBoost on 7 structure features) and applies them to
new flights. See `phenoapp/docs/guidance.md`.

Developed by **Dr. Muhammad Ibrahim**, Research Scientist, DPIRD,
under the supervision of **Dr. Hammad Khan**, Senior Research Scientist, DPIRD,
APPN Director for DPIRD node.

---

## Features

8 tabs covering the full phenotyping workflow:

1. **Project** — load LAS, grid (.shp/.gpkg/.geojson), optional pre-built canopy raster.
2. **Generate Grid** — create a grid by clicking 4 corners + setting plot dimensions.
3. **Edit** — drag/rotate/scale plot polygons over the canopy with mouse-wheel zoom.
4. **Visualize** — 2D top-down view always; 3D PyVista viewer optional.
5. **Traits** — compute any subset of 23 physical traits per plot.
6. **Statistics** — automatic distribution plots, field heatmaps, bank box-plots, correlation matrix.
7. **Guidance** — embedded reference for every trait + computation methods.
8. **About** — credits with DPIRD and APPN logos.

Caches:
- canopy raster (.tif) cached next to LAS
- ground-classified normalised LAS (`<name>.smrf_norm.las`) cached if SMRF is enabled

---

## Run from source

```bash
# Create a clean conda environment
conda create -n phenoapp python=3.11 -y
conda activate phenoapp

# Geospatial stack — conda-forge is essential on Windows
conda install -c conda-forge pdal python-pdal gdal geopandas shapely \
    rasterio fiona laspy lazrs pyproj -y

# Other deps
pip install PyQt5 pyvista pyvistaqt scipy matplotlib pandas numpy \
            "shapely>=2.0"

# Optional: faster point cloud loading
pip install lazrs
```

Then:

```bash
cd plant_phenotyping_app
python run.py
```

## Build a standalone executable

PyInstaller produces a folder containing `PhenoApp.exe` (Windows) or
`PhenoApp` (macOS/Linux) plus all needed dependencies. No Python install
required for end users.

```bash
pip install pyinstaller
cd plant_phenotyping_app
pyinstaller phenoapp.spec --noconfirm
# Output: dist/PhenoApp/
# Run: dist/PhenoApp/PhenoApp     (or PhenoApp.exe on Windows)
```

The resulting folder is portable — copy it to any machine of the same OS.

---

## Folder layout

```
plant_phenotyping_app/
  run.py                  <- entry point
  phenoapp.spec           <- PyInstaller config
  README.md
  phenoapp/
    app.py                <- main window
    __init__.py
    core/                 <- UI-agnostic engine
      las_manager.py        LAS load + SMRF caching
      canopy_raster.py      Top-down density GeoTIFF
      grid_io.py            shp / gpkg / geojson read+write
      grid_gen.py           4-corner grid generation
      traits.py             23 traits with method docs
      extract.py            Per-plot clipping + traits
      project.py            Project state save/load
    ui/                   <- PyQt5 widgets
      project_tab.py
      generate_grid_tab.py
      edit_tab.py
      visualize_tab.py
      traits_tab.py
      statistics_tab.py
      guidance_tab.py
      about_tab.py
      canopy_view.py        Shared zoom/pan canvas
    assets/
      dpird_logo.png
      appn_logo.png
    docs/
      guidance.md         <- shown in Guidance tab
```

---

## Workflow tutorial

1. **First-time setup**: launch app, go to **Project** tab.
2. Click **Browse** next to "Point cloud" → pick your `.las` / `.laz`.
3. Either:
   - **Browse** for an existing grid file, or
   - Click **No grid yet — generate** to use the Generate Grid tab.
4. Tick **Use proper ground classification (SMRF)** for accurate heights.
   This runs once; the result is cached.
5. Click **Load project**. The canopy raster is built (also cached).
6. Switch to **Generate Grid** if needed. Click 4 corners (BL→BR→TR→TL),
   set banks/rows/dimensions, **Preview**, then **Save**.
7. Switch to **Edit**, click **Reload from Project**.
   Drag/rotate/scale plots until they sit cleanly over the canopy.
   Click **Save Grid** when aligned.
8. Switch to **Traits**, pick the metrics you want (defaults are sensible),
   click **Compute Traits**.
9. Switch to **Statistics** to view graphs.

---

## Default traits computed

- Plant height: h_max, h_mean, h_median, **h_p95**, h_p99, h_std, h_p99_p50, skewness, kurtosis
- Cover: cover_frac, point density, vertical profile, gap fraction
- Volume: voxel volume, convex hull, alpha-shape, surface area
- Shape: canopy extent, lodging angle, roughness, row count
- Counts: n_points, n_canopy

See the **Guidance** tab in the app for full definitions and methods.

---

## Web version

A Flask + Leaflet web version is available separately. The desktop and web
versions share the same `core/` engine; only the UI layer differs.
