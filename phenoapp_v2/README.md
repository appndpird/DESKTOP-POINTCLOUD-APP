# Plant Phenotyping from Point Clouds

Desktop application for extracting plot polygons and per-plot physical traits
from UAV LiDAR point clouds.

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
