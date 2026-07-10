# User Guide — Plant Phenotyping from Point Clouds

This app extracts physical phenotypic traits from a 3-D point cloud of a plot
trial. It is designed for UAV LiDAR data of breeding-style trials with regular
plot grids.

---

## Workflow at a glance

1. **Project** — load the LAS, the grid (or generate one), and optionally a pre-built canopy raster.
2. **Generate Grid** *(optional)* — if no grid file exists, click 4 corners and define banks/rows/dimensions. You can draw on either the LiDAR **canopy raster** or a loaded **RGB orthomosaic** (see below).
3. **Edit** — drag/rotate/scale plot polygons until each one sits cleanly over its plot. Save the aligned grid.
4. **Visualize** — sanity-check alignment in 2D (always) or 3D (PyVista, optional).
5. **Traits** — pick which traits to compute, run extraction.
6. **Statistics** — auto-generated graphs from the metrics CSV.

---

## Why a top-down canopy raster?

Point clouds are unwieldy at full resolution; rendering 100 M+ points
interactively is slow. The app pre-computes a 2-D density raster of the canopy
*at the source resolution* (no point downsampling — every point contributes to
its 5 cm pixel) and uses that raster for the editor and visualizations. The
underlying point cloud is only loaded when you compute traits.

The canopy raster is **cached on disk** next to your LAS, so subsequent runs
are instant.

---

## Generating a plot grid on an orthomosaic

You can draw the plot grid on top of an **RGB orthomosaic** instead of (or as
well as) the LiDAR canopy raster — plot edges, alleys and gaps are often much
easier to see on true-colour imagery.

In the **Generate Grid** tab:

1. Set **Base** to *Orthomosaic (RGB)* (or click **Load Orthomosaic…**).
2. Pick a **georeferenced GeoTIFF** ortho. It must be in the **same CRS as your
   point cloud** (DPIRD standard: `EPSG:7850`, GDA2020 / MGA zone 50) so the
   corners you click map to the correct world coordinates.
3. Click the 4 trial corners in the chosen order, set the layout
   (banks / rows / plot size / gaps), then **Preview** and **Save** exactly as
   for the canopy raster.

Notes:
- The ortho is only used for *drawing* — traits are always computed from the
  point cloud. The saved grid works identically for both.
- Large orthomosaics are automatically downsampled for display (long edge
  capped at ~4000 px); corner picking still uses full-precision world
  coordinates, so grid accuracy is unaffected.
- The colour-map / background controls only affect the single-band canopy
  raster; they are ignored while an ortho is shown.

---

## Why proper ground classification?

Plant height is `point_elevation - ground_elevation`. If you don't classify
ground points, "height" is whatever Z value the scanner recorded — possibly
hundreds of metres above sea level.

The app uses **PDAL's Simple Morphological Filter (SMRF)** when the
*"Use proper ground classification"* tickbox is on. SMRF builds a model of the
bare ground beneath the canopy, then computes Height-Above-Ground (HAG) for
every point. This is the standard method in the LiDAR / phenotyping
literature.

The classified, height-normalized LAS is **cached** as `<name>.smrf_norm.las`
next to your source LAS. Subsequent loads skip the classification step.

For relatively flat fields, the per-plot 5th percentile fallback (used when
SMRF is off) gives reasonable relative heights but slightly compressed
absolute values.

---

## Trait reference

### Height & vertical structure

| Trait | What it captures | When to use |
|---|---|---|
| **h_max** | Maximum height in the plot | Quick QC; sensitive to noise spikes |
| **h_mean** | Mean of all heights | Stable for uniform canopies |
| **h_median** (h_p50) | 50th percentile | Robust alternative to mean |
| **h_p95** ⭐ | 95th percentile — *recommended plant height* | Standard in literature; robust to noise |
| **h_p99** | 99th percentile | Less smoothing than p95 |
| **h_std** | Standard deviation of heights | Canopy uniformity; high = uneven stand |
| **h_p99_p50** | p99 − p50 (vertical spread) | Distinguishes flat-topped vs gradient canopies |
| **h_skew** | Skewness of Z distribution | Positive = top-heavy canopy |
| **h_kurt** | Kurtosis of Z distribution | High = sharp peak in distribution |

> **Recommendation**: report **h_p95** as your primary plant-height value.
> It is robust against the ~1% of stray high points (insects, multipath, birds)
> that contaminate UAV LiDAR.

### Canopy cover & density

| Trait | Definition | Notes |
|---|---|---|
| **cover_frac** | Fraction of points above height_cut (default 0.15 m) | Strong biomass proxy, especially early-season |
| **pt_density** | Points per m² of plot area | QC for scanning quality |
| **vert_profile** | Density per 10 cm slice from 0–3 m | JSON-encoded list; opens up LAI-style analysis |
| **gap_frac** | 1 − cover_frac | Inverse of cover |

### Volume & biomass proxies

| Trait | Method | Trade-off |
|---|---|---|
| **vol_voxel** | Count occupied 5 cm voxels × voxel volume | Most defensible; standard in literature |
| **vol_chull** | 3-D scipy ConvexHull | Fast, overestimates |
| **vol_alpha** | Concave-hull volume (column-integrated) | Closer to true canopy envelope |
| **surf_area** | Triangulated canopy-top area × roughness factor | For light-interception models |
| **biomass_pvi** | Plant Volume Index = cover_frac × h_p95 × plot_area (m³) | Standard single-metric UAV biomass proxy |
| **biomass_kg** | biomass_pvi × k | PVI scaled to kg by one calibration coefficient |

---

## Estimating biomass (kg) from LiDAR

The point cloud gives you **proxies** (PVI, voxel volume, height, cover); turning
those into **kilograms** needs field calibration. The app offers two routes,
both driven by a small **ground-truth CSV** of harvested weights.

### The ground-truth CSV — what goes in it

This is the "excel sheet" for biomass. It is a plain CSV (open/edit in Excel)
with exactly two columns:

| Column | Meaning |
|---|---|
| `Plot_ID` | The plot identifier — **must match** the `Plot_ID` in your grid / `plot_metrics.csv` (e.g. `1001`, `1002`, …) |
| `biomass_kg` | The measured biomass for that plot, from **cut-and-weigh** sampling (fresh or dry — just be consistent) |

You do **not** need every plot — even 10–30 sampled plots are enough to
calibrate, as long as they span low-to-high biomass. Leave un-sampled rows blank.

> Click **"Save biomass template CSV…"** in the Traits tab to generate this file
> pre-filled with your actual plot IDs, then just type in the weights.

### Route 1 — single coefficient *k* (PVI model)

1. Tick **biomass_pvi** (and **biomass_kg**) and **Compute Traits** →
   `plot_metrics.csv` now has a `biomass_pvi` column.
2. Harvest & weigh some plots; fill the ground-truth CSV.
3. Click **"Fit k from CSV…"**. It least-squares fits `biomass_kg = k × PVI`
   (through the origin) and reports *k*, R² and RMSE.
4. **Compute Traits** again to write the calibrated `biomass_kg` for every plot.

Starting-point *k* values (no field data yet): wheat ≈ 0.28, barley ≈ 0.25,
fodder grass / pasture ≈ 0.10. These are rough — always calibrate when you can.

### Route 2 — multiple-metric model (recommended)

A single index rarely captures biomass across a whole season. The app can fit a
**multiple linear regression** on several LiDAR metrics at once:

```
biomass_kg = b0 + b1·h_p95 + b2·cover_frac + b3·vol_voxel
```

1. Tick **h_p95**, **cover_frac** and **vol_voxel**, then **Compute Traits**.
2. Fill the ground-truth CSV as above.
3. Click **"Fit multi-metric model from CSV…"**. It solves the coefficients by
   least squares, reports the equation, R² (and adjusted R²) and RMSE, and
   writes a **`biomass_pred_kg`** column for every plot in `plot_metrics.csv`.

This usually beats the single-*k* model because height, canopy cover and 3-D
occupancy each carry independent information about standing biomass. Compare the
two R² values to see which model your data supports; view `biomass_pred_kg` in
the Statistics tab.

### Which LiDAR metrics matter for biomass?

- **vol_voxel** — 3-D occupancy; the most defensible single volume proxy.
- **h_p95** — robust plant height (avoid h_max, which chases noise spikes).
- **cover_frac** — ground cover; dominates early-season biomass signal.
- **biomass_pvi** — combines the above into one index.
Use several together (Route 2) rather than relying on any one.

### Geometric & shape

| Trait | What it captures |
|---|---|
| **canopy_extent** | XY extent of canopy points (max of width/length) |
| **lodging_angle** | Angle (deg) between principal axis and vertical via PCA |
| **roughness** | Std dev of canopy-top elevations across 10 cm cells |
| **row_count** | Detected number of crop rows (peak counting in cross-row density) |

### Bookkeeping

- **n_points** — total points falling inside the plot polygon
- **n_canopy** — points above the height cutoff (used for cover_frac)

---

## Method details

### Voxel volume (vol_voxel)
Each point is assigned to a 5 cm cube. The number of occupied cubes above ground
is multiplied by the cube volume (1.25 × 10⁻⁴ m³). Gives a 3-D occupancy estimate
that is robust to point density variations within a plot (a denser sub-region
doesn't double-count voxels).

### Lodging angle
Principal Component Analysis (PCA) on the centred (x, y, z) coordinates returns
three eigenvectors. The vector with the largest eigenvalue is the principal
axis. Lodging angle is its deviation from vertical (z-axis). 0° = upright
canopy; 90° = horizontal (severely lodged).

### Row count
The plot polygon's minimum rotated rectangle gives the row direction. Points
are projected onto the cross-row direction; peaks in the resulting 1-D density
correspond to crop rows.

### Surface roughness
Each 10 cm × 10 cm cell within the plot is summarized by its zmax. Roughness =
std dev of these zmax values. Smooth canopy = low value; uneven canopy = high.

---

## Tips for accurate results

1. **Always tick the SMRF option** for the first run on a new LAS.
   The cache makes subsequent runs fast.
2. **Verify alignment visually** in the Edit tab before computing traits.
   A misaligned plot polygon catches half a row + buffer + half another row,
   producing meaningless statistics.
3. **Report h_p95 for plant height**, not h_max.
4. **Use voxel volume** as the primary biomass proxy unless you have a specific
   reason to prefer convex/alpha hulls.
5. **Inspect the Statistics tab heatmap** after extraction — anomalous
   plots (extremely low or extremely high values) are usually alignment
   artefacts that you can fix in the Edit tab and re-run.

---

## File outputs

- `aligned_grid.gpkg` — your edited plot grid
- `<las>_canopy.tif` — top-down canopy density raster (cached)
- `<las>.smrf_norm.las` — height-normalized LAS (cached, only if SMRF is enabled)
- `plots_las/plot_<id>_<B/R>.las` — one LAS per plot
- `plot_metrics.csv` — per-plot metrics (one row per plot, one column per trait)

---

## Credits

Developed by **Dr. Muhammad Ibrahim**, Research Scientist, DPIRD,
under the supervision of **Dr. Hammad Khan**, Senior Research Scientist, DPIRD,
APPN Director for DPIRD node.
