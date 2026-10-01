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
| **biomass_pvi** | Plant Volume Index = cover_frac × h_p95 × region_area (m³ per plot) | Standard single-metric UAV biomass proxy |
| **biomass_kg** | biomass_pvi × k | Total biomass in the sampled region (kg per plot) |
| **biomass_kg_ha** | k × cover_frac × h_p95 × 10 000 | The same calibration as a density (kg/ha), comparable across plot sizes |

---

## Correcting the plot grid to the crop (Edit tab → "Refine plots to crop")

A grid drawn from corners or a trial plan is rarely exactly on the crop: the
seeder starts and stops differently in each range and drilling direction, so
plots are individually shifted along the sowing direction and their sown length
differs from the nominal length. **Auto-align** fixes only a rigid shift of the
whole grid. **Refine plots to crop** corrects every plot individually from the
canopy height model (building the CHM first if the project has none):

* **Along the plot** – the CHM is averaged across the plot's inner width and
  the two positions where it falls to half its plateau (the bare alleys) mark
  the crop ends. The polygon is re-centred on them and its length set to the
  crop length minus a 0.20 m margin at each end (never more than ±0.5 m from
  the original length).
* **Across rows** – neighbouring plots usually touch at maturity, so there are
  no edges to find; the furrow minima expected at ±half the plot pitch are
  located on the range-median profile and applied as one shift per range,
  only when a plausible furrow pair is visible.
* **Safeguards** – a plot whose crop edges are implausible keeps its length and
  takes its range's median offset; the report flags it (`refine_ok = False`).
* **Raster source** – the LiDAR CHM by default; if the project has an RGB
  orthomosaic you can choose it instead. A crop/soil index is built from the
  ortho (excess-green, or excess-blue for false-colour composites where the crop
  renders blue – the polarity is picked automatically from plot-vs-alley
  contrast) and saved as `grid_refine_veg_index.tif`.
* **Automatic method choice** – along the plot the tool first tries half-height
  crop edges; when the plot ends are shaded or ragged (edges found on < 70 % of
  plots, or the crop reads < 85 % of the polygon length) it switches to centring
  each plot between the two alleys, keeping the length. Across rows it uses each
  plot's own two gaps when ≥ 60 % of plots show clear gaps (bare gaps between
  plots, e.g. small-plot trials), otherwise one shift per range. Gaps and alleys
  are located by the centre of their low run, not the single lowest pixel, so
  wide flat alleys do not bias the centre.

Outputs next to the grid: `<grid>_refit.shp` (extra columns `plot_len`,
`along_off`, `across_off`, `crop_len`, `refine_ok`), `<grid>_refit_report.csv`
and `<grid>_refit_qa.png` (before/after zooms of the largest corrections). The
Traits tab uses the refined grid from then on. On the 2025 Muresk NUE trial the
original polygons had one end in the alley on 46 of 128 plots; after refinement
none did. Plant heights were unchanged (r = 0.999 between grids), the sampled
area grew 5 % to the true sown core, and the biomass proxies' correlation with
harvested biomass rose by 0.01 to 0.02 - a small but free gain.

---

## Surface models: DSM, DTM and CHM for the whole trial

Drone crop height is always a **surface minus a terrain model**. The Traits tab
can write the three classic rasters for the whole trial, on one grid, plus maps
with every plot labelled with its plant height:

| Product | What it is | How it is made |
|---|---|---|
| **DSM** (digital surface model) | Everything the laser hit: crop, soil, tracks | Maximum z per cell (default 0.10 m) |
| **DTM** (digital terrain model) | Bare ground only | One of the four methods below, evaluated on the DSM grid |
| **CHM** (canopy height model) | Crop height above ground | DSM − DTM, negatives clipped to 0 |
| **Annotated maps** (PNG) | CHM / DTM / DSM with plot outlines and a number per plot | Label = `h_p99` from `plot_metrics.csv` (cm) if Compute Traits has run, else CHM p99 |

Why one raster set for the trial and not one per plot: the ground is only
observable *outside* the plots (alleys, tracks), so a terrain model built inside
a plot polygon rides on the canopy bottom and compresses the height. One smooth
surface constrained by every alley is far better determined than 128 little
ones, it is what every photogrammetry pipeline does, and a saved DTM from one
date can be reused on the next.

**DTM methods** (drop-down next to the *Export* button):

1. **Exterior ground surface** (default) — a low-order surface fitted to the
   lowest returns in the alleys around all plots. Works under a closed canopy.
2. **Ground inside each plot** — each plot's 1st-percentile return becomes its
   ground ("ground inside the zone"); use for raised beds / vegetables.
3. **SMRF height-above-ground** — the PDAL ground classification loaded on the
   Project tab, gridded as the median of z − HAG.
4. **External GeoTIFF** — a DTM from another date (e.g. a bare-soil or seedling
   flight). It is resampled onto the DSM grid and **checked against this
   flight's alley ground**: the median offset is removed, and if the two
   disagree by more than 15 cm robust SD the export refuses, because the flights
   are not co-registered well enough (RTK on both dates or shared GCPs are
   required for this method).

Outputs go to `<project>/surface_models/` as `<las name>_DSM.tif`, `_DTM.tif`,
`_CHM.tif`, three `_annotated.png` maps and `_CHM_zonal_stats.csv` (per-plot
max, p99, p95, mean and cover fraction read from the CHM).

> **CHM statistics read higher than point statistics.** A CHM cell holds the
> highest of ~50 returns, so the p99 of CHM cells sits several centimetres above
> the p99 of the points themselves (about +19 cm on a 4500 pts/m² wheat cloud at
> 0.10 m cells). Use the point-based traits (`h_p95`, `h_p99`) for calibration
> against ruler heights, and the rasters for mapping, QGIS and reuse.

---

## Estimating biomass from LiDAR

The point cloud gives you **proxies** (PVI, voxel volume, height, cover); turning
those into **biomass** needs field calibration. The app offers two routes,
both driven by a small **ground-truth CSV** of harvested weights.

> **Units.** Ground truth may be in **any** unit — kg/ha, t/ha, g/m², kg/m² or a
> whole-plot weight in kg. Everything is converted to an area density (kg/m²)
> before fitting, and per-plot LiDAR volumes are divided by the plot region
> area, so *k* and the model coefficients **do not depend on plot size** and can
> be reused on another trial. Results are reported in **kg/ha**; per-plot totals
> (kg) are written alongside for convenience.

### The ground-truth CSV — what goes in it

This is the "excel sheet" for biomass. It is a plain CSV (open/edit in Excel)
with two columns:

| Column | Meaning |
|---|---|
| `Plot_ID` | The plot identifier — **must match** the `Plot_ID` in your grid / `plot_metrics.csv` (e.g. `1001`, `1002`, …) |
| `biomass_kg_ha` | The measured biomass for that plot from **cut-and-weigh** sampling, in kg/ha (fresh or dry — just be consistent) |

The unit is read from the column name: `biomass_kg_ha`, `biomass_t_ha`,
`biomass_g_m2`, `biomass_kg_m2`, or `biomass_kg` for a whole-plot weight in kg.
If your column is just `biomass`, pick the unit in the **Ground-truth unit** box
next to the fit buttons (it also overrides the column name when set).

You do **not** need every plot — even 10–30 sampled plots are enough to
calibrate, as long as they span low-to-high biomass. Leave un-sampled rows blank.

> Click **"Save biomass template CSV…"** in the Traits tab to generate this file
> pre-filled with your actual plot IDs, then just type in the weights.

### Route 1 — single coefficient *k* (PVI model)

1. Tick **biomass_pvi** (and **biomass_kg**) and **Compute Traits** →
   `plot_metrics.csv` now has a `biomass_pvi` column.
2. Harvest & weigh some plots; fill the ground-truth CSV.
3. Click **"Fit k from CSV…"**. It least-squares fits
   `biomass (kg/m²) = k × cover_frac × h_p95` (through the origin; this is
   `k × PVI / region_area`) and reports *k* in kg per m³ of canopy, R², RMSE
   and leave-one-out RMSE in kg/ha.
4. **Compute Traits** again to write the calibrated `biomass_kg` (per plot) and
   `biomass_kg_ha` for every plot.

Starting-point *k* values (no field data yet): wheat ≈ 0.28, barley ≈ 0.25,
fodder grass / pasture ≈ 0.10. These are rough — always calibrate when you can.

### Route 2 — multiple-metric model (recommended)

A single index rarely captures biomass across a whole season. The app can fit a
**multiple linear regression** on several LiDAR metrics at once:

```
biomass_kg_ha = b0 + b1·h_p95 + b2·cover_frac + b3·(vol_voxel / region_area)
```

1. Tick **h_p95**, **cover_frac** and **vol_voxel**, then **Compute Traits**.
2. Fill the ground-truth CSV as above.
3. Click **"Fit multi-metric model from CSV…"**. It solves the coefficients by
   least squares, reports the equation, R² (and adjusted R²), RMSE and the
   leave-one-out RMSE in kg/ha, and writes **`biomass_pred_kg_ha`** and
   **`biomass_pred_kg`** (per plot) columns for every plot in `plot_metrics.csv`.

This usually beats the single-*k* model because height, canopy cover and 3-D
occupancy each carry independent information about standing biomass. Compare the
two R² values to see which model your data supports; view `biomass_pred_kg_ha`
in the Statistics tab.

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

## VNIR spectral indices (tab 8) and the band viewer

The Biomass tab computes four fixed indices (NDVI, NDRE-740, WBI, NDWI970) as
ratios of the per-band plot means. The **VNIR Spectral** tab opens the full
*Awesome Spectral Indices* catalogue (Montero et al. 2023, Scientific Data;
bundled JSON, MIT licence) plus PhenoApp narrow-band extras (red-edge position
REP, NDRE 720/790, PRI, Vogelmann, WBI...). Indices are grouped by application
domain (vegetation, water, soil, ...) with a checkbox per index; hover an index
for its formula, bands and reference. "Recommended (biomass)" ticks the set that
carried biomass information on the 2025 DPIRD trials.

How a hyperspectral cube feeds a catalogue formula:

* **broad bands** (N, R, G, B, RE1, RE2, RE3, N2...) are the mean of every cube
  band whose wavelength falls inside the catalogue's range for that band
  (e.g. N = 760-900 nm, RE1 = 695-715 nm);
* **narrow bands** (R705, R850...) use the closest cube band;
* **excluded ranges** are never used. Defaults for the GOBI/GRYFN cubes:
  0-415 nm (calibration spike on the first bands), 755-770 nm (O2-A residual)
  and 928-962 nm (water vapour). Edit them in the tab.

Two computation modes:

* **Per pixel, then plot mean** (recommended): the index is evaluated on every
  pixel of the sampling region and averaged; the QC file reports the per-plot
  standard deviation and the **valid fraction**. A pixel is valid only when
  every band feeding the index is > 0. GRYFN cubes are unsigned integers, so
  negative reflectance (dark canopy in the blue, and the darkest red pixels)
  is clipped to 0: blue-based indices (EVI, VARI, ExG, TGI, SIPI, mND705...)
  can be valid on only half the canopy pixels or less. A warning lists such
  indices - do not use them for calibration on those cubes.
* **On the plot mean spectrum**: fast, reuses the Biomass-tab spectra when the
  grid matches, and reproduces the Biomass-tab NDVI/NDRE columns exactly.

Outputs next to the metrics CSV: `<las>_spectral_indices.csv` (Plot_ID + one
column per index), `_qc.csv` (sd, valid fraction), `_provenance.json`
(formulas, band-to-wavelength mapping, exclusions, region - keep it with the
data, as the APPN plot-delineation protocol asks for tool settings), and
optionally one GeoTIFF per index over the trial extent (`index_rasters/`).
The Biomass tab merges the CSV into the Ridge / kernel-ridge feature pool when
the checkbox "Add the indices computed on the VNIR Spectral tab" is on.

**Muresk NUE 2025-09-30 (anthesis, 128 plots) screening**, correlation with
dry biomass: REP 0.44, LCI 0.42, S2REP 0.41, NDRE-740 0.36, MTCI 0.29,
NDRE-720 0.26, NDREI 0.23, NDVI 0.14, WBI 0.04. Red-edge position beats
every NDVI-type index on a closed canopy.

### Band viewer (sub-tab "View")

Shows the **whole orthomosaic** (decimated), **selected plots at full
resolution** straight from the cube, or the **per-plot cube files** as a
gallery. Display as a NIR/red/green composite, true colour, any custom
wavelength triplet, a single wavelength, or any catalogue index; overlay the
plot polygons and IDs; click a pixel to plot its full spectrum (excluded
ranges shaded, count of clipped bands shown).

Why plots "look wrong" in QGIS: a 172-band GeoTIFF opens as bands 1/2/3 =
399/403/406 nm, the noisy sensor-edge bands, so every plot renders as
orange noise, while the GRYFN `*.rgb.tif` preview is a false-colour product
in which crop renders blue. Per-plot cubes written by PhenoApp now carry a
`.qml` sidecar (NIR/red/green composite) that QGIS applies automatically.

**No holes in the per-plot cubes (v2.2.1).** Pixel values inside the polygon
are exact copies of the orthomosaic, including the zeros GRYFN writes where
reflectance went negative (most of 400-500 nm, a few percent of pixels at
670 nm). Earlier files declared `nodata = 0`, so QGIS made every such pixel
transparent and the plots looked holed, although the same zeros are present
in the orthomosaic (rendered black there because the ENVI file has no nodata
tag). The writer now stores the plot polygon as an internal GDAL mask and sets
no nodata tag: only the area outside the polygon is transparent, every pixel
inside is shown, and the values are still bit-exact. The Biomass-tab option
"Also write per-plot VNIR cubes" uses this writer.

## Canopy-top plant height (cth_* traits) and reference targets

`h_p95` / `h_p99` are percentiles of *all* points in the plot, so they depend
on how many returns penetrate the canopy (flying height, scan rate, density,
growth stage, wind), and they inherit any drift of the trial-wide ground
surface. The canopy-top traits describe the upper canopy surface on a
plot-local ground instead:

1. noise filter: statistical outlier removal on the upper canopy slice and
   isolated points more than 25 cm above the canopy;
2. plot-local ground: robust plane through 25 cm alley cells (5th percentile
   of z) in a 0.25-1.0 m ring around the plot, neighbours excluded; falls back
   to the trial-wide surface when the ring has too few cells;
3. density normalisation: 2 cm voxel thinning;
4. top-surface raster: 5 cm cells, highest point per cell, plot edge trimmed
   by 15 cm;
5. metrics: `cth_p95` (recommended), `cth_p90`, `cth_p99`, `cth_max`,
   `cth_mean`, plus `cth_pt_p99` for continuity;
6. quality flags: `cth_cover`, `cth_ground_cells`, `cth_ground_rms`,
   `cth_ground_offset` (local minus trial-wide ground), `cth_n_noise`, `cth_ok`.

Muresk NUE 2025 (128 wheat plots, ruler at maturity): on the anthesis flight
the trial-wide ground drifted 10 cm between ranges and `h_p99` reached only
r 0.31; `cth_p95` on the plot-local ground reached r 0.68, error SD 3.8 cm,
slope 0.89, range drift 2.7 cm, which is at the noise level of the ruler
itself (about 4 cm; two flights agree with each other at r 0.84 but each
agrees with the ruler at no better than 0.56-0.69). On the maturity flight the
alleys had been disturbed and the local ground was no better than the
trial-wide surface: watch `cth_ground_offset`, and if it varies by more than
about 5 cm across the trial, prefer the trial-wide option or an earlier
flight.

**Why LiDAR reads below the ruler without calibration.** The ruler is held to
the tip of the tallest heads at a few spots; the LiDAR canopy-top statistic
is a percentile of a surface sampled by finite laser footprints, and thin
awns and ear tips return few points, so even the 95th percentile of the cell
maxima sits 5-15 cm below the tallest tip. The offset is systematic and stable
across ranges once the ground is right, so it is removed by a per-flight
calibration. **Reference targets** make that calibration independent of hand
measurements: place three or four rigid targets of known height (60, 90,
120 cm) in the trial for every flight, list them in a CSV
(`name,E,N,height_m[,radius_m]`, working CRS) in the "Canopy-top height" row,
and the tool measures them with the same pipeline, fits known = offset +
scale x measured (offset only with fewer than three targets) and writes
`cth_*_cal` columns plus `<metrics>_targets.csv` and
`_target_calibration.json`. Keep a 15-20 plot hand-measured subset each season
to confirm, and pool seasons once several are available.

## Height Models (tab 8)

Trains plant-height models on the Traits-tab metrics (the canopy-top `cth_*`
heights, cover fraction, roughness, point density and percentile heights must be
ticked) against a measured-height CSV (`Plot_ID, height_cm`). Four models can be
ticked, validated leave-one-out or 5-fold:

| Model | Features | When to use |
|---|---|---|
| Linear calibration on `cth_p95` | 1 | reporting within a trial; the reference (Muresk 2025: LOOCV 3.8 cm, at the ruler's noise floor) |
| Ridge, 7 structure features | cth_p95, tip thinness, cth_cover, cover_frac, roughness, vertical spread, point density | best within a flight by 0.1-0.4 cm; does not transfer |
| Random forest, 7 structure features | same | the model to carry to another date or trial (5.3 cm raw, 4.9 cm with a per-flight offset) |
| XGBoost, 7 structure features, monotone in cth_p95 | same | alternative boosted model (needs the xgboost package) |

Outputs next to the metrics CSV: `<metrics>_height_predictions.csv` (measured,
full-fit `pred_*` and held-out `predcv_*` per model), `<metrics>_height_models.json`
(metrics, calibration line, feature lists) and one `<metrics>_height_model_<key>.joblib`
per model. "Apply a saved model" predicts every plot of another flight's metrics CSV
with an optional per-flight offset (from reference targets or a few measured plots)
and writes `<metrics>_height_applied_<key>.csv`. Always plot `predcv_*`, not `pred_*`,
when judging accuracy.

## LiDAR plot viewer (Visualize tab)

Besides the whole-trial cloud, the Visualize tab can open the per-plot LAS
files written by the Traits tab: pick the folder, select one or many plots,
colour by elevation, height above the plot ground (z minus the plot's 1st
percentile), intensity or plot ID; plot outlines and IDs are drawn above the
canopy. Display is decimated to "Max points"; analysis always uses every
point.

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
