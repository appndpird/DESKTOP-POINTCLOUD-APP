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
Traits tab uses the refined grid from then on. On the trial used for validation
more than a third of the original polygons had one end in the alley; after
refinement none did. Plant heights were unchanged (r = 0.999 between grids), the sampled
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
carried biomass information on pasture and wheat trials.

How a hyperspectral cube feeds a catalogue formula:

* **broad bands** (N, R, G, B, RE1, RE2, RE3, N2...) are the mean of every cube
  band whose wavelength falls inside the catalogue's range for that band
  (e.g. N = 760-900 nm, RE1 = 695-715 nm);
* **narrow bands** (R705, R850...) use the closest cube band;
* **excluded ranges** are never used. Defaults for GRYFN-processed cubes:
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

**Anthesis screening on a wheat trial**, correlation with
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

Wheat trial, ruler heights at maturity: on the anthesis flight
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
| Linear calibration on `cth_p95` | 1 | reporting within a trial; the reference (LOOCV 3.8 cm on the validation trial, at the ruler's noise floor) |
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

## PhenoApp v3 (2026-10-06): LiDAR preprocessing and classification, cleaned VNIR, fused ML

**LiDAR v3 (Traits tab, ticked by default).** Each plot goes through the standard forestry sequence (noise removal,
ground classification, height normalisation, canopy classification, canopy metrics), but per plot and tied to the
plot-local ground: the noise rule (default: isolated points more than 30 cm above the canopy; statistical outlier
variants selectable) flags noise (class 7); a robust plane through the alley ring (25 cm cells, 5th
percentile) refined with in-plot points within 8 cm is the ground (class 2, |h| <= 8 cm); heights are normalised
to it; points above 10 cm are canopy (class 5), 8-10 cm low vegetation (class 3). Tick "Write classified per-plot
LAS" to get one classified LAS per plot in <project>/plots_las_classified. Features are computed on canopy points
only (the legacy h_* columns stay for continuity): H50 / H_mean, H90, H95, H99, H99.9 (robust maximum), top-N mean,
spread and shape statistics, 10 density layers, canopy cover on a 5 cm grid, voxel canopy volume (5 cm), PVI, gap
fraction Pgap (ground returns / all returns - the Ouster cloud is single-return, so return counts cannot be used),
20 cm gap-fraction layers, LAI proxy = -ln(Pgap)/0.5, 3D profile area, canopy-top P90/P95/P99 (5 cm cell maxima,
15 cm edge trim), rumple index, roughness, return fractions and intensity normalised by the alley ground
(I_canopy_mean, I_canopy_p90, I_canopy_x_cover). Columns go into the metrics CSV (duplicated names get a _v3
suffix) and into <metrics>_lidar_v3.csv.

**VNIR v3 (VNIR Spectral tab, button "Cleaned VNIR v3 features").** Clipped band values (0) are treated as missing,
the excluded wavelength ranges are never used, and a soil mask is derived per flight from the NDVI histogram
(Otsu split bounded to 0.15-0.50; disabled automatically when a senesced canopy shows no soil/crop contrast).
Thirty indices are averaged over all valid pixels and over vegetation pixels (mean and sd), with the vegetation
fraction, broad-band reflectances, red-edge derivative features, Guyot red-edge position, PCA scores of the
vegetation spectra and QC flags (red_ok = the 670 nm band is valid on more than half of the pixels; vnir_qc_ok).
Output <metrics>_vnir_v3.csv (+ _spectra.csv, _band_valid_fraction.csv). v3.0.1: a band or index is NaN for a plot
when it is valid on fewer than half of the plot's pixels (clipped reflectance), lies in an excluded range, or the cube
fails the flight-level reflectance check (NDVI ~ 0 everywhere: a radiance cube); red-clipped plots get `red_ok = 0`
and NaN for every red-based index; `vnir_reflectance_ok` and `vnir_qc_ok` are written per plot. NaN cells are meant to
be left out of any analysis or model, never filled.

**Height Models tab** gained the v3 sets: LiDAR core (21 canopy features) and LiDAR + VNIR (43 features) with PLS,
ridge, Gaussian process and random forest, each with random-forest top-10 feature selection inside every training
fold, and repeated 10-fold (5x) validation as the default. On the validation trial: anthesis R2 0.50-0.51 / RMSE 3.6 cm;
maturity LiDAR+VNIR R2 0.48 / 3.7 cm against 0.25 / 4.4 cm for the single cth_p95 calibration.

**Biomass ML tab (new).** Fused LiDAR v3 + VNIR v3 features (plus CHM-weighted indices: index x cover, index x H95,
NDRE x PVI) against any per-plot target (default biomass_kg_ha); families LiDAR_core / VNIR_core / Fused_core /
Fused_all; learners Ridge, PLS, SVR, GPR, RandomForest, ExtraTrees, XGBoost, LightGBM; nested repeated 10-fold;
writes held-out and full-fit predictions, a results JSON, one joblib per model and a top-feature ranking; a saved
model can be applied to another flight with an offset.

**PCA families (v3.3.1), Biomass ML and Height Models.** The core families keep a few features chosen by random-forest
importance inside each fold. The PCA families take the opposite route: every available feature of a block is
standardised and reduced to the principal components that hold 95 % of its variance, and the components are the inputs
of the learner. The scaler and the PCA are part of the model pipeline, so they are fitted inside every training fold and
travel with the saved model. `LiDAR_PCA` and `VNIR_PCA` use one block each; `Fused_PCA` fits one PCA per modality and
concatenates the components (late fusion); `Fused_PCA_joint` fits one PCA on both modalities together (early fusion);
`VNIR_bands_PCA` and `Fused_bands_PCA` work on the mean vegetation spectrum itself (log10 reflectance, SNV per plot,
only the bands usable on at least 95 % of the plots and outside the excluded ranges; the spectra CSV written by the VNIR
Spectral tab is picked up next to the VNIR v3 CSV). Results show the components kept per block. Typical numbers: about
80 LiDAR features reduce to 12 components and 130 VNIR features to 9; the fused PCA families were the best biomass
and height models in the 9 October rebuild, a little above the selected-feature families, and they are less sensitive
to which single feature happens to be chosen in a fold. The Height Models tab offers the same sets (LiDAR PCA, LiDAR +
VNIR PCA per modality, joint PCA) with ridge, Gaussian process and random forest learners.

**Validation choice.** The literature (review of 2020-2026 studies) uses k-fold or single random splits and rarely
tests independent sites; k-fold pooled across dates is optimistic. In this tool: repeated 10-fold with nested
feature selection is the default (stable, honest within a trial); LOOCV gives nearly the same numbers for n ~ 100
but a noisier estimate; the real test of transfer is a held-out flight or trial, which the research scripts
(Height Prediction Models6-10-06, Biomass methods 6-10-26) report separately.

## PhenoApp v3.2 (2026-10-09): noise rule, interior gap metrics, labelled per-plot LAS

**Noise rule (Traits tab, next to LiDAR v3).** Which points are flagged as noise (class 7) before the ground plane,
normalisation, classification and features. Options: `gap` (points more than 30 cm above the 99.9th percentile; the new
default), `none`, `legacy` (the canopy-top filter: SOR on the top 10 % of points, k=8, 2 sd, plus a 25 cm gap) and `sor`
(k=10 on all points at 3 sd, the v3.0 default, or 5 sd). Flagged points are kept and marked unless "Drop flagged noise
points" is ticked; every metric ignores them either way.

Why the default changed: on the validation flight (30,000 pts/m2, single-return Ouster) the gap
rule flagged a single point in the whole trial, while every statistical-outlier variant removed sparse canopy tips (heads
and awns): 0.3 % of points at 5 sd, 1.5 % at 3 sd, about half of them in the top 5 % of the canopy. Against the ruler
heights the canopy-top P95 scored r 0.70 / LOOCV RMSE 3.70 cm unfiltered, 0.69 / 3.76 cm with the legacy filter, 0.68 /
3.79 cm at 5 sd and 0.66 / 3.88 cm at 3 sd. A global neighbour-distance threshold flags wherever points are naturally
sparse, and in a cereal canopy that is the top. Use an SOR only for clouds with genuine multipath or air returns, or
apply it below the 90th height percentile.

**Interior gap metrics.** The refit plot polygons include 10-20 cm of alley soil along the long sides, and the laser also
sees soil in inter-row gaps and thin patches. On the validation flight 68 % of the ground returns sat within 20 cm of the
polygon edge: whole-polygon Pgap 8.3 % against 3.5 % on the interior. The v3 features therefore add, on the 20 cm-trimmed
interior, `Pgap_interior`, `LAI_proxy_interior` (= -ln(Pgap_interior)/0.5), `cover_2cm_interior` (share of 2 cm cells
with a canopy point) and `ground_visible_2cm_interior` (share of 2 cm cells with a ground return), plus `cover_2cm`,
`ground_visible_2cm`, `ground_edge_frac` and the 20 cm layers `Pgap_interior_below_*`. At this density `cover_5cm` is 1.0
in half of the plots, so `LIDAR_V3_CORE` (Height Models and Biomass ML tabs) now uses `cover_2cm_interior`,
`ground_visible_2cm_interior`, `Pgap_interior` and `LAI_proxy_interior` in place of `cover_5cm`, `Pgap` and `LAI_proxy`
(the old columns are still written). The density layers `dens_1..10` now span the 10 cm canopy threshold to H999
(`dens_1` was always 0 before). On this sensor the return-based columns (`frac_first_return`, `frac_single_return`,
`mean_returns_per_pulse`, `frac_canopy_first`) are constant and should be left out of models.

**Labelled per-plot LAS.** "Write classified per-plot LAS" now writes classification 2 / 3 / 5 / 7, the LAS withheld bit
on noise points, a float32 `HeightAboveGround` extra dimension (z minus the plot-local alley plane, for every point) and a
sidecar JSON with the plane coefficients (g = c0 + c1 (x - x0) + c2 (y - y0)), ring statistics, class counts and the rule
used, so downstream tools (including the sparse 3D CNN tensors) can read normalised heights without re-deriving the ground.

## Deep Models tab (v3.1): two-stream network for biomass and height

The network (sparse 3D CNN over the plot cloud voxelised at 2 cm with height, normalised intensity and return
number, plus a transformer over 512 cleaned VNIR pixel spectra, late fusion with a trial/stage embedding) runs
in a separate Python environment that has torch and spconv, because PhenoApp's own environment does not.
The tab finds such an environment automatically ("Detect automatically" probes the running interpreter and the
conda environments on the machine), reports the GPU, and selects the device itself: CUDA when available,
otherwise CPU, where spconv's native algorithm is used with the same weights (slower). Step 2 builds the plot
tensors from the project LAS, VNIR cube and grid with the LiDAR v3 preprocessing and the cleaned VNIR pixels
(dl_data/ next to the metrics CSV). Step 3 either scores every plot with the bundled pretrained ensemble (five
fold models per target, trained on wheat plots at anthesis and maturity; predictions and the fold spread are
written to <metrics>_dl_<target>_<variant>_pretrained_predictions.csv, with metrics when a ground truth was
given) or trains and cross-validates the network on the current trial (fold weights, CV predictions and a
JSON summary in <metrics>_dl_<target>_training). Expect the feature models to be as good or better until
several hundred labelled plots per crop are available: on the validation trial the network reached biomass R2 0.22
against 0.35 for the fused ridge, and height 3.9 cm against 3.6 cm.

### Using the pretrained deep models, step by step

The bundled ensembles live in `phenoapp/assets/dl_models/`: `two_stream_<target>_fold0-4.pt` (LiDAR + VNIR) and
`lidar_only_<target>_fold0-4.pt` for biomass (kg/ha dry) and height (cm), five fold models each, trained for 100 epochs
per fold on 760 wheat plots (biomass) and 256 plots (height) of two trials at anthesis and maturity, with the common band
policy: `common_bands.csv` next to the weights lists the 111 bands used for every plot (522-645, 691-754, 772-926 and
965-1000 nm), and the runner applies it automatically. A prediction is the mean of the fold models; the spread between
them is written next to it as an uncertainty. Use `lidar_only` for a flight without a usable reflectance cube. Held-out
scores of these ensembles (10-fold, per dataset): biomass R² 0.27 / RMSE 1,636 kg/ha at anthesis and 0.13 / 1,340 at
maturity on the trial with ruler heights, 0.07 / 1,467 and 0.05 / 1,848 on the second trial; height 3.7 cm at anthesis,
4.1 cm at maturity. The LiDAR-only ensemble scores the same within the noise, and the VNIR-only variant carries no
within-trial signal at this sample size, so the spectral stream is kept for its small gain at anthesis, not as a model
on its own.

1. **Environment.** The network needs torch and spconv, which are not in PhenoApp's own environment. Create one once
   (conda or venv; Python 3.11, torch 2.6 with CUDA 12.4, `spconv-cu124`) and tick nothing else. On an older GPU without
   prebuilt spconv kernels (Pascal, e.g. TITAN Xp) spconv compiles them at first use and needs the CUDA 12.4 compiler
   headers in the same environment (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`, `cuda-nvrtc-dev`, `cuda-cuxxfilt` from
   conda-forge; the first run then takes a few minutes longer). On the Deep Models tab press "Detect automatically" or
   pick the interpreter; the status line reports torch, the GPU and whether spconv is present. Without spconv only the
   VNIR-only variant can run.
2. **Inputs.** The Project tab needs the flight LAS and an aligned plot grid; the VNIR Spectral tab needs the reflectance
   cube of the same flight. The cube must be reflectance (the flight-level check is applied; a radiance or DN cube makes
   every spectral value unusable and the plots are scored by the LiDAR stream alone). Give the dataset a name and the
   growth stage (anthesis or maturity): the stage selects the trial/stage embedding the ensemble was trained with.
   A ground-truth CSV (Plot_ID plus biomass_kg_ha and/or height_cm) is optional for prediction and required for training.
3. **Prepare the plot tensors.** One file per plot is written to `dl_data/<name>/` next to the metrics CSV, with the
   classified cloud (noise removed, height above the plot-local ground, normalised intensity, return number), the
   cleaned VNIR pixels of the plot region, the per-plot usable-band mask (NaN rule) and the bookkeeping in `index.csv`.
   The same tensors serve prediction and training; prepare them once per flight.
4. **Predict.** Choose the target, keep variant `two_stream`, leave the weights folder at the bundled path (or point it at
   a folder of fold models trained elsewhere) and press "Predict with pretrained ensemble". Output:
   `<metrics>_dl_<target>_<variant>_pretrained_predictions.csv` with one row per plot: the prediction, the fold spread,
   the measured value when given, and the metrics (R2, RMSE, rRMSE, MAE, bias, r, accuracy) printed in the log.
5. **Read the result with the right expectation.** The ensemble was trained on wheat plots of two trials at anthesis and
   maturity, and each training trial has its own embedding in the network. A dataset it was not trained on is scored with
   every training embedding of its stage and the predictions are averaged, so no single trial's offset is imposed, but
   the trial's own offset is still unknown: compare the predictions with a few measured plots and apply an offset, or
   treat the output as a ranking. Within a trial with its own ground truth, the feature models of the Height Models and
   Biomass ML tabs were better in every test so far (biomass R² 0.37 against 0.27 for the network at anthesis; height
   3.3 cm against 3.7 cm).
6. **Train or cross-validate on your own plots.** With a ground-truth CSV, "Train / cross-validate on this trial" runs
   k-fold cross-validation (default 10 folds, 100 epochs per fold, training-time augmentation: random 180-degree
   rotation, mirror across the row axis, 0-30 % point dropout, 1 cm jitter, pixel bootstrap and a 0.9-1.1 brightness
   factor) and then fits a model on all plots. Outputs in `<metrics>_dl_<target>_training/`: fold weights, held-out
   predictions per plot and a JSON summary. Several hundred labelled plots per crop are needed before the network beats
   the feature models; with 100-250 plots use it as a check, not as the reporting model.
7. **Band policy.** The bundled models use the fixed common band list (111 bands, `common_bands.csv` next to the
   weights), which is also the default when you train from the tab, so every flight is represented by the same bands
   and the network cannot read band availability as a fingerprint of the trial. The alternative, `per_plot`, uses every
   band usable on a plot (the NaN rule written as `band_mask` in the tensor files) and tells the network which bands
   were absent; it keeps the most data per plot but was not better in the comparison (biomass R² 0.23 against 0.27 with
   the common list at anthesis, 0.07 against 0.13 at maturity; height equal). A model trained with one policy must be
   applied with the same policy; the runner reads the policy from the weights folder.

### Features used by each model

| Tab / model | Inputs |
|---|---|
| Traits: canopy-top height | `cth_p95`: 95th percentile of the 5 cm cell maxima above the plot-local ground |
| Height Models: linear calibration | `cth_p95` only (ruler = a + b x cth_p95) |
| Height Models: 7-feature ridge / RF / XGBoost | `cth_p95`, `tip_thin` (cth_max - cth_p95), `cth_cover`, `cover_frac`, `roughness`, `vspread` (h_p99 - h_median), `pt_density` |
| Height Models / Biomass ML: LiDAR v3 core (22) | canopy-only percentiles `H50`, `H95`, `H99`, `H999`, `H_topN_mean`, `H_mean`, `H_sd`, `H_crr`; canopy-top `cth_p95`, `cth_p99`, `tip_thin`; cover and gaps on the 20 cm interior `cover_2cm_interior`, `ground_visible_2cm_interior`, `Pgap_interior`, `LAI_proxy_interior`; `vox_volume_m3_per_m2`, `profile_area_m`, `canopy_pts_per_m2`, `roughness`, `rumple`; normalised intensity `I_canopy_mean`, `I_canopy_p90` |
| Height Models / Biomass ML: VNIR v3 core (22) | vegetation-masked indices `NDRE740_veg`, `NDRE720_veg`, `REP_veg`, `LCI_veg`, `MTCI_veg`, `CIRE_veg`, `NDVI_nb_veg`, `OSAVI_veg`, `GNDVI_veg`, `kNDVI_veg`, `NIRv_veg`, `PRI_veg`, `WBI_veg`; `fcover_vnir`; reflectance `refl_N`, `refl_RE1`, `refl_R`; red-edge shape `red_edge_slope_max`, `red_edge_pos_deriv_nm`; spectral scores `spec_pc1..3` |
| Biomass ML: LiDAR_core | the LiDAR biomass list (25): `H95`, `H99`, `H_mean`, `H50`, `H_sd`, `H_crr`, `cth_p95`, `cover_5cm`, `Pgap`, `LAI_proxy`, `vox_volume_m3_per_m2`, `PVI_m`, `profile_area_m`, `canopy_pts_per_m2`, `roughness`, `rumple`, `I_canopy_mean`, `I_canopy_p90`, `I_canopy_x_cover`, `Pgap_below_020cm`, `Pgap_below_040cm`, `Pgap_below_060cm`, `dens_1`, `dens_5`, `dens_9` |
| Biomass ML: VNIR_core | VNIR v3 core + `WDRVI_veg`, `NDVI_nb`, `NDRE740`, `OSAVI`, `refl_RE2`, `refl_G`, `red_depth` |
| Biomass ML: Fused_core | LiDAR_core + VNIR_core + 9 canopy-weighted indices (`NDRE740_veg_x_cover`, `OSAVI_veg_x_cover`, `GNDVI_veg_x_cover`, `NDVI_nb_veg_x_cover`, `NDRE740_veg_x_H95`, `OSAVI_veg_x_H95`, `REP_veg_x_H95`, `LCI_veg_x_H95`, `NDRE_x_PVI`) |
| Biomass ML: Fused_all | every numeric v3 column of the metrics and VNIR tables |
| All of the above | random-forest importance keeps the top 12 (Biomass ML) or top 10 (Height Models) features inside each training fold; families with fewer features use all |
| PCA families (Biomass ML) and PCA sets (Height Models) | every finite LiDAR feature (about 70-85 columns) and / or every finite VNIR feature (about 130), each block standardised and reduced to the components holding 95 % of its variance inside the fold; the band variants use the mean vegetation spectrum (log10, SNV) over the bands usable on >= 95 % of the plots |
| Deep Models: LiDAR stream | the raw classified plot cloud, up to 24,000 points, voxelised at 2 cm: height above ground, log normalised intensity, return number |
| Deep Models: VNIR stream | 512 cleaned pixel spectra per plot, all 172 bands after log10 and SNV, with the usable-band mask as a second input; invalid bands are zero and flagged |
| Deep Models: fusion | 128-d LiDAR embedding + 128-d spectral embedding + 8-d trial/stage embedding -> regression head |

## Training, testing and inference: a user guide

This section is the end-to-end recipe for anyone who wants to fit a biomass or height model on their own trial, test it
on held-out plots or on another flight, and predict plots that have no ground truth. The same text is shipped as
`USER_GUIDE.md` in the repository.

### 1. What you need

| Item | Requirement |
|---|---|
| Point cloud | one LAS/LAZ per flight, projected CRS in metres, intensity and return number present (single-return clouds are fine) |
| Plot grid | shapefile / GeoPackage / GeoJSON with one polygon per plot and a `Plot_ID` column; align it on the Edit tab and run "Refine plots to crop" once |
| VNIR cube | the ENVI reflectance orthomosaic of the same flight (reflectance x 10,000 or 0-1; a radiance / DN cube is detected and refused for spectral features) |
| Ground truth | CSV with `Plot_ID` and `biomass_kg_ha` (dry matter per hectare) and / or `height_cm`; one row per measured plot; the IDs must match the grid |
| Settings kept identical across flights | noise rule, region mode and band width, excluded wavelength ranges, canopy threshold; a model only transfers to features extracted the same way |

### 2. Extract the features (identical for training and inference)

1. **Point Cloud QC** (tab 2): run it once per flight; fix tilt, CRS or noise problems before anything else.
2. **Project** (tab 1): load the LAS and the grid. **Edit** (tab 4): align, then "Refine plots to crop", save the grid.
3. **Traits** (tab 6): keep "LiDAR v3" ticked, the noise rule on "Gap only", tick the canopy-top heights; tick "Write classified per-plot LAS" if you want the labelled clouds. Compute. Outputs: `<metrics>.csv` and `<metrics>_lidar_v3.csv` with the per-plot QC column `qc_ok`.
4. **VNIR Spectral** (tab 11): load the cube, run "VNIR v3". Outputs next to the metrics CSV: `_vnir_v3.csv` (features, NaN where unusable, with `vnir_qc_ok`, `red_ok`, `vnir_reflectance_ok`), `_vnir_v3_spectra.csv`, `_vnir_v3_band_valid_fraction.csv`, `_vnir_v3_band_usable.csv`.
5. Look at the usable-band table before modelling: a band column that is 0 on most plots (clipped blue or red) means the indices built on it are NaN and will be left out; a flight whose every band is 0 is a radiance cube.

### 3. Train and validate

**Plant height, Height Models tab (tab 8).** Inputs: the metrics CSV, the ground-truth CSV, optionally the `_vnir_v3.csv` for the fused sets. Tick the models (defaults: linear calibration, 7-feature ridge and random forest, LiDAR core PLS, LiDAR + VNIR PLS, LiDAR PCA ridge, fused PCA ridge). Validation: leave-one-out for the headline number on fewer than about 150 plots, repeated 10-fold for a stable comparison of many models, 5-fold for speed. Press "Fit and validate". Outputs: `<metrics>_height_predictions.csv` (measured, full-fit `pred_<model>` and held-out `predcv_<model>` per plot), `<metrics>_height_models.json` (metrics, features, calibration line), one `<metrics>_height_model_<key>.joblib` per model.

**Biomass or any per-plot trait, Biomass ML tab (tab 9).** Inputs: metrics CSV, VNIR v3 CSV, ground-truth CSV, the spectra CSV is found automatically; set the target column. Families: start with LiDAR core, fused core, LiDAR PCA and fused PCA; add the spectral-band PCA families when the spectra are clean; learners: ridge, PLS and random forest first, the others for comparison. Validation: repeated 10-fold with nested selection (default). Outputs: `<metrics>_<target>_ml_predictions.csv`, `_ml_results.json`, one `_ml_model_<family>_<learner>.joblib` per pair, `_ml_top_features.csv`.

**Reading the table.** Report the held-out columns (cvR2, cvRMSE, rRMSE, accuracy), never the fit columns; a fit R² far above the cv R² is overfitting. Prefer the simplest model within about 0.02 R² of the best. The "within 5 cm" share (height) and rRMSE (biomass) are the numbers agronomists read. With fewer than about 60 labelled plots, use the linear calibration or ridge only.

### 4. Test on held-out plots or on another flight

- **Same flight:** the `predcv_` columns are already held-out predictions; plot them against the measured values (Statistics tab, or any spreadsheet) and read the residuals per block or treatment to see whether an error is systematic.
- **Another flight or trial:** extract the features there with the same settings, open "Apply a saved model" on the Height Models or Biomass ML tab, pick the `.joblib`, press "Predict every plot". Output: `<metrics>_height_applied_<key>.csv` or `<metrics>_<target>_ml_applied_<family>_<learner>.csv` with the prediction per plot and `model_ok` (0 when a required feature is NaN on that plot). Merge with the measured plots of that flight to get the test metrics; the snippet in section 6 does it in five lines.
- **Expect an offset.** Between dates and trials the structure-to-trait relation keeps its slope but not its intercept (different stage, variety set, ruler protocol, soil). Measure a handful of plots on the new flight (five is enough for height) and enter the mean difference as the offset on the apply panel; without it, use the predictions as a ranking, not as absolute values. Within one trial and date the models are at the noise floor of the ground truth; across trials expect R² well below the within-trial value for biomass.
- **Which model to carry:** random forest and the PCA families transfer between dates better than PLS; the linear cth_p95 calibration is the safest height model for a new trial once the offset is set.

### 5. Inference on a flight without ground truth

Extract features (section 2), apply the saved model (section 4) with the offset you established, then sanity-check: the predicted range should sit inside the training range (a saved model does not extrapolate reliably), `model_ok = 0` plots need their missing feature explained (outside the cube, red clipped, too few points), and the Statistics tab heatmap should show the field pattern, not the flight-line pattern. Keep the metrics CSV, the model file and the offset together with the predictions; the JSON next to the model records the features, the validation and the training inputs.

### 6. Command line and Python

The research scripts in the repository (`research/<date>/`) reproduce every table of the report from a dataset folder: `build_dataset.py` (per-plot extraction and QC for all flights), `height_models.py`, `biomass_models.py` (feature families, PCA families, nested validation, pooled and cross-trial schemes), `dl_prepare.py` and `dl_twostream.py` (network tensors and training; `--bands per_plot|common`, `--epochs`, `--folds`), `report_builder.py`. A saved feature model is applied in Python with:

```python
import joblib, pandas as pd
m = joblib.load("flight_biomass_kg_ha_ml_model_Fused_PCA_Ridge.joblib")      # dict: model, features, meta
X = pd.read_csv("new_flight_metrics_merged.csv")                             # metrics + VNIR v3 columns merged on Plot_ID
ok = X[m["features"]].notna().all(axis=1)
X.loc[ok, "predicted"] = m["model"].predict(X.loc[ok, m["features"]].to_numpy(float)) + offset
test = X.merge(gt, on="Plot_ID"); err = test.predicted - test.biomass_kg_ha      # RMSE = (err**2).mean()**0.5
```

PCA pipelines carry their scaler and PCA inside `model`, so the input is always the raw feature columns listed in `features`.

### 7. Metrics

R² = 1 - sum of squared errors / total variance of the measured values (can be negative on transfer); RMSE in the target's unit; rRMSE = RMSE / mean measured, in %; MAE; bias = mean(prediction - measured), positive means over-prediction; r = Pearson correlation; accuracy = 100 - mean absolute percentage error; "within 5 cm" = share of plots whose height error is below 5 cm. Held-out values come from plots the model never saw in that fold; fit values are on the training plots and only show how flexible the model is.

### 8. Checklist before trusting a number

- Point cloud QC passed; grid refined; same noise rule and region on every flight.
- VNIR cube is reflectance (`vnir_reflectance_ok = 1`); usable-band table inspected; red-based indices absent where `red_ok = 0`.
- Plots with `qc_ok = 0` or `vnir_qc_ok = 0` excluded from the families that need them (the tabs do this).
- Held-out metrics reported; model and its JSON kept with the predictions; offset documented when applied to another flight.
- Deep models: see "Using the pretrained deep models, step by step"; the feature models remain the reporting models until several hundred labelled plots per crop are available.

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
