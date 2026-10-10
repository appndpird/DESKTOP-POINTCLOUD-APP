# Per-plot dataset of 9 October 2026: LiDAR, VNIR and ground truth for height and biomass modelling

Built 2026-10-09 with PhenoApp v3.3 (v3.2 LiDAR pipeline with the gap noise rule; v3.0.1 VNIR QC with the NaN rule) from the raw combined
LAS and the GRYFN VNIR orthomosaic of each flight. Provenance: `manifest.json` (tool, version, feature lists, flight inputs), `build_dataset.log`.
Every plot of every grid is extracted; the ground truth and the validity flags say which plots can be modelled.

## Folder layout

```
dataset\
  README.md, manifest.json, dataset_summary.csv, build_dataset.log
  workbooks\                      trial workbooks (per-flight Spectra, VNIR, Bands_QC and Bands_ValidFrac sheets, NaN rule applied)
  <trial>\<flight>\
    ground_truth.csv              one row per plot with a measurement: Plot_ID, Plot, design columns (Range, Row, Block, Rep, N treatment,
                                  Variety ...), biomass_<stage>_kg_ha, biomass_kg_ha (= the target of this flight's stage), height_cm, height_source.
                                  NOTE height: Muresk has ONE ruler measurement per plot, taken at maturity; the same value is the height target of
                                  both Muresk flights (height_source says so), i.e. the anthesis-flight height models predict the final height.
                                  AGT has no height ground truth.
    features_lidar.csv            v3.2 LiDAR features for every plot of the grid (97 columns): qc_ok, counts, noise rule, ground plane quality,
                                  canopy-only percentiles (H50..H999, top-N mean), canopy-top P90/P95/P99, cover (5 cm, 2 cm, interior), gap fraction
                                  and 20 cm gap layers (whole polygon and 20 cm interior), LAI proxy, voxel volume, profile area, rumple, roughness,
                                  return and normalised intensity metrics, density layers
    features_vnir.csv             v3 VNIR features (181 columns): fcover_vnir, per-index value, _sd, _veg (vegetation-only) and _valid fraction,
                                  red-edge shape, spectral PCA scores; NaN wherever the NaN rule applies
    vnir_spectra.csv              mean reflectance spectrum per plot (172 bands, column = wavelength nm, reflectance x 10000); NaN = unusable band
    vnir_band_valid_fraction.csv  fraction of the plot's pixels where each band is valid (> 0, not clipped)
    vnir_band_usable.csv          1 = usable / 0 = NaN per band and plot, plus cube_reflectance_ok and n_usable_bands
    features_all.csv              merge for modelling: flags + ground truth + LiDAR + VNIR features (one row per grid plot)
    lidar_classified\             per-plot LAS with every point of the plot window: class 2 ground, 3 low vegetation, 5 canopy, 7 noise
                                  (kept, withheld bit), extra dimension HeightAboveGround; sidecar JSON with the ground plane, counts and rule
    vnir\                         per-plot VNIR cube (GeoTIFF, all bands, pixel-exact against the orthomosaic) with .qml and .aux.xml sidecars
    qc\                           the 8 October audit of this flight: plot_vs_cube_comparison.csv (per-plot pixel comparison with the orthomosaic),
                                  band tables (max abs diff, zero fraction, valid fraction), usable_plots_*.csv, valid_plots_for_modeling_*.csv,
                                  example figures
  agt\2025-11-13_merged\         AGT maturity: ground_truth.csv and features_all.csv with each plot taken from flight 1, or from flight 2
                                  where flight 1 has no usable cloud (column source_flight)
```

## Validity flags (features_all.csv)

- `has_ground_truth` : the plot has a biomass (or height) measurement for this stage.
- `valid_lidar`      : LiDAR pipeline qc_ok = 1 (enough points, ground plane found, polygon inside the cloud).
- `valid_vnir`       : strict VNIR flag = cube is reflectance AND >= 50 % of the polygon inside the cube AND the red band is not clipped (red_ok = 1).
- `valid_vnir_lenient`: as above but the red band may be clipped; red-based indices are NaN on those plots, the others are usable.
- `valid_fused`      : valid_lidar AND valid_vnir.

## NaN rule for bands and indices (MIN_VALID_FRAC = 0.5)

A band is NaN for a plot when it is valid on fewer than half of the plot's pixels (reflectance clipped to 0 by the processing), when it lies in an
excluded range (0-415, 755-770, 928-962 nm) or when the whole cube fails the reflectance check (AGT 2025-11-13: radiance cube, every band NaN).
An index built from a NaN band is NaN. NaN cells are left out of every analysis and model, never filled.

## Plot counts

| Trial | Flight | Stage | Plots in grid | Ground truth | LiDAR usable | VNIR reflectance | VNIR usable (lenient) | Red band ok | GT plots with usable LiDAR |
|---|---|---|---|---|---|---|---|---|---|
| muresk | 2025-09-30 | anthesis | 128 | 128 | 128 | yes | 128 | 128 | 128 |
| muresk | 2025-11-21 | maturity | 128 | 128 | 128 | yes | 126 | 128 | 128 |
| agt | 2025-09-22 | anthesis | 816 | 252 | 814 | yes | 814 | 337 | 252 |
| agt | 2025-11-13_f1 | maturity | 816 | 252 | 816 | no (radiance cube) | 0 | 728 | 252 |
| agt | 2025-11-13_f2 | maturity | 816 | 252 | 501 | no (radiance cube) | 0 | 221 | 163 |

Modelling sets: height = the two Muresk flights (ruler heights at maturity, 128 plots each); biomass = Muresk anthesis and maturity (128 each),
AGT anthesis (252) and AGT maturity merged (252). The two-stream tensors (`..\two_stream\dl_data`) hold the 760 labelled plots.

## Known data issues carried as flags, not fixed

- AGT 2025-11-13 (both flights): VNIR cubes are radiance, not reflectance; all VNIR features NaN; LiDAR only.
- AGT 2025-09-22: blue and red clipped to 0 on about half of the plots (red_ok = 0 on 479 of 816); red-based indices NaN there.
- AGT 2025-11-13 flight 2 covers part of the field (501 of 816 plots with a usable cloud).
