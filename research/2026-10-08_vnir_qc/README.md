# VNIR plot-level QC of 8 October 2026 (ENVI 6.2 / IDL 9.2, headless) and workbook build

`idl\` - run with `idl.exe -quiet -e "..."` (resolve_routine, /COMPILE_FULL_FILE); each driver `run_*.pro` sets the paths and calls the worker.
- `compare_vnir_plots.pro` / `run_compare.pro`   : every per-plot GeoTIFF against the ENVI orthomosaic, pixel by pixel, all bands (plot_vs_cube_comparison.csv, band_max_abs_diff.csv, plot_mean_spectra_tif_vs_cube.csv)
- `compare_flight.pro`, `compare_agt.pro`         : the same comparison for a whole flight / the AGT flights (band_max_abs_diff_and_zero_fraction.csv)
- `zero_fraction.pro` / `run_zero.pro`            : fraction of zero (clipped) pixels inside each polygon, per plot and per band, read from the cube itself
- `band_qc.pro`                                   : per-plot band valid fraction and the usable-band tables (band_valid_fraction_*.csv, usable_plots_*.csv, valid_plots_for_modeling_*.csv)
- `replicate_vnir_v3.pro` / `run_replicate.pro`   : re-computes the tool's VNIR v3 indices in IDL from the cube to check the feature extraction
- `maturity_spectra.pro` / `run_maturity.pro`     : mean spectra of the first plots of the maturity cubes (shows the AGT 2025-11-13 cubes are radiance, not reflectance)
- `inspect_f2.pro`                                : the two AGT flight-2 plots whose per-plot files lost 5 edge pixels (writer fixed in v3.0.1)
- `make_fig.pro` / `run_fig.pro`                  : true- and false-colour example figures, per-plot file next to the cube

`workbooks\` - `build_plot_traits_indices.py` builds the trial workbooks (`<flight>_Spectra`, `<flight>_VNIR`, `<flight>_Bands_QC`, `<flight>_Bands_ValidFrac` sheets, NaN rule) with the v3.3 tool;
`regenerate_plot_cubes.py` re-extracts the per-plot cubes; `tidy_workbooks.py` applies the sheet layout.

Outputs of the audit are stored with the dataset of 9 October: `Biomass_Height_2026-10-09\dataset\<trial>\<flight>\qc\`.
