"""Core engine for the phenotyping app — UI-agnostic."""
from .las_manager   import LASManager, quick_summary
from .canopy_raster import (build_canopy_raster, build_chm, raster_display_array,
                            ortho_display_array)
from .grid_io       import load_grid, save_grid, detect_format, GRID_FILE_FILTER
from .grid_gen      import generate_grid_from_corners
from .traits        import (TRAITS_CATALOG, TRAIT_KEYS, TRAIT_BY_KEY,
                            compute_plot_traits, fit_biomass_k,
                            fit_biomass_multi, apply_biomass_multi,
                            BIOMASS_MODEL_PREDICTORS)
from .units         import (BIOMASS_UNITS, UNIT_KEYS, UNIT_LABELS, AUTO_LABEL,
                            to_kg_m2, from_kg_m2, resolve_unit,
                            infer_unit_from_column, area_normalise)
from .surface_models import (build_surface_models, zonal_chm_stats,
                             annotate_plot_map)
from .extract       import extract_all_plots
from .auto_align    import auto_align_grid
from .regions       import plot_region
from .ground_model  import fit_exterior_ground, GroundModel
from .vnir          import VNIRCube, parse_envi_wavelengths
from .models        import (MODEL_SUITE, fit_model_suite, results_table,
                            save_models, load_models, apply_pls,
                            loocv_linear)
from .qc            import run_qc, format_report
