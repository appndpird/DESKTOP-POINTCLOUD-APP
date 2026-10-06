"""
Grid file I/O.

Reads/writes plot grid polygons in any of the common formats.

Public API
----------
load_grid(path, target_crs=None)         -> GeoDataFrame
save_grid(gdf, path)                     -> writes shp/gpkg/geojson based on extension
detect_format(path)                      -> 'shp' | 'gpkg' | 'geojson' | None
"""

from __future__ import annotations
import os
import geopandas as gpd


SHP_EXT     = {".shp"}
GPKG_EXT    = {".gpkg"}
GEOJSON_EXT = {".geojson", ".json"}
ALL_EXT     = SHP_EXT | GPKG_EXT | GEOJSON_EXT

GRID_FILE_FILTER = (
    "Plot grid files (*.shp *.gpkg *.geojson *.json);;"
    "Shapefile (*.shp);;GeoPackage (*.gpkg);;GeoJSON (*.geojson *.json);;"
    "All files (*)"
)


def detect_format(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in SHP_EXT:     return "shp"
    if ext in GPKG_EXT:    return "gpkg"
    if ext in GEOJSON_EXT: return "geojson"
    return None


DEFAULT_CRS = "EPSG:7850"   # GDA2020 / MGA zone 50 (DPIRD/WA standard)


def load_grid(path: str, target_crs: str | None = None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    gdf = gpd.read_file(path)
    # If the shapefile has no .prj sidecar / no CRS in the file, assume
    # EPSG:7850 (DPIRD WA standard) rather than silently loading raw coords.
    if gdf.crs is None:
        gdf = gdf.set_crs(DEFAULT_CRS, allow_override=True)
    if target_crs and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Normalize attribute column names: ensure Plot_ID, Bank, Row, B/R exist
    cols_lower = {c.lower(): c for c in gdf.columns}
    if "plot_id" not in [c.lower() for c in gdf.columns]:
        gdf["Plot_ID"] = range(1, len(gdf) + 1)
    if "bank" not in [c.lower() for c in gdf.columns]:
        gdf["Bank"] = 1
    if "row" not in [c.lower() for c in gdf.columns]:
        gdf["Row"] = range(1, len(gdf) + 1)
    if "b/r" not in [c.lower() for c in gdf.columns]:
        gdf["B/R"] = [f"B{b}R{r}" for b, r in zip(gdf["Bank"], gdf["Row"])]
    return gdf


def save_grid(gdf, path: str):
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"Unsupported grid extension: {path}")
    if fmt == "shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    elif fmt == "gpkg":
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path, driver="GeoJSON")
    return path
