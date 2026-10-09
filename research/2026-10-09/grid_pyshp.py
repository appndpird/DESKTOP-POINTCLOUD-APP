"""Grid loader without fiona / pyogrio (blocked on this machine): reads the plot shapefile with pyshp and returns the same
GeoDataFrame that phenoapp.core.load_grid would (Plot_ID, Bank, Row, B/R normalised, CRS EPSG:7850 unless the .prj says otherwise)."""
import os
import geopandas as gpd, shapefile
from shapely.geometry import Polygon, MultiPolygon
DEFAULT_CRS = "EPSG:7850"

def load_grid(path, target_crs=DEFAULT_CRS):
    sf = shapefile.Reader(path); fields = [f[0] for f in sf.fields[1:]]; recs = []
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record)); pts = sr.shape.points; parts = list(sr.shape.parts) + [len(pts)]
        rings = [Polygon(pts[parts[i]:parts[i + 1]]) for i in range(len(parts) - 1) if parts[i + 1] - parts[i] >= 4]
        rec["geometry"] = rings[0] if len(rings) == 1 else MultiPolygon(rings); recs.append(rec)
    gdf = gpd.GeoDataFrame(recs, geometry="geometry")
    crs = DEFAULT_CRS
    prj = os.path.splitext(path)[0] + ".prj"
    if os.path.exists(prj):
        txt = open(prj).read()
        if "MGA_Zone_50" in txt or "7850" in txt: crs = "EPSG:7850"
        elif "28350" in txt or ("GDA94" in txt and "Zone_50" in txt): crs = "EPSG:28350"
    gdf = gdf.set_crs(crs, allow_override=True)
    if target_crs and str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    low = [c.lower() for c in gdf.columns]
    if "plot_id" not in low: gdf["Plot_ID"] = range(1, len(gdf) + 1)
    if "bank" not in low: gdf["Bank"] = 1
    if "row" not in low: gdf["Row"] = range(1, len(gdf) + 1)
    if "b/r" not in low: gdf["B/R"] = [f"B{b}R{r}" for b, r in zip(gdf["Bank"], gdf["Row"])]
    return gdf
