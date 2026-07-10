"""
Canopy density rasterizer.

Generates a top-down density GeoTIFF from a LAS at full resolution
(no point downsampling). Output is cached on disk and reused on
subsequent runs.

Public API
----------
build_canopy_raster(
    las_manager, out_tif, resolution=0.05, height_cutoff=0.10,
    progress_cb=None, force=False)
    -> path to the GeoTIFF (created or reused)
"""

from __future__ import annotations
import os
import numpy as np


def build_canopy_raster(
    mgr,
    out_tif: str,
    resolution: float = 0.05,
    height_cutoff: float = 0.0,
    progress_cb=None,
    force: bool = False,
) -> str:
    """Build a 2D canopy density raster (.tif) from a loaded LAS.

    Parameters
    ----------
    mgr : LASManager  (already .load()ed)
    out_tif : str     output GeoTIFF path
    resolution : float    pixel size in metres (default 5 cm)
    height_cutoff : float HAG threshold for "canopy" points (default 0.0:
        include ground points too, so sparse / young canopies stay visible;
        bump higher to filter out ground if you want a canopy-only view)
    force : bool      ignore cache, rebuild
    """
    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    # --- cache check ---
    if not force and os.path.exists(out_tif):
        if os.path.getmtime(out_tif) >= os.path.getmtime(mgr.las_path):
            _p(100, "Using cached canopy raster")
            return out_tif

    import rasterio
    from rasterio.transform import from_origin

    _p(5, "Selecting canopy points...")
    x, y = mgr.x, mgr.y
    if mgr.hag is not None:
        # Use HAG to filter: keep only canopy points (above the cutoff)
        m = mgr.hag > height_cutoff
        x = x[m]; y = y[m]
        _p(15, f"Using {len(x):,} canopy points (HAG > {height_cutoff} m)")
    else:
        _p(15, f"Using all {len(x):,} points (no HAG available)")

    # --- raster grid ---
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    nx = int(np.ceil((xmax - xmin) / resolution))
    ny = int(np.ceil((ymax - ymin) / resolution))
    _p(25, f"Rasterizing {nx} x {ny} pixels @ {resolution} m")

    # --- 2D histogram (point density per cell) ---
    H, _, _ = np.histogram2d(
        x, y, bins=[nx, ny],
        range=[[xmin, xmin + nx * resolution],
               [ymin, ymin + ny * resolution]]
    )
    _p(70, "Density computed; writing GeoTIFF...")

    # rasterio expects rows = north-down; histogram2d gives (nx, ny)
    # so transpose and flip Y
    density = H.T[::-1, :].astype("float32")

    transform = from_origin(xmin, ymin + ny * resolution, resolution, resolution)
    crs = str(mgr.crs) if mgr.crs else None

    with rasterio.open(
        out_tif, "w",
        driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs=crs, transform=transform,
        compress="lzw", tiled=True,
    ) as dst:
        dst.write(density, 1)

    _p(100, f"Wrote {out_tif}")
    return out_tif


def build_chm(
    mgr,
    out_tif: str,
    resolution: float = 0.5,
    smooth_sigma: float = 1.0,
    progress_cb=None,
    force: bool = False,
) -> str:
    """Build a Canopy Height Model GeoTIFF from a loaded LAS.

    Each cell is the MAX HAG (HeightAboveGround) of points falling in it,
    matching the MATLAB pc2dem(..., CornerFillMethod="max") approach. Cells
    with no points are filled by nearest-neighbour from adjacent valid cells,
    then optionally smoothed with a Gaussian. Requires SMRF/HAG (so
    LASManager must have been loaded with use_smrf=True).
    """
    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    if mgr.hag is None:
        raise RuntimeError(
            "CHM requires HeightAboveGround. Tick 'Use proper ground "
            "classification (SMRF)' in the Project tab and reload.")

    if not force and os.path.exists(out_tif):
        if os.path.getmtime(out_tif) >= os.path.getmtime(mgr.las_path):
            _p(100, "Using cached CHM")
            return out_tif

    import rasterio
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter, distance_transform_edt

    _p(5, f"Binning HAG at {resolution} m grid...")
    x, y, hag = mgr.x, mgr.y, mgr.hag
    # clip negative HAG (noise below ground) to zero like the MATLAB script
    hag = np.clip(hag, 0.0, None)

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    nx = int(np.ceil((xmax - xmin) / resolution))
    ny = int(np.ceil((ymax - ymin) / resolution))

    # cell index per point
    ix = np.clip(((x - xmin) / resolution).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - ymin) / resolution).astype(np.int64), 0, ny - 1)
    flat = iy * nx + ix

    _p(25, f"Computing max-z per cell ({nx} x {ny})...")
    # max-per-bin via sort + reduce-at
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    hag_s  = hag[order]
    # first index where each unique cell starts
    unique_flat, first_idx = np.unique(flat_s, return_index=True)
    # max within each run
    bounds = np.r_[first_idx, len(flat_s)]
    max_per_cell = np.empty(len(unique_flat), dtype="float32")
    for k in range(len(unique_flat)):
        max_per_cell[k] = hag_s[bounds[k]:bounds[k+1]].max()

    chm = np.full((ny, nx), np.nan, dtype="float32")
    cy = unique_flat // nx
    cx = unique_flat % nx
    chm[cy, cx] = max_per_cell

    _p(60, "Filling empty cells (nearest)...")
    mask_nan = np.isnan(chm)
    if mask_nan.any() and (~mask_nan).any():
        # fill NaNs with the nearest known cell value
        _, (yy, xx) = distance_transform_edt(mask_nan, return_indices=True)
        chm = chm[yy, xx]

    if smooth_sigma > 0:
        _p(80, f"Gaussian smoothing (sigma={smooth_sigma})...")
        chm = gaussian_filter(chm, sigma=smooth_sigma)

    # rasterio: rows north-down -> flip Y
    chm = chm[::-1, :]
    transform = from_origin(xmin, ymin + ny * resolution, resolution, resolution)
    crs = str(mgr.crs) if mgr.crs else None

    _p(90, "Writing GeoTIFF...")
    with rasterio.open(
        out_tif, "w",
        driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs=crs, transform=transform,
        compress="lzw", tiled=True, nodata=-9999.0,
    ) as dst:
        dst.write(chm.astype("float32"), 1)

    _p(100, f"Wrote CHM {out_tif}")
    return out_tif


COLORMAPS = ("viridis", "plasma", "inferno", "magma", "terrain",
             "gray", "gray_r", "Greens", "YlGn")


def raster_display_array(
    tif_path: str,
    stretch_pct=(5, 95),
    colormap: str = "viridis",
    gamma: float = 0.55,
    background_rgb=(20, 20, 20),
) -> tuple:
    """Read a canopy GeoTIFF and return (display_rgba_uint8, bounds, crs).

    - `stretch_pct`: percentile range of non-zero values that maps to 0..1.
      Defaults to (5, 99.5) — clips both noise floor and outliers.
    - `gamma`: <1 brightens dark mid-tones (helps sparse-canopy rasters where
      most non-zero values are near the noise floor); set to 1.0 for linear.
    - `colormap`: any name from COLORMAPS (matplotlib).
    - `background_rgb`: RGB triple used to fill zero/transparent cells, so the
      raster sits against a neutral background instead of pure black.

    Returns an HxWx4 RGBA uint8 array (background_rgb where data == 0).
    """
    import rasterio
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        b   = src.bounds
        crs = src.crs

    nz = arr[arr > 0]
    if nz.size:
        vmin, vmax = np.percentile(nz, stretch_pct)
        if vmax <= vmin:
            vmax = float(nz.max()) + 1.0
    else:
        vmin, vmax = float(arr.min()), float(arr.max() + 1)

    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    if gamma and gamma != 1.0:
        norm = np.power(norm, gamma)

    try:
        from matplotlib import colormaps as mpl_colormaps
        cmap = mpl_colormaps.get_cmap(colormap)
    except Exception:
        from matplotlib import cm as mpl_cm
        cmap = mpl_cm.get_cmap(colormap)
    rgba = (cmap(norm) * 255).astype("uint8")     # HxWx4

    # Where the raster has no data (== 0 in our histogram), paint the neutral
    # background colour and full alpha so it doesn't go pure-black.
    zero_mask = arr <= 0
    if zero_mask.any():
        bg_r, bg_g, bg_b = background_rgb
        rgba[zero_mask, 0] = bg_r
        rgba[zero_mask, 1] = bg_g
        rgba[zero_mask, 2] = bg_b
        rgba[zero_mask, 3] = 255

    return rgba, (b.left, b.bottom, b.right, b.top), crs


def ortho_display_array(
    tif_path: str,
    max_dim: int = 4000,
    stretch_pct=(2, 98),
    dst_crs=None,
) -> tuple:
    """Read an RGB(A) orthomosaic GeoTIFF and return (display_rgba_uint8, bounds, crs).

    Handles the real-world orthomosaic cases:
      - 3+ band imagery  -> first three bands used as R, G, B
      - 4-band imagery   -> 4th band treated as an alpha/mask if it looks like one
      - 1-band imagery   -> shown as greyscale
      - 8-bit data       -> used directly; 16-bit / float -> per-band percentile
                            stretch to 0..255 for a sensible display
      - nodata / zero-alpha pixels -> made fully transparent

    If `dst_crs` is given and differs from the file's CRS, the ortho is
    reprojected to `dst_crs` on the fly (via a WarpedVRT) so the returned bounds
    are in the working CRS. This is essential when the ortho is in geographic
    degrees (e.g. EPSG:4326) but the app works in projected metres (EPSG:7850) —
    without it, clicked corners are in degrees and the grid geometry is wrong.

    Large orthos are decimated on read (rasterio out_shape) so a multi-gigapixel
    mosaic never blows up memory: the long edge is capped at `max_dim` pixels.
    This is display-only downsampling — corner picking still happens in true
    world coordinates, so grid accuracy is unaffected.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.crs import CRS as RioCRS
    from rasterio.vrt import WarpedVRT

    want_crs = RioCRS.from_user_input(dst_crs) if dst_crs else None

    with rasterio.open(tif_path) as src:
        need_warp = (want_crs is not None and src.crs is not None
                     and src.crs != want_crs)
        vrt = WarpedVRT(src, crs=want_crs, resampling=Resampling.bilinear) \
            if need_warp else None
        ds = vrt if vrt is not None else src
        try:
            b = ds.bounds
            crs = ds.crs
            count = ds.count
            H, W = ds.height, ds.width
            nodata = ds.nodata

            # decimation factor to keep the long edge <= max_dim
            scale = max(1.0, max(H, W) / float(max_dim))
            out_h = max(1, int(round(H / scale)))
            out_w = max(1, int(round(W / scale)))

            n_read = min(count, 4)
            data = ds.read(
                indexes=list(range(1, n_read + 1)),
                out_shape=(n_read, out_h, out_w),
                resampling=Resampling.average,
            )  # (bands, out_h, out_w)
        finally:
            if vrt is not None:
                vrt.close()

    data = data.astype("float32")

    # Split into colour bands + optional alpha
    if n_read == 1:
        bands = [data[0], data[0], data[0]]
        alpha_band = None
    else:
        bands = [data[0], data[1], data[2]]
        # A 4th band is usually an alpha/mask (0 = transparent). Treat any
        # band whose values are only 0 or its own max as a mask.
        alpha_band = data[3] if n_read >= 4 else None

    def _to_u8(band):
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            return np.zeros(band.shape, dtype="uint8")
        vmax = float(finite.max())
        # 8-bit imagery: pass through untouched
        if vmax <= 255.0 and finite.min() >= 0.0:
            return np.clip(band, 0, 255).astype("uint8")
        lo, hi = np.percentile(finite, stretch_pct)
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
        return out.astype("uint8")

    r8, g8, b8 = (_to_u8(bd) for bd in bands)
    rgba = np.dstack([r8, g8, b8, np.full(r8.shape, 255, dtype="uint8")])

    # Transparency: explicit alpha band, or nodata, or all-zero pixels
    transparent = np.zeros(r8.shape, dtype=bool)
    if alpha_band is not None:
        transparent |= (alpha_band <= 0)
    if nodata is not None:
        transparent |= (data[0] == nodata)
    transparent |= (r8 == 0) & (g8 == 0) & (b8 == 0)
    if transparent.any():
        rgba[transparent, 3] = 0

    return rgba, (b.left, b.bottom, b.right, b.top), crs
