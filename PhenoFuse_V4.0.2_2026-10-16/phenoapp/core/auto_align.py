"""Auto-align a plot grid to a canopy raster via FFT cross-correlation.

Given:
  - a canopy raster (GeoTIFF — what the Edit tab already displays)
  - a list of plot polygons in the same CRS

Compute the rigid translation (dx, dy in metres) that best matches the
grid mask to the canopy mask, and return shifted polygons.

Approach:
  1. Read the canopy raster and threshold to a binary mask of "vegetation"
  2. Rasterize the plot polygons into a mask aligned to the same grid
  3. Phase-correlate the two masks via FFT - peak = shift (in pixels)
  4. Convert pixel shift to world metres via the raster transform
  5. Apply translation to every polygon

This handles translation only (no rotation/scale). For best results,
load a CHM (build_chm) rather than a density raster - CHM has cleaner
edges, but the density raster works too.
"""
from __future__ import annotations
import numpy as np


def auto_align_grid(canopy_tif: str,
                    polygons: list,
                    canopy_threshold_pct: float = 60.0,
                    max_search_metres: float = 5.0,
                    progress_cb=None) -> tuple[list, float, float]:
    """Return (shifted_polygons, dx_m, dy_m).

    canopy_threshold_pct : percentile of the canopy raster above which a
        cell is treated as vegetation (60% works for density rasters; for
        a CHM, the threshold is in metres - bypass via threshold=...).
    max_search_metres : clamp peak search to this radius. Stops large
        false-positive shifts when canopy is sparse.
    """
    def _p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import xy as transform_xy
    from shapely.affinity import translate as sh_translate

    _p(5, "Reading canopy raster...")
    with rasterio.open(canopy_tif) as src:
        canopy = src.read(1).astype("float32")
        tr = src.transform
        crs = src.crs
        pix_x = abs(tr.a)
        pix_y = abs(tr.e)

    # threshold canopy -> binary
    finite = canopy[np.isfinite(canopy)]
    if finite.size == 0:
        raise RuntimeError("Canopy raster is empty.")
    thr = float(np.percentile(finite[finite > 0], canopy_threshold_pct)) \
        if (finite > 0).any() else 0.0
    canopy_mask = (canopy > thr).astype("float32")
    _p(20, f"Canopy mask: threshold={thr:.3f}, "
            f"{int(canopy_mask.sum())} active pixels")

    if canopy_mask.sum() == 0:
        raise RuntimeError("Canopy mask has no active pixels - "
                           "reduce canopy_threshold_pct.")

    _p(30, "Rasterizing plot polygons...")
    shapes = [(g, 1) for g in polygons if g is not None and not g.is_empty]
    grid_mask = rasterize(
        shapes,
        out_shape=canopy_mask.shape,
        transform=tr, fill=0, dtype="uint8",
    ).astype("float32")

    if grid_mask.sum() == 0:
        raise RuntimeError("No plot polygons overlap the canopy raster - "
                           "shift the grid manually first.")

    # zero-mean for proper cross-correlation
    cm = canopy_mask - canopy_mask.mean()
    gm = grid_mask   - grid_mask.mean()

    _p(50, "FFT cross-correlation...")
    # cross-correlation = IFFT( FFT(canopy) * conj(FFT(grid)) )
    F1 = np.fft.rfft2(cm)
    F2 = np.fft.rfft2(gm)
    xcorr = np.fft.irfft2(F1 * np.conj(F2), s=cm.shape)
    # fftshift so the zero-shift is at the centre
    xcorr = np.fft.fftshift(xcorr)

    _p(70, "Locating peak shift...")
    H, W = xcorr.shape
    cy, cx = H // 2, W // 2

    # clamp search to max_search_metres around the centre
    max_dy = int(max_search_metres / pix_y)
    max_dx = int(max_search_metres / pix_x)
    y0, y1 = max(0, cy - max_dy), min(H, cy + max_dy + 1)
    x0, x1 = max(0, cx - max_dx), min(W, cx + max_dx + 1)
    window = xcorr[y0:y1, x0:x1]
    py, px = np.unravel_index(np.argmax(window), window.shape)
    py += y0
    px += x0

    dy_pix = py - cy
    dx_pix = px - cx
    # raster rows increase southward -> flip dy sign for world Y
    dx_m =  dx_pix * pix_x
    dy_m = -dy_pix * pix_y
    _p(85, f"Best shift: dx={dx_m:.3f} m  dy={dy_m:.3f} m")

    _p(95, "Applying translation to polygons...")
    shifted = [sh_translate(g, xoff=dx_m, yoff=dy_m) if g is not None else None
               for g in polygons]
    _p(100, "Auto-align complete.")
    return shifted, dx_m, dy_m
