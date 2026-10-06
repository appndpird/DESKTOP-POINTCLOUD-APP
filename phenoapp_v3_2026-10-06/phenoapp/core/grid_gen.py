"""
Grid generation from 4 corners + plot dimensions + buffers.

Given:
  - 4 corner points of the trial (in working CRS coordinates), in order
    bottom-left, bottom-right, top-right, top-left
  - n_banks (rows of plots — wide alleys between banks)
  - n_rows  (plots per bank — narrow gaps between plots)
  - plot_w, plot_l (planted area per plot, m)
  - gap_row, gap_bank (m)

Build a regular plot grid that fills the trial extent.

Public API
----------
generate_grid_from_corners(corners, n_banks, n_rows,
                           plot_w, plot_l, gap_row, gap_bank,
                           crs) -> GeoDataFrame
"""

from __future__ import annotations
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def generate_grid_from_corners(
    corners: list[tuple[float, float]],
    n_banks: int,
    n_rows: int,
    plot_w: float,
    plot_l: float,
    gap_row: float,
    gap_bank: float,
    crs: str,
    fit_to_corners: bool = False,
) -> gpd.GeoDataFrame:
    if len(corners) != 4:
        raise ValueError("Need exactly 4 corners (BL, BR, TR, TL)")

    bl, br, tr, tl = [np.asarray(c, dtype=float) for c in corners]

    # "Fit" mode: stretch the whole n_banks x n_rows grid to fill the
    # quadrilateral defined by the 4 clicked corners (handles rotated and
    # slightly non-rectangular trials). plot_w/plot_l/gaps are used only as
    # ratios to place the plot vs alley boundaries; absolute sizes come from
    # the corner span.
    if fit_to_corners:
        return _grid_fit_corners(bl, br, tr, tl, n_banks, n_rows,
                                 plot_w, plot_l, gap_row, gap_bank, crs)

    # Determine the orientation from the bottom edge BL -> BR (row direction)
    row_vec = br - bl
    row_len = np.linalg.norm(row_vec)
    if row_len == 0:
        raise ValueError("BL and BR are the same point")
    ux = row_vec / row_len   # unit along row direction
    # Bank direction = BL -> TL
    bank_vec = tl - bl
    bank_len = np.linalg.norm(bank_vec)
    if bank_len == 0:
        raise ValueError("BL and TL are the same point")
    uy = bank_vec / bank_len   # unit along bank direction

    # Build axis-aligned grid centred at (0, 0), oriented so row_dir = +X, bank_dir = +Y
    polys, attrs = [], []
    for b in range(n_banks):
        for r in range(n_rows):
            x0 = r * (plot_w + gap_row)
            y0 = b * (plot_l + gap_bank)
            poly = Polygon([
                (x0,           y0),
                (x0 + plot_w,  y0),
                (x0 + plot_w,  y0 + plot_l),
                (x0,           y0 + plot_l),
            ])
            polys.append(poly)
            attrs.append({
                "Plot_ID": (b + 1) * 1000 + (r + 1),
                "B/R":     f"Rg{b+1}Rw{r+1}",
                "Range":   b + 1,          # 1..n_banks (ranges)
                "Row":     r + 1,          # 1..n_rows
                "Bank":    b + 1,          # kept = Range for back-compat
            })

    gdf = gpd.GeoDataFrame(attrs, geometry=polys, crs=crs)

    # Rotate from local +X axis to ux direction
    angle = math.degrees(math.atan2(ux[1], ux[0]))
    gdf["geometry"] = gdf.geometry.apply(lambda g: rotate(g, angle, origin=(0, 0)))

    # Translate so that plot (1,1) BL is at the user's BL corner
    gdf["geometry"] = gdf.geometry.apply(lambda g: translate(g, bl[0], bl[1]))

    return gdf


def _slot_edges(n: int, size: float, gap: float):
    """Fractional [start, end] positions of each of `n` plots along [0, 1],
    given a repeating (plot `size`, `gap`) pattern. The alleys between plots
    keep their proportion of the total span."""
    n = max(int(n), 1)
    size = max(float(size), 1e-9)
    gap = max(float(gap), 0.0)
    total = n * size + (n - 1) * gap
    if total <= 0:
        total = 1.0
    edges, pos = [], 0.0
    for _ in range(n):
        edges.append((pos / total, (pos + size) / total))
        pos += size + gap
    return edges


def _bilinear(bl, br, tr, tl, u, v):
    """Point at parametric (u, v) in the quad. u runs BL->BR, v runs BL->TL."""
    return ((1 - u) * (1 - v) * bl + u * (1 - v) * br
            + u * v * tr + (1 - u) * v * tl)


def _grid_fit_corners(bl, br, tr, tl, n_banks, n_rows,
                      plot_w, plot_l, gap_row, gap_bank, crs):
    """Build an n_banks x n_rows grid that exactly fills the BL/BR/TR/TL quad
    by bilinear interpolation, keeping the plot:gap proportions."""
    u_edges = _slot_edges(n_rows, plot_w, gap_row)    # along row dir (BL->BR)
    v_edges = _slot_edges(n_banks, plot_l, gap_bank)  # along bank dir (BL->TL)

    polys, attrs = [], []
    for b, (v0, v1) in enumerate(v_edges):
        for r, (u0, u1) in enumerate(u_edges):
            p00 = _bilinear(bl, br, tr, tl, u0, v0)
            p10 = _bilinear(bl, br, tr, tl, u1, v0)
            p11 = _bilinear(bl, br, tr, tl, u1, v1)
            p01 = _bilinear(bl, br, tr, tl, u0, v1)
            polys.append(Polygon([tuple(p00), tuple(p10),
                                  tuple(p11), tuple(p01)]))
            attrs.append({
                "Plot_ID": (b + 1) * 1000 + (r + 1),
                "B/R":     f"Rg{b+1}Rw{r+1}",
                "Range":   b + 1,          # 1..n_banks (ranges)
                "Row":     r + 1,          # 1..n_rows
                "Bank":    b + 1,          # kept = Range for back-compat
            })
    return gpd.GeoDataFrame(attrs, geometry=polys, crs=crs)
