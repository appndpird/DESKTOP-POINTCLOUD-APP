"""
Plot sampling regions.

Validated on the 2025 DPIRD Fodder trials: computing plot statistics over
the CENTRAL BAND of each plot (matching where the ground-truth mower strip
is cut, and clear of edge effects / neighbour overhang) consistently beats
whole-plot means for both LiDAR and spectral biomass models. Cherry-picking
"best looking" sub-areas (tallest / densest patch) makes models WORSE,
because the ground truth is cut indiscriminately - sample representatively,
match the support, never flatter the plot.

Public API
----------
plot_region(polygon, mode="whole", band_width=0.5, inset=0.10) -> polygon
"""

from __future__ import annotations
import math


def plot_region(polygon, mode: str = "whole",
                band_width: float = 0.5,
                inset: float = 0.10):
    """Return the sampling region for a plot polygon.

    mode = "whole"   : polygon shrunk inward by `inset` metres
    mode = "central" : a `band_width`-wide band along the plot's long axis
                       (inset from the ends), intersected with the polygon
    """
    if mode not in ("whole", "central"):
        raise ValueError(f"unknown region mode: {mode}")

    if mode == "whole":
        r = polygon.buffer(-inset)
        return r if (not r.is_empty and r.area > 0) else polygon

    # ---- central band along the long axis ----
    from shapely.geometry import LineString

    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    # find the two side midpoint pairs; long axis connects midpoints of the
    # two SHORT sides
    sides = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % 4]
        sides.append((math.hypot(x2 - x1, y2 - y1),
                      ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
    # two shortest sides -> their midpoints define the centreline
    order = sorted(range(4), key=lambda i: sides[i][0])
    m1 = sides[order[0]][1]
    m2 = sides[order[1]][1]
    line = LineString([m1, m2])
    # inset the ends
    L = line.length
    if L > 2 * inset + 0.5:
        p1 = line.interpolate(inset)
        p2 = line.interpolate(L - inset)
        line = LineString([p1, p2])
    band = line.buffer(band_width / 2.0, cap_style=2)   # flat caps
    r = band.intersection(polygon)
    return r if (not r.is_empty and r.area > 0) else polygon
