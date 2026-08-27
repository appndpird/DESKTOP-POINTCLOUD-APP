"""
Exterior ground-surface model.

Motivation (validated on the 2025 DPIRD Fodder trials): with single-return
LiDAR over a dense, closed sward the laser almost never reaches the soil
inside plots. Ground classifiers that look for the lowest local surface
(SMRF, per-plot p1-of-Z) therefore ride along the *canopy bottom*, which
compresses height-above-ground differentially by canopy density and can
invert biomass relationships entirely.

The robust alternative implemented here: take ground observations ONLY from
the exterior of the plot polygons (mowed alleys, tracks, bare surrounds,
where the laser genuinely sees soil), then fit a smooth low-order surface
and evaluate it underneath the plots. On the Busselton fodder trial this
doubled the LiDAR fresh-biomass LOOCV R^2 (0.13 -> 0.25) versus the vendor
DTM, while an SMRF ground flipped it negative.

Term selection is condition-number guarded: a term (e.g. v^2) whose
constraints are degenerate (alleys only north+south of the trial) is
dropped automatically instead of bowing metres between the constraints.

Public API
----------
fit_exterior_ground(x, y, z, plots_gdf, buffer_m=0.3, margin_m=6.0,
                    cell=1.0, pctl=5.0) -> GroundModel
GroundModel.predict(x, y) -> ground z
GroundModel.hag(x, y, z)  -> height above ground
GroundModel.diagnostics   -> dict
"""

from __future__ import annotations
import numpy as np


class GroundModel:
    def __init__(self, origin, axes, terms, coef, diagnostics):
        self.origin = origin          # (x0, y0)
        self.axes = axes              # 2x2 rotation (rows = unit axes)
        self.terms = terms            # list of term names, e.g. ["1","u","v","u2"]
        self.coef = coef
        self.diagnostics = diagnostics

    # ---- term expansion -------------------------------------------------
    @staticmethod
    def _expand(u, v, terms):
        cols = []
        for t in terms:
            if   t == "1":  cols.append(np.ones_like(u))
            elif t == "u":  cols.append(u)
            elif t == "v":  cols.append(v)
            elif t == "uv": cols.append(u * v)
            elif t == "u2": cols.append(u * u)
            elif t == "v2": cols.append(v * v)
            else: raise ValueError(t)
        return np.column_stack(cols)

    def _to_uv(self, x, y):
        dx = np.asarray(x, float) - self.origin[0]
        dy = np.asarray(y, float) - self.origin[1]
        u = dx * self.axes[0, 0] + dy * self.axes[0, 1]
        v = dx * self.axes[1, 0] + dy * self.axes[1, 1]
        return u, v

    def predict(self, x, y):
        u, v = self._to_uv(x, y)
        return self._expand(u, v, self.terms) @ self.coef

    def hag(self, x, y, z):
        return np.asarray(z, float) - self.predict(x, y)


def fit_exterior_ground(x, y, z, plots_gdf,
                        buffer_m: float = 0.3,
                        margin_m: float = 4.0,
                        cell: float = 1.0,
                        pctl: float = 5.0,
                        cond_max: float = 1.0e4) -> GroundModel:
    """Fit a smooth ground surface from points OUTSIDE the plot polygons.

    x, y, z    : full point arrays (working CRS, metres)
    plots_gdf  : GeoDataFrame of plot polygons
    buffer_m   : safety buffer around each plot excluded from 'ground'
                 (canopy overhang)
    margin_m   : how far beyond the grid bounding box to still accept ground
    cell       : grid cell size for per-cell ground observations (m)
    pctl       : per-cell percentile of z taken as the ground observation
    cond_max   : max design-matrix condition number when adding curvature
                 terms; degenerate terms are dropped automatically
    """
    from shapely.ops import unary_union
    from shapely.vectorized import contains as sv_contains

    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)

    plots_union = unary_union(list(plots_gdf.geometry)).buffer(buffer_m)
    minx, miny, maxx, maxy = plots_gdf.total_bounds
    inbox = ((x > minx - margin_m) & (x < maxx + margin_m) &
             (y > miny - margin_m) & (y < maxy + margin_m))
    # exterior = in the margin box but not inside any (buffered) plot
    idx = np.where(inbox)[0]
    inside = sv_contains(plots_union, x[idx], y[idx])
    ext = idx[~inside]
    if len(ext) < 500:
        raise RuntimeError(
            f"Only {len(ext)} exterior points found around the plots - "
            "cannot fit an exterior ground surface. Increase the margin or "
            "check that the LAS extends beyond the trial.")

    xe, ye, ze = x[ext], y[ext], z[ext]

    # ---- local PCA frame (stabilises the polynomial numerically) ----
    x0, y0 = float(xe.mean()), float(ye.mean())
    P = np.column_stack([xe - x0, ye - y0])
    cov = np.cov(P, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    a1 = evecs[:, -1]; a2 = evecs[:, 0]           # major, minor axes
    axes = np.vstack([a1, a2])
    ue = P @ a1; ve = P @ a2

    # ---- per-cell ground observations ----
    iu = np.floor(ue / cell).astype(np.int64)
    iv = np.floor(ve / cell).astype(np.int64)
    keys = iu * 1_000_003 + iv
    order = np.argsort(keys)
    ks = keys[order]; zs = ze[order]; us = ue[order]; vs = ve[order]
    uk, st = np.unique(ks, return_index=True)
    en = np.append(st[1:], len(zs))
    ou, ov, oz = [], [], []
    for s, e in zip(st, en):
        if e - s >= 20:                    # need enough returns per cell
            ou.append(us[s:e].mean())
            ov.append(vs[s:e].mean())
            oz.append(np.percentile(zs[s:e], pctl))
    ou = np.array(ou); ov = np.array(ov); oz = np.array(oz)
    if len(oz) < 8:
        raise RuntimeError(
            f"Only {len(oz)} exterior ground cells - not enough to fit a "
            "surface. Reduce 'cell' or check the exterior coverage.")

    # ---- guarded term selection ----
    # Base plane is always kept; curvature terms are added one at a time and
    # kept only if the design stays well-conditioned AND the term genuinely
    # reduces robust residuals. This automatically rejects terms that the
    # exterior geometry cannot constrain (e.g. v^2 when alleys exist only on
    # two sides of the trial).
    def fit(terms):
        A = GroundModel._expand(ou, ov, terms)
        # column scaling for a meaningful condition number
        s = np.linalg.norm(A, axis=0); s[s == 0] = 1.0
        cond = np.linalg.cond(A / s)
        coef, *_ = np.linalg.lstsq(A, oz, rcond=None)
        resid = A @ coef - oz
        return coef, float(np.sqrt(np.mean(resid ** 2))), cond

    terms = ["1", "u", "v"]
    coef, rms, _ = fit(terms)
    for extra in ("u2", "uv", "v2"):
        trial = terms + [extra]
        c2, rms2, cond2 = fit(trial)
        if cond2 < cond_max and rms2 < rms * 0.97:
            terms, coef, rms = trial, c2, rms2

    # robust re-fit: drop cells > 3*rms once (vegetation clumps in the alley)
    A = GroundModel._expand(ou, ov, terms)
    resid = A @ coef - oz
    keep = np.abs(resid) < max(3 * rms, 0.05)
    if keep.sum() >= 8 and keep.sum() < len(oz):
        coef, *_ = np.linalg.lstsq(A[keep], oz[keep], rcond=None)
        resid = GroundModel._expand(ou[keep], ov[keep], terms) @ coef - oz[keep]
        rms = float(np.sqrt(np.mean(resid ** 2)))

    diag = {
        "n_exterior_points": int(len(ext)),
        "n_ground_cells":    int(len(oz)),
        "terms":             list(terms),
        "rms_resid_m":       rms,
        "cell_m":            cell,
        "percentile":        pctl,
    }
    return GroundModel((x0, y0), axes, terms, coef, diag)
