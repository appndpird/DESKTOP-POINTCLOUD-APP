"""
VNIR / hyperspectral orthomosaic support (ENVI .bin/.hdr cubes).

Provides per-plot spectral indices and mean spectra for biomass / dry-matter
modelling, sampled over the same plot regions as the LiDAR traits.

Index cheat-sheet (validated on dense pasture):
  NDRE  (R800-R740)/(R800+R740) : best single fresh-biomass predictor on
                                  dense pasture (red band of NDVI saturates).
                                  NOTE: this is the 740/800 nm variant, not the
                                  720/790 nm NDRE of Barnes et al. (2000) nor the
                                  ASI NDREI (705 nm); the VNIR Spectral tab offers all.
  NDVI  (R800-R670)/(R800+R670) : classic greenness; saturates on closed swards.
  WBI   R900/R970               : 970 nm water absorption; predicts dry-matter
                                  fraction (moisture), r ~ -0.8 on pasture.
  NDWI970 (R850-R970)/(R850+R970): alternative water index.

Public API
----------
VNIRCube(bin_path)
    .wavelengths                 np.ndarray (nm), parsed from the .hdr
    .band_at(nm)                 1-based band index closest to wavelength
    .plot_table(plots, regions)  DataFrame of indices + coverage per plot
    .plot_spectra(plots, regions)-> (spectra [n_plots x n_bands], n_px)
"""

from __future__ import annotations
import os
import re
import numpy as np


DEFAULT_INDEX_BANDS = {
    "R550": 550.0, "R670": 670.0, "R705": 705.0, "R740": 740.0,
    "R800": 800.0, "R850": 850.0, "R900": 900.0, "R970": 970.0,
}


def _find_header(bin_path: str) -> str:
    for cand in (bin_path + ".hdr",
                 os.path.splitext(bin_path)[0] + ".hdr"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"No ENVI header (.hdr) found next to {bin_path}")


def parse_envi_wavelengths(hdr_path: str) -> np.ndarray:
    """Parse the wavelength list from an ENVI header."""
    with open(hdr_path, "r", errors="ignore") as f:
        text = f.read()
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        raise RuntimeError(f"No 'wavelength = {{...}}' block in {hdr_path}")
    vals = [float(t) for t in re.split(r"[,\s]+", m.group(1).strip()) if t]
    return np.asarray(vals, float)


class VNIRCube:
    def __init__(self, bin_path: str):
        import rasterio
        self.path = bin_path
        self.hdr = _find_header(bin_path)
        self.wavelengths = parse_envi_wavelengths(self.hdr)
        self._src = rasterio.open(bin_path)
        if self._src.count != len(self.wavelengths):
            # tolerate mismatch but warn via attribute
            self.wavelengths = self.wavelengths[: self._src.count]
        self.crs = self._src.crs
        self.res = self._src.res

    def close(self):
        try: self._src.close()
        except Exception: pass

    # ------------------------------------------------------------------
    def band_at(self, nm: float) -> int:
        """1-based rasterio band index closest to a wavelength (nm)."""
        return int(np.argmin(np.abs(self.wavelengths - nm))) + 1

    # ------------------------------------------------------------------
    def _labels(self, regions):
        """Rasterize the plot regions onto the cube grid. 0 = background."""
        from rasterio import features
        shapes = [(geom, i + 1) for i, geom in enumerate(regions)]
        lab = features.rasterize(
            shapes, out_shape=(self._src.height, self._src.width),
            transform=self._src.transform, fill=0, dtype="int32",
            all_touched=False)
        return lab

    # ------------------------------------------------------------------
    def plot_table(self, plots, regions, progress_cb=None):
        """Per-plot mean reflectance at key bands + spectral indices.

        plots   : GeoDataFrame (for Plot_ID etc.)
        regions : list of sampling polygons (same order/length as plots)
        Returns a pandas DataFrame. Nodata (0) pixels are excluded; coverage
        = valid pixels / region pixels, so cropped or striped cubes are
        visible in the output rather than silently biasing means.
        Indices are ratios of the per-band region means (index of means),
        computed on the sampling REGION (whole plot = polygon inset 10 cm,
        or the central band) - not the full polygon used by
        write_plot_cubes. The VNIR Spectral tab computes per-pixel indices.
        """
        import pandas as pd
        lab = self._labels(regions).ravel()
        n_plots = len(regions)
        region_px = np.bincount(lab, minlength=n_plots + 1)[1:]

        from .spectral_indices import MIN_VALID_FRAC
        band_means = {}
        band_valid = {}
        valid_px = None
        items = list(DEFAULT_INDEX_BANDS.items())
        for bi, (name, nm) in enumerate(items):
            band = self._src.read(self.band_at(nm)).astype(np.float64).ravel()
            good = band > 0
            gl = np.where(good, lab, 0)
            cnt = np.bincount(gl, minlength=n_plots + 1)[1:]
            ssum = np.bincount(gl, weights=band, minlength=n_plots + 1)[1:]
            frac = np.where(region_px > 0, cnt / np.maximum(region_px, 1), 0.0)
            with np.errstate(invalid="ignore"):
                # NaN when the band is valid on fewer than MIN_VALID_FRAC of the region pixels (clipped band):
                # the indices built from it then come out NaN as well and cannot enter a model unnoticed
                band_means[name] = np.where((cnt > 0) & (frac >= MIN_VALID_FRAC), ssum / np.maximum(cnt, 1), np.nan)
            band_valid[name + "_valid"] = frac
            if name == "R800":
                # coverage is referenced to the NIR band: dropout rates are
                # band-dependent (short wavelengths are zeroed far more often)
                # and R800 is the band every index depends on
                valid_px = cnt
            if progress_cb:
                progress_cb(int(100 * (bi + 1) / len(items)),
                            f"VNIR band {name}")

        df = pd.DataFrame(band_means)
        eps = 1e-9
        df["NDVI"]    = (df.R800 - df.R670) / (df.R800 + df.R670 + eps)
        df["NDRE"]    = (df.R800 - df.R740) / (df.R800 + df.R740 + eps)
        df["WBI"]     = df.R900 / (df.R970 + eps)
        df["NDWI970"] = (df.R850 - df.R970) / (df.R850 + df.R970 + eps)
        # flight-level reflectance check: a radiance / DN cube gives NDVI ~ 0 in every plot -> nothing is usable
        med_ndvi = float(np.nanmedian(df["NDVI"])) if np.isfinite(df["NDVI"]).any() else np.nan
        refl_ok = not (np.isfinite(med_ndvi) and med_ndvi < 0.05)
        if not refl_ok:
            for c in list(band_means) + ["NDVI", "NDRE", "WBI", "NDWI970"]:
                df[c] = np.nan
        df.insert(0, "Plot_ID", list(plots["Plot_ID"]) if "Plot_ID" in plots
                  else list(range(1, n_plots + 1)))
        for k, v in band_valid.items():
            df[k] = v
        df["vnir_px"] = valid_px
        with np.errstate(invalid="ignore", divide="ignore"):
            df["vnir_coverage"] = np.where(region_px > 0,
                                           valid_px / np.maximum(region_px, 1),
                                           0.0)
        df["vnir_reflectance_ok"] = int(refl_ok)
        self.last_reflectance_ok = refl_ok
        return df

    # ------------------------------------------------------------------
    def plot_spectra(self, plots, regions, progress_cb=None, min_valid_frac=None, excluded=None):
        """Per-plot mean spectrum over ALL bands. Returns (spectra, valid_px).

        spectra : (n_plots, n_bands) float array in cube units. A band is NaN for a plot when
                  - it is valid (non-zero) on fewer than `min_valid_frac` of the region pixels
                    (default MIN_VALID_FRAC = 0.5; clipped reflectance),
                  - it lies in an excluded wavelength range (default DEFAULT_EXCLUDED_NM), or
                  - the cube fails the flight-level reflectance check (NDVI ~ 0 in every plot: a
                    radiance / DN cube), in which case every value is NaN.
                  Per-band valid fractions are kept in self.last_band_valid_frac (n_plots x n_bands)
                  and the reflectance flag in self.last_reflectance_ok, so the caller can write them
                  next to the spectra table.
        valid_px: valid pixels per plot in the first band (legacy).
        """
        from .spectral_indices import MIN_VALID_FRAC, DEFAULT_EXCLUDED_NM, _excluded_mask
        mvf = MIN_VALID_FRAC if min_valid_frac is None else float(min_valid_frac)
        excl = _excluded_mask(np.asarray(self.wavelengths, float), DEFAULT_EXCLUDED_NM if excluded is None else excluded)
        lab = self._labels(regions).ravel()
        n_plots = len(regions)
        region_px = np.bincount(lab, minlength=n_plots + 1)[1:]
        nb = self._src.count
        spectra = np.full((n_plots, nb), np.nan)
        vfrac = np.zeros((n_plots, nb))
        cnt_ref = None
        for b in range(1, nb + 1):
            band = self._src.read(b).astype(np.float64).ravel()
            good = band > 0
            gl = np.where(good, lab, 0)
            cnt = np.bincount(gl, minlength=n_plots + 1)[1:]
            ssum = np.bincount(gl, weights=band, minlength=n_plots + 1)[1:]
            frac = np.where(region_px > 0, cnt / np.maximum(region_px, 1), 0.0)
            vfrac[:, b - 1] = frac
            ok = (cnt > 0) & (frac >= mvf)
            if b - 1 < len(excl) and excl[b - 1]:
                ok[:] = False
            spectra[ok, b - 1] = ssum[ok] / cnt[ok]
            if cnt_ref is None:
                cnt_ref = cnt
            if progress_cb and b % 10 == 0:
                progress_cb(int(100 * b / nb), f"VNIR spectra band {b}/{nb}")
        b670 = self.band_at(670) - 1; b800 = self.band_at(800) - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (spectra[:, b800] - spectra[:, b670]) / (spectra[:, b800] + spectra[:, b670])
        refl_ok = not (np.isfinite(ndvi).any() and float(np.nanmedian(ndvi)) < 0.05)
        if not refl_ok:
            spectra[:] = np.nan
        self.last_band_valid_frac = vfrac
        self.last_reflectance_ok = refl_ok
        return spectra, cnt_ref

    # ------------------------------------------------------------------
    def write_plot_cubes(self, plots, out_dir, progress_cb=None,
                         compress="deflate", name_col="B/R", clip="polygon",
                         name_fmt="plot_{pid}_{name}", plot_ids=None):
        """Write one multi-band GeoTIFF per plot: every cube band, every pixel
        value exactly as in the orthomosaic (same grid, CRS, dtype; lossless
        DEFLATE), wavelengths as band descriptions and in a 'wavelengths_nm' tag.

        clip = "polygon" : the bounding window of the plot; pixels outside the
                           polygon are flagged by an INTERNAL GDAL MASK (per
                           dataset) and written as 0. No nodata tag is set, so
                           a pixel that is 0 in the orthomosaic (GRYFN clips
                           negative reflectance to 0) stays a visible value in
                           QGIS instead of a transparent hole; only the area
                           outside the polygon is transparent.
        clip = "bbox"    : the bounding window as is, nothing masked.
        name_fmt         : file name pattern with {pid} and {name}
        plot_ids         : optional list of Plot_IDs to write (default all)

        Bands are read once each for the whole trial and sliced per plot, so
        the cube is streamed one band at a time; all per-plot files are kept
        open and written band by band.

        Returns a DataFrame index: Plot_ID, name, path, width, height, px_inside
        (also written to <out_dir>/plots_vnir_index.csv).
        """
        import pandas as pd, rasterio
        from rasterio.windows import Window, from_bounds
        from rasterio.features import geometry_mask
        os.makedirs(out_dir, exist_ok=True)
        if plot_ids is not None:
            keep = set(int(p) for p in plot_ids)
            plots = plots[plots["Plot_ID"].astype(int).isin(keep)]
        src = self._src; tr = src.transform; nb = src.count
        geoms = list(plots.geometry)
        pids = list(plots["Plot_ID"]) if "Plot_ID" in plots else list(range(1, len(geoms) + 1))
        names = list(plots[name_col]) if name_col in plots else [f"P{p}" for p in pids]

        # union read window built from the per-plot INTEGER windows (rounding the
        # union bounds separately could be one pixel short on the last row/column)
        wins = []
        for g in geoms:
            w = from_bounds(*g.bounds, tr).round_offsets().round_lengths()
            try:
                w = w.intersection(Window(0, 0, src.width, src.height))
            except Exception:
                continue
            if w.width > 0 and w.height > 0:
                wins.append(w)
        if not wins:
            raise RuntimeError("None of the plots overlap the VNIR cube.")
        r_lo = min(int(w.row_off) for w in wins); c_lo = min(int(w.col_off) for w in wins)
        r_hi = max(int(w.row_off + w.height) for w in wins); c_hi = max(int(w.col_off + w.width) for w in wins)
        uw = Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
        u_r0, u_c0 = int(uw.row_off), int(uw.col_off)

        specs = []
        for pid, nm, g in zip(pids, names, geoms):
            w = from_bounds(*g.bounds, tr).round_offsets().round_lengths()
            try:
                w = w.intersection(Window(0, 0, src.width, src.height))
            except Exception:                 # plot entirely outside the cube
                specs.append(None); continue
            if w.width <= 0 or w.height <= 0:
                specs.append(None); continue
            wt = rasterio.windows.transform(w, tr)
            if clip == "bbox":
                inside = np.ones((int(w.height), int(w.width)), bool)
            else:
                inside = ~geometry_mask([g], out_shape=(int(w.height), int(w.width)), transform=wt, invert=False)
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(nm))
            path = os.path.join(out_dir, name_fmt.format(pid=pid, name=safe) + ".tif")
            prof = dict(driver="GTiff", width=int(w.width), height=int(w.height), count=nb, dtype=src.dtypes[0],
                        crs=src.crs, transform=wt, compress=compress, tiled=True,
                        blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
            dst = rasterio.open(path, "w", **prof)
            for b in range(1, nb + 1):
                dst.set_band_description(b, f"{self.wavelengths[b-1]:.1f} nm")
            dst.update_tags(Plot_ID=str(pid), plot_name=str(nm), source=os.path.basename(self.path),
                            wavelengths_nm=",".join(f"{w_:.2f}" for w_ in self.wavelengths),
                            clip=("bounding box, nothing masked" if clip == "bbox" else
                                  "plot polygon via internal GDAL mask; pixel values inside are exact copies "
                                  "of the orthomosaic (0 = clipped reflectance, NOT nodata); outside = masked"),
                            pixel_values="lossless copy of the orthomosaic (uint16 reflectance x scale factor)")
            specs.append(dict(pid=pid, name=nm, path=path, dst=dst, inside=inside,
                              r0=int(w.row_off) - u_r0, c0=int(w.col_off) - u_c0, h=int(w.height), w=int(w.width)))
        try:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                for b in range(1, nb + 1):
                    band = src.read(b, window=uw)
                    for sp in specs:
                        if sp is None:
                            continue
                        sub = band[sp["r0"]:sp["r0"] + sp["h"], sp["c0"]:sp["c0"] + sp["w"]]
                        if sub.shape != sp["inside"].shape:
                            # the union read came back short on its last row/column (rounding at the edge of the
                            # union window; seen on two plots of one flight): read this plot's own
                            # window straight from the cube instead of zero-padding, so no polygon pixel is lost
                            sub = src.read(b, window=Window(sp["c0"] + u_c0, sp["r0"] + u_r0, sp["w"], sp["h"]),
                                           boundless=True, fill_value=0)
                        sp["dst"].write(np.where(sp["inside"], sub, 0).astype(band.dtype), b)
                    if progress_cb and (b % 8 == 0 or b == nb):
                        progress_cb(int(100 * b / nb), f"per-plot VNIR cubes: band {b}/{nb}")
                if clip != "bbox":
                    for sp in specs:
                        if sp is not None:
                            sp["dst"].write_mask((sp["inside"] * 255).astype("uint8"))
        finally:
            for sp in specs:
                if sp is not None:
                    sp["dst"].close()
        # QGIS opens a 172-band GeoTIFF as bands 1/2/3 (the noisy 399-406 nm sensor
        # edge) - a .qml sidecar makes it open as a NIR/red/green composite instead.
        from .spectral_indices import write_qml_sidecar, guess_scale
        scale = guess_scale(self)
        for sp in specs:
            if sp is not None:
                try: write_qml_sidecar(sp["path"], self.wavelengths, "cir", scale)
                except Exception: pass
        rows = [dict(Plot_ID=sp["pid"], name=sp["name"], path=sp["path"], width=sp["w"], height=sp["h"],
                     px_inside=int(sp["inside"].sum())) for sp in specs if sp is not None]
        idx = pd.DataFrame(rows); idx.to_csv(os.path.join(out_dir, "plots_vnir_index.csv"), index=False)
        return idx
