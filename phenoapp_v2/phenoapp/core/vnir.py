"""
VNIR / hyperspectral orthomosaic support (ENVI .bin/.hdr cubes).

Provides per-plot spectral indices and mean spectra for biomass / dry-matter
modelling, sampled over the same plot regions as the LiDAR traits.

Index cheat-sheet (validated on the 2025 DPIRD Fodder trials):
  NDRE  (R800-R740)/(R800+R740) : best single fresh-biomass predictor on
                                  dense pasture (red band of NDVI saturates).
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
        """
        import pandas as pd
        lab = self._labels(regions).ravel()
        n_plots = len(regions)
        region_px = np.bincount(lab, minlength=n_plots + 1)[1:]

        band_means = {}
        valid_px = None
        items = list(DEFAULT_INDEX_BANDS.items())
        for bi, (name, nm) in enumerate(items):
            band = self._src.read(self.band_at(nm)).astype(np.float64).ravel()
            good = band > 0
            gl = np.where(good, lab, 0)
            cnt = np.bincount(gl, minlength=n_plots + 1)[1:]
            ssum = np.bincount(gl, weights=band, minlength=n_plots + 1)[1:]
            with np.errstate(invalid="ignore"):
                band_means[name] = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
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
        df.insert(0, "Plot_ID", list(plots["Plot_ID"]) if "Plot_ID" in plots
                  else list(range(1, n_plots + 1)))
        df["vnir_px"] = valid_px
        with np.errstate(invalid="ignore", divide="ignore"):
            df["vnir_coverage"] = np.where(region_px > 0,
                                           valid_px / np.maximum(region_px, 1),
                                           0.0)
        return df

    # ------------------------------------------------------------------
    def plot_spectra(self, plots, regions, progress_cb=None):
        """Per-plot mean spectrum over ALL bands. Returns (spectra, valid_px).

        spectra : (n_plots, n_bands) float array, NaN where no valid pixels.
        """
        lab = self._labels(regions).ravel()
        n_plots = len(regions)
        nb = self._src.count
        spectra = np.full((n_plots, nb), np.nan)
        cnt_ref = None
        for b in range(1, nb + 1):
            band = self._src.read(b).astype(np.float64).ravel()
            good = band > 0
            gl = np.where(good, lab, 0)
            cnt = np.bincount(gl, minlength=n_plots + 1)[1:]
            ssum = np.bincount(gl, weights=band, minlength=n_plots + 1)[1:]
            ok = cnt > 0
            spectra[ok, b - 1] = ssum[ok] / cnt[ok]
            if cnt_ref is None:
                cnt_ref = cnt
            if progress_cb and b % 10 == 0:
                progress_cb(int(100 * b / nb), f"VNIR spectra band {b}/{nb}")
        return spectra, cnt_ref
