"""
Spectral index catalogue and evaluation for VNIR hyperspectral cubes.

The catalogue is the *Awesome Spectral Indices* (ASI) database (Montero et
al. 2023, Scientific Data 10:197; MIT licence), bundled as JSON under
phenoapp/assets/spectral_indices/, plus a small set of narrow-band
hyperspectral indices that ASI does not carry (red-edge position, WBI,
PRI, Vogelmann, the legacy PhenoApp NDRE).

How a hyperspectral cube feeds an index
---------------------------------------
ASI formulas are written in terms of *broad* bands (N, R, G, B, RE1, ...)
and *narrow* bands (R705, R850, ...).  For a cube with a wavelength per
band we synthesise them:

  broad band  -> mean of every cube band whose centre wavelength falls
                 inside the ASI wavelength range for that band
                 (bands.json: N = 760-900 nm, RE1 = 695-715 nm, ...)
  narrow band -> the single cube band closest to the requested wavelength
  R<a>_<b>    -> mean of the cube bands between a and b nm

Bands whose wavelength lies in an *excluded range* (sensor edge, O2-A
absorption at 761 nm, water vapour 930-960 nm, ...) are never used.  A
pixel is valid for an index only when every cube band feeding it is > 0:
the GRYFN cubes store reflectance as unsigned integers, so negative
reflectance (dark canopy in the blue) is clipped to 0 = nodata.  The per-
index valid fraction is reported so that indices built on unreliable bands
are visible rather than silently biased.

Public API
----------
IndexCatalogue()                      loads ASI + PhenoApp extras
  .entries: dict acronym -> IndexDef
  .domains()                          ordered list of category names
  .by_domain(wavelengths, excluded)   {domain: [IndexDef]} computable here
  .computable(acronym, wavelengths, excluded) -> bool
RECOMMENDED_BIOMASS                   preset list of acronyms
compute_plot_indices(cube, plots, regions, acronyms, ...) -> (df, qc_df, meta)
compute_index_window(cube, acronym, window, ...)          -> 2-D array
write_index_rasters(cube, acronyms, out_dir, ...)        -> [paths]
write_qml_sidecar(raster_path, wavelengths)              -> QGIS style path
"""

from __future__ import annotations
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# catalogue
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _asset_dir() -> str:
    cands = []
    if hasattr(sys, "_MEIPASS"):
        cands.append(os.path.join(sys._MEIPASS, "phenoapp", "assets", "spectral_indices"))
        cands.append(os.path.join(sys._MEIPASS, "assets", "spectral_indices"))
    cands.append(os.path.join(os.path.dirname(_HERE), "assets", "spectral_indices"))
    for c in cands:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("spectral_indices assets folder not found")


# Default wavelength ranges (nm) that are never used to build a band.
# Validated on the Muresk 2025 GOBI cubes: the first three bands (399-406 nm)
# carry a calibration spike, 410-500 nm is mostly clipped to 0 in canopy,
# 761 nm is the O2-A absorption residual and 930-960 nm the water-vapour
# feature. Users can edit the list in the tab.
DEFAULT_EXCLUDED_NM = [(0.0, 415.0), (755.0, 770.0), (928.0, 962.0)]
# A band (or an index) is reported for a plot only when it is valid (non-zero) on at least this fraction of the
# plot's region pixels. Below it the value is NaN: GRYFN clips negative reflectance to 0, so the surviving pixels
# are the brighter ones and their mean is biased, not representative. Applies to plot_spectra, plot_table,
# compute_plot_indices and the VNIR v3 features, so every per-plot band table carries NaN where the band is unusable.
MIN_VALID_FRAC = 0.5


def band_usable(wl, valid_frac, reflectance_ok=True, frac_in_cube=None, excluded=None, min_valid_frac=None):
    """Per plot and band: 1 = usable, 0 = do not use (treat as NaN).

    wl             : wavelengths (n_bands)
    valid_frac     : (n_plots, n_bands) fraction of the plot's region pixels with a valid (non-zero) value
    reflectance_ok : flight-level flag (False for a radiance / DN cube -> everything 0)
    frac_in_cube   : optional (n_plots,) fraction of the region inside the cube (> 0.9 required)
    A band is usable when valid on >= MIN_VALID_FRAC of the pixels and outside the excluded ranges.
    This is the table written as *_band_usable.csv next to every per-plot spectra table, and the rule
    behind the NaN cells in those tables.
    """
    wl = np.asarray(wl, float); vf = np.asarray(valid_frac, float)
    mvf = MIN_VALID_FRAC if min_valid_frac is None else float(min_valid_frac)
    ex = _excluded_mask(wl, DEFAULT_EXCLUDED_NM if excluded is None else excluded)
    ok = (vf >= mvf) & ~ex[None, :]
    if frac_in_cube is not None:
        ok &= (np.asarray(frac_in_cube, float) > 0.9)[:, None]
    if not reflectance_ok:
        ok[:] = False
    return ok.astype(np.int8)

# PhenoApp narrow-band extras (not in ASI, or hyperspectral variants of it).
_EXTRA = [
    dict(acronym="NDRE740", name="Normalised Difference Red Edge (740/800, PhenoApp legacy 'NDRE')",
         formula="(R800 - R740) / (R800 + R740)", bands=["R800", "R740"], domain="hyperspectral narrow-band",
         reference="PhenoApp v2 (validated on the 2025 DPIRD Fodder trials); Gitelson & Merzlyak (1994) form with 740 nm"),
    dict(acronym="NDRE720", name="Normalised Difference Red Edge (720/790, Barnes et al. 2000)",
         formula="(R790 - R720) / (R790 + R720)", bands=["R790", "R720"], domain="hyperspectral narrow-band",
         reference="Barnes, E.M. et al. (2000) Coincident detection of crop water stress, nitrogen status and canopy "
                   "density using ground-based multispectral data. Proc. 5th Int. Conf. Precision Agriculture"),
    dict(acronym="REP", name="Red-Edge Position, linear four-point interpolation (Guyot & Baret 1988), nm",
         formula="700 + 40 * ((R670 + R780) / 2 - R700) / (R740 - R700)", bands=["R670", "R780", "R700", "R740"],
         domain="hyperspectral narrow-band",
         reference="Guyot, G. & Baret, F. (1988) Utilisation de la haute resolution spectrale pour suivre l'etat des "
                   "couverts vegetaux. Proc. 4th Int. Colloquium on Spectral Signatures of Objects in Remote Sensing, "
                   "ESA SP-287, 279-286"),
    dict(acronym="WBI", name="Water Band Index (900/970)", formula="R900 / R970", bands=["R900", "R970"],
         domain="hyperspectral narrow-band",
         reference="Penuelas, J. et al. (1993) The reflectance at the 950-970 nm region as an indicator of plant water "
                   "status. Int. J. Remote Sensing 14(10), 1887-1905"),
    dict(acronym="NDWI970", name="Normalised Difference Water Index (850/970)",
         formula="(R850 - R970) / (R850 + R970)", bands=["R850", "R970"], domain="hyperspectral narrow-band",
         reference="PhenoApp v2 variant of the Penuelas et al. (1993) 970 nm water feature"),
    dict(acronym="PRI", name="Photochemical Reflectance Index (531/570)", formula="(R531 - R570) / (R531 + R570)",
         bands=["R531", "R570"], domain="hyperspectral narrow-band",
         reference="Gamon, J.A., Penuelas, J. & Field, C.B. (1992) A narrow-waveband spectral index that tracks diurnal "
                   "changes in photosynthetic efficiency. Remote Sensing of Environment 41, 35-44"),
    dict(acronym="VOG1", name="Vogelmann Red Edge Index 1 (740/720)", formula="R740 / R720", bands=["R740", "R720"],
         domain="hyperspectral narrow-band",
         reference="Vogelmann, J.E., Rock, B.N. & Moss, D.M. (1993) Red edge spectral measurements from sugar maple "
                   "leaves. Int. J. Remote Sensing 14, 1563-1575"),
    dict(acronym="VOG2", name="Vogelmann Red Edge Index 2", formula="(R734 - R747) / (R715 + R726)",
         bands=["R734", "R747", "R715", "R726"], domain="hyperspectral narrow-band",
         reference="Vogelmann, J.E., Rock, B.N. & Moss, D.M. (1993) Int. J. Remote Sensing 14, 1563-1575"),
    dict(acronym="CIre705", name="Chlorophyll Index red edge (800/705)", formula="R800 / R705 - 1",
         bands=["R800", "R705"], domain="hyperspectral narrow-band",
         reference="Gitelson, A.A., Gritz, Y. & Merzlyak, M.N. (2003) Relationships between leaf chlorophyll content "
                   "and spectral reflectance. J. Plant Physiology 160, 271-282"),
    dict(acronym="NDVI_nb", name="NDVI narrow-band (800/670)", formula="(R800 - R670) / (R800 + R670)",
         bands=["R800", "R670"], domain="hyperspectral narrow-band",
         reference="PhenoApp v2 (matches the Biomass-tab NDVI column)"),
]

# Preset: indices that carried biomass information on the 2025 DPIRD trials
# (Fodder pastures + Muresk NUE wheat) - red-edge family first.
RECOMMENDED_BIOMASS = ["NDRE740", "NDRE720", "REP", "S2REP", "LCI", "MTCI", "NDREI", "CIRE", "CIre705", "VOG1",
                       "NDVI_nb", "OSAVI", "SAVI", "GNDVI", "kNDVI", "NIRv", "WDRVI", "MCARI", "TCARI", "TCARIOSAVI",
                       "mND705", "PRI", "WBI", "NDWI970"]

DOMAIN_ORDER = ["hyperspectral narrow-band", "vegetation", "water", "soil", "burn", "snow", "urban", "clouds",
                "geology"]
DOMAIN_LABELS = {
    "hyperspectral narrow-band": "Hyperspectral narrow-band (PhenoApp)",
    "vegetation": "Vegetation", "water": "Water", "soil": "Soil", "burn": "Burn", "snow": "Snow / ice",
    "urban": "Urban", "clouds": "Clouds", "geology": "Geology",
}

_NARROW_RE = re.compile(r"^R(\d{3,4})$")
_RANGE_RE = re.compile(r"^R(\d{3,4})_(\d{3,4})$")
_FUNCS_OK = {"kernel", "log", "tanh", "max", "exp", "sqrt"}


@dataclass
class IndexDef:
    acronym: str
    name: str
    formula: str
    bands: list
    domain: str
    constants: dict = field(default_factory=dict)      # name -> default value
    constants_doc: dict = field(default_factory=dict)  # name -> description
    modalities: list = field(default_factory=list)
    reference: str = ""
    link: str = ""
    source: str = "ASI"
    unsupported: str = ""   # non-empty -> cannot be evaluated (reason)

    def tooltip(self) -> str:
        c = ", ".join(f"{k}={v}" for k, v in self.constants.items())
        s = f"{self.acronym}: {self.name}\n\nformula: {self.formula}\nbands: {', '.join(self.bands)}"
        if c:
            s += f"\nconstants: {c}"
        if self.reference:
            s += f"\n\n{self.reference}"
        if self.link:
            s += f"\n{self.link}"
        return s


class IndexCatalogue:
    def __init__(self, asset_dir: str | None = None):
        d = asset_dir or _asset_dir()
        with open(os.path.join(d, "spectral-indices-dict.json"), encoding="utf-8") as f:
            raw = json.load(f)["SpectralIndices"]
        with open(os.path.join(d, "bands.json"), encoding="utf-8") as f:
            self.bands = json.load(f)          # code -> {min_wavelength, max_wavelength, long_name, ...}
        self.entries: dict[str, IndexDef] = {}
        for acr, v in raw.items():
            cls = v.get("classification", {}) or {}
            src = v.get("source", {}) or {}
            meta = src.get("source_metadata", {}) or {}
            cite = ""
            if isinstance(meta, dict):
                cite = (meta.get("how_to_cite", {}) or {}).get("apa", "") or ""
            consts = {k: (c or {}).get("default_value") for k, c in (v.get("constants") or {}).items()}
            cdoc = {k: (c or {}).get("description", "") for k, c in (v.get("constants") or {}).items()}
            e = IndexDef(acronym=acr, name=v.get("name", acr), formula=v.get("formula", ""),
                         bands=list(v.get("bands", [])), domain=cls.get("application_domain", "other"),
                         constants=consts, constants_doc=cdoc, modalities=list(cls.get("sensing_modalities", [])),
                         reference=cite, link=src.get("source_link", "") or "", source="ASI")
            if "radar" in e.modalities or "thermal" in e.modalities:
                e.unsupported = "radar / thermal index"
            elif "spatial_max" in e.formula or v.get("reductions"):
                e.unsupported = "needs a spatial reduction over the whole image"
            elif v.get("external_variables"):
                e.unsupported = "needs external variables (" + ", ".join(v["external_variables"]) + ")"
            elif any(c is None for c in consts.values()):
                e.unsupported = "constant without a default value: " + ", ".join(
                    k for k, c in consts.items() if c is None)
            self.entries[acr] = e
        for x in _EXTRA:
            self.entries[x["acronym"]] = IndexDef(
                acronym=x["acronym"], name=x["name"], formula=x["formula"], bands=x["bands"], domain=x["domain"],
                modalities=["hyperspectral"], reference=x["reference"], source="PhenoApp")
        self.asset_dir = d

    # ------------------------------------------------------------------
    def domains(self) -> list:
        present = {e.domain for e in self.entries.values()}
        return [d for d in DOMAIN_ORDER if d in present] + sorted(present - set(DOMAIN_ORDER))

    def band_indices(self, band: str, wavelengths, excluded=None):
        """0-based cube band indices used to synthesise `band`, or None."""
        return resolve_band(band, np.asarray(wavelengths, float), self.bands, excluded)

    def computable(self, acronym: str, wavelengths, excluded=None) -> bool:
        e = self.entries.get(acronym)
        if e is None or e.unsupported:
            return False
        return all(self.band_indices(b, wavelengths, excluded) is not None for b in e.bands)

    def by_domain(self, wavelengths, excluded=None, include_uncomputable=False) -> dict:
        out = {d: [] for d in self.domains()}
        for e in sorted(self.entries.values(), key=lambda x: x.acronym.lower()):
            if include_uncomputable or self.computable(e.acronym, wavelengths, excluded):
                out[e.domain].append(e)
        return {d: v for d, v in out.items() if v}


# ----------------------------------------------------------------------
# band synthesis
# ----------------------------------------------------------------------
def _excluded_mask(wl, excluded):
    m = np.zeros(len(wl), bool)
    for lo, hi in (excluded or []):
        m |= (wl >= lo) & (wl <= hi)
    return m


def parse_excluded(text: str):
    """'0-415, 755-770, 928-962' -> [(0,415),(755,770),(928,962)]"""
    out = []
    for part in re.split(r"[;,]", text or ""):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([0-9.]+)\s*[-:]\s*([0-9.]+)$", part)
        if not m:
            raise ValueError(f"cannot parse excluded range '{part}' (use e.g. 755-770)")
        lo, hi = float(m.group(1)), float(m.group(2))
        out.append((min(lo, hi), max(lo, hi)))
    return out


def resolve_band(band: str, wl: np.ndarray, band_table: dict, excluded=None):
    """Return 0-based cube band indices for an ASI band name, or None if it
    cannot be built from this cube (wavelength outside range / all excluded)."""
    ex = _excluded_mask(wl, excluded)
    ok = ~ex
    if not ok.any():
        return None
    m = _RANGE_RE.match(band)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        sel = np.where(ok & (wl >= lo) & (wl <= hi))[0]
        return sel if len(sel) else None
    m = _NARROW_RE.match(band)
    if m:
        nm = float(m.group(1))
        step = float(np.median(np.diff(wl))) if len(wl) > 1 else 5.0
        if nm < wl.min() - step or nm > wl.max() + step:
            return None
        cand = np.where(ok)[0]
        i = cand[np.argmin(np.abs(wl[cand] - nm))]
        if abs(wl[i] - nm) > max(2.0 * step, 8.0):   # nearest usable band too far away
            return None
        return np.array([i])
    if band in band_table:
        lo, hi = float(band_table[band]["min_wavelength"]), float(band_table[band]["max_wavelength"])
        sel = np.where(ok & (wl >= lo) & (wl <= hi))[0]
        return sel if len(sel) else None
    return None


# ----------------------------------------------------------------------
# safe formula evaluation
# ----------------------------------------------------------------------
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


def _kernel(a, b):
    """RBF kernel as used by spyndex for the kernel indices (sigma = (a+b)/2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    sigma = 0.5 * (a + b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(-((a - b) ** 2) / (2.0 * sigma ** 2))


_FUNC_IMPL = {"kernel": _kernel, "log": np.log, "tanh": np.tanh, "exp": np.exp, "sqrt": np.sqrt,
              "max": lambda *a: np.maximum.reduce([np.asarray(x, float) for x in a])}


def _eval_node(node, env):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise KeyError(f"unknown symbol '{node.id}'")
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("bad unary operator")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _BIN[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS_OK:
        args = [_eval_node(a, env) for a in node.args]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return _FUNC_IMPL[node.func.id](*args)
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def evaluate_formula(formula: str, env: dict):
    """Evaluate an ASI formula with numpy semantics on the arrays in env.
    Only arithmetic, numbers, names and the whitelisted functions are allowed."""
    tree = ast.parse(formula.strip(), mode="eval")
    return np.asarray(_eval_node(tree, env), dtype=np.float64)


def constants_env(e: IndexDef, overrides: dict | None = None) -> dict:
    env = {k: float(v) for k, v in e.constants.items() if v is not None}
    for k, v in (overrides or {}).items():
        if k in e.constants:
            env[k] = float(v)
    return env


# ----------------------------------------------------------------------
# cube helpers
# ----------------------------------------------------------------------
def parse_scale(hdr_path: str, default: float = 1.0) -> float:
    """'reflectance scale factor' from an ENVI header (GRYFN writes 10000)."""
    try:
        with open(hdr_path, "r", errors="ignore") as f:
            m = re.search(r"reflectance\s+scale\s+factor\s*=\s*([0-9.eE+-]+)", f.read(), re.I)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return default


def guess_scale(cube) -> float:
    """Scale factor for a VNIRCube: header value, else 10000 for integer cubes."""
    s = parse_scale(getattr(cube, "hdr", "") or "", default=0.0)
    if s > 0:
        return s
    try:
        return 10000.0 if np.issubdtype(np.dtype(cube._src.dtypes[0]), np.integer) else 1.0
    except Exception:
        return 1.0


def _needed_bands(cat: IndexCatalogue, acronyms, wl, excluded):
    """{band_name: idx array} for every band used by the chosen indices."""
    need = {}
    for a in acronyms:
        e = cat.entries[a]
        for b in e.bands:
            if b not in need:
                idx = cat.band_indices(b, wl, excluded)
                if idx is None:
                    raise ValueError(f"{a}: band {b} cannot be built from this cube "
                                     "(outside its wavelength range or fully excluded)")
                need[b] = idx
    return need


def read_band(src, band_1based: int, window=None, out_shape=None):
    """Read one cube band, optionally windowed and/or decimated (nearest)."""
    from rasterio.enums import Resampling
    kw = {}
    if window is not None:
        kw["window"] = window
    if out_shape is not None:
        kw["out_shape"] = tuple(int(v) for v in out_shape)
        kw["resampling"] = Resampling.nearest
    return src.read(band_1based, **kw)


def synthesise_bands(src, need: dict, window, scale: float, progress_cb=None, sel=None, out_shape=None):
    """Stream the cube once and build every synthesised band in `need`.

    src       : open rasterio dataset
    need      : {band_name: 0-based cube band indices}
    window    : rasterio Window (or None for the full image)
    sel       : optional flat pixel indices inside the window to keep (memory)
    out_shape : optional (rows, cols) decimated read (viewer use)
    Returns {band_name: float64 array (window shape, or 1-D if sel)} with NaN
    where any member band is 0 (clipped / nodata).
    """
    cube_bands = sorted({int(i) for idx in need.values() for i in idx})
    first = read_band(src, cube_bands[0] + 1, window, out_shape)
    shape = first.shape
    if sel is not None:
        shape = (len(sel),)
    acc = {b: np.zeros(shape, np.float64) for b in need}
    cnt = {b: np.zeros(shape, np.int32) for b in need}
    for k, bi in enumerate(cube_bands):
        band = first if k == 0 else read_band(src, bi + 1, window, out_shape)
        band = band.astype(np.float64)
        if sel is not None:
            band = band.ravel()[sel]
        good = band > 0
        for b, idx in need.items():
            if bi in idx:
                acc[b] += np.where(good, band, 0.0)
                cnt[b] += good
        if progress_cb and (k % 5 == 0 or k == len(cube_bands) - 1):
            progress_cb(int(70 * (k + 1) / len(cube_bands)), f"reading cube band {bi + 1}")
    vals = {}
    for b, idx in need.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            v = acc[b] / np.maximum(cnt[b], 1) / scale
        vals[b] = np.where(cnt[b] == len(idx), v, np.nan)       # strict: every member band valid
    return vals


# ----------------------------------------------------------------------
# per-plot computation
# ----------------------------------------------------------------------
def compute_plot_indices(cube, plots, regions, acronyms, mode: str = "pixel",
                         excluded=None, constants=None, catalogue: IndexCatalogue | None = None,
                         progress_cb=None, spectra=None):
    """Per-plot spectral indices.

    cube      : phenoapp.core.vnir.VNIRCube
    plots     : GeoDataFrame with Plot_ID
    regions   : list of sampling polygons (same order as plots)
    acronyms  : list of catalogue acronyms to compute
    mode      : 'pixel'  - index evaluated per pixel, then averaged per plot
                           (mean, sd; valid fraction reported)
                'mean'   - index evaluated on the plot mean spectrum
                           (zeros excluded per band, like the Biomass tab)
    excluded  : list of (lo, hi) nm ranges never used (DEFAULT_EXCLUDED_NM)
    constants : {acronym: {const: value}} overrides
    spectra   : optional (n_plots, n_bands) mean spectra in cube units to
                reuse in 'mean' mode

    Returns (df, qc, meta): df has Plot_ID + one column per index (mean);
    qc has <acr>_sd and <acr>_valid (fraction of region pixels valid);
    meta is a provenance dict (bands used, formulas, settings).
    """
    import pandas as pd
    cat = catalogue or IndexCatalogue()
    wl = np.asarray(cube.wavelengths, float)
    excluded = DEFAULT_EXCLUDED_NM if excluded is None else list(excluded)
    acronyms = [a for a in acronyms if a in cat.entries]
    if not acronyms:
        raise ValueError("no indices selected")
    need = _needed_bands(cat, acronyms, wl, excluded)
    scale = guess_scale(cube)
    src = cube._src
    lab = cube._labels(regions).ravel()
    n_plots = len(regions)
    region_px = np.bincount(lab, minlength=n_plots + 1)[1:]
    sel = np.flatnonzero(lab)
    lab_sel = lab[sel]
    pids = list(plots["Plot_ID"]) if "Plot_ID" in plots else list(range(1, n_plots + 1))
    n_read = len({int(i) for idx in need.values() for i in idx})
    meta = dict(mode=mode, scale_factor=scale, excluded_nm=[list(map(float, r)) for r in excluded], cube=cube.path,
                n_cube_bands_read=n_read,
                bands={b: dict(cube_bands=[int(i) + 1 for i in idx],
                               wavelengths_nm=[round(float(wl[i]), 2) for i in idx]) for b, idx in need.items()},
                indices={a: dict(name=cat.entries[a].name, formula=cat.entries[a].formula,
                                 bands=cat.entries[a].bands,
                                 constants=constants_env(cat.entries[a], (constants or {}).get(a)),
                                 source=cat.entries[a].source, reference=cat.entries[a].reference)
                         for a in acronyms})

    cols, qc = {}, {}
    if mode == "pixel":
        vals = synthesise_bands(src, need, None, scale, progress_cb=progress_cb, sel=sel)
        for j, a in enumerate(acronyms):
            e = cat.entries[a]
            env = {b: vals[b] for b in e.bands}
            env.update(constants_env(e, (constants or {}).get(a)))
            r = evaluate_formula(e.formula, env)
            ok = np.isfinite(r)
            gl = np.where(ok, lab_sel, 0)
            n = np.bincount(gl, minlength=n_plots + 1)[1:]
            s = np.bincount(gl, weights=np.where(ok, r, 0.0), minlength=n_plots + 1)[1:]
            s2 = np.bincount(gl, weights=np.where(ok, r * r, 0.0), minlength=n_plots + 1)[1:]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
                var = np.where(n > 1, s2 / np.maximum(n, 1) - mean ** 2, np.nan)
            vfrac = np.where(region_px > 0, n / np.maximum(region_px, 1), 0.0)
            cols[a] = np.where(vfrac >= MIN_VALID_FRAC, mean, np.nan)       # NaN when the index is valid on too few pixels
            qc[f"{a}_sd"] = np.where(vfrac >= MIN_VALID_FRAC, np.sqrt(np.maximum(var, 0)), np.nan)
            qc[f"{a}_valid"] = vfrac
            if progress_cb:
                progress_cb(70 + int(28 * (j + 1) / max(len(acronyms), 1)), f"index {a}")
    elif mode == "mean":
        if spectra is None:
            cube_bands = sorted({int(i) for idx in need.values() for i in idx})
            spectra = np.full((n_plots, len(wl)), np.nan)
            for k, bi in enumerate(cube_bands):
                band = src.read(bi + 1).ravel()[sel].astype(np.float64)
                good = band > 0
                gl = np.where(good, lab_sel, 0)
                n = np.bincount(gl, minlength=n_plots + 1)[1:]
                s = np.bincount(gl, weights=np.where(good, band, 0.0), minlength=n_plots + 1)[1:]
                frac = np.where(region_px > 0, n / np.maximum(region_px, 1), 0.0)
                spectra[:, bi] = np.where((n > 0) & (frac >= MIN_VALID_FRAC), s / np.maximum(n, 1), np.nan)   # NaN: band unusable for this plot
                if progress_cb and (k % 5 == 0 or k == len(cube_bands) - 1):
                    progress_cb(int(70 * (k + 1) / len(cube_bands)), f"mean spectrum band {bi + 1}")
        spectra = np.asarray(spectra, float)
        vals = {b: np.nanmean(spectra[:, idx], axis=1) / scale for b, idx in need.items()}
        for a in acronyms:
            e = cat.entries[a]
            env = {b: vals[b] for b in e.bands}
            env.update(constants_env(e, (constants or {}).get(a)))
            r = evaluate_formula(e.formula, env)
            cols[a] = np.where(np.isfinite(r), r, np.nan)
    else:
        raise ValueError("mode must be 'pixel' or 'mean'")

    df = pd.DataFrame({"Plot_ID": pids, **cols})
    qcdf = pd.DataFrame({"Plot_ID": pids, **qc})
    if progress_cb:
        progress_cb(100, "spectral indices done")
    return df, qcdf, meta


def summarise_indices(df, qc=None) -> str:
    """Plain-text summary table (mean, sd across plots, valid fraction)."""
    lines = [f"{'index':14s} {'mean':>10s} {'sd':>10s} {'min':>10s} {'max':>10s} {'valid%':>7s}"]
    for c in df.columns:
        if c == "Plot_ID":
            continue
        v = df[c].to_numpy(float)
        valid = ""
        if qc is not None and f"{c}_valid" in qc.columns:
            valid = f"{100 * float(np.nanmean(qc[f'{c}_valid'])):6.1f}"
        lines.append(f"{c:14s} {np.nanmean(v):10.4g} {np.nanstd(v):10.4g} {np.nanmin(v):10.4g} "
                     f"{np.nanmax(v):10.4g} {valid:>7s}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# index arrays / rasters for viewing
# ----------------------------------------------------------------------
def compute_index_window(cube, acronym, window=None, excluded=None, constants=None,
                         catalogue: IndexCatalogue | None = None, progress_cb=None, out_shape=None):
    """2-D float32 index array over a rasterio window of the cube (NaN = invalid).
    `out_shape` reads the window decimated (viewer)."""
    cat = catalogue or IndexCatalogue()
    wl = np.asarray(cube.wavelengths, float)
    excluded = DEFAULT_EXCLUDED_NM if excluded is None else list(excluded)
    need = _needed_bands(cat, [acronym], wl, excluded)
    vals = synthesise_bands(cube._src, need, window, guess_scale(cube), progress_cb=progress_cb, out_shape=out_shape)
    e = cat.entries[acronym]
    env = {b: vals[b] for b in e.bands}
    env.update(constants_env(e, (constants or {}).get(acronym)))
    r = evaluate_formula(e.formula, env).astype(np.float32)
    r[~np.isfinite(r)] = np.nan
    return r


def write_index_rasters(cube, acronyms, out_dir, bounds=None, excluded=None, constants=None,
                        catalogue: IndexCatalogue | None = None, progress_cb=None, base_name="index"):
    """Write one float32 GeoTIFF per index over `bounds` (minx, miny, maxx,
    maxy; default = whole cube). Invalid pixels = NaN. Returns paths."""
    import rasterio
    from rasterio.windows import from_bounds, Window
    cat = catalogue or IndexCatalogue()
    wl = np.asarray(cube.wavelengths, float)
    excluded = DEFAULT_EXCLUDED_NM if excluded is None else list(excluded)
    acronyms = [a for a in acronyms if a in cat.entries]
    need = _needed_bands(cat, acronyms, wl, excluded)
    src = cube._src
    if bounds is None:
        w = Window(0, 0, src.width, src.height)
    else:
        w = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        w = w.intersection(Window(0, 0, src.width, src.height))
    tr = rasterio.windows.transform(w, src.transform)
    H, W = int(w.height), int(w.width)
    vals = synthesise_bands(src, need, w, guess_scale(cube), progress_cb=progress_cb)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for j, a in enumerate(acronyms):
        e = cat.entries[a]
        env = {b: vals[b] for b in e.bands}
        env.update(constants_env(e, (constants or {}).get(a)))
        r = evaluate_formula(e.formula, env).astype(np.float32)
        r[~np.isfinite(r)] = np.nan
        p = os.path.join(out_dir, f"{base_name}_{a}.tif")
        with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=1, dtype="float32", crs=src.crs,
                           transform=tr, nodata=np.nan, compress="deflate", tiled=True, BIGTIFF="IF_SAFER") as dst:
            dst.write(r, 1)
            dst.set_band_description(1, f"{a}: {e.name}")
            dst.update_tags(formula=e.formula, bands=",".join(e.bands), source=e.source)
        paths.append(p)
        if progress_cb:
            progress_cb(70 + int(30 * (j + 1) / len(acronyms)), f"raster {a}")
    return paths


# ----------------------------------------------------------------------
# QGIS style sidecars for multi-band cubes
# ----------------------------------------------------------------------
def qgis_multiband_qml(red_band: int, green_band: int, blue_band: int, maxima=(6000, 1500, 1300),
                       minima=(0, 0, 0)) -> str:
    """Minimal QGIS 3 layer style: multiband colour with a fixed stretch.
    Saved next to a raster as <name>.qml it is applied automatically on load,
    so a 172-band cube opens as a sensible composite instead of bands 1/2/3."""
    def ce(tag, lo, hi):
        return (f"<{tag}><minValue>{lo}</minValue><maxValue>{hi}</maxValue>"
                f"<algorithm>StretchToMinimumMaximum</algorithm></{tag}>")
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            "<qgis version=\"3.34\" styleCategories=\"AllStyleCategories\" hasScaleBasedVisibilityFlag=\"0\">\n"
            "  <pipe>\n"
            f"    <rasterrenderer type=\"multibandcolor\" redBand=\"{red_band}\" greenBand=\"{green_band}\" "
            f"blueBand=\"{blue_band}\" alphaBand=\"-1\" opacity=\"1\" nodataColor=\"\">\n"
            "      <rasterTransparency/>\n"
            f"      {ce('redContrastEnhancement', minima[0], maxima[0])}\n"
            f"      {ce('greenContrastEnhancement', minima[1], maxima[1])}\n"
            f"      {ce('blueContrastEnhancement', minima[2], maxima[2])}\n"
            "    </rasterrenderer>\n"
            "    <brightnesscontrast brightness=\"0\" contrast=\"0\" gamma=\"1\"/>\n"
            "    <huesaturation saturation=\"0\" grayscaleMode=\"0\" colorizeOn=\"0\" colorizeRed=\"255\" "
            "colorizeGreen=\"128\" colorizeBlue=\"128\" colorizeStrength=\"100\"/>\n"
            "    <rasterresampler maxOversampling=\"2\"/>\n"
            "  </pipe>\n"
            "  <blendMode>0</blendMode>\n"
            "</qgis>\n")


COMPOSITES = {
    "cir": ((800.0, 670.0, 550.0), (0.60, 0.15, 0.13)),   # NIR / red / green: robust on GRYFN cubes
    "rgb": ((640.0, 550.0, 470.0), (0.15, 0.13, 0.07)),   # true colour (blue is clipped on many cubes)
}


def composite_bands(wavelengths, composite: str = "cir", custom_nm=None):
    """1-based band triplet + reflectance maxima for a display composite."""
    wl = np.asarray(wavelengths, float)
    if custom_nm is not None:
        nm, mx = tuple(custom_nm), (0.6, 0.6, 0.6)
    else:
        nm, mx = COMPOSITES[composite]
    return tuple(int(np.argmin(np.abs(wl - x))) + 1 for x in nm), mx


def write_qml_sidecar(raster_path: str, wavelengths, composite: str = "cir", scale: float = 10000.0) -> str:
    """Write <raster>.qml (multiband colour composite). Returns the qml path."""
    (r, g, b), mx = composite_bands(wavelengths, composite)
    qml = os.path.splitext(raster_path)[0] + ".qml"
    with open(qml, "w", encoding="utf-8") as f:
        f.write(qgis_multiband_qml(r, g, b, maxima=tuple(int(m * scale) for m in mx)))
    return qml
