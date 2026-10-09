"""
VNIR cube access without rasterio / GDAL (their DLLs are blocked on this machine).

ENVICube(bin_path)   drop-in for phenoapp.core.vnir.VNIRCube as far as vnir_features_all / guess_scale / band_usable need it:
    .wavelengths, .hdr, .path, ._src (count, dtypes, height, width, read(b)), ._labels(regions), .close()
    The cube is read band by band from the ENVI BSQ file with numpy.memmap (uint16 or float32, any byte order).
    _labels rasterises the sampling polygons with the same pixel-centre rule as rasterio.features.rasterize(all_touched=False).

read_plot_tif(path) -> (array (bands, rows, cols), (x0, y0, pixel_size), wavelengths)   per-plot GeoTIFF via tifffile
"""
import os, re, numpy as np
from shapely import contains_xy

_DT = {1: "uint8", 2: "int16", 3: "int32", 4: "float32", 5: "float64", 12: "uint16", 13: "uint32"}

def _hdr(bin_path):
    for cand in (bin_path + ".hdr", os.path.splitext(bin_path)[0] + ".hdr"):
        if os.path.exists(cand): return cand
    raise FileNotFoundError(f"no ENVI header next to {bin_path}")

def _parse_hdr(hdr):
    txt = open(hdr, "r", errors="ignore").read()
    def val(key, default=None):
        m = re.search(r"^\s*" + key + r"\s*=\s*(\{[^}]*\}|[^\n]+)", txt, re.I | re.M)
        return m.group(1).strip() if m else default
    d = dict(samples=int(val("samples")), lines=int(val("lines")), bands=int(val("bands")), dtype=_DT[int(val("data type"))],
             interleave=val("interleave", "bsq").lower(), byte_order=int(val("byte order", "0")), offset=int(val("header offset", "0")))
    wl = val("wavelength"); d["wavelengths"] = np.array([float(t) for t in re.split(r"[,\s]+", wl.strip("{}").strip()) if t]) if wl else None
    mi = val("map info")
    if mi:
        parts = [p.strip() for p in mi.strip("{}").split(",")]
        d["x0"], d["y0"], d["psx"], d["psy"] = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        d["ref_px"], d["ref_py"] = float(parts[1]), float(parts[2])
        # ENVI map info: pixel (1,1) = upper-left corner of the upper-left pixel -> tie point of the (0,0) corner
        d["x0"] -= (d["ref_px"] - 1.0) * d["psx"]; d["y0"] += (d["ref_py"] - 1.0) * d["psy"]
    return d

class _Src:
    def __init__(self, path, h):
        self.h = h; self.count = h["bands"]; self.height = h["lines"]; self.width = h["samples"]; self.dtypes = [h["dtype"]]
        dt = np.dtype(h["dtype"]).newbyteorder(">" if h["byte_order"] == 1 else "<")
        shape = {"bsq": (h["bands"], h["lines"], h["samples"]), "bil": (h["lines"], h["bands"], h["samples"]), "bip": (h["lines"], h["samples"], h["bands"])}[h["interleave"]]
        self._mm = np.memmap(path, dtype=dt, mode="r", offset=h["offset"], shape=shape)
        self.res = (h["psx"], h["psy"])
    def read(self, b):
        """band b (1-based) as a (rows, cols) array in native dtype (little-endian copy)"""
        il = self.h["interleave"]
        a = self._mm[b - 1] if il == "bsq" else (self._mm[:, b - 1, :] if il == "bil" else self._mm[:, :, b - 1])
        return np.ascontiguousarray(a).astype(np.dtype(self.h["dtype"]), copy=False)
    def close(self):
        try: del self._mm
        except Exception: pass

class ENVICube:
    def __init__(self, bin_path):
        self.path = bin_path; self.hdr = _hdr(bin_path); self.h = _parse_hdr(self.hdr)
        self.wavelengths = self.h["wavelengths"]; self._src = _Src(bin_path, self.h); self.res = self._src.res
        self.x0, self.y0, self.ps = self.h["x0"], self.h["y0"], self.h["psx"]
    def close(self): self._src.close()
    def _labels(self, regions):
        """int32 (rows, cols): 1-based region index of the polygon whose interior contains the pixel centre, 0 elsewhere"""
        H, W, ps = self._src.height, self._src.width, self.ps
        lab = np.zeros((H, W), np.int32)
        for i, g in enumerate(regions):
            minx, miny, maxx, maxy = g.bounds
            c0 = max(int(np.floor((minx - self.x0) / ps)), 0); c1 = min(int(np.ceil((maxx - self.x0) / ps)), W - 1)
            r0 = max(int(np.floor((self.y0 - maxy) / ps)), 0); r1 = min(int(np.ceil((self.y0 - miny) / ps)), H - 1)
            if c1 < c0 or r1 < r0: continue
            cols = np.arange(c0, c1 + 1); rows = np.arange(r0, r1 + 1)
            xc = self.x0 + (cols + 0.5) * ps; yc = self.y0 - (rows + 0.5) * ps
            XX, YY = np.meshgrid(xc, yc)
            m = contains_xy(g, XX.ravel(), YY.ravel()).reshape(XX.shape)
            sub = lab[r0:r1 + 1, c0:c1 + 1]; sub[m & (sub == 0)] = i + 1
        return lab

def read_plot_tif(path):
    """per-plot GeoTIFF (bands, rows, cols) + georeference (x0, y0 of the upper-left corner, pixel size) + wavelengths"""
    import tifffile, xml.etree.ElementTree as ET
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]; a = t.series[0].asarray()
        if a.ndim == 3 and a.shape[0] != pg.samplesperpixel and a.shape[2] == pg.samplesperpixel: a = np.moveaxis(a, 2, 0)
        tags = {tg.name: tg.value for tg in pg.tags.values()}
        scale = tags.get("ModelPixelScaleTag"); tie = tags.get("ModelTiepointTag")
        x0, y0, ps = (float(tie[3]) - float(tie[0]) * float(scale[0]), float(tie[4]) + float(tie[1]) * float(scale[1]), float(scale[0])) if scale is not None and tie is not None else (np.nan, np.nan, np.nan)
        wl = None; md = tags.get("GDAL_METADATA")
        if md:
            try:
                root = ET.fromstring(md)
                for it in root.iter("Item"):
                    if it.get("name") == "wavelengths_nm": wl = np.array([float(v) for v in it.text.split(",")])
                if wl is None:
                    desc = {int(it.get("sample")): float(it.text.split()[0]) for it in root.iter("Item") if it.get("role") == "description" and it.get("sample") is not None}
                    if desc: wl = np.array([desc[i] for i in sorted(desc)])
            except Exception: pass
    return a, (x0, y0, ps), wl
