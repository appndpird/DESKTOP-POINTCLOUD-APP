"""
File-name conventions of the per-flight feature tables (PhenoFuse 4.0) and their v3 fall-backs.

New layout (one folder per modality next to the LAS):
    plots_lidar/lidar_features.csv
    plots_vnir/vnir_features.csv  + vnir_spectra.csv + vnir_band_valid_fraction.csv + vnir_band_usable.csv
                                  + vnir_provenance.json (excluded wavelength ranges, band policy)
    plots_rgb/rgb_features.csv
v3 layout (still readable):
    plot_metrics.csv (+ plot_metrics_lidar_v3.csv), plot_metrics_vnir_v3.csv, plot_metrics_vnir_v3_spectra.csv,
    plot_metrics_vnir_v3_band_valid_fraction.csv
"""
from __future__ import annotations
import os
import json

VNIR_SIDECARS = {"spectra": "vnir_spectra.csv", "valid_fraction": "vnir_band_valid_fraction.csv",
                 "usable": "vnir_band_usable.csv", "provenance": "vnir_provenance.json"}
_V3_SIDECARS = {"spectra": "_vnir_v3_spectra.csv", "valid_fraction": "_vnir_v3_band_valid_fraction.csv",
                "usable": "_vnir_v3_band_usable.csv", "provenance": "_vnir_v3_provenance.json"}


def vnir_sidecar(vnir_csv: str, kind: str) -> str:
    """Path of a VNIR side table for a features CSV in either layout (may not exist)."""
    if not vnir_csv:
        return ""
    d, fn = os.path.split(vnir_csv)
    if fn.endswith("_vnir_v3.csv"):
        return os.path.join(d, fn.replace("_vnir_v3.csv", _V3_SIDECARS[kind]))
    return os.path.join(d, VNIR_SIDECARS[kind])


def existing(path: str) -> str:
    return path if (path and os.path.exists(path)) else ""


def read_excluded(vnir_csv: str, default=None):
    """Excluded wavelength ranges recorded with a VNIR feature table (vnir_provenance.json), else `default`."""
    p = existing(vnir_sidecar(vnir_csv, "provenance"))
    if p:
        try:
            with open(p) as f:
                ex = json.load(f).get("excluded_nm")
            if ex:
                return [(float(a), float(b)) for a, b in ex]
        except Exception:
            pass
    if default is None:
        from .spectral_indices import DEFAULT_EXCLUDED_NM
        return DEFAULT_EXCLUDED_NM
    return default


def write_provenance(vnir_csv: str, **meta):
    p = vnir_sidecar(vnir_csv, "provenance")
    try:
        with open(p, "w") as f:
            json.dump(meta, f, indent=1, default=str)
    except Exception:
        pass
    return p
