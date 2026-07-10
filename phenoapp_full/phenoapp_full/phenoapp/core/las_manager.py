"""
Core LAS manager.

Loads a LAS file once, optionally runs proper ground classification
(PDAL SMRF) to compute Height-Above-Ground, and caches the result on
disk so subsequent runs are instant.

Public API
----------
LASManager(las_path, work_crs, use_smrf=False, cache_dir=None)
    .load(progress_cb=None)        Load points + (optional) HAG.
    .x, .y, .z                      arrays in working CRS metres
    .hag                            height-above-ground (None if not computed)
    .header                         laspy header (for writing per-plot LAS)
    .points                         laspy structured array (raw point records)
    .crs                            CRS string read from the LAS

Caching
-------
Cache key = (las_path, mtime, file size, use_smrf).
Cached normalized LAS is written next to the LAS as
    <name>.smrf_norm.las
Re-loading is then a fast laspy.read of the cached file.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import hashlib
import tempfile
import subprocess
import numpy as np
import laspy


def _find_pdal_exe() -> str | None:
    """Locate pdal.exe. In a PyInstaller bundle it sits next to the
    bundled DLLs (sys._MEIPASS); otherwise fall back to PATH."""
    if getattr(sys, "frozen", False):
        candidate = os.path.join(sys._MEIPASS, "pdal.exe")
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("pdal") or shutil.which("pdal.exe")


class LASManager:
    def __init__(self, las_path: str, work_crs: str,
                 use_smrf: bool = False, cache_dir: str | None = None):
        self.las_path = las_path
        self.work_crs = work_crs
        self.use_smrf = use_smrf
        self.cache_dir = cache_dir or os.path.dirname(las_path)
        os.makedirs(self.cache_dir, exist_ok=True)

        # populated by load()
        self.x = self.y = self.z = self.hag = None
        self.header = None
        self.points = None
        self.crs = None
        self._loaded = False
        self._smrf_cache_path = self._smrf_cache_filename()

    # ------------------------------------------------------------------
    # Cache filename
    # ------------------------------------------------------------------
    def _smrf_cache_filename(self) -> str:
        base = os.path.splitext(os.path.basename(self.las_path))[0]
        return os.path.join(self.cache_dir, f"{base}.smrf_norm.las")

    def _cache_valid(self) -> bool:
        """SMRF cache is valid if it exists AND is newer than source LAS."""
        if not os.path.exists(self._smrf_cache_path):
            return False
        return os.path.getmtime(self._smrf_cache_path) >= os.path.getmtime(self.las_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, progress_cb=None):
        if self._loaded:
            return self

        def _p(pct, msg):
            if progress_cb: progress_cb(pct, msg)

        if self.use_smrf:
            if self._cache_valid():
                _p(5, "Loading cached normalized LAS...")
                self._read_las(self._smrf_cache_path, expect_hag=True)
            else:
                _p(5, "Running SMRF ground classification (one-time)...")
                self._run_smrf_pipeline(progress_cb=progress_cb)
                _p(85, "Loading normalized LAS...")
                self._read_las(self._smrf_cache_path, expect_hag=True)
        else:
            _p(5, "Loading raw LAS (no ground classification)...")
            self._read_las(self.las_path, expect_hag=False)

        _p(100, f"Loaded {len(self.x):,} points")
        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Read LAS into numpy arrays
    # ------------------------------------------------------------------
    def _read_las(self, path: str, expect_hag: bool):
        las = laspy.read(path)
        self.x = np.asarray(las.x)
        self.y = np.asarray(las.y)
        self.z = np.asarray(las.z)
        self.points = las.points
        self.header = las.header
        try:
            self.crs = las.header.parse_crs()
        except Exception:
            self.crs = None

        if expect_hag and "HeightAboveGround" in las.point_format.dimension_names:
            self.hag = np.asarray(las["HeightAboveGround"])
        elif expect_hag:
            # Fallback: cache had no HAG column, recompute z minus per-tile-min
            self.hag = self.z - float(np.percentile(self.z, 1))
        else:
            self.hag = None

    # ------------------------------------------------------------------
    # SMRF ground classification + HAG via PDAL
    # ------------------------------------------------------------------
    def _run_smrf_pipeline(self, progress_cb=None):
        pdal_exe = _find_pdal_exe()
        if pdal_exe is None:
            raise RuntimeError(
                "pdal.exe not found. The frozen build should include it next to "
                "PhenoApp.exe; for a source run, install via:\n"
                "  conda install -c conda-forge pdal"
            )

        pipeline = {
            "pipeline": [
                self.las_path,
                {"type": "filters.smrf"},          # ground classification
                {"type": "filters.hag_nn"},        # HeightAboveGround dim
                {
                    "type": "writers.las",
                    "filename": self._smrf_cache_path,
                    "extra_dims": "HeightAboveGround=float32",
                    "forward":    "all",
                }
            ]
        }
        if progress_cb:
            progress_cb(10, "PDAL: classifying ground (SMRF)...")

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False, encoding="utf-8") as fh:
            json.dump(pipeline, fh)
            pipeline_json = fh.name
        try:
            # CREATE_NO_WINDOW so the subprocess doesn't flash a console.
            creationflags = 0x08000000 if os.name == "nt" else 0
            proc = subprocess.run(
                [pdal_exe, "pipeline", pipeline_json],
                capture_output=True, text=True, creationflags=creationflags,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pdal pipeline failed (exit {proc.returncode}):\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
        finally:
            try:
                os.unlink(pipeline_json)
            except OSError:
                pass

        if progress_cb:
            progress_cb(80, "Ground classification complete.")

    # ------------------------------------------------------------------
    def get_z_for_metrics(self) -> np.ndarray:
        """Return whichever Z-array should be used for height metrics."""
        return self.hag if self.hag is not None else self.z


def quick_summary(mgr: LASManager) -> dict:
    """A tiny helper for the UI to display sanity stats."""
    if not mgr._loaded:
        return {}
    return {
        "n_points": int(len(mgr.x)),
        "x_range": (float(mgr.x.min()), float(mgr.x.max())),
        "y_range": (float(mgr.y.min()), float(mgr.y.max())),
        "z_range": (float(mgr.z.min()), float(mgr.z.max())),
        "has_hag": mgr.hag is not None,
        "crs": str(mgr.crs) if mgr.crs else "unknown",
    }
