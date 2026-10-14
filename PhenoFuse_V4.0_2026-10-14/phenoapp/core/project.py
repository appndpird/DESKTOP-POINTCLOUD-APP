"""
Application configuration and project state (PhenoFuse 4.0).

A 'project' is a JSON file holding the inputs of one FLIGHT (LAS, VNIR cube, RGB orthomosaic, grid) and the
output locations. Every output of a flight lives next to its LAS in three per-modality folders:

    <flight folder>/plots_lidar/   per-plot LAS (classified when LiDAR v3 is on) + lidar_features.csv (ONE table)
    <flight folder>/plots_vnir/    per-plot VNIR cubes + vnir_features.csv, vnir_spectra.csv, band QC tables, vnir_indices.csv
    <flight folder>/plots_rgb/     per-plot RGB orthomosaic crops + rgb_features.csv
    <flight folder>/Models/        Height models/ and Biomass models/ (one sub-folder per training run)

Picking a different LAS resets every derived path to the new flight's folder (v3 kept the first flight's
paths, so a second flight was written into the first flight's folder).

Project state is shared across all tabs of the UI; tabs read from and write to the same singleton.
"""

from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, asdict, field

LIDAR_DIR, VNIR_DIR, RGB_DIR, MODELS_DIR = "plots_lidar", "plots_vnir", "plots_rgb", "Models"
LIDAR_CSV, VNIR_CSV, RGB_CSV, VNIR_INDICES_CSV = "lidar_features.csv", "vnir_features.csv", "rgb_features.csv", "vnir_indices.csv"


def default_flight_name(las_path: str) -> str:
    """'20250930_NUE_I_DPIRD_Muresk_F_Gobi_LiDAR_CombinedPointCloud.las' -> '20250930_NUE_I_DPIRD_Muresk_F_Gobi'."""
    b = os.path.splitext(os.path.basename(las_path or ""))[0]
    for suf in ("_LiDAR_CombinedPointCloud", "_CombinedPointCloud", "_LiDAR", "_noextent", "_PointCloud", "_pointcloud", "_las"):
        b = re.sub(re.escape(suf) + r"(?=_|$)", "", b)
    return b.strip("_") or "flight"


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _under(p: str, d: str) -> bool:
    return os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(os.path.abspath(d)))


@dataclass
class ProjectState:
    project_file:  str = ""           # where this state was saved/loaded from
    las_path:      str = ""
    grid_path:     str = ""
    canopy_tif:    str = ""
    ortho_tif:     str = ""            # RGB orthomosaic (grid drawing + RGB Features tab)
    work_crs:      str = "EPSG:7850"   # GDA2020 / MGA zone 50 (DPIRD WA)
    use_smrf:      bool = False
    out_dir:       str = ""           # per-plot LAS folder  (plots_lidar)
    out_csv:       str = ""           # LiDAR features table (plots_lidar/lidar_features.csv)
    saved_grid:    str = ""           # aligned grid output (.gpkg)
    # ---- v2 additions ----
    vnir_path:     str = ""           # ENVI .bin hyperspectral orthomosaic
    gcp_csv:       str = ""           # surveyed GCP CSV (E,N,H) for QC
    lastools_dir:  str = ""           # optional LAStools bin folder (free tools)
    ground_mode:   str = "hag"        # "hag" | "exterior"
    region_mode:   str = "whole"      # "whole" | "central"
    band_width:    float = 0.5        # central band width (m)
    vnir_csv:      str = ""           # VNIR features table (plots_vnir/vnir_features.csv)
    vnir_spectra:  str = ""           # per-plot spectra .npz output
    spectral_csv:  str = ""           # per-plot spectral-index table (VNIR Spectral Features tab)
    spectral_sel:  str = ""           # comma-separated index acronyms last computed
    spectral_excl: str = ""           # excluded wavelength ranges text
    plots_vnir_dir: str = ""          # folder of per-plot VNIR cubes
    # ---- v4 (PhenoFuse) additions ----
    rgb_csv:       str = ""           # RGB features table (plots_rgb/rgb_features.csv)
    plots_rgb_dir: str = ""           # folder of per-plot RGB crops
    models_dir:    str = ""           # <flight folder>/Models
    flight_name:   str = ""           # name of this flight inside a dataset (e.g. 20250930_NUE_I_DPIRD_Muresk_F_Gobi)
    trial_name:    str = ""           # trial the flight belongs to (groups the plots of several flights)
    stage:         str = "anthesis"   # growth stage label (anthesis | maturity | ...)
    dataset_dir:   str = ""           # dataset folder used by the Dataset / modelling tabs
    gt_csv:        str = ""           # ground-truth CSV of this flight (Plot_ID + biomass / height columns)

    # runtime-only (never serialized)
    las_loaded:        bool = field(default=False, repr=False)
    canopy_built:      bool = field(default=False, repr=False)

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("las_loaded", "canopy_built"):
            d.pop(k, None)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ProjectState":
        d = json.loads(text)
        valid = {f for f in cls.__dataclass_fields__.keys()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, path: str = ""):
        path = path or self.project_file
        if not path:
            raise ValueError("No project_file path set")
        with open(path, "w") as f:
            f.write(self.to_json())
        self.project_file = path

    @classmethod
    def load(cls, path: str) -> "ProjectState":
        with open(path) as f:
            s = cls.from_json(f.read())
        s.project_file = path
        return s

    # ------------------------------------------------------------------
    @property
    def proj_dir(self) -> str:
        return os.path.dirname(self.las_path) if self.las_path else ""

    _DERIVED = ("canopy_tif", "out_dir", "out_csv", "saved_grid", "vnir_csv", "vnir_spectra", "spectral_csv",
                "plots_vnir_dir", "rgb_csv", "plots_rgb_dir", "models_dir")

    def set_las(self, las_path: str):
        """Point the project at a (new) flight: derived paths of another flight folder are reset, and flight-level
        inputs (VNIR cube, orthomosaic, ground truth) are cleared when they live in the previous flight's folder."""
        las_path = las_path or ""
        old_dir = self.proj_dir
        self.las_path = las_path
        new_dir = self.proj_dir
        if old_dir and new_dir and not _same(old_dir, new_dir):
            for k in self._DERIVED:
                setattr(self, k, "")
            for k in ("vnir_path", "ortho_tif", "gt_csv"):
                p = getattr(self, k)
                if p and _same(os.path.dirname(p), old_dir):
                    setattr(self, k, "")
            self.flight_name = ""
        self.derive_default_paths()

    def derive_default_paths(self, force: bool = False):
        if not self.las_path:
            return
        proj_dir = self.proj_dir
        base = os.path.splitext(os.path.basename(self.las_path))[0]
        for k in self._DERIVED:
            p = getattr(self, k)
            # forced, or a derived path that lives in ANOTHER folder than the LAS (previous flight): re-derive it
            if force or (p and not _under(p, proj_dir)):
                setattr(self, k, "")
        if not self.canopy_tif:
            self.canopy_tif = os.path.join(proj_dir, f"{base}_canopy.tif")
        if not self.out_dir:
            self.out_dir = os.path.join(proj_dir, LIDAR_DIR)
        if not self.out_csv:
            self.out_csv = os.path.join(self.out_dir, LIDAR_CSV)
        if not self.saved_grid:
            self.saved_grid = os.path.join(proj_dir, "aligned_grid.gpkg")
        if not self.plots_vnir_dir:
            self.plots_vnir_dir = os.path.join(proj_dir, VNIR_DIR)
        if not self.vnir_csv:
            self.vnir_csv = os.path.join(self.plots_vnir_dir, VNIR_CSV)
        if not self.vnir_spectra:
            self.vnir_spectra = os.path.join(self.plots_vnir_dir, "vnir_spectra.npz")
        if not self.spectral_csv:
            self.spectral_csv = os.path.join(self.plots_vnir_dir, VNIR_INDICES_CSV)
        if not self.plots_rgb_dir:
            self.plots_rgb_dir = os.path.join(proj_dir, RGB_DIR)
        if not self.rgb_csv:
            self.rgb_csv = os.path.join(self.plots_rgb_dir, RGB_CSV)
        if not self.models_dir:
            self.models_dir = os.path.join(proj_dir, MODELS_DIR)
        if not self.flight_name:
            self.flight_name = default_flight_name(self.las_path)

    # ------------------------------------------------------------------
    def lidar_features_csv(self) -> str:
        """The LiDAR feature table of this flight (new layout, else the v3 plot_metrics.csv next to the LAS)."""
        cands = [self.out_csv]
        if self.proj_dir:
            cands += [os.path.join(self.proj_dir, LIDAR_DIR, LIDAR_CSV), os.path.join(self.proj_dir, "plot_metrics.csv")]
        for c in cands:
            if c and os.path.exists(c):
                return c
        return self.out_csv

    def vnir_features_csv(self) -> str:
        """The VNIR feature table of this flight (new layout, else the v3 <metrics>_vnir_v3.csv)."""
        cands = [self.vnir_csv]
        if self.proj_dir:
            cands += [os.path.join(self.proj_dir, VNIR_DIR, VNIR_CSV), os.path.join(self.proj_dir, "plot_metrics_vnir_v3.csv")]
        if self.out_csv:
            cands.append(os.path.splitext(self.out_csv)[0] + "_vnir_v3.csv")
        for c in cands:
            if c and os.path.exists(c):
                return c
        return self.vnir_csv

    def rgb_features_csv(self) -> str:
        for c in (self.rgb_csv, os.path.join(self.proj_dir, RGB_DIR, RGB_CSV) if self.proj_dir else ""):
            if c and os.path.exists(c):
                return c
        return self.rgb_csv

    def grid_file(self) -> str:
        return self.saved_grid if (self.saved_grid and os.path.exists(self.saved_grid)) else self.grid_path


# Singleton instance (each tab imports and reads/writes this)
_STATE = ProjectState()

def state() -> ProjectState:
    return _STATE

def set_state(new: ProjectState):
    global _STATE
    _STATE = new
