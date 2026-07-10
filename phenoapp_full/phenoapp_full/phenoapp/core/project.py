"""
Application configuration and project state.

A 'project' is just a JSON file holding:
  - LAS path
  - Grid path
  - Canopy TIF path
  - Working CRS
  - SMRF flag
  - Output dirs

Project state is shared across all tabs of the UI; tabs read from and
write to the same singleton.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict, field


@dataclass
class ProjectState:
    project_file:  str = ""           # where this state was saved/loaded from
    las_path:      str = ""
    grid_path:     str = ""
    canopy_tif:    str = ""
    ortho_tif:     str = ""            # optional RGB orthomosaic (for grid drawing)
    work_crs:      str = "EPSG:7850"   # GDA2020 / MGA zone 50 (DPIRD WA)
    use_smrf:      bool = False
    out_dir:       str = ""           # for per-plot LAS
    out_csv:       str = ""           # for traits CSV
    saved_grid:    str = ""           # aligned grid output (.gpkg)

    # runtime-only (never serialized)
    las_loaded:        bool = field(default=False, repr=False)
    canopy_built:      bool = field(default=False, repr=False)

    def to_json(self) -> str:
        d = asdict(self)
        # strip runtime-only fields
        for k in ("las_loaded", "canopy_built"):
            d.pop(k, None)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ProjectState":
        d = json.loads(text)
        # filter unknown keys to be forward-compatible
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

    # Defaults derived from LAS path
    def derive_default_paths(self):
        if not self.las_path:
            return
        proj_dir = os.path.dirname(self.las_path)
        base = os.path.splitext(os.path.basename(self.las_path))[0]
        if not self.canopy_tif:
            self.canopy_tif = os.path.join(proj_dir, f"{base}_canopy.tif")
        if not self.out_dir:
            self.out_dir = os.path.join(proj_dir, "plots_las")
        if not self.out_csv:
            self.out_csv = os.path.join(proj_dir, "plot_metrics.csv")
        if not self.saved_grid:
            self.saved_grid = os.path.join(proj_dir, "aligned_grid.gpkg")


# Singleton instance (each tab imports and reads/writes this)
_STATE = ProjectState()

def state() -> ProjectState:
    return _STATE

def set_state(new: ProjectState):
    global _STATE
    _STATE = new
