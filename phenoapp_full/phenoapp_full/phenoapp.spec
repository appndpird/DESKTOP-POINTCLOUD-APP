# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PhenoApp. Build with:
       pyinstaller phenoapp.spec --noconfirm
   IMPORTANT: invoke from a shell where <conda-env>/Library/bin is on PATH so
   PyInstaller's binary-dep scanner can resolve proj_9.dll/gdal.dll/etc.
"""
import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = []
binaries = []
hiddenimports = []

datas += [("phenoapp/assets", "phenoapp/assets")]
datas += [("phenoapp/docs",   "phenoapp/docs")]

for pkg in ("pyproj", "rasterio", "fiona", "shapely", "geopandas",
            "laspy", "pyvista", "pyvistaqt"):
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

# Bundle PROJ + GDAL data dirs so the rthooks can point at them.
_env_root   = os.path.dirname(sys.executable)
_proj_share = os.path.join(_env_root, "Library", "share", "proj")
if os.path.isdir(_proj_share):
    datas.append((_proj_share, "Library/share/proj"))
_gdal_share = os.path.join(_env_root, "Library", "share", "gdal")
if os.path.isdir(_gdal_share):
    datas.append((_gdal_share, "Library/share/gdal"))

# Bundle pdal.exe at bundle root. las_manager._run_smrf_pipeline shells out to
# it (subprocess) instead of using python-pdal, because conda's libpdalpython
# fails DllMain inside the PyInstaller bundle. pdal.exe loads pdalcpp.dll +
# transitive deps from the same directory, which already contain everything it
# needs (PyInstaller's auto-deps brought them in via libpdalpython.pyd).
_pdal_exe = os.path.join(_env_root, "Library", "bin", "pdal.exe")
if os.path.isfile(_pdal_exe):
    binaries.append((_pdal_exe, "."))

hiddenimports += [
    "scipy.spatial", "scipy.signal", "scipy.stats",
    "scipy.ndimage", "scipy.ndimage._filters", "scipy.ndimage._morphology",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5.QtPrintSupport",
]

a = Analysis(
    ['run.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhenoApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='phenoapp/assets/appn_logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PhenoApp',
)
