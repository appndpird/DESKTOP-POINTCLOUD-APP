# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Plant Phenotyping app.

Build with:
    pyinstaller phenoapp.spec --noconfirm
or directly:
    pyinstaller --noconfirm --windowed --name PhenoApp \\
        --add-data "phenoapp/assets:phenoapp/assets" \\
        --add-data "phenoapp/docs:phenoapp/docs" \\
        --hidden-import rasterio.sample --hidden-import rasterio._shim \\
        --hidden-import rasterio.vrt --hidden-import rasterio.control \\
        --hidden-import laspy.lib --hidden-import laspy.copc \\
        --collect-data pyproj --collect-data rasterio \\
        run.py

Output ends up in dist/PhenoApp/  (or dist/PhenoApp.exe on Windows --onefile).
"""
import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Bundle data files for libraries that need them at runtime
datas = []
datas += [("phenoapp/assets", "phenoapp/assets")]
datas += [("phenoapp/docs",   "phenoapp/docs")]
datas += collect_data_files("rasterio")
datas += collect_data_files("pyproj")
datas += collect_data_files("geopandas")
datas += collect_data_files("shapely")
datas += collect_data_files("fiona", include_py_files=False) if True else []

hiddenimports = []
hiddenimports += collect_submodules("rasterio")
hiddenimports += collect_submodules("laspy")
hiddenimports += collect_submodules("pyproj")
hiddenimports += [
    "scipy.spatial", "scipy.signal", "scipy.stats",
    "matplotlib.backends.backend_qt5agg",
    "PyQt5.QtPrintSupport",
]

a = Analysis(
    ['run.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter'],   # not needed; saves ~30 MB
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---- One-folder build (recommended; faster startup, easier to debug) ----
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhenoApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                            # GUI app, no console
    icon='phenoapp/assets/appn_logo.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhenoApp',
)
