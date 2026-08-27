"""Runtime hook injected by PyInstaller. Runs before any app import.
Adds the bundled conda Library/bin to the DLL search path (so pyproj/rasterio/
fiona/gdal can resolve their native deps) WITHOUT putting them in the app root
(which would clash with Qt5/VTK's own copies of zlib/libpng/icu/libxml2)."""
import os, sys

if hasattr(sys, "_MEIPASS"):
    _root    = sys._MEIPASS
    _lib_bin = os.path.join(_root, "Library", "bin")

    if os.path.isdir(_lib_bin):
        try:
            os.add_dll_directory(_lib_bin)
        except (AttributeError, OSError):
            pass
        # Also prepend to PATH so child processes (e.g. PDAL CLI) inherit it.
        os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")

    _proj = os.path.join(_root, "Library", "share", "proj")
    if os.path.isdir(_proj):
        os.environ.setdefault("PROJ_DATA", _proj)
        os.environ.setdefault("PROJ_LIB",  _proj)

    _gdal = os.path.join(_root, "Library", "share", "gdal")
    if os.path.isdir(_gdal):
        os.environ.setdefault("GDAL_DATA", _gdal)
