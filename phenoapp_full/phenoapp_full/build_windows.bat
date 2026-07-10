@echo off
REM Build PhenoApp.exe on Windows
REM Run from the directory that contains this file.

echo === PhenoApp Windows build ===
echo.

REM Make sure we're in a conda env with all deps
where pyinstaller >nul 2>nul
if errorlevel 1 (
    echo ERROR: pyinstaller not found. Install it with:
    echo     pip install pyinstaller
    exit /b 1
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

pyinstaller phenoapp.spec --noconfirm
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo === Build complete ===
echo Run:  dist\PhenoApp\PhenoApp.exe
echo.
