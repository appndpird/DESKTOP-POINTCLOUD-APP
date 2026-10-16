@echo off
REM Build PhenoFuse on Windows.
REM IMPORTANT: uses the phenoapp environment's OWN PyInstaller by full path.
REM A bare `pyinstaller` can resolve to another Python on PATH (e.g.
REM AppData\...\Python311) whose site-packages lack PyQt5 - the build then
REM "succeeds" but the exe dies with ModuleNotFoundError: PyQt5.
REM ENVROOT must point at the env that has PyQt5 + pyinstaller installed.
REM Default: the phenoapp conda env of the current user. Override with
REM     set ENVROOT=C:\path\to\env  (before calling this script)

if "%ENVROOT%"=="" set ENVROOT=%USERPROFILE%\.conda\envs\phenoapp
set PATH=%ENVROOT%;%ENVROOT%\Library\bin;%ENVROOT%\Scripts;%PATH%

echo === PhenoFuse Windows build (env: %ENVROOT%) ===
if not exist "%ENVROOT%\Scripts\pyinstaller.exe" (
    echo ERROR: %ENVROOT%\Scripts\pyinstaller.exe not found.
    echo Set ENVROOT to the conda env that has PyQt5, rasterio, laspy, scikit-learn and pyinstaller, e.g.
    echo     set ENVROOT=C:\Users\^<you^>\.conda\envs\phenoapp
    echo Install PyInstaller into it with:  pip install pyinstaller
    exit /b 1
)

cd /d "%~dp0"
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

"%ENVROOT%\Scripts\pyinstaller.exe" phenoapp.spec --noconfirm
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo === Build complete ===
echo Verify before shipping:
echo   1. dist\PhenoFuse\_internal\PyQt5 folder exists
echo   2. dist\PhenoFuse\PhenoFuse.exe opens and phenoapp_crash.log stays clean
echo.
