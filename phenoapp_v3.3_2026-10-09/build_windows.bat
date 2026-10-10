@echo off
REM Build PhenoApp v2 on Windows.
REM IMPORTANT: uses the phenoapp environment's OWN PyInstaller by full path.
REM A bare `pyinstaller` can resolve to another Python on PATH (e.g.
REM AppData\...\Python311) whose site-packages lack PyQt5 - the build then
REM "succeeds" but the exe dies with ModuleNotFoundError: PyQt5.
REM ENVROOT must point at the env that has PyQt5 + pyinstaller installed.
REM On this PC that is the Python 3.12 venv at %USERPROFILE%\phenoapp-venv.

set ENVROOT=C:\Users\appn\phenoapp-venv
set PATH=%ENVROOT%;%ENVROOT%\Library\bin;%ENVROOT%\Scripts;%PATH%

echo === PhenoApp v2 Windows build (env: %ENVROOT%) ===
if not exist "%ENVROOT%\Scripts\pyinstaller.exe" (
    echo ERROR: %ENVROOT%\Scripts\pyinstaller.exe not found.
    echo Install into the env with:  pip install pyinstaller
    exit /b 1
)

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
echo   1. dist\PhenoApp\_internal\PyQt5 folder exists
echo   2. dist\PhenoApp\PhenoApp.exe opens and phenoapp_crash.log stays clean
echo.
