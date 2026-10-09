$ErrorActionPreference = "Continue"
# Two-stream network on the COMMON band list (dataset\vnir_common_bands.csv, common_95: 111 bands identical for all flights).
# Starts after the feature-model run (biomass_models.py) has released the CPU; runs next to the per-plot-band run on cuda:1.
$py = "C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe"
$env:PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library\bin;C:\Users\appn\miniforge3\envs\phenoapp_dl;$env:PATH"
$env:CUDA_PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library"
$d = "C:\Users\appn\pipeline_2026-10-09"
$out = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream"
Set-Location $d
function Running($pattern) { @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like $pattern }).Count }
"[{0}] waiting for biomass_models.py to finish" -f (Get-Date -Format "HH:mm:ss")
while ((Running "*biomass_models.py*") -gt 0) { Start-Sleep -Seconds 60 }
"[{0}] feature-model run finished; starting the common-band two-stream runs on cuda:1" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target biomass --epochs 100 --device cuda:1 --bands common | Out-File "$out\dl_biomass_commonbands.stdout.log" -Encoding utf8
"[{0}] biomass (common bands) exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
& $py -u "$d\dl_twostream.py" --target height --epochs 100 --device cuda:1 --bands common | Out-File "$out\dl_height_commonbands.stdout.log" -Encoding utf8
"[{0}] height (common bands) exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
Get-Content "$out\dl_biomass_commonbands.log" -Tail 12
Get-Content "$out\dl_height_commonbands.log" -Tail 12
