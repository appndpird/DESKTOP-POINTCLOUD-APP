$ErrorActionPreference = "Continue"
# Two-stream on the plots that have every valid band (Muresk maturity flight, 128 plots); bands = all except the first bands (0-415 nm)
# and the water-vapour tail (928-962 nm). Starts when the 138-band Muresk run has finished (GPU memory).
$py = "C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe"
$env:PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library\bin;C:\Users\appn\miniforge3\envs\phenoapp_dl;$env:PATH"
$env:CUDA_PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library"
$d = "C:\Users\appn\pipeline_2026-10-09"
$out = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream"
$cb = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\dataset\vnir_muresk_maturity_validplots_bands.csv"
Set-Location $d
function Running($pattern) { @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like $pattern }).Count }
"[{0}] waiting for the 138-band Muresk run" -f (Get-Date -Format "HH:mm:ss")
while ((Running "*muresk_fullbands*") -gt 0) { Start-Sleep -Seconds 60 }
"[{0}] starting the valid-plots (Muresk maturity, 157 bands) run" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target biomass --datasets muresk_maturity --bands common --bands-csv $cb --variants two_stream,lidar_only,vnir_only --tag _maturity_validplots --epochs 100 --device cuda:1 | Out-File "$out\dl_biomass_commonbands_maturity_validplots.stdout.log" -Encoding utf8
"[{0}] exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
Get-Content "$out\dl_biomass_commonbands_maturity_validplots.log" -Tail 12
