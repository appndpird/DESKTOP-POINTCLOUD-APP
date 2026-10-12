$ErrorActionPreference = "Continue"
# Muresk-only two-stream with EVERY usable band per plot (excluded ranges included); starts when the 138-band Muresk run has finished (GPU memory).
$py = "C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe"
$env:PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library\bin;C:\Users\appn\miniforge3\envs\phenoapp_dl;$env:PATH"
$env:CUDA_PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library"
$d = "C:\Users\appn\pipeline_2026-10-09"
$out = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream"
Set-Location $d
function Running($pattern) { @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like $pattern }).Count }
"[{0}] waiting for the 138-band Muresk run" -f (Get-Date -Format "HH:mm:ss")
while ((Running "*muresk_fullbands*") -gt 0) { Start-Sleep -Seconds 60 }
"[{0}] starting Muresk all-bands run" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target biomass --datasets muresk_anthesis,muresk_maturity --bands per_plot_all --variants two_stream,lidar_only,vnir_only --tag _muresk --epochs 100 --device cuda:1 | Out-File "$out\dl_biomass_allbands_muresk.stdout.log" -Encoding utf8
"[{0}] exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
Get-Content "$out\dl_biomass_allbands_muresk.log" -Tail 14
