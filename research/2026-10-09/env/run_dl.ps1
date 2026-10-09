$ErrorActionPreference = "Continue"
$py = "C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe"
$d = "C:\Users\appn\pipeline_2026-10-09"
$out = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream"
Set-Location $d
"[{0}] biomass: two_stream, lidar_only, vnir_only - 10 folds x 100 epochs on cuda:1" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target biomass --epochs 100 --device cuda:1 | Out-File "$out\dl_biomass.stdout.log" -Encoding utf8
"[{0}] biomass exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
"[{0}] height: two_stream, lidar_only, vnir_only - 10 folds x 100 epochs on cuda:1" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target height --epochs 100 --device cuda:1 | Out-File "$out\dl_height.stdout.log" -Encoding utf8
"[{0}] height exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
Get-Content "$out\dl_biomass.log" -Tail 12
Get-Content "$out\dl_height.log" -Tail 12
