$ErrorActionPreference = "Continue"
# Two-stream reruns with folds GROUPED BY PHYSICAL PLOT (both flights of a plot in the same fold), common band list.
# Height first (the earlier height CV was optimistic: both flights share one ruler value per plot), then biomass.
$py = "C:\Users\appn\miniforge3\envs\phenoapp_dl\python.exe"
$env:PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library\bin;C:\Users\appn\miniforge3\envs\phenoapp_dl;$env:PATH"
$env:CUDA_PATH = "C:\Users\appn\miniforge3\envs\phenoapp_dl\Library"
$d = "C:\Users\appn\pipeline_2026-10-09"
$out = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09\two_stream"
Set-Location $d
"[{0}] height, grouped folds, common bands, two_stream + lidar_only" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target height --epochs 100 --device cuda:1 --bands common --variants two_stream,lidar_only --tag _grouped | Out-File "$out\dl_height_commonbands_grouped.stdout.log" -Encoding utf8
"[{0}] height exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
"[{0}] biomass, grouped folds, common bands, two_stream + lidar_only" -f (Get-Date -Format "HH:mm:ss")
& $py -u "$d\dl_twostream.py" --target biomass --epochs 100 --device cuda:1 --bands common --variants two_stream,lidar_only --tag _grouped | Out-File "$out\dl_biomass_commonbands_grouped.stdout.log" -Encoding utf8
"[{0}] biomass exit {1}" -f (Get-Date -Format "HH:mm:ss"), $LASTEXITCODE
Get-Content "$out\dl_height_commonbands_grouped.log" -Tail 8
Get-Content "$out\dl_biomass_commonbands_grouped.log" -Tail 12
