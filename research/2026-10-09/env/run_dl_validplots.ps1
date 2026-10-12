$ErrorActionPreference = "Continue"
$E = "C:\Users\appn\miniforge3\envs\phenoapp_dl"; $env:PATH = "$E\Library\bin;$E;$env:PATH"; $env:CUDA_PATH = "$E\Library"
$P = "C:\Users\appn\pipeline_2026-10-09"; Set-Location $P
$D = "D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass_Height_2026-10-09"; $out = "$D\two_stream"
function Running($pat) { @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like $pat }).Count }
# GPU 0: rule B now, then rule A vnir_only; GPU 1: rule A two_stream after the running VNIR-only grouped run
$jobB = Start-Process -FilePath "$E\python.exe" -ArgumentList '-u', "$P\dl_twostream.py", '--target', 'biomass', '--plots', "`"$D\dataset\valid_plots_ruleB.csv`"", '--bands', 'common', '--variants', 'vnir_only,two_stream', '--tag', '__ruleB', '--epochs', '100', '--device', 'cuda:0' -RedirectStandardOutput "$out\dl_ruleB.stdout.log" -RedirectStandardError "$out\dl_ruleB.stderr.log" -WindowStyle Hidden -PassThru
"[{0}] rule B started on cuda:0 (pid {1})" -f (Get-Date -Format "HH:mm:ss"), $jobB.Id
while ((Running "*dl_twostream.py*vnir_only --tag _grouped*") -gt 0) { Start-Sleep -Seconds 60 }
"[{0}] cuda:1 free - rule A two_stream" -f (Get-Date -Format "HH:mm:ss")
$jobA = Start-Process -FilePath "$E\python.exe" -ArgumentList '-u', "$P\dl_twostream.py", '--target', 'biomass', '--plots', "`"$D\dataset\valid_plots_ruleA.csv`"", '--bands', 'common', '--variants', 'two_stream', '--tag', '__ruleA', '--epochs', '100', '--device', 'cuda:1' -RedirectStandardOutput "$out\dl_ruleA_ts.stdout.log" -RedirectStandardError "$out\dl_ruleA_ts.stderr.log" -WindowStyle Hidden -PassThru
$jobB.WaitForExit()
"[{0}] rule B done - rule A vnir_only on cuda:0" -f (Get-Date -Format "HH:mm:ss")
& "$E\python.exe" -u "$P\dl_twostream.py" --target biomass --plots "$D\dataset\valid_plots_ruleA.csv" --bands common --variants vnir_only --tag __ruleA --epochs 100 --device cuda:0 *> "$out\dl_ruleA_vnir.stdout.log"
$jobA.WaitForExit()
"[{0}] all valid-plot network runs done" -f (Get-Date -Format "HH:mm:ss")