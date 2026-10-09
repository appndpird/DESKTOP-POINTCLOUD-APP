$ErrorActionPreference = "Continue"
$log = "C:\Users\appn\scratch_2026-10-09\conda_setup.log"
function L($m) { $s = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m; Add-Content -Path $log -Value $s -Encoding utf8; Write-Output $s }

# 1. Miniforge (conda with the conda-forge channel; BSD licence, no defaults-channel terms) - user-level install
$dl = "C:\Users\appn\downloads_miniforge"; New-Item -ItemType Directory -Force $dl | Out-Null
$inst = "$dl\Miniforge3-Windows-x86_64.exe"
$pre = "C:\Users\appn\miniforge3"
if (-not (Test-Path "$pre\Scripts\conda.exe")) {
    L "downloading Miniforge3-Windows-x86_64.exe from github.com/conda-forge/miniforge (latest release)"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $inst -UseBasicParsing
    L ("downloaded {0:N1} MB" -f ((Get-Item $inst).Length / 1MB))
    L "installing silently to $pre (JustMe, no PATH change, not registered as system Python)"
    $p = Start-Process -FilePath $inst -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$pre" -Wait -PassThru
    L "installer exit code $($p.ExitCode)"
}
$conda = "$pre\Scripts\conda.exe"
if (-not (Test-Path $conda)) { L "conda.exe not found - abort"; exit 1 }
L "conda: $(& $conda --version)"

# 2. environment
$envpy = "$pre\envs\phenoapp_dl\python.exe"
if (-not (Test-Path $envpy)) {
    L "creating env phenoapp_dl (python 3.11, conda-forge)"
    & $conda create -y -n phenoapp_dl python=3.11 | Out-File $log -Append -Encoding utf8
}
L "env python: $(& $envpy --version)"

# 3. libraries (pip inside the env; torch from the official cu124 index)
L "pip: torch 2.6.0 cu124 (about 2.5 GB download)"
& $envpy -m pip install --no-cache-dir --disable-pip-version-check torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 | Out-File $log -Append -Encoding utf8
L "pip: spconv-cu124 2.3.8 + pipeline libraries"
& $envpy -m pip install --no-cache-dir --disable-pip-version-check spconv-cu124==2.3.8 numpy pandas scipy scikit-learn laspy lazrs shapely tifffile pyshp matplotlib xgboost lightgbm openpyxl | Out-File $log -Append -Encoding utf8

# 4. import test
L "import test"
& $envpy -c "import torch, spconv.pytorch as s, numpy, laspy, sklearn, pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), '| spconv', getattr(s, '__version__', 'ok'), '| numpy', numpy.__version__)" | Out-File $log -Append -Encoding utf8
& $envpy -m pip freeze | Out-File "C:\Users\appn\scratch_2026-10-09\phenoapp_dl_requirements.txt" -Encoding utf8
L "SETUP DONE"
