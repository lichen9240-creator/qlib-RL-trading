param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($Python) {
        & $Python -m venv .venv
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 --seed .venv
    } else {
        py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python 3.12 environment failed.' }
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installing requirements failed.' }
    $env:PYTHONUTF8 = '1'
    & $taskPython code/apply_compatibility.py
    if ($LASTEXITCODE -ne 0) { throw 'Applying compatibility fixes failed.' }
    & $taskPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'The CUDA verification failed.' }
} finally {
    Pop-Location
}
