param(
    [ValidateSet('train', 'evaluate', 'verify', 'analyze')][string]$Stage = 'evaluate',
    [string]$Python = '',
    [string[]]$ScriptArguments = @()
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$taskPython = $Python
if (-not $taskPython) { $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 to create the Python environment.' }
$scriptName = @{ train = 'train.py'; evaluate = 'evaluate.py'; verify = 'verify_data.py'; analyze = 'analyze_results.py' }[$Stage]
& $taskPython -u (Join-Path $PSScriptRoot "code\$scriptName") @ScriptArguments
if ($LASTEXITCODE -ne 0) { throw "$scriptName failed with exit code $LASTEXITCODE" }
