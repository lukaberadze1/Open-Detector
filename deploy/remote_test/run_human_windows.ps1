# Sets up Open Detector and tests the built-in human profile on Windows. Run from the unzipped folder:
#   powershell -ExecutionPolicy Bypass -File .\run_human_windows.ps1
# Add -NoFull under 16 GB of memory, or -TryOne for a quick run on one document first.
param([switch]$NoFull, [switch]$TryOne)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it first (see START_HUMAN.md), open a new PowerShell window, then run this again:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Write-Host "Creating the Python 3.12 environment..."
    uv venv --python 3.12 .venv
}
Write-Host "Installing packages (the first time downloads about 1 GB)..."
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
$env:PYTHONUTF8 = "1"
$extra = @()
if ($NoFull) { $extra += "--no-full" }
if ($TryOne) { $extra += "--documents", "1" }
.venv\Scripts\python.exe deploy\remote_test\run_human.py @extra
