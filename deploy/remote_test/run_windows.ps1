# Sets up Open Detector and runs the remote tests on Windows. Run from the unzipped folder:
#   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
# Add -NoGenerate to skip step 4 (it needs a Gemini key in .env).
param([switch]$NoGenerate)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it first (see START_HERE.md), open a new PowerShell window, then run this again:"
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
if ($NoGenerate) { $extra += "--no-generate" }
.venv\Scripts\python.exe deploy\remote_test\run_tests.py @extra
