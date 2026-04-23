# PowerShell: run project scripts with C:\venv\cv_ca2 (no `cd /d` — that is cmd.exe only).
# Usage: .\run_with_venv.ps1 pneumonia_classification.py --baseline
$venvPy = "C:\venv\cv_ca2\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error "Missing $venvPy. Create it: py -3.13 -m venv C:\venv\cv_ca2"
    exit 1
}
Set-Location -LiteralPath $PSScriptRoot
& $venvPy @args
