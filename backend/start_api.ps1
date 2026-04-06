# Run from repo: always start API from the backend folder so imports resolve.
Set-Location $PSScriptRoot
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Error "venv not found. Run: python -m venv venv ; .\venv\Scripts\Activate.ps1 ; pip install -r requirements.txt"
    exit 1
}
. .\venv\Scripts\Activate.ps1
Write-Host "Starting API on http://127.0.0.1:8000 — if /api/franchise/* 404s, stop other uvicorn/Python using port 8000 first."
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
